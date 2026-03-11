"""
流水线编排服务（对标原 SearchService._run_pipeline）。

支持：
- 完整流水线执行
- 只运行指定阶段
- 从任意阶段继续（自动复用历史输出）
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import ResearchConfig
from ..models import StageMetadata, StageResult
from ..services.events import event_bus
from ..services.knowledge_base import knowledge_base_service
from ..services.llm import LLMService
from ..services.output import output_service
from ..stages import STAGE_REGISTRY
from ..stages.base import StageContext

logger = logging.getLogger(__name__)


class PipelineService:
    async def run(
        self,
        config: ResearchConfig,
        *,
        session_dir: Path | None = None,
        session_id: str = "default",
        base_dir: Path | None = None,
    ) -> dict[int, StageResult]:
        """
        执行研究流水线。

        参数:
            config: 完整配置
            session_dir: 指定 session 目录（用于断点续跑）
            session_id: 事件 ID
            base_dir: 知识库和输出的基础目录
        """
        pipe_cfg = config.pipeline
        base = base_dir or Path(".")
        output_base = base / pipe_cfg.output_dir

        # 确定 session 目录
        if session_dir:
            if not session_dir.is_dir():
                raise FileNotFoundError(f"指定的 session 目录不存在: {session_dir}")
            logger.info("使用已有 session 目录: %s", session_dir)
        elif pipe_cfg.continue_from:
            session_dir = output_service.resolve_session_dir(
                output_base,
                pipe_cfg.continue_from,
                base_dir=base,
            )
            logger.info("沿指定 session 继续: %s", session_dir)
        elif pipe_cfg.resume_from:
            session_dir = output_service.find_latest_session(output_base)
            if session_dir:
                logger.info("从最新 session 继续: %s", session_dir)
            else:
                session_dir = output_service.create_session_dir(output_base)
        else:
            session_dir = output_service.create_session_dir(output_base)

        # 保存配置快照
        output_service.save_session_config(session_dir, asdict(config))

        # 加载知识库
        knowledge = knowledge_base_service.load(config.knowledge_base, base)

        # 构建 LLM 服务
        llm = LLMService(config.llm)

        event_bus.publish(session_id, "pipeline_started", f"流水线启动，输出目录: {session_dir}")
        if not llm.is_configured:
            event_bus.publish(session_id, "llm_warning", "LLM 未配置，将使用降级模式。")

        # 确定要运行的阶段
        selected_stages = sorted(set(pipe_cfg.stages))
        force_rerun_from = pipe_cfg.resume_from

        # 收集前序阶段输出（用于断点续跑）
        previous_outputs: dict[int, dict[str, Any]] = {}
        # 按序执行各阶段
        results: dict[int, StageResult] = {}
        stages_status: dict[int, str] = {}

        if pipe_cfg.continue_from:
            force_rerun_from = self._find_first_unfinished_stage(session_dir, selected_stages)
            if force_rerun_from is None:
                for stage_num in selected_stages:
                    reused = self._load_completed_stage_result(session_dir, stage_num)
                    if reused is None:
                        continue
                    results[stage_num] = reused
                    stages_status[stage_num] = reused.metadata.status
                output_service.write_summary(session_dir, stages_status, pipeline_status="completed")
                event_bus.publish(session_id, "pipeline_finished", "指定阶段均已完成，无需继续执行。")
                return results

            logger.info("检测到第一个未完成阶段: Stage %d", force_rerun_from)

        if force_rerun_from:
            for stage_num in range(1, force_rerun_from):
                reused = self._load_completed_stage_result(session_dir, stage_num)
                if reused is None:
                    continue
                previous_outputs[stage_num] = reused.data
                stages_status[stage_num] = reused.metadata.status
                logger.info("复用 Stage %d 历史输出", stage_num)
                if stage_num in selected_stages:
                    results[stage_num] = reused
            selected_stages = [s for s in selected_stages if s >= force_rerun_from]

        output_service.write_summary(session_dir, stages_status, pipeline_status="running")

        for stage_num in selected_stages:
            stage_cls = STAGE_REGISTRY.get(stage_num)
            if not stage_cls:
                logger.warning("未注册的阶段: %d", stage_num)
                continue

            stage = stage_cls()
            ctx = StageContext(
                config=config,
                llm=llm,
                knowledge=knowledge,
                session_dir=session_dir,
                previous_outputs=previous_outputs,
            )

            event_bus.publish(
                session_id,
                f"stage_{stage_num}_started",
                f"Stage {stage_num} ({stage.stage_name}) 开始执行",
            )
            output_service.write_summary(
                session_dir,
                stages_status,
                pipeline_status="running",
                current_stage=stage_num,
            )

            result = await stage.execute(
                ctx,
                allow_existing_output=force_rerun_from is None,
            )
            results[stage_num] = result
            previous_outputs[stage_num] = result.data

            status = result.metadata.status
            stages_status[stage_num] = status
            output_service.write_summary(session_dir, stages_status, pipeline_status="running")
            event_bus.publish(
                session_id,
                f"stage_{stage_num}_finished",
                f"Stage {stage_num} ({stage.stage_name}) 完成 [{status}]",
                {
                    "status": status,
                    "duration_ms": result.metadata.duration_ms,
                    "degraded": result.metadata.degraded,
                },
            )

            if not result.success and status == "failed":
                output_service.write_summary(session_dir, stages_status, pipeline_status="failed")
                event_bus.publish(session_id, "pipeline_failed",
                                 f"Stage {stage_num} 失败，流水线终止。")
                break

        # 写入最终摘要
        output_service.write_final_summary(session_dir, stages_status)
        event_bus.publish(session_id, "pipeline_finished", "流水线执行完毕。")

        return results

    @staticmethod
    def _load_stage_output(session_dir: Path, stage_num: int) -> dict[str, Any] | None:
        """尝试加载某阶段的历史 JSON 输出。"""
        stage_dir = session_dir / f"stage_{stage_num}"
        if not stage_dir.is_dir():
            return None
        for json_file in stage_dir.glob("*.json"):
            if json_file.name == "metadata.json":
                continue
            try:
                return json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
        return None

    @staticmethod
    def _load_stage_metadata(session_dir: Path, stage_num: int) -> dict[str, Any] | None:
        meta_path = session_dir / f"stage_{stage_num}" / "metadata.json"
        if not meta_path.is_file():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @classmethod
    def _load_completed_stage_result(cls, session_dir: Path, stage_num: int) -> StageResult | None:
        metadata = cls._load_stage_metadata(session_dir, stage_num)
        if metadata is None or metadata.get("status") != "completed":
            return None

        data = cls._load_stage_output(session_dir, stage_num)
        if data is None:
            return None

        stage_name = metadata.get("stage_name", f"stage_{stage_num}")
        md_path = session_dir / f"stage_{stage_num}" / f"{stage_name}_report.md"
        markdown = "(从历史输出复用)"
        if md_path.is_file():
            try:
                markdown = md_path.read_text(encoding="utf-8")
            except OSError:
                pass

        return StageResult(
            stage_number=stage_num,
            stage_name=stage_name,
            success=True,
            data=data,
            metadata=StageMetadata(**metadata),
            markdown=markdown,
        )

    @classmethod
    def _find_first_unfinished_stage(cls, session_dir: Path, stages: list[int]) -> int | None:
        for stage_num in sorted(stages):
            metadata = cls._load_stage_metadata(session_dir, stage_num)
            if metadata is None or metadata.get("status") != "completed":
                return stage_num
            if cls._load_stage_output(session_dir, stage_num) is None:
                return stage_num
        return None


pipeline_service = PipelineService()
