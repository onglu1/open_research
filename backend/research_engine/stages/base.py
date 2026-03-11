"""
阶段基类（对标原 CompanyCollector 的 timed_collect 模式）。

每个阶段统一实现：
- check_existing_output: 断点续跑
- run: 核心逻辑
- fallback: 降级策略
- save_output: 双格式输出（JSON + Markdown）
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from ..config import ResearchConfig
from ..models import KnowledgeBaseContent, StageMetadata, StageResult
from ..services.llm import LLMService

logger = logging.getLogger(__name__)


@dataclass
class StageContext:
    """阶段执行上下文，由 PipelineService 创建并传入。"""
    config: ResearchConfig
    llm: LLMService
    knowledge: KnowledgeBaseContent
    session_dir: Path
    previous_outputs: dict[int, dict[str, Any]] = field(default_factory=dict)

    def get_previous(self, stage_number: int) -> dict[str, Any]:
        return self.previous_outputs.get(stage_number, {})


class StageBase:
    """五阶段流水线的抽象基类。"""
    stage_number: int = 0
    stage_name: str = "base"

    @property
    def output_dir_name(self) -> str:
        return f"stage_{self.stage_number}"

    # ── 断点续跑 ──────────────────────────────────────────────

    def check_existing_output(self, session_dir: Path) -> dict[str, Any] | None:
        """检查是否已有该阶段的历史输出，有则直接返回。"""
        json_path = session_dir / self.output_dir_name / f"{self.stage_name}.json"
        meta_path = session_dir / self.output_dir_name / "metadata.json"
        if json_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") == "completed":
                    logger.info("Stage %d 已有完成的历史输出，复用: %s", self.stage_number, json_path)
                    return json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("读取历史输出失败: %s", e)
        return None

    # ── 核心执行 ──────────────────────────────────────────────

    async def run(self, ctx: StageContext) -> StageResult:
        """子类必须实现此方法。"""
        raise NotImplementedError

    # ── 降级策略 ──────────────────────────────────────────────

    async def fallback(self, ctx: StageContext, error: Exception) -> StageResult:
        """默认降级：返回空结果并标记 degraded。"""
        logger.error("Stage %d 降级: %s", self.stage_number, error)
        meta = StageMetadata(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            status="degraded",
            degraded=True,
            degraded_reason=str(error),
        )
        return StageResult(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            success=False,
            data={},
            metadata=meta,
            markdown=f"# Stage {self.stage_number}: {self.stage_name}\n\n**降级**: {error}\n",
        )

    # ── 带计时的执行包装 ──────────────────────────────────────

    async def execute(self, ctx: StageContext, *, allow_existing_output: bool = True) -> StageResult:
        """完整执行流程：检查历史 -> run -> 降级 -> 保存。"""
        existing = self.check_existing_output(ctx.session_dir) if allow_existing_output else None
        if existing is not None:
            meta = StageMetadata(
                stage_number=self.stage_number,
                stage_name=self.stage_name,
                status="completed",
            )
            return StageResult(
                stage_number=self.stage_number,
                stage_name=self.stage_name,
                success=True,
                data=existing,
                metadata=meta,
                markdown="(从历史输出复用)",
            )

        start = perf_counter()
        now_str = datetime.now(timezone.utc).isoformat()
        try:
            result = await self.run(ctx)
            elapsed = int((perf_counter() - start) * 1000)
            result.metadata.started_at = now_str
            result.metadata.finished_at = datetime.now(timezone.utc).isoformat()
            result.metadata.duration_ms = elapsed
            result.metadata.status = "completed" if result.success else "degraded"
        except Exception as error:
            elapsed = int((perf_counter() - start) * 1000)
            result = await self.fallback(ctx, error)
            result.metadata.started_at = now_str
            result.metadata.finished_at = datetime.now(timezone.utc).isoformat()
            result.metadata.duration_ms = elapsed

        self.save_output(ctx.session_dir, result)
        return result

    # ── 输出持久化 ────────────────────────────────────────────

    def save_output(self, session_dir: Path, result: StageResult) -> None:
        """保存 JSON + Markdown + metadata 三文件到阶段目录。"""
        stage_dir = session_dir / self.output_dir_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        # JSON 数据
        json_path = stage_dir / f"{self.stage_name}.json"
        json_path.write_text(
            json.dumps(result.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Markdown 报告
        md_path = stage_dir / f"{self.stage_name}_report.md"
        md_path.write_text(result.markdown, encoding="utf-8")

        # 元信息
        meta_path = stage_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(asdict(result.metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Stage %d 输出已保存到 %s (状态: %s, 耗时: %dms)",
            self.stage_number, stage_dir, result.metadata.status, result.metadata.duration_ms,
        )
