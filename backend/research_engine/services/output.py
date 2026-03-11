"""
输出归档服务：按时间戳创建 session 目录，管理各阶段输出。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class OutputService:
    def create_session_dir(self, output_base: str | Path) -> Path:
        """创建以时间戳命名的 session 目录。"""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = Path(output_base) / ts
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("创建 session 目录: %s", session_dir)
        return session_dir

    def find_latest_session(self, output_base: str | Path) -> Path | None:
        """找到最新的 session 目录（用于 resume）。"""
        base = Path(output_base)
        if not base.is_dir():
            return None
        dirs = sorted(
            [d for d in base.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        return dirs[0] if dirs else None

    def resolve_session_dir(
        self,
        output_base: str | Path,
        session_ref: str,
        *,
        base_dir: str | Path | None = None,
    ) -> Path:
        """将配置中的 session 名或路径解析为实际目录。"""
        ref = Path(session_ref)
        candidates: list[Path] = []

        if ref.is_absolute():
            candidates.append(ref)
        else:
            if base_dir is not None:
                candidates.append(Path(base_dir) / ref)
            candidates.append(Path(output_base) / ref)

        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        raise FileNotFoundError(f"未找到可继续的 session 目录: {session_ref}")

    def save_session_config(self, session_dir: Path, config_dict: dict) -> None:
        """保存本次运行的配置快照。"""
        path = session_dir / "config_snapshot.json"
        path.write_text(
            json.dumps(config_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_summary(
        self,
        session_dir: Path,
        stages_status: dict[int, str],
        *,
        pipeline_status: str = "running",
        current_stage: int | None = None,
    ) -> None:
        """写入当前运行摘要。"""
        summary = {
            "session_dir": str(session_dir),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": pipeline_status,
            "current_stage": current_stage,
            "stages": stages_status,
        }
        path = session_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        md_lines = [
            "# Research Session Summary\n",
            f"**Session:** {session_dir.name}",
            f"**Updated:** {summary['updated_at']}",
            f"**Status:** {pipeline_status}\n",
            "## Stage Status\n",
        ]
        for stage_num in sorted(stages_status):
            md_lines.append(f"- Stage {stage_num}: **{stages_status[stage_num]}**")

        md_path = session_dir / "summary.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

    def write_final_summary(self, session_dir: Path, stages_status: dict[int, str]) -> None:
        """写入最终的运行摘要。"""
        self.write_summary(session_dir, stages_status, pipeline_status="completed")


output_service = OutputService()
