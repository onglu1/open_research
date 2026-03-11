"""
知识库服务：从 knowledge/ 目录加载用户已有研究、idea、论文笔记。

在不同阶段中提供上下文：
- Stage 1: 已有研究方向约束
- Stage 3: 已有 idea 用于去重
- Stage 4: 论文笔记加入排序上下文
- Stage 5: 背景信息补充深度分析
"""

from __future__ import annotations

from pathlib import Path

from ..config import KnowledgeBaseConfig
from ..models import KnowledgeBaseContent


class KnowledgeBaseService:
    def load(self, config: KnowledgeBaseConfig, base_dir: Path | None = None) -> KnowledgeBaseContent:
        """根据配置加载知识库内容，文件不存在时静默跳过。"""
        base = base_dir or Path(".")

        existing_research = self._read_file(base / config.existing_research) if config.existing_research else ""
        existing_ideas = self._read_file(base / config.existing_ideas) if config.existing_ideas else ""

        paper_notes: list[str] = []
        paper_notes_files: list[str] = []
        if config.paper_notes_dir:
            notes_dir = base / config.paper_notes_dir
            if notes_dir.is_dir():
                for p in sorted(notes_dir.glob("*.md")):
                    content = self._read_file(p)
                    if content:
                        paper_notes.append(content)
                        paper_notes_files.append(str(p.name))

        return KnowledgeBaseContent(
            existing_research=existing_research,
            existing_ideas=existing_ideas,
            paper_notes=paper_notes,
            paper_notes_files=paper_notes_files,
        )

    @staticmethod
    def _read_file(path: Path) -> str:
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def summarize_for_stage(self, kb: KnowledgeBaseContent, stage_number: int) -> str:
        """为指定阶段生成知识库摘要，作为 LLM prompt 的上下文补充。"""
        parts: list[str] = []

        if stage_number == 1 and kb.existing_research:
            parts.append(f"[已有研究方向]\n{kb.existing_research[:3000]}")

        if stage_number == 3 and kb.existing_ideas:
            parts.append(f"[已有 Idea（用于去重）]\n{kb.existing_ideas[:3000]}")

        if stage_number in (4, 5) and kb.paper_notes:
            combined = "\n---\n".join(kb.paper_notes[:10])
            parts.append(f"[已读论文笔记]\n{combined[:4000]}")

        if stage_number == 5 and kb.existing_research:
            parts.append(f"[研究背景]\n{kb.existing_research[:2000]}")

        return "\n\n".join(parts) if parts else ""


knowledge_base_service = KnowledgeBaseService()
