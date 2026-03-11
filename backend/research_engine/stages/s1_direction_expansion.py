"""
Stage 1 — Direction Expansion: 根据核心研究方向生成平行/交叉方向。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from ..models import (
    CrossPollinationOpportunity,
    DirectionExpansionResult,
    ResearchDirection,
    StageMetadata,
    StageResult,
)
from ..prompts import direction_expansion as prompts
from ..services.knowledge_base import knowledge_base_service
from ..services.llm import extract_json
from .base import StageBase, StageContext

logger = logging.getLogger(__name__)


class DirectionExpansionStage(StageBase):
    stage_number = 1
    stage_name = "direction_expansion"

    async def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.config
        rd = cfg.research_direction
        exp = cfg.expansion

        kb_summary = knowledge_base_service.summarize_for_stage(ctx.knowledge, 1)

        # 构建 prompt
        ext_dirs = [{"name": d.name, "name_en": d.name_en, "description": d.description}
                     for d in exp.external_directions]
        par_dirs = [{"name": d.name, "name_en": d.name_en, "description": d.description}
                     for d in exp.parallel_directions]

        user_prompt = prompts.build_user_prompt(
            direction_name=rd.name,
            direction_description=rd.description,
            keywords_en=rd.keywords_en,
            categories=rd.categories,
            strategies=exp.strategies,
            external_directions=ext_dirs,
            parallel_directions=par_dirs,
            forced_crosses=exp.forced_crosses,
            knowledge_context=kb_summary,
        )

        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = await ctx.llm.chat_json(messages)
        except ValueError:
            # JSON 解析失败，尝试文本降级
            raw_text = await ctx.llm.chat(messages)
            raw = self._text_fallback(raw_text, cfg)

        # 如果 LLM 不可用，使用配置构建基础结果
        if not raw:
            raw = self._heuristic_result(cfg)

        data = raw if isinstance(raw, dict) else {}
        result_obj = self._build_result(data, cfg)
        data_dict = asdict(result_obj)

        md = self._render_markdown(result_obj)

        return StageResult(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            success=True,
            data=data_dict,
            metadata=StageMetadata(stage_number=self.stage_number, stage_name=self.stage_name),
            markdown=md,
        )

    def _build_result(self, data: dict, cfg) -> DirectionExpansionResult:
        """从 LLM 返回的 dict 构建类型化结果，合并用户配置。"""
        rd = cfg.research_direction
        core = data.get("core_direction", {})
        core_dir = ResearchDirection(
            name=core.get("name", rd.name),
            name_en=core.get("name_en", ""),
            description=core.get("description", rd.description),
            relationship="core",
            keywords_en=core.get("keywords_en", rd.keywords_en),
            keywords_cn=core.get("keywords_cn", rd.keywords_cn),
            categories=core.get("categories", rd.categories),
            key_researchers=core.get("key_researchers", []),
            relevant_venues=core.get("relevant_venues", []),
        )

        parallel = []
        for item in data.get("parallel_directions", []):
            parallel.append(ResearchDirection(
                name=item.get("name", ""),
                name_en=item.get("name_en", ""),
                description=item.get("description", ""),
                relationship=item.get("relationship", "parallel"),
                strategies=item.get("strategies", []),
                keywords_en=item.get("keywords_en", []),
                keywords_cn=item.get("keywords_cn", []),
                categories=item.get("categories", []),
                cross_value=item.get("cross_value", ""),
                key_researchers=item.get("key_researchers", []),
                relevance_score=float(item.get("relevance_score", 0)),
                relevant_venues=item.get("relevant_venues", []),
            ))

        cross_ops = []
        for item in data.get("cross_pollination_opportunities", []):
            cross_ops.append(CrossPollinationOpportunity(
                direction_a=item.get("direction_a", ""),
                direction_b=item.get("direction_b", ""),
                description=item.get("description", ""),
                potential_value=item.get("potential_value", ""),
            ))

        return DirectionExpansionResult(
            core_direction=core_dir,
            parallel_directions=parallel,
            cross_pollination_opportunities=cross_ops,
            relevant_venues=data.get("relevant_venues", []),
        )

    @staticmethod
    def _heuristic_result(cfg) -> dict:
        """LLM 不可用时，基于配置生成最低限度结果。"""
        rd = cfg.research_direction
        exp = cfg.expansion
        directions = []
        for d in exp.parallel_directions:
            directions.append({
                "name": d.name, "name_en": d.name_en,
                "description": d.description, "relationship": "parallel",
                "strategies": [], "keywords_en": [], "keywords_cn": [],
                "categories": [], "cross_value": "", "key_researchers": [],
                "relevance_score": 0.7, "relevant_venues": [],
            })
        for d in exp.external_directions:
            directions.append({
                "name": d.name, "name_en": d.name_en,
                "description": d.description, "relationship": "external",
                "strategies": [], "keywords_en": [], "keywords_cn": [],
                "categories": [], "cross_value": "", "key_researchers": [],
                "relevance_score": 0.5, "relevant_venues": [],
            })
        return {
            "core_direction": {
                "name": rd.name, "description": rd.description,
                "keywords_en": rd.keywords_en, "keywords_cn": rd.keywords_cn,
                "categories": rd.categories,
            },
            "parallel_directions": directions,
            "cross_pollination_opportunities": [],
            "relevant_venues": [],
        }

    @staticmethod
    def _text_fallback(raw_text: str, cfg) -> dict:
        """LLM 返回非 JSON 时的文本模式降级。"""
        try:
            return extract_json(raw_text)
        except ValueError:
            logger.warning("Stage 1 文本降级也失败，使用 heuristic")
            return DirectionExpansionStage._heuristic_result(cfg)

    @staticmethod
    def _render_markdown(result: DirectionExpansionResult) -> str:
        lines = [
            "# Stage 1: Direction Expansion Report\n",
            f"## Core Direction: {result.core_direction.name}\n",
            f"{result.core_direction.description}\n",
            f"**Keywords (EN):** {', '.join(result.core_direction.keywords_en)}\n",
            f"**Categories:** {', '.join(result.core_direction.categories)}\n",
        ]
        if result.core_direction.relevant_venues:
            lines.append(f"**Venues:** {', '.join(result.core_direction.relevant_venues)}\n")

        lines.append(f"\n## Expanded Directions ({len(result.parallel_directions)})\n")
        for i, d in enumerate(result.parallel_directions, 1):
            lines.append(f"### {i}. {d.name} ({d.name_en})")
            lines.append(f"- **Relationship:** {d.relationship}")
            lines.append(f"- **Relevance:** {d.relevance_score:.2f}")
            lines.append(f"- **Description:** {d.description}")
            if d.keywords_en:
                lines.append(f"- **Keywords:** {', '.join(d.keywords_en)}")
            if d.cross_value:
                lines.append(f"- **Cross-pollination value:** {d.cross_value}")
            lines.append("")

        if result.cross_pollination_opportunities:
            lines.append(f"\n## Cross-Pollination Opportunities ({len(result.cross_pollination_opportunities)})\n")
            for op in result.cross_pollination_opportunities:
                lines.append(f"- **{op.direction_a} x {op.direction_b}**: {op.description}")
                if op.potential_value:
                    lines.append(f"  - Potential value: {op.potential_value}")
            lines.append("")

        if result.relevant_venues:
            lines.append(f"\n## Relevant Venues\n")
            for v in result.relevant_venues:
                lines.append(f"- {v}")

        return "\n".join(lines)
