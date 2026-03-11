"""
Stage 3 — Idea Discovery: 基于多方向的热点、研究空白和趋势信号生成候选 Idea。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from ..models import (
    IdeaDiscoveryResult,
    ResearchIdea,
    StageMetadata,
    StageResult,
)
from ..prompts import idea_generation as prompts
from ..services.knowledge_base import knowledge_base_service
from ..services.llm import extract_json
from .base import StageBase, StageContext

logger = logging.getLogger(__name__)


class IdeaDiscoveryStage(StageBase):
    stage_number = 3
    stage_name = "idea_discovery"

    async def run(self, ctx: StageContext) -> StageResult:
        s2_data = ctx.get_previous(2)
        cfg = ctx.config
        idea_cfg = cfg.idea_generation

        # 汇总所有方向的分析结果
        scans = s2_data.get("direction_scans", [])
        scan_summaries = []
        all_gaps: list[str] = []
        all_trends: list[str] = []
        all_topics: list[str] = []

        for scan in scans:
            name = scan.get("direction_name", "")
            summary = scan.get("landscape_summary", "")
            scan_summaries.append(f"[{name}] {summary}")
            all_gaps.extend(scan.get("research_gaps", []))
            all_trends.extend(scan.get("trend_signals", []))
            all_topics.extend(scan.get("hot_topics", []))

        # 资源约束描述
        rc = cfg.resource_constraints
        resource_str = (
            f"GPU: {rc.gpu}, Time: {rc.time_budget_months} months, "
            f"Team: {rc.team_size}, Compute: {rc.compute_budget}"
        ) if rc.gpu else ""

        kb_summary = knowledge_base_service.summarize_for_stage(ctx.knowledge, 3)

        user_prompt = prompts.build_user_prompt(
            direction_scans_summary="\n\n".join(scan_summaries),
            research_gaps=all_gaps,
            trend_signals=all_trends,
            hot_topics=all_topics,
            strategies=idea_cfg.strategies,
            max_ideas=idea_cfg.max_ideas,
            encourage_cross=idea_cfg.encourage_cross_direction,
            resource_constraints=resource_str,
            knowledge_context=kb_summary,
        )

        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = await ctx.llm.chat_json(messages)
        except ValueError:
            raw_text = await ctx.llm.chat(messages)
            raw = self._text_fallback(raw_text)

        if not raw:
            raw = {"ideas": [], "generation_strategies_used": []}

        data = raw if isinstance(raw, dict) else {"ideas": raw}
        result_obj = self._build_result(data)
        data_dict = asdict(result_obj)
        md = self._render_markdown(result_obj)

        return StageResult(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            success=bool(result_obj.ideas),
            data=data_dict,
            metadata=StageMetadata(stage_number=self.stage_number, stage_name=self.stage_name),
            markdown=md,
        )

    @staticmethod
    def _build_result(data: dict) -> IdeaDiscoveryResult:
        ideas = []
        for item in data.get("ideas", []):
            ideas.append(ResearchIdea(
                id=item.get("id", ""),
                title_cn=item.get("title_cn", ""),
                title_en=item.get("title_en", ""),
                one_liner=item.get("one_liner", ""),
                description=item.get("description", ""),
                motivation=item.get("motivation", ""),
                method_sketch=item.get("method_sketch", ""),
                novelty_points=item.get("novelty_points", []),
                expected_contributions=item.get("expected_contributions", []),
                validation_plan=item.get("validation_plan", ""),
                source_directions=item.get("source_directions", []),
                discovery_strategy=item.get("discovery_strategy", ""),
                related_gaps=item.get("related_gaps", []),
                related_papers=item.get("related_papers", []),
                estimated_effort=item.get("estimated_effort", ""),
                risk_factors=item.get("risk_factors", []),
                preliminary_score=float(item.get("preliminary_score", 0)),
            ))
        return IdeaDiscoveryResult(
            ideas=ideas,
            generation_strategies_used=data.get("generation_strategies_used", []),
        )

    @staticmethod
    def _text_fallback(raw_text: str) -> dict:
        try:
            return extract_json(raw_text)
        except ValueError:
            logger.warning("Stage 3 文本降级也失败")
            return {"ideas": [], "generation_strategies_used": []}

    @staticmethod
    def _render_markdown(result: IdeaDiscoveryResult) -> str:
        lines = [
            "# Stage 3: Idea Discovery Report\n",
            f"**Total ideas generated:** {len(result.ideas)}",
            f"**Strategies used:** {', '.join(result.generation_strategies_used)}\n",
        ]

        for idea in result.ideas:
            lines.append(f"## {idea.id}: {idea.title_cn}")
            lines.append(f"**{idea.title_en}**\n")
            lines.append(f"*{idea.one_liner}*\n")
            lines.append(f"**Strategy:** {idea.discovery_strategy} | "
                         f"**Effort:** {idea.estimated_effort} | "
                         f"**Score:** {idea.preliminary_score:.1f}\n")
            if idea.source_directions:
                lines.append(f"**Source directions:** {', '.join(idea.source_directions)}")

            lines.append(f"\n### Motivation\n{idea.motivation}\n")
            lines.append(f"### Method Sketch\n{idea.method_sketch}\n")

            if idea.novelty_points:
                lines.append("### Novelty")
                for n in idea.novelty_points:
                    lines.append(f"- {n}")
                lines.append("")

            if idea.expected_contributions:
                lines.append("### Expected Contributions")
                for c in idea.expected_contributions:
                    lines.append(f"- {c}")
                lines.append("")

            if idea.risk_factors:
                lines.append("### Risk Factors")
                for r in idea.risk_factors:
                    lines.append(f"- {r}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)
