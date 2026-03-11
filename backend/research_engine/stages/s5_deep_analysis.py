"""
Stage 5 — Deep Analysis: 对高优先级 Idea 生成接近 Research Proposal 的完整报告。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from ..models import (
    DeepAnalysis,
    DeepAnalysisResult,
    ExperimentDesign,
    KeyPaperAnalysis,
    Milestone,
    StageMetadata,
    StageResult,
)
from ..prompts import deep_analysis as prompts
from ..services.knowledge_base import knowledge_base_service
from ..services.llm import extract_json
from .base import StageBase, StageContext

logger = logging.getLogger(__name__)


class DeepAnalysisStage(StageBase):
    stage_number = 5
    stage_name = "deep_analysis"

    async def run(self, ctx: StageContext) -> StageResult:
        s4_data = ctx.get_previous(4)
        s2_data = ctx.get_previous(2)
        s3_data = ctx.get_previous(3)

        ranked = s4_data.get("ranked_ideas", [])
        # 只对 S / A tier 做深度分析
        high_priority = [r for r in ranked if r.get("tier") in ("S", "A")]
        if not high_priority:
            high_priority = ranked[:3]

        ideas_by_id = {idea.get("id", ""): idea for idea in s3_data.get("ideas", [])}

        # 构建前沿上下文摘要
        frontier_ctx = self._build_frontier_context(s2_data)

        rc = ctx.config.resource_constraints
        resource_str = (
            f"GPU: {rc.gpu}, Time: {rc.time_budget_months} months, "
            f"Team: {rc.team_size}, Compute: {rc.compute_budget}"
        ) if rc.gpu else ""

        kb_summary = knowledge_base_service.summarize_for_stage(ctx.knowledge, 5)

        analyses: list[DeepAnalysis] = []
        for ranked_idea in high_priority:
            idea_id = ranked_idea.get("idea_id", "")
            full_idea = ideas_by_id.get(idea_id, ranked_idea)

            idea_json = json.dumps({**full_idea, **ranked_idea}, ensure_ascii=False, indent=1)
            user_prompt = prompts.build_user_prompt(
                idea_json=idea_json[:6000],
                frontier_context=frontier_ctx[:4000],
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
                try:
                    raw = extract_json(raw_text)
                except ValueError:
                    logger.warning("Idea %s 深度分析 JSON 解析失败", idea_id)
                    raw = {"idea_id": idea_id, "executive_summary": raw_text[:1000] if raw_text else "分析失败"}

            if not raw:
                raw = {"idea_id": idea_id, "executive_summary": "LLM 不可用，无法生成深度分析。"}

            analysis = self._build_analysis(raw, ranked_idea)
            analyses.append(analysis)
            logger.info("Idea [%s] 深度分析完成", idea_id)

        result_obj = DeepAnalysisResult(analyses=analyses)
        data_dict = asdict(result_obj)
        md = self._render_markdown(result_obj)

        return StageResult(
            stage_number=self.stage_number,
            stage_name=self.stage_name,
            success=bool(analyses),
            data=data_dict,
            metadata=StageMetadata(stage_number=self.stage_number, stage_name=self.stage_name),
            markdown=md,
        )

    @staticmethod
    def _build_frontier_context(s2_data: dict) -> str:
        parts = []
        for scan in s2_data.get("direction_scans", []):
            name = scan.get("direction_name", "")
            summary = scan.get("landscape_summary", "")
            gaps = scan.get("research_gaps", [])
            parts.append(f"[{name}] {summary}")
            if gaps:
                parts.append("  Gaps: " + "; ".join(gaps[:5]))
        return "\n".join(parts)

    @staticmethod
    def _build_analysis(data: dict, ranked_idea: dict) -> DeepAnalysis:
        kpa = []
        for item in data.get("key_paper_analysis", []):
            kpa.append(KeyPaperAnalysis(
                paper_title=item.get("paper_title", ""),
                paper_url=item.get("paper_url", ""),
                relevance=item.get("relevance", ""),
                key_findings=item.get("key_findings", ""),
                limitations=item.get("limitations", ""),
            ))

        exp = data.get("experiment_design", {})
        experiment = ExperimentDesign(
            research_questions=exp.get("research_questions", []),
            datasets=exp.get("datasets", []),
            baselines=exp.get("baselines", []),
            metrics=exp.get("metrics", []),
            ablations=exp.get("ablations", []),
            expected_results=exp.get("expected_results", ""),
        )

        timeline = []
        for item in data.get("timeline", []):
            timeline.append(Milestone(
                name=item.get("name", ""),
                duration=item.get("duration", ""),
                deliverables=item.get("deliverables", []),
            ))

        return DeepAnalysis(
            idea_id=data.get("idea_id", ranked_idea.get("idea_id", "")),
            title_cn=data.get("title_cn", ranked_idea.get("title_cn", "")),
            title_en=data.get("title_en", ranked_idea.get("title_en", "")),
            executive_summary=data.get("executive_summary", ""),
            related_work_review=data.get("related_work_review", ""),
            key_paper_analysis=kpa,
            technical_approach=data.get("technical_approach", ""),
            system_architecture=data.get("system_architecture", ""),
            key_components=data.get("key_components", []),
            experiment_design=experiment,
            risk_assessment=data.get("risk_assessment", ""),
            resource_risks=data.get("resource_risks", ""),
            timeline=timeline,
            target_venues=data.get("target_venues", []),
            go_no_go_verdict=data.get("go_no_go_verdict", ""),
        )

    @staticmethod
    def _render_markdown(result: DeepAnalysisResult) -> str:
        lines = [
            "# Stage 5: Deep Analysis Report\n",
            f"**Ideas analyzed:** {len(result.analyses)}\n",
        ]

        for a in result.analyses:
            lines.append(f"---\n\n## {a.title_cn} ({a.title_en})")
            lines.append(f"**ID:** {a.idea_id}\n")

            if a.executive_summary:
                lines.append(f"### Executive Summary\n{a.executive_summary}\n")

            if a.related_work_review:
                lines.append(f"### Related Work\n{a.related_work_review}\n")

            if a.key_paper_analysis:
                lines.append("### Key Papers")
                for kp in a.key_paper_analysis:
                    lines.append(f"- **{kp.paper_title}** ([link]({kp.paper_url}))")
                    lines.append(f"  - Relevance: {kp.relevance}")
                    lines.append(f"  - Key findings: {kp.key_findings}")
                    lines.append(f"  - Limitations: {kp.limitations}")
                lines.append("")

            if a.technical_approach:
                lines.append(f"### Technical Approach\n{a.technical_approach}\n")

            if a.system_architecture:
                lines.append(f"### System Architecture\n{a.system_architecture}\n")

            if a.key_components:
                lines.append("### Key Components")
                for c in a.key_components:
                    lines.append(f"- {c}")
                lines.append("")

            exp = a.experiment_design
            if exp.research_questions:
                lines.append("### Experiment Design")
                lines.append("**Research Questions:**")
                for q in exp.research_questions:
                    lines.append(f"1. {q}")
                if exp.datasets:
                    lines.append(f"\n**Datasets:** {', '.join(exp.datasets)}")
                if exp.baselines:
                    lines.append(f"**Baselines:** {', '.join(exp.baselines)}")
                if exp.metrics:
                    lines.append(f"**Metrics:** {', '.join(exp.metrics)}")
                if exp.ablations:
                    lines.append("**Ablation studies:**")
                    for ab in exp.ablations:
                        lines.append(f"- {ab}")
                if exp.expected_results:
                    lines.append(f"\n**Expected results:** {exp.expected_results}")
                lines.append("")

            if a.risk_assessment:
                lines.append(f"### Risk Assessment\n{a.risk_assessment}\n")

            if a.timeline:
                lines.append("### Timeline")
                for ms in a.timeline:
                    lines.append(f"- **{ms.name}** ({ms.duration})")
                    for d in ms.deliverables:
                        lines.append(f"  - {d}")
                lines.append("")

            if a.target_venues:
                lines.append(f"**Target venues:** {', '.join(a.target_venues)}\n")

            if a.go_no_go_verdict:
                lines.append(f"### Go/No-Go Verdict\n**{a.go_no_go_verdict}**\n")

        return "\n".join(lines)
