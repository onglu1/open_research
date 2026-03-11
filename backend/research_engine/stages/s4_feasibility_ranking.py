"""
Stage 4 — Feasibility Ranking: 对候选 Idea 进行量化评估和排序。

支持对 Stage 3 的全部 idea 进行打乱、精简、分批评估，避免因 prompt
长度截断导致只分析前几条 idea 的问题。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict

from ..models import (
    DimensionScore,
    FeasibilityRankingResult,
    RankedIdea,
    StageMetadata,
    StageResult,
)
from ..prompts import feasibility_ranking as prompts
from ..services.knowledge_base import knowledge_base_service
from ..services.llm import extract_json
from .base import StageBase, StageContext

logger = logging.getLogger(__name__)

# 精简 idea 时保留的核心字段
_CONDENSED_FIELDS = [
    "id", "title_cn", "title_en", "one_liner",
    "motivation", "novelty_points", "risk_factors",
    "estimated_effort", "preliminary_score", "source_directions",
    "discovery_strategy",
]
# method_sketch 截断长度
_METHOD_SKETCH_MAX = 500


class FeasibilityRankingStage(StageBase):
    stage_number = 4
    stage_name = "feasibility_ranking"

    # ── 精简 idea，去掉冗长字段以让更多 idea 放入 prompt ──

    @staticmethod
    def _condense_idea(idea: dict) -> dict:
        """保留评估所需的核心字段，截断过长的 method_sketch。"""
        condensed = {k: idea[k] for k in _CONDENSED_FIELDS if k in idea}
        sketch = idea.get("method_sketch", "")
        if sketch:
            condensed["method_sketch"] = (
                sketch[:_METHOD_SKETCH_MAX] + "..." if len(sketch) > _METHOD_SKETCH_MAX else sketch
            )
        return condensed

    # ── 将 idea 列表切分为多个 batch ──

    @staticmethod
    def _split_into_batches(
        ideas: list[dict],
        max_per_batch: int,
        max_chars: int,
    ) -> list[list[dict]]:
        """按数量和字符上限将 idea 切分为若干批次。"""
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_chars = 0

        for idea in ideas:
            idea_json = json.dumps(idea, ensure_ascii=False)
            idea_len = len(idea_json)

            need_new_batch = (
                (max_per_batch and len(current_batch) >= max_per_batch)
                or (max_chars and current_chars + idea_len > max_chars and current_batch)
            )
            if need_new_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(idea)
            current_chars += idea_len

        if current_batch:
            batches.append(current_batch)
        return batches

    async def run(self, ctx: StageContext) -> StageResult:
        s3_data = ctx.get_previous(3)
        cfg = ctx.config
        feas_cfg = cfg.feasibility

        ideas = s3_data.get("ideas", [])
        if not ideas:
            return StageResult(
                stage_number=self.stage_number,
                stage_name=self.stage_name,
                success=True,
                data=asdict(FeasibilityRankingResult()),
                metadata=StageMetadata(stage_number=self.stage_number, stage_name=self.stage_name),
                markdown="# Stage 4: Feasibility Ranking\n\n无候选 Idea 可供评估。\n",
            )

        # 打乱顺序，消除位置偏见
        if feas_cfg.shuffle_ideas:
            ideas = list(ideas)
            random.shuffle(ideas)
            logger.info("Stage 4: 已打乱 %d 条 idea 的顺序", len(ideas))

        rc = cfg.resource_constraints
        resource_str = (
            f"GPU: {rc.gpu}, Time: {rc.time_budget_months} months, "
            f"Team: {rc.team_size}, Compute: {rc.compute_budget}"
        ) if rc.gpu else ""

        dimensions_dict = {
            name: {"weight": d.weight, "description": d.description}
            for name, d in feas_cfg.dimensions.items()
        }
        kb_summary = knowledge_base_service.summarize_for_stage(ctx.knowledge, 4)

        # 精简 idea 内容
        condensed_ideas = [self._condense_idea(idea) for idea in ideas]

        # 按配置切分批次
        batches = self._split_into_batches(
            condensed_ideas,
            max_per_batch=feas_cfg.max_ideas_per_batch,
            max_chars=feas_cfg.max_prompt_chars,
        )
        logger.info(
            "Stage 4: %d 条 idea 分为 %d 批评估",
            len(ideas), len(batches),
        )

        # 逐批调用 LLM，合并结果
        all_ranked: list[dict] = []
        for batch_idx, batch in enumerate(batches, 1):
            logger.info("Stage 4: 评估第 %d/%d 批（%d 条 idea）", batch_idx, len(batches), len(batch))
            batch_raw = await self._evaluate_batch(
                ctx, batch, dimensions_dict, feas_cfg, resource_str, kb_summary,
                batch_idx=batch_idx, total_batches=len(batches),
            )
            all_ranked.extend(batch_raw.get("ranked_ideas", []))

        # 如果所有批次都失败，使用启发式降级
        if not all_ranked:
            merged = self._heuristic_ranking(ideas, feas_cfg)
        else:
            merged = {
                "ranked_ideas": all_ranked,
                "ranking_rationale": (
                    f"对全部 {len(ideas)} 条 idea 分 {len(batches)} 批完成评估，"
                    "按 weighted_total 统一排序。"
                ),
            }

        result_obj = self._build_result(merged)
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

    async def _evaluate_batch(
        self,
        ctx: StageContext,
        batch: list[dict],
        dimensions_dict: dict,
        feas_cfg,
        resource_str: str,
        kb_summary: str,
        *,
        batch_idx: int,
        total_batches: int,
    ) -> dict:
        """对单个批次的 idea 调用 LLM 评估，返回 ranked_ideas 列表。"""
        ideas_json = json.dumps(batch, ensure_ascii=False, indent=1)

        batch_note = ""
        if total_batches > 1:
            batch_note = (
                f"\n[NOTE: This is batch {batch_idx}/{total_batches}. "
                f"Score each idea on its own merits using absolute criteria.]\n"
            )

        user_prompt = prompts.build_user_prompt(
            ideas_json=batch_note + ideas_json,
            dimensions=dimensions_dict,
            tier_thresholds=feas_cfg.tier_thresholds,
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
            raw = self._text_fallback(raw_text, batch, feas_cfg)

        if not raw:
            raw = self._heuristic_ranking(batch, feas_cfg)

        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _build_result(data: dict) -> FeasibilityRankingResult:
        ranked = []
        for item in data.get("ranked_ideas", []):
            dim_scores = {}
            for dim_name, dim_data in item.get("dimension_scores", {}).items():
                if isinstance(dim_data, dict):
                    dim_scores[dim_name] = DimensionScore(
                        score=float(dim_data.get("score", 0)),
                        rationale=dim_data.get("rationale", ""),
                    )
                else:
                    dim_scores[dim_name] = DimensionScore(score=float(dim_data))

            ranked.append(RankedIdea(
                idea_id=item.get("idea_id", ""),
                title_cn=item.get("title_cn", ""),
                title_en=item.get("title_en", ""),
                one_liner=item.get("one_liner", ""),
                dimension_scores=dim_scores,
                weighted_total=float(item.get("weighted_total", 0)),
                tier=item.get("tier", "C"),
                recommendation=item.get("recommendation", ""),
                next_steps=item.get("next_steps", []),
                target_venues=item.get("target_venues", []),
                time_estimate=item.get("time_estimate", ""),
                key_dependencies=item.get("key_dependencies", []),
            ))

        ranked.sort(key=lambda x: x.weighted_total, reverse=True)

        return FeasibilityRankingResult(
            ranked_ideas=ranked,
            ranking_rationale=data.get("ranking_rationale", ""),
        )

    @staticmethod
    def _heuristic_ranking(ideas: list[dict], feas_cfg) -> dict:
        """LLM 不可用时的规则排序。"""
        ranked = []
        thresholds = feas_cfg.tier_thresholds
        for idea in ideas:
            score = float(idea.get("preliminary_score", 50))
            tier = "C"
            for t, threshold in sorted(thresholds.items(), key=lambda x: -x[1]):
                if score >= threshold:
                    tier = t
                    break
            ranked.append({
                "idea_id": idea.get("id", ""),
                "title_cn": idea.get("title_cn", ""),
                "title_en": idea.get("title_en", ""),
                "one_liner": idea.get("one_liner", ""),
                "dimension_scores": {},
                "weighted_total": score,
                "tier": tier,
                "recommendation": "需要 LLM 进行详细评估",
                "next_steps": [],
                "target_venues": [],
                "time_estimate": idea.get("estimated_effort", ""),
                "key_dependencies": [],
            })
        ranked.sort(key=lambda x: x["weighted_total"], reverse=True)
        return {
            "ranked_ideas": ranked,
            "ranking_rationale": "基于初步评分的规则排序（LLM 降级模式）。",
        }

    @staticmethod
    def _text_fallback(raw_text: str, ideas: list[dict], feas_cfg) -> dict:
        try:
            return extract_json(raw_text)
        except ValueError:
            logger.warning("Stage 4 文本降级也失败，使用 heuristic")
            return FeasibilityRankingStage._heuristic_ranking(ideas, feas_cfg)

    @staticmethod
    def _render_markdown(result: FeasibilityRankingResult) -> str:
        lines = [
            "# Stage 4: Feasibility Ranking Report\n",
            f"**Total ideas ranked:** {len(result.ranked_ideas)}\n",
        ]

        if result.ranking_rationale:
            lines.append(f"## Ranking Rationale\n{result.ranking_rationale}\n")

        # 按 Tier 分组
        tiers: dict[str, list[RankedIdea]] = {}
        for idea in result.ranked_ideas:
            tiers.setdefault(idea.tier, []).append(idea)

        for tier in ["S", "A", "B", "C"]:
            group = tiers.get(tier, [])
            if not group:
                continue
            lines.append(f"\n## Tier {tier} ({len(group)} ideas)\n")
            for idea in group:
                lines.append(f"### {idea.idea_id}: {idea.title_cn} ({idea.title_en})")
                lines.append(f"*{idea.one_liner}*\n")
                lines.append(f"**Weighted Total: {idea.weighted_total:.1f}** | Tier: {idea.tier}\n")

                if idea.dimension_scores:
                    lines.append("| Dimension | Score | Rationale |")
                    lines.append("|-----------|-------|-----------|")
                    for dim, ds in idea.dimension_scores.items():
                        lines.append(f"| {dim} | {ds.score:.1f} | {ds.rationale} |")
                    lines.append("")

                if idea.recommendation:
                    lines.append(f"**Recommendation:** {idea.recommendation}\n")
                if idea.target_venues:
                    lines.append(f"**Target venues:** {', '.join(idea.target_venues)}")
                if idea.time_estimate:
                    lines.append(f"**Time estimate:** {idea.time_estimate}")
                if idea.next_steps:
                    lines.append("\n**Next steps:**")
                    for ns in idea.next_steps:
                        lines.append(f"- {ns}")
                lines.append("\n---\n")

        return "\n".join(lines)
