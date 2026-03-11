"""
Stage 2 — Frontier Scan: 针对核心方向和扩展方向执行学术前沿扫描。

本阶段主要通过论文采集器（arXiv / Semantic Scholar）抓取论文，
再由 LLM 生成态势摘要、热点主题、研究空白等分析。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from ..models import (
    CollectedPaper,
    DirectionScanResult,
    FrontierScanResult,
    PaperQuery,
    StageMetadata,
    StageResult,
)
from ..paper_collectors.runner import paper_collector_runner
from ..services.llm import extract_json
from .base import StageBase, StageContext

logger = logging.getLogger(__name__)

SCAN_SYSTEM_PROMPT = """\
You are a research frontier analyst. Given a set of recent papers for a \
research direction, produce a structured analysis. Return strict JSON:
{
  "landscape_summary": str (2-3 paragraphs),
  "hot_topics": [str],
  "research_gaps": [str],
  "trend_signals": [str],
  "key_methods": [str]
}
Do NOT wrap in markdown code fences.
"""


class FrontierScanStage(StageBase):
    stage_number = 2
    stage_name = "frontier_scan"

    async def run(self, ctx: StageContext) -> StageResult:
        s1_data = ctx.get_previous(1)
        directions = self._collect_directions(s1_data, ctx.config)

        search_cfg = ctx.config.paper_search
        all_scans: list[DirectionScanResult] = []
        total_raw = 0
        total_deduped = 0

        for direction in directions:
            query = PaperQuery(
                keywords=direction.get("keywords_en", []),
                categories=direction.get("categories", []),
                direction_name=direction.get("name", ""),
                max_results=search_cfg.max_papers_per_direction,
                time_range_days=search_cfg.time_range_days,
            )

            results = await paper_collector_runner.run(query, search_cfg)
            papers = paper_collector_runner.merge_and_dedupe(results, search_cfg)

            raw_count = sum(len(r.papers) for r in results if r.status != "error")
            total_raw += raw_count
            total_deduped += len(papers)

            # 用 LLM 对论文集做态势分析
            analysis = await self._analyze_direction(ctx, direction, papers)

            scan = DirectionScanResult(
                direction_name=direction.get("name", ""),
                landscape_summary=analysis.get("landscape_summary", ""),
                hot_topics=analysis.get("hot_topics", []),
                research_gaps=analysis.get("research_gaps", []),
                trend_signals=analysis.get("trend_signals", []),
                key_methods=analysis.get("key_methods", []),
                paper_count=len(papers),
                papers=papers,
            )
            all_scans.append(scan)
            logger.info("方向 [%s] 采集完成: %d 篇论文", direction.get("name", ""), len(papers))

        result_obj = FrontierScanResult(
            direction_scans=all_scans,
            total_papers=total_raw,
            deduplicated_papers=total_deduped,
        )
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

    @staticmethod
    def _collect_directions(s1_data: dict, config) -> list[dict]:
        """从 Stage 1 输出中提取所有要扫描的方向（核心 + 扩展）。"""
        directions: list[dict] = []
        rd = config.research_direction

        # 核心方向
        core = s1_data.get("core_direction", {})
        directions.append({
            "name": core.get("name", rd.name),
            "keywords_en": core.get("keywords_en", rd.keywords_en),
            "categories": core.get("categories", rd.categories),
        })

        # 扩展方向
        for d in s1_data.get("parallel_directions", []):
            if d.get("keywords_en") or d.get("categories"):
                directions.append({
                    "name": d.get("name", ""),
                    "keywords_en": d.get("keywords_en", []),
                    "categories": d.get("categories", []),
                })

        return directions

    async def _analyze_direction(
        self,
        ctx: StageContext,
        direction: dict,
        papers: list[CollectedPaper],
    ) -> dict:
        """用 LLM 分析单个方向的论文集。"""
        if not papers:
            return {"landscape_summary": "该方向未检索到论文。",
                    "hot_topics": [], "research_gaps": [], "trend_signals": [], "key_methods": []}

        # 构建论文摘要给 LLM（限制 token 用量）
        paper_summaries = []
        for p in papers[:30]:
            paper_summaries.append(
                f"- [{p.published_date}] {p.title} (citations: {p.citation_count})\n"
                f"  Abstract: {p.abstract[:300]}"
            )
        paper_text = "\n".join(paper_summaries)

        user_prompt = (
            f"Direction: {direction.get('name', '')}\n"
            f"Number of papers: {len(papers)}\n\n"
            f"--- Recent papers ---\n{paper_text}\n\n"
            "Analyze the research landscape for this direction."
        )

        messages = [
            {"role": "system", "content": SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            return await ctx.llm.chat_json(messages)
        except ValueError:
            raw = await ctx.llm.chat(messages)
            try:
                return extract_json(raw)
            except ValueError:
                logger.warning("方向 [%s] LLM 分析 JSON 解析失败，使用空分析", direction.get("name"))
                return {"landscape_summary": raw[:500] if raw else "",
                        "hot_topics": [], "research_gaps": [], "trend_signals": [], "key_methods": []}

    @staticmethod
    def _render_markdown(result: FrontierScanResult) -> str:
        lines = [
            "# Stage 2: Frontier Scan Report\n",
            f"**Total papers fetched:** {result.total_papers}",
            f"**After deduplication:** {result.deduplicated_papers}",
            f"**Directions scanned:** {len(result.direction_scans)}\n",
        ]

        for scan in result.direction_scans:
            lines.append(f"## {scan.direction_name} ({scan.paper_count} papers)\n")
            if scan.landscape_summary:
                lines.append(f"{scan.landscape_summary}\n")

            if scan.hot_topics:
                lines.append("### Hot Topics")
                for t in scan.hot_topics:
                    lines.append(f"- {t}")
                lines.append("")

            if scan.research_gaps:
                lines.append("### Research Gaps")
                for g in scan.research_gaps:
                    lines.append(f"- {g}")
                lines.append("")

            if scan.trend_signals:
                lines.append("### Trend Signals")
                for s in scan.trend_signals:
                    lines.append(f"- {s}")
                lines.append("")

            if scan.key_methods:
                lines.append("### Key Methods")
                for m in scan.key_methods:
                    lines.append(f"- {m}")
                lines.append("")

            if scan.papers:
                lines.append("### Representative Papers")
                for p in scan.papers[:10]:
                    cite_info = f" (citations: {p.citation_count})" if p.citation_count else ""
                    lines.append(f"- **{p.title}**{cite_info}")
                    lines.append(f"  - Authors: {', '.join(p.authors[:5])}")
                    lines.append(f"  - Date: {p.published_date} | Source: {p.source}")
                    lines.append(f"  - URL: {p.url}")
                lines.append("")

        return "\n".join(lines)
