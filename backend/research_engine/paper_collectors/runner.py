"""
论文采集运行器（对标原 CareerCollectorRunner）。

并发执行多个论文源采集，做去重、打分和排序。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from ..config import PaperSearchConfig
from ..models import CollectedPaper, CollectorRunResult, PaperQuery
from .sources import REGISTERED_COLLECTORS

logger = logging.getLogger(__name__)


class PaperCollectorRunner:
    def __init__(self) -> None:
        self._collectors_by_name = {c.collector_name: c for c in REGISTERED_COLLECTORS}

    async def run(
        self,
        query: PaperQuery,
        search_config: PaperSearchConfig,
    ) -> list[CollectorRunResult]:
        """对配置中启用的所有论文源并发执行采集。"""
        enabled_sources = search_config.sources
        collectors = [
            self._collectors_by_name[name]
            for name in enabled_sources
            if name in self._collectors_by_name
        ]

        if not collectors:
            logger.warning("没有可用的论文采集器（配置的来源: %s）", enabled_sources)
            return []

        tasks = [
            collector.timed_collect(query, search_config)
            for collector in collectors
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    @staticmethod
    def merge_and_dedupe(results: list[CollectorRunResult], search_config: PaperSearchConfig) -> list[CollectedPaper]:
        """合并多个来源的论文，去重并按分数排序。"""
        seen_titles: set[str] = set()
        seen_ids: set[str] = set()
        merged: list[CollectedPaper] = []

        for result in results:
            if result.status == "error":
                logger.warning("跳过失败的来源 %s: %s", result.source_name, result.error)
                continue
            for paper in result.papers:
                # 基于标题去重（忽略大小写和空格）
                title_key = paper.title.lower().strip()
                if title_key in seen_titles:
                    continue
                if paper.paper_id in seen_ids:
                    continue
                seen_titles.add(title_key)
                seen_ids.add(paper.paper_id)
                paper.relevance_score = PaperCollectorRunner._score_paper(paper, search_config)
                merged.append(paper)

        merged.sort(key=lambda p: p.relevance_score, reverse=True)
        return merged

    @staticmethod
    def _score_paper(paper: CollectedPaper, config: PaperSearchConfig) -> float:
        """基于配置的权重为论文打分。"""
        weights = config.score_weights

        # 引用分（对数尺度，满分 100）
        citation_score = min(100.0, (paper.citation_count ** 0.5) * 10)

        # 时效分（越新越高）
        recency_score = 50.0
        if paper.published_date:
            try:
                pub = datetime.fromisoformat(paper.published_date)
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                days_ago = (datetime.now(timezone.utc) - pub).days
                recency_score = max(0.0, 100.0 - days_ago * 0.5)
            except ValueError:
                pass

        # 相关性分（暂以 50 为基础，后续由 LLM 细化）
        relevance_score = 50.0

        total = (
            citation_score * weights.get("citation_count", 0.3)
            + recency_score * weights.get("recency", 0.3)
            + relevance_score * weights.get("relevance", 0.4)
        )
        return round(total, 2)


paper_collector_runner = PaperCollectorRunner()
