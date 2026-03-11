"""
论文采集器基类（对标原 career_collectors/base.py）。

每个论文源（arXiv, Semantic Scholar 等）实现一个 PaperCollector 子类。
"""

from __future__ import annotations

import logging
from time import perf_counter

from ..config import PaperSearchConfig
from ..models import CollectedPaper, CollectorRunResult, PaperQuery

logger = logging.getLogger(__name__)


class PaperCollector:
    """论文采集器基类，子类需实现 collect() 方法。"""
    collector_name: str = "base"

    async def collect(
        self,
        query: PaperQuery,
        search_config: PaperSearchConfig,
    ) -> list[CollectedPaper]:
        raise NotImplementedError

    async def timed_collect(
        self,
        query: PaperQuery,
        search_config: PaperSearchConfig,
    ) -> CollectorRunResult:
        """带计时和错误处理的采集包装（对标 CompanyCollector.timed_collect）。"""
        started_at = perf_counter()
        try:
            papers = await self.collect(query, search_config)
            status = "success" if papers else "empty"
            return CollectorRunResult(
                source_name=self.collector_name,
                status=status,
                papers=papers,
                stats={"returned_papers": len(papers)},
                duration_ms=int((perf_counter() - started_at) * 1000),
            )
        except Exception as error:
            logger.error("%s 采集失败: %s", self.collector_name, error, exc_info=True)
            return CollectorRunResult(
                source_name=self.collector_name,
                status="error",
                papers=[],
                stats={},
                error=str(error) or error.__class__.__name__,
                duration_ms=int((perf_counter() - started_at) * 1000),
            )
