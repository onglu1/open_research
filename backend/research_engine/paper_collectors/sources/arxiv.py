"""
arXiv 论文采集器：通过 arXiv Atom API 抓取论文。

注意事项：
- arXiv API 建议至少间隔 3 秒请求一次
- 返回的是 Atom XML feed，需手动解析
- 按 submittedDate 排序获取最新论文
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

from ...config import PaperSearchConfig
from ...models import CollectedPaper, PaperQuery
from ...services.retry import get_retry_wait_seconds
from ..base import PaperCollector

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivCollector(PaperCollector):
    collector_name = "arxiv"

    async def collect(
        self,
        query: PaperQuery,
        search_config: PaperSearchConfig,
    ) -> list[CollectedPaper]:
        search_terms = self._build_query(query, search_config)
        max_results = min(query.max_results, search_config.max_papers_per_direction)

        params = {
            "search_query": search_terms,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        logger.info("arXiv 查询: %s (max=%d)", search_terms[:120], max_results)

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "research-engine/0.1.0"},
        ) as client:
            max_attempts = max(1, search_config.request_max_retries + 1)
            response: httpx.Response | None = None
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await client.get(ARXIV_API_URL, params=params)
                    response.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TimeoutException) as error:
                    last_error = error
                    if attempt >= max_attempts:
                        raise

                    wait_seconds = get_retry_wait_seconds(attempt)
                    logger.warning(
                        "arXiv 请求失败，将在 %d 秒后重试 (%d/%d): %s",
                        wait_seconds,
                        attempt,
                        search_config.request_max_retries,
                        error,
                    )
                    await asyncio.sleep(wait_seconds)

            if response is None:
                raise RuntimeError(f"arXiv 请求失败: {last_error}")

        papers = self._parse_feed(response.text, search_config)

        # arXiv 要求至少间隔 3 秒
        interval = max(search_config.request_interval_seconds, 3.0)
        await asyncio.sleep(interval)

        logger.info("arXiv 返回 %d 篇论文", len(papers))
        return papers

    @staticmethod
    def _build_query(query: PaperQuery, config: PaperSearchConfig) -> str:
        """构建 arXiv API 的查询字符串。"""
        parts: list[str] = []

        if query.keywords:
            kw_clause = " OR ".join(f'all:"{kw}"' for kw in query.keywords)
            parts.append(f"({kw_clause})")

        if query.categories:
            cat_clause = " OR ".join(f"cat:{cat}" for cat in query.categories)
            parts.append(f"({cat_clause})")

        search = " AND ".join(parts) if parts else "all:machine learning"

        if config.filter_words:
            for word in config.filter_words:
                search += f' ANDNOT all:"{word}"'

        return search

    @staticmethod
    def _parse_feed(xml_text: str, config: PaperSearchConfig) -> list[CollectedPaper]:
        """解析 arXiv Atom feed XML。"""
        root = ET.fromstring(xml_text)
        papers: list[CollectedPaper] = []

        cutoff = datetime.now(timezone.utc) - timedelta(days=config.time_range_days)

        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                paper = ArxivCollector._parse_entry(entry, cutoff)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning("解析 arXiv entry 失败: %s", e)

        return papers

    @staticmethod
    def _parse_entry(entry: ET.Element, cutoff: datetime) -> CollectedPaper | None:
        published_str = (entry.findtext(f"{ATOM_NS}published") or "").strip()
        if published_str:
            try:
                pub_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                if pub_date < cutoff:
                    return None
            except ValueError:
                pass

        arxiv_id_raw = entry.findtext(f"{ATOM_NS}id") or ""
        arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw

        title = (entry.findtext(f"{ATOM_NS}title") or "").strip().replace("\n", " ")
        abstract = (entry.findtext(f"{ATOM_NS}summary") or "").strip().replace("\n", " ")

        authors = [
            (a.findtext(f"{ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]

        categories = [
            c.get("term", "")
            for c in entry.findall(f"{ARXIV_NS}primary_category") + entry.findall(f"{ATOM_NS}category")
            if c.get("term")
        ]
        categories = list(dict.fromkeys(categories))

        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        return CollectedPaper(
            paper_id=f"arxiv:{arxiv_id}",
            title=title,
            authors=authors,
            abstract=abstract,
            url=arxiv_id_raw,
            pdf_url=pdf_url,
            source="arxiv",
            published_date=published_str[:10],
            categories=categories,
            citation_count=0,
            venue="arXiv preprint",
        )
