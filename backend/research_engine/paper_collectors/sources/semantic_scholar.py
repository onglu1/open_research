"""
Semantic Scholar 论文采集器：通过 S2 Academic Graph API 抓取论文。

API 文档: https://api.semanticscholar.org/api-docs/graph

注意事项：
- 免费额度有请求频率限制（100 req/5min），需控制间隔
- 可获取引用数和有影响力引用数（influential citations）
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx

from ...config import PaperSearchConfig
from ...models import CollectedPaper, PaperQuery
from ...services.retry import get_retry_wait_seconds
from ..base import PaperCollector

logger = logging.getLogger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,abstract,authors,url,year,venue,citationCount,influentialCitationCount,publicationDate,externalIds"


class SemanticScholarCollector(PaperCollector):
    collector_name = "semantic_scholar"

    async def collect(
        self,
        query: PaperQuery,
        search_config: PaperSearchConfig,
    ) -> list[CollectedPaper]:
        keyword_str = " ".join(query.keywords) if query.keywords else query.direction_name
        max_results = min(query.max_results, search_config.max_papers_per_direction, 100)

        cutoff_year = (datetime.now(timezone.utc) - timedelta(days=search_config.time_range_days)).year

        params = {
            "query": keyword_str,
            "limit": max_results,
            "fields": S2_FIELDS,
            "year": f"{cutoff_year}-",
        }

        logger.info("Semantic Scholar 查询: %s (max=%d)", keyword_str[:80], max_results)

        all_papers: list[CollectedPaper] = []
        offset = 0

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "research-engine/0.1.0"},
        ) as client:
            while offset < max_results:
                params["offset"] = offset
                try:
                    data = await self._request_page(client, params, offset, search_config.request_max_retries)
                except httpx.HTTPStatusError as e:
                    logger.error("S2 API HTTP 错误: %s", e)
                    break
                except Exception as e:
                    logger.error("S2 API 请求失败: %s", e)
                    break

                items = data.get("data", [])
                if not items:
                    break

                for item in items:
                    paper = self._normalize(item, search_config)
                    if paper:
                        all_papers.append(paper)

                total_available = data.get("total", 0)
                offset += len(items)
                if offset >= total_available or offset >= max_results:
                    break

                await asyncio.sleep(search_config.request_interval_seconds)

        logger.info("Semantic Scholar 返回 %d 篇论文", len(all_papers))
        return all_papers

    @staticmethod
    async def _request_page(
        client: httpx.AsyncClient,
        params: dict[str, str | int],
        offset: int,
        max_retries: int,
    ) -> dict:
        last_error: Exception | None = None
        max_attempts = max(1, max_retries + 1)

        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.get(S2_SEARCH_URL, params=params)
                if response.status_code == 429:
                    if attempt >= max_attempts:
                        logger.warning(
                            "S2 API 在 offset=%d 连续触发速率限制 %d 次，跳过该批次",
                            offset,
                            max_retries,
                        )
                        return {"data": []}

                    retry_after = response.headers.get("Retry-After")
                    try:
                        retry_after_seconds = max(1, int(retry_after)) if retry_after else 0
                    except ValueError:
                        retry_after_seconds = 0

                    wait_seconds = max(retry_after_seconds, get_retry_wait_seconds(attempt))
                    logger.warning(
                        "S2 API 速率限制，将在 %d 秒后重试 (%d/%d)",
                        wait_seconds,
                        attempt,
                        max_retries,
                    )
                    await asyncio.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, httpx.TimeoutException) as error:
                last_error = error
                if attempt >= max_attempts:
                    raise

                wait_seconds = get_retry_wait_seconds(attempt)
                logger.warning(
                    "S2 API 请求失败，将在 %d 秒后重试 (%d/%d): %s",
                    wait_seconds,
                    attempt,
                    max_retries,
                    error,
                )
                await asyncio.sleep(wait_seconds)

        raise RuntimeError(f"S2 API 请求失败: {last_error}")

    @staticmethod
    def _normalize(item: dict, config: PaperSearchConfig) -> CollectedPaper | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        # 过滤词检查
        title_lower = title.lower()
        for fw in config.filter_words:
            if fw.lower() in title_lower:
                return None

        authors = [
            a.get("name", "")
            for a in (item.get("authors") or [])
            if a.get("name")
        ]

        external_ids = item.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")
        doi = external_ids.get("DOI", "")

        paper_id = item.get("paperId", "")
        s2_url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""

        pub_date = item.get("publicationDate") or ""
        if not pub_date and item.get("year"):
            pub_date = f"{item['year']}-01-01"

        pdf_url = ""
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return CollectedPaper(
            paper_id=f"s2:{paper_id}" if paper_id else f"doi:{doi}",
            title=title,
            authors=authors[:10],
            abstract=(item.get("abstract") or "")[:2000],
            url=s2_url,
            pdf_url=pdf_url,
            source="semantic_scholar",
            published_date=pub_date[:10],
            categories=[],
            citation_count=item.get("citationCount") or 0,
            influential_citation_count=item.get("influentialCitationCount") or 0,
            venue=(item.get("venue") or "").strip(),
        )
