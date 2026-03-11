"""论文采集器测试。"""

from __future__ import annotations

import httpx
import pytest

from research_engine.config import PaperSearchConfig
from research_engine.models import PaperQuery
from research_engine.paper_collectors.base import PaperCollector
from research_engine.paper_collectors.runner import PaperCollectorRunner
from research_engine.paper_collectors.sources.arxiv import ArxivCollector
from research_engine.paper_collectors.sources.semantic_scholar import SemanticScholarCollector


class TestArxivCollector:
    def test_build_query_with_keywords_and_categories(self):
        query = PaperQuery(
            keywords=["LLM inference", "speculative decoding"],
            categories=["cs.CL", "cs.LG"],
        )
        config = PaperSearchConfig()
        result = ArxivCollector._build_query(query, config)
        assert 'all:"LLM inference"' in result
        assert "cat:cs.CL" in result

    def test_build_query_with_filter_words(self):
        query = PaperQuery(keywords=["test"])
        config = PaperSearchConfig(filter_words=["survey"])
        result = ArxivCollector._build_query(query, config)
        assert "ANDNOT" in result
        assert "survey" in result

    def test_parse_empty_feed(self):
        xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        config = PaperSearchConfig(time_range_days=365)
        papers = ArxivCollector._parse_feed(xml, config)
        assert papers == []

    @pytest.mark.asyncio
    async def test_collect_honors_configured_retry_count(self, monkeypatch):
        calls = 0

        class DummyAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                request = httpx.Request("GET", "https://export.arxiv.org/api/query")
                raise httpx.ConnectError("network down", request=request)

        async def _noop_sleep(_seconds: float):
            return None

        monkeypatch.setattr(
            "research_engine.paper_collectors.sources.arxiv.httpx.AsyncClient",
            lambda *args, **kwargs: DummyAsyncClient(),
        )
        monkeypatch.setattr(
            "research_engine.paper_collectors.sources.arxiv.asyncio.sleep",
            _noop_sleep,
        )

        collector = ArxivCollector()
        query = PaperQuery(keywords=["test"])
        config = PaperSearchConfig(request_max_retries=2)

        with pytest.raises(httpx.ConnectError):
            await collector.collect(query, config)

        assert calls == 3


class TestSemanticScholarCollector:
    def test_normalize_empty_title_returns_none(self):
        config = PaperSearchConfig()
        result = SemanticScholarCollector._normalize({"title": ""}, config)
        assert result is None

    def test_normalize_filters_by_filter_words(self):
        config = PaperSearchConfig(filter_words=["survey"])
        result = SemanticScholarCollector._normalize(
            {"title": "A Survey of LLMs", "paperId": "abc"}, config
        )
        assert result is None

    def test_normalize_valid_paper(self):
        config = PaperSearchConfig()
        item = {
            "title": "Efficient LLM Inference",
            "paperId": "abc123",
            "abstract": "We propose...",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2025,
            "venue": "NeurIPS",
            "citationCount": 42,
            "influentialCitationCount": 5,
            "publicationDate": "2025-06-01",
            "externalIds": {"ArXiv": "2506.00001"},
            "url": "https://example.com",
        }
        paper = SemanticScholarCollector._normalize(item, config)
        assert paper is not None
        assert paper.title == "Efficient LLM Inference"
        assert paper.citation_count == 42
        assert paper.source == "semantic_scholar"
        assert "Alice" in paper.authors

    @pytest.mark.asyncio
    async def test_request_page_honors_configured_retry_count(self, monkeypatch):
        calls = 0

        class DummyResponse:
            status_code = 429
            headers = {}

            def json(self):
                return {"data": []}

            def raise_for_status(self):
                return None

        class DummyClient:
            async def get(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                return DummyResponse()

        async def _noop_sleep(_seconds: float):
            return None

        monkeypatch.setattr(
            "research_engine.paper_collectors.sources.semantic_scholar.asyncio.sleep",
            _noop_sleep,
        )

        data = await SemanticScholarCollector._request_page(
            DummyClient(),
            {"query": "test"},
            0,
            max_retries=2,
        )

        assert data == {"data": []}
        assert calls == 3


class TestPaperCollectorRunner:
    def test_score_paper_basic(self):
        from research_engine.models import CollectedPaper
        config = PaperSearchConfig()
        paper = CollectedPaper(
            paper_id="test:1",
            title="Test",
            citation_count=100,
            published_date="2025-01-01",
        )
        score = PaperCollectorRunner._score_paper(paper, config)
        assert 0 <= score <= 100
