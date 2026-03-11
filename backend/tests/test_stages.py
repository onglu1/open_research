"""阶段执行测试（使用降级 LLM）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_engine.config import ResearchConfig
from research_engine.models import KnowledgeBaseContent, StageResult
from research_engine.services.llm import LLMService
from research_engine.stages.base import StageContext
from research_engine.stages.s1_direction_expansion import DirectionExpansionStage
from research_engine.stages.s3_idea_discovery import IdeaDiscoveryStage
from research_engine.stages.s4_feasibility_ranking import FeasibilityRankingStage


def _make_ctx(
    config: ResearchConfig,
    llm: LLMService,
    tmp_path: Path,
    previous: dict | None = None,
) -> StageContext:
    return StageContext(
        config=config,
        llm=llm,
        knowledge=KnowledgeBaseContent(),
        session_dir=tmp_path,
        previous_outputs=previous or {},
    )


class TestDirectionExpansionStage:
    @pytest.mark.asyncio
    async def test_heuristic_fallback(self, sample_config, fallback_llm, tmp_path):
        """LLM 不可用时应使用 heuristic 结果。"""
        stage = DirectionExpansionStage()
        ctx = _make_ctx(sample_config, fallback_llm, tmp_path)
        result = await stage.run(ctx)
        assert isinstance(result, StageResult)
        assert "core_direction" in result.data
        assert "parallel_directions" in result.data

    @pytest.mark.asyncio
    async def test_save_and_reload(self, sample_config, fallback_llm, tmp_path):
        """输出应能保存并被 check_existing_output 识别。"""
        stage = DirectionExpansionStage()
        ctx = _make_ctx(sample_config, fallback_llm, tmp_path)
        result = await stage.execute(ctx)
        assert result.success or result.metadata.status == "degraded"

        reloaded = stage.check_existing_output(tmp_path)
        assert reloaded is not None
        assert "core_direction" in reloaded


class TestIdeaDiscoveryStage:
    @pytest.mark.asyncio
    async def test_no_frontier_data(self, sample_config, fallback_llm, tmp_path):
        """Stage 2 输出为空时不应崩溃。"""
        stage = IdeaDiscoveryStage()
        ctx = _make_ctx(sample_config, fallback_llm, tmp_path, previous={2: {}})
        result = await stage.run(ctx)
        assert isinstance(result, StageResult)


class TestFeasibilityRankingStage:
    @pytest.mark.asyncio
    async def test_heuristic_ranking(self, sample_config, fallback_llm, tmp_path, sample_s3_output):
        """LLM 不可用时应使用 heuristic 排序。"""
        stage = FeasibilityRankingStage()
        ctx = _make_ctx(sample_config, fallback_llm, tmp_path, previous={3: sample_s3_output})
        result = await stage.run(ctx)
        assert isinstance(result, StageResult)
        ranked = result.data.get("ranked_ideas", [])
        assert len(ranked) > 0
        assert ranked[0].get("idea_id") == "IDEA-001"
