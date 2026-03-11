"""pytest 配置与公共 fixture。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from research_engine.config import (
    ExpansionConfig,
    ExternalDirection,
    FeasibilityConfig,
    IdeaGenerationConfig,
    KnowledgeBaseConfig,
    LLMConfig,
    PaperSearchConfig,
    PipelineConfig,
    ResearchConfig,
    ResearchDirectionConfig,
    ResourceConstraints,
)
from research_engine.models import KnowledgeBaseContent
from research_engine.services.llm import LLMService


@pytest.fixture
def sample_config(tmp_path: Path) -> ResearchConfig:
    """构建一个用于测试的最小配置。"""
    return ResearchConfig(
        research_direction=ResearchDirectionConfig(
            name="测试方向",
            description="这是一个测试研究方向",
            keywords_en=["test", "unit testing"],
            keywords_cn=["测试"],
            categories=["cs.SE"],
        ),
        expansion=ExpansionConfig(
            strategies=["cross_domain"],
            parallel_directions=[
                ExternalDirection(name="平行方向A", name_en="Parallel A", description="desc A"),
            ],
        ),
        paper_search=PaperSearchConfig(
            time_range_days=30,
            max_papers_per_direction=5,
            sources=["arxiv"],
            request_interval_seconds=0.1,
        ),
        idea_generation=IdeaGenerationConfig(max_ideas=3),
        feasibility=FeasibilityConfig(),
        resource_constraints=ResourceConstraints(gpu="1x GPU", time_budget_months=3, team_size=1),
        llm=LLMConfig(provider="fallback"),
        knowledge_base=KnowledgeBaseConfig(),
        pipeline=PipelineConfig(output_dir=str(tmp_path / "output")),
    )


@pytest.fixture
def empty_knowledge() -> KnowledgeBaseContent:
    return KnowledgeBaseContent()


@pytest.fixture
def fallback_llm() -> LLMService:
    """不调用外部 API 的 LLM 服务（降级模式）。"""
    return LLMService(LLMConfig(provider="fallback"))


@pytest.fixture
def sample_s1_output() -> dict:
    """Stage 1 的样例输出。"""
    return {
        "core_direction": {
            "name": "测试方向",
            "name_en": "Test Direction",
            "description": "desc",
            "keywords_en": ["test"],
            "keywords_cn": ["测试"],
            "categories": ["cs.SE"],
            "key_researchers": [],
            "relevant_venues": ["ICSE"],
            "relationship": "core",
            "strategies": [],
            "cross_value": "",
            "relevance_score": 1.0,
        },
        "parallel_directions": [
            {
                "name": "平行方向A",
                "name_en": "Parallel A",
                "description": "desc A",
                "relationship": "parallel",
                "strategies": [],
                "keywords_en": ["parallel test"],
                "keywords_cn": [],
                "categories": ["cs.SE"],
                "cross_value": "",
                "key_researchers": [],
                "relevance_score": 0.7,
                "relevant_venues": [],
            },
        ],
        "cross_pollination_opportunities": [],
        "relevant_venues": ["ICSE", "FSE"],
    }


@pytest.fixture
def sample_s2_output() -> dict:
    """Stage 2 的样例输出。"""
    return {
        "direction_scans": [
            {
                "direction_name": "测试方向",
                "landscape_summary": "该方向近期关注自动化测试。",
                "hot_topics": ["自动测试生成", "变异测试"],
                "research_gaps": ["跨语言测试迁移研究不足"],
                "trend_signals": ["LLM 辅助测试"],
                "key_methods": ["符号执行", "模糊测试"],
                "paper_count": 5,
                "papers": [],
            },
        ],
        "total_papers": 5,
        "deduplicated_papers": 5,
    }


@pytest.fixture
def sample_s3_output() -> dict:
    """Stage 3 的样例输出。"""
    return {
        "ideas": [
            {
                "id": "IDEA-001",
                "title_cn": "测试 Idea",
                "title_en": "Test Idea",
                "one_liner": "一个测试用的想法",
                "description": "desc",
                "motivation": "motivation",
                "method_sketch": "method",
                "novelty_points": ["new"],
                "expected_contributions": ["contrib"],
                "validation_plan": "plan",
                "source_directions": ["测试方向"],
                "discovery_strategy": "gap_filling",
                "related_gaps": ["gap"],
                "related_papers": [],
                "estimated_effort": "medium",
                "risk_factors": ["risk"],
                "preliminary_score": 72.0,
            },
        ],
        "generation_strategies_used": ["gap_filling"],
    }
