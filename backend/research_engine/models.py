"""
全领域数据模型：纯 Pydantic / dataclass，不依赖数据库 ORM。

每个阶段的输入与输出都由此处的模型定义，阶段间以 JSON 序列化传递。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ── 通用 ─────────────────────────────────────────────────────

@dataclass
class StageMetadata:
    """阶段执行元信息，随每阶段输出一起持久化到 metadata.json。"""
    stage_number: int
    stage_name: str
    status: str = "pending"          # pending / running / completed / failed / degraded
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int = 0
    error: str | None = None
    degraded: bool = False
    degraded_reason: str | None = None


# ── Stage 1: Direction Expansion ─────────────────────────────

@dataclass
class ResearchDirection:
    """一个研究方向（核心方向或扩展方向）。"""
    name: str = ""
    name_en: str = ""
    description: str = ""
    relationship: str = ""           # core / parallel / cross / external
    strategies: list[str] = field(default_factory=list)
    keywords_en: list[str] = field(default_factory=list)
    keywords_cn: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    cross_value: str = ""
    key_researchers: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    relevant_venues: list[str] = field(default_factory=list)


@dataclass
class CrossPollinationOpportunity:
    direction_a: str = ""
    direction_b: str = ""
    description: str = ""
    potential_value: str = ""


@dataclass
class DirectionExpansionResult:
    """Stage 1 的完整输出。"""
    core_direction: ResearchDirection = field(default_factory=ResearchDirection)
    parallel_directions: list[ResearchDirection] = field(default_factory=list)
    cross_pollination_opportunities: list[CrossPollinationOpportunity] = field(default_factory=list)
    relevant_venues: list[str] = field(default_factory=list)


# ── Stage 2: Frontier Scan ───────────────────────────────────

@dataclass
class CollectedPaper:
    """标准化论文记录（对标 NormalizedJobDraft）。"""
    paper_id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    source: str = ""                 # arxiv / semantic_scholar / crossref
    published_date: str = ""
    categories: list[str] = field(default_factory=list)
    citation_count: int = 0
    influential_citation_count: int = 0
    venue: str = ""
    relevance_score: float = 0.0
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectionScanResult:
    """单个方向的前沿扫描结果。"""
    direction_name: str = ""
    landscape_summary: str = ""
    hot_topics: list[str] = field(default_factory=list)
    research_gaps: list[str] = field(default_factory=list)
    trend_signals: list[str] = field(default_factory=list)
    key_methods: list[str] = field(default_factory=list)
    paper_count: int = 0
    papers: list[CollectedPaper] = field(default_factory=list)


@dataclass
class FrontierScanResult:
    """Stage 2 的完整输出。"""
    direction_scans: list[DirectionScanResult] = field(default_factory=list)
    total_papers: int = 0
    deduplicated_papers: int = 0


# ── Stage 3: Idea Discovery ─────────────────────────────────

@dataclass
class ResearchIdea:
    """一条候选研究 idea。"""
    id: str = ""
    title_cn: str = ""
    title_en: str = ""
    one_liner: str = ""
    description: str = ""
    motivation: str = ""
    method_sketch: str = ""
    novelty_points: list[str] = field(default_factory=list)
    expected_contributions: list[str] = field(default_factory=list)
    validation_plan: str = ""
    source_directions: list[str] = field(default_factory=list)
    discovery_strategy: str = ""
    related_gaps: list[str] = field(default_factory=list)
    related_papers: list[str] = field(default_factory=list)
    estimated_effort: str = ""
    risk_factors: list[str] = field(default_factory=list)
    preliminary_score: float = 0.0


@dataclass
class IdeaDiscoveryResult:
    """Stage 3 的完整输出。"""
    ideas: list[ResearchIdea] = field(default_factory=list)
    generation_strategies_used: list[str] = field(default_factory=list)


# ── Stage 4: Feasibility Ranking ─────────────────────────────

@dataclass
class DimensionScore:
    score: float = 0.0
    rationale: str = ""


@dataclass
class RankedIdea:
    """经过排序的 idea，在 ResearchIdea 基础上增加评分细节。"""
    idea_id: str = ""
    title_cn: str = ""
    title_en: str = ""
    one_liner: str = ""
    dimension_scores: dict[str, DimensionScore] = field(default_factory=dict)
    weighted_total: float = 0.0
    tier: str = ""                   # S / A / B / C
    recommendation: str = ""
    next_steps: list[str] = field(default_factory=list)
    target_venues: list[str] = field(default_factory=list)
    time_estimate: str = ""
    key_dependencies: list[str] = field(default_factory=list)


@dataclass
class FeasibilityRankingResult:
    """Stage 4 的完整输出。"""
    ranked_ideas: list[RankedIdea] = field(default_factory=list)
    ranking_rationale: str = ""


# ── Stage 5: Deep Analysis ──────────────────────────────────

@dataclass
class KeyPaperAnalysis:
    paper_title: str = ""
    paper_url: str = ""
    relevance: str = ""
    key_findings: str = ""
    limitations: str = ""


@dataclass
class ExperimentDesign:
    research_questions: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    ablations: list[str] = field(default_factory=list)
    expected_results: str = ""


@dataclass
class Milestone:
    name: str = ""
    duration: str = ""
    deliverables: list[str] = field(default_factory=list)


@dataclass
class DeepAnalysis:
    """单个高优先级 idea 的深度分析报告。"""
    idea_id: str = ""
    title_cn: str = ""
    title_en: str = ""
    executive_summary: str = ""
    related_work_review: str = ""
    key_paper_analysis: list[KeyPaperAnalysis] = field(default_factory=list)
    technical_approach: str = ""
    system_architecture: str = ""
    key_components: list[str] = field(default_factory=list)
    experiment_design: ExperimentDesign = field(default_factory=ExperimentDesign)
    risk_assessment: str = ""
    resource_risks: str = ""
    timeline: list[Milestone] = field(default_factory=list)
    target_venues: list[str] = field(default_factory=list)
    go_no_go_verdict: str = ""


@dataclass
class DeepAnalysisResult:
    """Stage 5 的完整输出。"""
    analyses: list[DeepAnalysis] = field(default_factory=list)


# ── 论文采集相关（供 paper_collectors 使用）───────────────────

@dataclass
class PaperQuery:
    """向论文源发起的查询请求。"""
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    direction_name: str = ""
    max_results: int = 50
    time_range_days: int = 180


@dataclass
class CollectorRunResult:
    """单个采集器的运行结果（对标原 CollectorRunResult）。"""
    source_name: str = ""
    status: str = "pending"          # success / empty / error
    papers: list[CollectedPaper] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0


# ── 知识库 ───────────────────────────────────────────────────

@dataclass
class KnowledgeBaseContent:
    """加载后的个人知识库内容。"""
    existing_research: str = ""
    existing_ideas: str = ""
    paper_notes: list[str] = field(default_factory=list)
    paper_notes_files: list[str] = field(default_factory=list)


# ── 阶段上下文与结果 ────────────────────────────────────────

@dataclass
class StageResult:
    """每个阶段统一返回的结果容器。"""
    stage_number: int = 0
    stage_name: str = ""
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    metadata: StageMetadata = field(default_factory=lambda: StageMetadata(0, ""))
    markdown: str = ""
