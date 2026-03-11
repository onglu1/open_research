"""
配置系统：支持 YAML 文件加载 + 环境变量替换。

YAML 中形如 ${ENV_VAR} 的值会自动替换为对应环境变量。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _substitute_env(value: Any) -> Any:
    """递归替换字符串中的 ${VAR} 占位符。"""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            return os.environ.get(match.group(1), match.group(0))
        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    return value


# ── 各配置段的数据类 ──────────────────────────────────────────

@dataclass
class ResearchDirectionConfig:
    name: str = ""
    description: str = ""
    keywords_en: list[str] = field(default_factory=list)
    keywords_cn: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class ExternalDirection:
    name: str = ""
    name_en: str = ""
    description: str = ""


@dataclass
class ExpansionConfig:
    strategies: list[str] = field(default_factory=lambda: [
        "cross_domain", "methodology_transfer", "application_driven",
    ])
    external_directions: list[ExternalDirection] = field(default_factory=list)
    parallel_directions: list[ExternalDirection] = field(default_factory=list)
    forced_crosses: list[list[str]] = field(default_factory=list)


@dataclass
class PaperSearchConfig:
    time_range_days: int = 180
    max_papers_per_direction: int = 50
    filter_words: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=lambda: ["arxiv", "semantic_scholar"])
    request_max_retries: int = 3
    score_weights: dict[str, float] = field(default_factory=lambda: {
        "citation_count": 0.3,
        "recency": 0.3,
        "relevance": 0.4,
    })
    request_interval_seconds: float = 1.5


@dataclass
class FeasibilityDimension:
    weight: float = 0.2
    description: str = ""


@dataclass
class IdeaGenerationConfig:
    max_ideas: int = 20
    strategies: list[str] = field(default_factory=lambda: [
        "gap_filling", "method_combination", "cross_domain_transfer",
    ])
    encourage_cross_direction: bool = True


@dataclass
class FeasibilityConfig:
    dimensions: dict[str, FeasibilityDimension] = field(default_factory=lambda: {
        "novelty": FeasibilityDimension(weight=0.25),
        "feasibility": FeasibilityDimension(weight=0.25),
        "impact": FeasibilityDimension(weight=0.20),
        "effort": FeasibilityDimension(weight=0.15),
        "risk": FeasibilityDimension(weight=0.15),
    })
    tier_thresholds: dict[str, int] = field(default_factory=lambda: {
        "S": 85, "A": 70, "B": 55, "C": 0,
    })
    shuffle_ideas: bool = True
    max_ideas_per_batch: int = 8
    max_prompt_chars: int = 30000


@dataclass
class ResourceConstraints:
    gpu: str = ""
    time_budget_months: int = 6
    team_size: int = 1
    compute_budget: str = "moderate"


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_retries: int = 3


@dataclass
class KnowledgeBaseConfig:
    existing_research: str = ""
    existing_ideas: str = ""
    paper_notes_dir: str = ""


@dataclass
class PipelineConfig:
    stages: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    resume_from: int | None = None
    continue_from: str | None = None
    output_dir: str = "output"


# ── 顶层配置 ──────────────────────────────────────────────────

@dataclass
class ResearchConfig:
    research_direction: ResearchDirectionConfig = field(default_factory=ResearchDirectionConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    paper_search: PaperSearchConfig = field(default_factory=PaperSearchConfig)
    idea_generation: IdeaGenerationConfig = field(default_factory=IdeaGenerationConfig)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    resource_constraints: ResourceConstraints = field(default_factory=ResourceConstraints)
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# ── 加载与构建 ────────────────────────────────────────────────

def _coerce_dataclass_value(field_type: Any, raw: Any) -> Any:
    """按注解类型将原始 YAML 值递归转换为 dataclass 嵌套结构。"""
    if raw is None:
        return None

    if hasattr(field_type, "__dataclass_fields__") and isinstance(raw, dict):
        return _build_dataclass(field_type, raw)

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is list and isinstance(raw, list):
        item_type = args[0] if args else Any
        return [_coerce_dataclass_value(item_type, item) for item in raw]

    if origin is dict and isinstance(raw, dict):
        value_type = args[1] if len(args) > 1 else Any
        return {key: _coerce_dataclass_value(value_type, value) for key, value in raw.items()}

    if origin in (tuple, set) and isinstance(raw, origin):
        item_type = args[0] if args else Any
        return origin(_coerce_dataclass_value(item_type, item) for item in raw)

    if origin is not None and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_dataclass_value(non_none_args[0], raw)

    return raw


def _build_dataclass(cls: type, data: dict | None) -> Any:
    """从字典递归构建嵌套 dataclass 实例。"""
    if data is None:
        return cls()

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for f in cls.__dataclass_fields__.values():
        if f.name not in data:
            continue
        raw = data[f.name]
        field_type = type_hints.get(f.name, f.type)
        kwargs[f.name] = _coerce_dataclass_value(field_type, raw)

    return cls(**kwargs)


def load_config(path: str | Path) -> ResearchConfig:
    """从 YAML 文件加载完整配置，自动做环境变量替换。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    raw = _substitute_env(raw)
    return _build_dataclass(ResearchConfig, raw)


# ── 全局 Settings（不依赖 YAML，提供基础路径） ────────────────

@dataclass(frozen=True)
class Settings:
    api_host: str = os.getenv("RESEARCH_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("RESEARCH_API_PORT", "38417"))
    storage_dir: Path = Path(
        os.getenv("RESEARCH_STORAGE_DIR", str(Path(__file__).resolve().parents[1] / "storage"))
    )

    @property
    def log_dir(self) -> Path:
        return self.storage_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.storage_dir / "cache"


settings = Settings()

for _d in [settings.storage_dir, settings.log_dir, settings.cache_dir]:
    _d.mkdir(parents=True, exist_ok=True)
