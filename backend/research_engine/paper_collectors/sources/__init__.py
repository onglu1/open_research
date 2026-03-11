from .arxiv import ArxivCollector
from .semantic_scholar import SemanticScholarCollector

REGISTERED_COLLECTORS = [
    ArxivCollector(),
    SemanticScholarCollector(),
]

__all__ = ["ArxivCollector", "SemanticScholarCollector", "REGISTERED_COLLECTORS"]
