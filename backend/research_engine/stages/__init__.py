from .s1_direction_expansion import DirectionExpansionStage
from .s2_frontier_scan import FrontierScanStage
from .s3_idea_discovery import IdeaDiscoveryStage
from .s4_feasibility_ranking import FeasibilityRankingStage
from .s5_deep_analysis import DeepAnalysisStage

STAGE_REGISTRY: dict[int, type] = {
    1: DirectionExpansionStage,
    2: FrontierScanStage,
    3: IdeaDiscoveryStage,
    4: FeasibilityRankingStage,
    5: DeepAnalysisStage,
}

__all__ = [
    "DirectionExpansionStage",
    "FrontierScanStage",
    "IdeaDiscoveryStage",
    "FeasibilityRankingStage",
    "DeepAnalysisStage",
    "STAGE_REGISTRY",
]
