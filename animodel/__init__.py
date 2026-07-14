"""
animodel — model anime vkusu z MAL exportu + doporučení.

Veřejné API:
    from animodel import Config, TasteModel, Enricher, Recommender
    from animodel.mal import parse_export, split_by_status
    from animodel import report
"""
from .config import Config, ModelCfg, EnrichCfg, RecommendCfg
from .taste import TasteModel, Title, AttrEffect, Interaction, Triple, Cluster
from .enrich import Enricher, Enriched
from .recommend import Recommender, Recommendation

__all__ = [
    "Config", "ModelCfg", "EnrichCfg", "RecommendCfg",
    "TasteModel", "Title", "AttrEffect", "Interaction", "Triple", "Cluster",
    "Enricher", "Enriched", "Recommender", "Recommendation",
]
__version__ = "1.0.0"
