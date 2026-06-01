"""
config.py — POUZE laditelné parametry. Žádné seznamy atributů!
Atributy se objevují automaticky z dat (to je hlavní rozdíl oproti starému
přístupu s ručně udržovaným config.yaml).

Načítá volitelný uživatelský config.yaml a překrývá jím defaulty.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelCfg:
    shrinkage_k: float = 8.0          # síla smrštění efektů k nule (vyšší = konzervativnější)
    min_attr_count: float = 4.0       # min. efektivní počet titulů, aby atribut vstoupil
    interaction_min_count: float = 8.0
    interaction_min_lift: float = 0.30
    n_clusters: int | None = None     # None = automaticky podle siluety (4–7)
    aggregate_franchises: bool = True # sequel/prequel → jeden vážený datový bod


@dataclass
class EnrichCfg:
    use_anilist: bool = True
    anilist_min_rank: int = 30        # ignoruj okrajové AniList tagy (rank < 30 %)
    include_studios: bool = True


@dataclass
class RecommendCfg:
    high_score: float = 8.0           # od jaké známky brát titul jako "oblíbený" (seed)
    candidates_per_seed: int = 25     # kolik doporučení/podobných tahat na seed (Jikan/AniList)
    max_seeds: int = 40
    # váhy kompozitního skóre pro řazení doporučení (z-skóry se sčítají)
    w_taste_fit: float = 1.0          # shoda s mými afinitními efekty + klastry
    w_cf: float = 1.0                 # collaborative signál (rec. graf / podobní uživatelé)
    w_quality: float = 0.3            # mírná preference vyššího komunitního skóre
    min_community: float = 6.5        # nedoporučuj pod tímto komunitním skóre
    top_n: int = 40                   # kolik doporučení vypsat
    use_user_cf: bool = False         # zapnout user-based CF přes AniList (drahé, pomalé)
    user_cf_min_overlap: int = 15
    user_cf_top_users: int = 50


@dataclass
class Config:
    mal_export: str = "animelist.xml"
    cache_dir: str = "cache"
    out_dir: str = "output"
    model: ModelCfg = field(default_factory=ModelCfg)
    enrich: EnrichCfg = field(default_factory=EnrichCfg)
    recommend: RecommendCfg = field(default_factory=RecommendCfg)

    @staticmethod
    def load(path: str | None) -> "Config":
        cfg = Config()
        if path and Path(path).exists():
            import yaml
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            for k, v in raw.items():
                if k in ("model", "enrich", "recommend") and isinstance(v, dict):
                    sub = getattr(cfg, k)
                    for kk, vv in v.items():
                        if hasattr(sub, kk):
                            setattr(sub, kk, vv)
                elif hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def as_dict(self):
        return asdict(self)
