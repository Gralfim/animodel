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
    side_story_weight: float = 0.5    # příspěvek vedlejšího obsahu (OVA/speciál/
                                      # side story) do franšízové váhy vs. 1.0
                                      # hlavní řady; 1.0 = bez rozlišení,
                                      # 0.0 = vedlejší obsah z modelu vyřadit
    intensity_lexicon: str = "intensity.yaml"  # osa náročnosti: generuj přes
                                      # --gen-intensity, hodnoty uprav ručně;
                                      # když soubor neexistuje, použije se
                                      # vestavěný default (intensity.py)


@dataclass
class EnrichCfg:
    use_jikan: bool = True            # False = nouzový AniList-only režim
                                       # (--no-jikan): žánry/synopse/dekáda/
                                       # franšízy se berou z AniListu, MAL rec
                                       # graf se přeskočí. Pro výpadky Jikanu.
    use_anilist: bool = True
    anilist_min_rank: int = 30        # ignoruj okrajové AniList tagy (rank < 30 %)
    include_studios: bool = True
    include_staff: bool = False       # signál po režisérech/scenáristech; navíc
                                       # 1 Jikan volání na titul (/staff endpoint),
                                       # proto default vypnuto -- zapni, když chceš
                                       # cenu za první běh a mít to napojené
    use_shikimori: bool = False       # další nezávislý zdroj "podobných anime"
                                       # kandidátů (viz sources/shikimori.py) --
                                       # default vypnuto, není naživo ověřené


@dataclass
class RecommendCfg:
    high_score: float = 8.0           # od jaké známky brát titul jako "oblíbený" (seed)
    candidates_per_seed: int = 25     # kolik doporučení/podobných tahat na seed (Jikan/AniList)
    max_seeds: int = 40
    seeds_per_franchise: int = 2      # max. seedů z jedné franšízy (nejlépe
                                      # hodnocené řady vyhrávají); 0 = bez limitu.
                                      # Bez něj pětiřadá oblíbená franšíza sebere
                                      # 5 slotů a hlasuje 5x skoro stejným rec grafem.
    # váhy kompozitního skóre pro řazení doporučení (z-skóry se sčítají)
    w_taste_fit: float = 1.0          # shoda s mými afinitními efekty + klastry
    w_cf: float = 1.0                 # collaborative signál (rec. graf / podobní uživatelé)
    w_quality: float = 0.3            # mírná preference vyššího komunitního skóre
    min_community: float = 6.5        # nedoporučuj pod tímto komunitním skóre
    top_n: int = 40                   # kolik doporučení ve globálním přehledu
    top_per_cluster: int = 15         # kolik doporučení na náladu v per-klastr pohledu
    use_user_cf: bool = False         # zapnout user-based CF přes AniList (drahé, pomalé)
    user_cf_min_overlap: int = 4      # min. počet sdílených (nišových) seedů s uživatelem
    user_cf_top_users: int = 120      # kolik nejpodobnějších uživatelů použít
    user_cf_seed_count: int = 25      # kolik nejméně populárních seedů použít
    user_cf_users_per_seed: int = 100 # kolik uživatelů stáhnout na jeden seed


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
