"""
config.py — POUZE laditelné parametry. Žádné seznamy atributů!
Atributy se objevují automaticky z dat (to je hlavní rozdíl oproti starému
přístupu s ručně udržovaným config.yaml).

Načítá volitelný uživatelský config.yaml a překrývá jím defaulty.
Neznámé klíče hlásí warningem -- při přejmenování/odstranění parametru by
jinak staré klíče v uživatelově configu tiše přestaly působit (přesně to
se stalo u user_cf_min_overlap/user_cf_top_users při senpai redesignu).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ModelCfg:
    shrinkage_k: float = 8.0          # síla smrštění efektů k nule (vyšší = konzervativnější)
    min_attr_count: float = 4.0       # min. efektivní počet titulů, aby atribut vstoupil
    interaction_min_count: float = 8.0  # min. VÁŽENÁ podpora páru (Σ w_a·w_b·w_titulu)
    interaction_min_lift: float = 0.30  # práh na SMRŠTĚNÝ lift (n/(n+K), jako efekty);
                                        # když synergií prochází málo, sniž (např. 0.2)
    interaction_triples: bool = False   # EXPERIMENT: hierarchické synergie trojic
                                        # nad klastrovými signaturami (jádra nálad);
                                        # sdílí prahy interaction_min_count/_lift
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
    # MAL data (žánry/relations/rec graf/…) se tahají z Jikan-KOMPATIBILNÍHO
    # API. Default je Tenrai -- 1:1 mirror Jikan v4 schématu, za Cloudflare,
    # výrazně spolehlivější (Jikan má od 2026-07 trvalé 504). Cache je společná
    # (klíč dle endpointu, ne hostu) a schéma identické, takže přepnutí je
    # bezpečné i s teplou cache. Jikan zůstává jako fallback v config.example.
    anime_api_base_url: str = "https://api.tenrai.org/v1"
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
    # váhy kompozitního skóre pro řazení doporučení (z-skóry se sčítají).
    # Item-CF (graf podobnosti) a user-CF jsou ODDĚLENÉ složky s vlastním
    # z-skóre -- dřív sdílely jeden kbelík a šikmé rozdělení hlasů grafu
    # user-CF utopilo a přebilo i model vkusu (změřeno 2026-07: |příspěvek|
    # 6.4 vs 1.6; hlasy grafu se navíc před z-skórem tlumí log1p).
    w_taste_fit: float = 1.0          # shoda s mými afinitními efekty + klastry
    w_cf: float = 0.8                 # graf podobnosti (MAL/AniList/Shikimori)
    w_user_cf: float = 0.6            # user-based CF (podobní uživatelé)
    w_quality: float = 0.3            # mírná preference vyššího komunitního skóre
    # prahy minimální síly hrany v grafu podobnosti -- slabé hrany (jednotky
    # hlasů, automatická doporučení) jsou spíš šum než skutečná podobnost
    min_mal_rec_votes: int = 5        # MAL: min. počet uživatelských hlasů
    min_anilist_rec_rating: int = 3   # AniList: min. čistý rating doporučení
    min_community: float = 6.5        # nedoporučuj pod tímto komunitním skóre
    top_n: int = 40                   # kolik doporučení ve globálním přehledu
    top_per_cluster: int = 15         # kolik doporučení na náladu v per-klastr pohledu
    use_user_cf: bool = False         # zapnout user-based CF přes AniList (drahé, pomalé)
    # -- senpai pipeline (viz usercf.py): discovery přes nišové tituly ->
    #    plné seznamy kandidátů -> podobnost na plném překryvu -> pár senpai
    user_cf_seed_count: int = 50      # kolik nejméně populárních titulů použít k discovery
    user_cf_users_per_seed: int = 100 # kolik sledujících vzorkovat na jeden seed
    user_cf_min_sample_overlap: int = 2   # min. sdílených nišových seedů pro kvalifikaci
                                      # (1 může být náhoda, 2+ je vzorec)
    user_cf_candidate_pool: int = 200 # kolik kandidátů vyhodnotit na PLNÝCH seznamech
    user_cf_senpai_count: int = 20    # kolik nejpodobnějších uživatelů (senpai) vybrat
    user_cf_min_full_overlap: int = 40    # min. společně ohodnocených titulů (plný překryv)
    user_cf_shrink_k: float = 50.0    # smrštění podobnosti n/(n+K) -- málo překryvu, málo důvěry
    user_cf_fav_score: float = 9.0    # od jaké MÉ známky je titul "oblíbený" pro
                                      # výpočet pokrytí senpaiem
    user_cf_fav_miss_penalty: float = 0.3
                                      # lehký negativní signál: senpai, který mé
                                      # oblíbené tituly nemá ohodnocené ANI na PTW,
                                      # ztrácí až tolik ze skóre (0.3 = -30 % při
                                      # nulovém pokrytí; 0 = penalizaci vypnout).
                                      # Dropnutý-bez-známky se počítá jako nepokrytý;
                                      # dropnutý SE známkou je řádný (a výmluvný)
                                      # datový bod v podobnosti.
    user_cf_exclude_users: list = field(default_factory=list)
                                      # AniList jména vyloučená z hledání senpai
                                      # (vlastní/alt účty -- import vlastního MAL
                                      # seznamu má podobnost 1.00 a je k ničemu).
                                      # Jméno z MAL exportu přidává cli.py automaticky.


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
                        else:
                            log.warning(
                                f"config {path}: neznámý klíč '{k}.{kk}' -- "
                                f"ignoruje se (překlep, nebo parametr už neexistuje?)"
                            )
                elif hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    log.warning(f"config {path}: neznámý klíč '{k}' -- ignoruje se")
        return cfg

    def as_dict(self):
        return asdict(self)
