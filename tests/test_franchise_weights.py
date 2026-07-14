"""Franšízové váhy s rozlišením hlavní řady vs. vedlejší obsah
(OVA/speciály/side story), propsání vah do klastrování a limit seedů
na franšízu v doporučeních."""
import math

import pytest

from animodel.attributes import AttrValue
from animodel.config import Config
from animodel.enrich import Enricher, Enriched, _is_side_content, SIDE_FORMATS
from animodel.mal import MalEntry
from animodel.recommend import Recommender
from animodel.taste import TasteModel, Title


# ── detekce vedlejšího obsahu ────────────────────────────────────────────

def _enriched(jikan=None, anilist=None):
    return Enriched(mal_id=1, title="X", title_en="", community=8.0,
                    attrs={}, jikan=jikan, anilist=anilist)


def test_side_detected_by_jikan_type():
    assert _is_side_content(_enriched(jikan={"type": "OVA"})) is True
    assert _is_side_content(_enriched(jikan={"type": "Special"})) is True
    assert _is_side_content(_enriched(jikan={"type": "TV"})) is False
    assert _is_side_content(_enriched(jikan={"type": "Movie"})) is False


def test_side_detected_by_anilist_format():
    assert _is_side_content(_enriched(anilist={"format": "SPECIAL"})) is True
    assert _is_side_content(_enriched(anilist={"format": "OVA"})) is True
    # ONA záměrně NENÍ vedlejší -- plnohodnotné série vycházejí jako ONA
    assert _is_side_content(_enriched(anilist={"format": "ONA"})) is False
    assert _is_side_content(_enriched(anilist={"format": "TV"})) is False


def test_side_detected_by_parent_story_relation():
    jikan = {"type": "TV", "relations": [
        {"relation": "Parent story", "entry": [{"mal_id": 5, "type": "anime"}]},
    ]}
    assert _is_side_content(_enriched(jikan=jikan)) is True

    anilist = {"format": "TV", "relations": {"edges": [
        {"relationType": "PARENT", "node": {"idMal": 5, "type": "ANIME"}},
    ]}}
    assert _is_side_content(_enriched(anilist=anilist)) is True
    # PARENT na manga uzel (předloha) není vedlejšost anime
    anilist_manga = {"format": "TV", "relations": {"edges": [
        {"relationType": "PARENT", "node": {"idMal": 5, "type": "MANGA"}},
    ]}}
    assert _is_side_content(_enriched(anilist=anilist_manga)) is False


# ── výpočet vah ve build_titles ──────────────────────────────────────────

def _media(mal_id, fmt="TV", relations=None):
    return {
        "id": mal_id * 10, "idMal": mal_id,
        "title": {"romaji": f"Anime {mal_id}", "english": ""},
        "genres": ["Comedy"], "tags": [], "studios": {"nodes": []},
        "format": fmt, "source": "MANGA", "averageScore": 80,
        "seasonYear": 2020, "startDate": {"year": 2020},
        "description": "", "relations": {"edges": relations or []},
    }


class FakeAniListClient:
    def __init__(self, media_by_id):
        self.media = media_by_id

    def get_anime_batch(self, mal_ids, show_progress=True):
        return {mid: self.media[mid] for mid in mal_ids if mid in self.media}


def _entry(mal_id, score):
    return MalEntry(mal_id=mal_id, title=f"T{mal_id}", type="TV", episodes=12,
                    watched_episodes=12, score=score, status="Completed",
                    start_date="", finish_date="", rewatched=0)


def _cfg(side_w=None):
    cfg = Config()
    cfg.enrich.use_jikan = False
    if side_w is not None:
        cfg.model.side_story_weight = side_w
    return cfg


def _franchise_media():
    """2 hlavní řady (TV, sequel/prequel) + 1 OVA side story."""
    m1 = _media(1, "TV", [{"relationType": "SEQUEL", "node": {"idMal": 2, "type": "ANIME"}}])
    m2 = _media(2, "TV", [
        {"relationType": "PREQUEL", "node": {"idMal": 1, "type": "ANIME"}},
        {"relationType": "SIDE_STORY", "node": {"idMal": 3, "type": "ANIME"}},
    ])
    m3 = _media(3, "OVA", [{"relationType": "PARENT", "node": {"idMal": 2, "type": "ANIME"}}])
    return {1: m1, 2: m2, 3: m3}


def test_side_content_gets_reduced_weight_within_franchise():
    enricher = Enricher(_cfg(), anilist=FakeAniListClient(_franchise_media()))
    titles = enricher.build_titles([_entry(1, 9), _entry(2, 8), _entry(3, 7)],
                                   show_progress=False)
    by_id = {t.mal_id: t for t in titles}

    # contrib = {1: 1.0, 2: 1.0, 3: 0.5} -> k_eff = 2.5
    k_eff = 2.5
    assert by_id[1].weight == pytest.approx(1.0 / math.sqrt(k_eff))
    assert by_id[2].weight == pytest.approx(1.0 / math.sqrt(k_eff))
    assert by_id[3].weight == pytest.approx(0.5 / math.sqrt(k_eff))
    # OVA má poloviční vliv oproti hlavním řadám
    assert by_id[3].weight == pytest.approx(by_id[1].weight * 0.5)
    # všichni členové sdílejí series_root
    roots = {t.series_root for t in titles}
    assert len(roots) == 1 and None not in roots


def test_side_story_weight_one_reproduces_plain_sqrt_k():
    enricher = Enricher(_cfg(side_w=1.0), anilist=FakeAniListClient(_franchise_media()))
    titles = enricher.build_titles([_entry(1, 9), _entry(2, 8), _entry(3, 7)],
                                   show_progress=False)
    for t in titles:
        assert t.weight == pytest.approx(1.0 / math.sqrt(3))


def test_standalone_ova_keeps_full_weight():
    """Vedlejšost se vyhodnocuje jen UVNITŘ franšíz -- samostatná OVA bez
    vazeb je plnohodnotný titul."""
    enricher = Enricher(_cfg(), anilist=FakeAniListClient({7: _media(7, "OVA")}))
    titles = enricher.build_titles([_entry(7, 9)], show_progress=False)
    assert titles[0].weight == 1.0
    assert titles[0].series_root is None


# ── váhy v klastrování ───────────────────────────────────────────────────

def test_cluster_mean_score_is_weighted():
    """Klastr s tituly o vahách 1.0 (score 8) a 0.25 (score 10) musí mít
    vážený průměr 8.4, ne nevážených 9.0."""
    titles = []
    for i in range(8):
        titles.append(Title(mal_id=100 + i, title=f"RC{i}", user_score=8.0,
                            community=7.5,
                            attrs={"comedy": AttrValue("genre", 1.0, "Comedy"),
                                   "romance": AttrValue("genre", 1.0, "Romance")},
                            weight=1.0))
    for i in range(8):
        titles.append(Title(mal_id=200 + i, title=f"RCw{i}", user_score=10.0,
                            community=7.5,
                            attrs={"comedy": AttrValue("genre", 1.0, "Comedy"),
                                   "romance": AttrValue("genre", 1.0, "Romance")},
                            weight=0.25))
    for i in range(16):
        titles.append(Title(mal_id=300 + i, title=f"DR{i}", user_score=9.0,
                            community=8.0,
                            attrs={"drama": AttrValue("genre", 1.0, "Drama"),
                                   "psychological": AttrValue("genre", 1.0, "Psychological")},
                            weight=1.0))

    m = TasteModel(shrinkage_k=8.0, min_attr_count=2.0).fit(titles, n_clusters=2)
    assert len(m.clusters) == 2
    romcom = next(c for c in m.clusters
                  if any(s[0] == "comedy" for s in c.signature))
    expected = (8.0 * 8 * 1.0 + 10.0 * 8 * 0.25) / (8 * 1.0 + 8 * 0.25)
    assert romcom.mean_user_score == pytest.approx(expected)   # 8.4
    assert romcom.size == 16   # size zůstává počet titulů (zobrazovací údaj)


# ── limit seedů na franšízu ──────────────────────────────────────────────

class _StubModel:
    u_mean = 8.0
    clusters = []

    def top_effects(self, n=40, sign=1):
        return []


class _StubEnricher:
    jikan = None
    anilist = None
    shikimori = None


def _seed_title(mal_id, score, root=None):
    return Title(mal_id=mal_id, title=f"T{mal_id}", user_score=score,
                 community=8.0, attrs={}, series_root=root)


def test_seeds_capped_per_franchise_best_seasons_win():
    cfg = Config()
    assert cfg.recommend.seeds_per_franchise == 2
    rec = Recommender(_StubModel(), _StubEnricher(), cfg)
    titles = [
        _seed_title(1, 10.0, root=1), _seed_title(2, 9.5, root=1),
        _seed_title(3, 9.0, root=1), _seed_title(4, 8.5, root=1),   # 4řadá franšíza
        _seed_title(10, 8.4),                                        # standalone
        _seed_title(11, 8.2, root=11), _seed_title(12, 8.1, root=11),
    ]
    seeds = rec._seeds(titles)
    ids = [t.mal_id for t in seeds]
    # z franšízy 1 jen dvě nejlepší řady; standalone a druhá franšíza nedotčené
    assert ids == [1, 2, 10, 11, 12]


def test_seeds_cap_zero_disables_limit():
    cfg = Config()
    cfg.recommend.seeds_per_franchise = 0
    rec = Recommender(_StubModel(), _StubEnricher(), cfg)
    titles = [_seed_title(i, 10.0 - i * 0.1, root=1) for i in range(1, 6)]
    assert len(rec._seeds(titles)) == 5


def test_seeds_cap_respects_max_seeds():
    cfg = Config()
    cfg.recommend.max_seeds = 3
    rec = Recommender(_StubModel(), _StubEnricher(), cfg)
    titles = [_seed_title(i, 9.0) for i in range(1, 10)]   # 9 standalone seedů
    assert len(rec._seeds(titles)) == 3
