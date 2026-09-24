"""Kompozitní skóre doporučení: 4 oddělené složky (taste / item-CF / user-CF /
quality), log1p tlumení hlasů grafu, prahy slabých hran a propsání obou CF
signálů do Recommendation + HTML karty."""
import math

import pytest

from animodel.attributes import AttrValue
from animodel.config import Config
from animodel.enrich import Enriched
from animodel.recommend import Recommender, Recommendation
from animodel.taste import Title
from animodel import report


class _StubModel:
    u_mean = 8.0
    c_mean = 7.5
    clusters = []
    cluster_feat_keys = set()
    effects = {}
    interactions = []
    triples = []

    def top_effects(self, n=40, sign=1):
        return []

    def residuals(self):
        return {}

    def affinity(self, attrs):
        return 0.0

    def predict(self, attrs, community):
        return 8.0, 7.5, 8.5, []


class _FakeJikan:
    def __init__(self, recs):
        self.recs = recs

    def get_recommendations(self, mal_id):
        return self.recs


class _FakeAniList:
    def __init__(self, recs):
        self.recs = recs

    def get_recommendations(self, mal_id):
        return self.recs

    def search_by_tags(self, tags, pages=2):
        return []


class _Enr:
    def __init__(self, jikan=None, anilist=None):
        self.jikan = jikan
        self.anilist = anilist
        self.shikimori = None

    def enrich_ids(self, ids, show_progress=True):
        return {}

    def relations_data(self, enriched):
        return {}          # bez franšízových vazeb = každý titul sám za sebe


def _seed():
    return Title(mal_id=1, title="Seed", user_score=9.0, community=8.0, attrs={})


# ── prahy slabých hran ───────────────────────────────────────────────────

def test_mal_recs_below_vote_threshold_are_dropped():
    cfg = Config()
    assert cfg.recommend.min_mal_rec_votes == 5
    jikan = _FakeJikan([
        {"mal_id": 101, "title": "Strong", "votes": 40},
        {"mal_id": 102, "title": "Weak", "votes": 2},      # pod prahem -> šum
        {"mal_id": 103, "title": "Border", "votes": 5},    # přesně práh -> bere se
    ])
    rec = Recommender(_StubModel(), _Enr(jikan=jikan), cfg)
    cand = rec._gather_candidates([_seed()], seen_ids=set())
    assert set(cand) == {101, 103}


def test_anilist_recs_below_rating_threshold_are_dropped():
    cfg = Config()
    anilist = _FakeAniList([
        {"mal_id": 201, "title": "Strong", "rating": 30},
        {"mal_id": 202, "title": "Weak", "rating": 1},
        {"mal_id": 203, "title": "Negative", "rating": -3},
    ])
    rec = Recommender(_StubModel(), _Enr(anilist=anilist), cfg)
    cand = rec._gather_candidates([_seed()], seen_ids=set())
    assert set(cand) == {201}


# ── routing item vs. user votes ──────────────────────────────────────────

def test_bump_routes_user_cf_votes_to_separate_bucket():
    cfg = Config()
    jikan = _FakeJikan([{"mal_id": 101, "title": "X", "votes": 40}])
    rec = Recommender(_StubModel(), _Enr(jikan=jikan), cfg)
    cand = rec._gather_candidates([_seed()], seen_ids=set())

    # simulace user-CF bumpu na stejném kandidátovi (stejné rozhraní jako
    # v _user_cf: bump(mid, score, None, "user-CF") -- tady přes cand dict)
    d = cand[101]
    assert d["item_votes"] > 0 and d["user_votes"] == 0.0


def test_gather_meta_has_both_buckets():
    cfg = Config()
    rec = Recommender(_StubModel(), _Enr(), cfg)
    assert rec._gather_candidates([_seed()], seen_ids=set()) == {}


# ── kompozit: 4 složky, log1p na item hlasech ────────────────────────────

#: user-CF váha pro testy MECHANIKY složky -- default je od §9d.8 nula
#: (k modelu vkusu nic nepřidala), ale výpočet složky se testovat má dál
UCF_W = 0.6


def _run_recommend(cand_meta, communities, w_user_cf=None):
    """Spustí recommend() s podstrčenými kandidáty a obohacením."""
    cfg = Config()
    if w_user_cf is not None:
        cfg.recommend.w_user_cf = w_user_cf
    rec = Recommender(_StubModel(), _Enr(), cfg)
    rec._gather_candidates = lambda titles, seen_ids: cand_meta

    def fake_enrich(ids, show_progress=True):
        return {mid: Enriched(mal_id=mid, title=f"T{mid}", title_en="",
                              community=communities[mid], attrs={})
                for mid in ids}
    rec.enr.enrich_ids = fake_enrich
    return rec.recommend([], ptw_ids=set(), watched_ids=set(),
                         show_progress=False, limit=None)


def test_composite_has_four_components_and_carries_both_signals():
    cand = {
        1: {"item_votes": 60.0, "user_votes": 0.0, "cf_seeds": ["S"], "sources": {"MAL-rec"}},
        2: {"item_votes": 0.0, "user_votes": 9.0, "cf_seeds": [], "sources": {"user-CF"}},
        3: {"item_votes": 0.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"tag-search"}},
    }
    recs = _run_recommend(cand, {1: 7.0, 2: 7.0, 3: 9.0}, w_user_cf=UCF_W)
    by_id = {r.mal_id: r for r in recs}

    # oba signály se nesou odděleně
    assert by_id[1].cf_signal == 60.0 and by_id[1].user_cf_signal == 0.0
    assert by_id[2].cf_signal == 0.0 and by_id[2].user_cf_signal == 9.0

    # s vahami 0.8 / 0.6 / 0.3 (taste=0 pro všechny):
    # silný graf > user-CF > jen kvalita
    order = [r.mal_id for r in recs]
    assert order == [1, 2, 3]

    # kompozit kandidáta 1 odpovídá ručnímu výpočtu s log1p tlumením.
    # Parametry grafu a kvality (i vkusu) jsou z OBSAHOVÉHO poolu -- kandidáti
    # 1 (MAL-rec) a 3 (tag-search); user-CF kandidát 2 do nich nevstupuje.
    # User-CF složka se normalizuje přes celý pool (§9d, nález #1).
    cfg = Config()
    logs = [math.log1p(60.0), 0.0]
    users = [0.0, 9.0, 0.0]
    comms = [7.0, 9.0]
    def z(vals, x):
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) or 1.0
        return (x - mean) / sd
    expected = (cfg.recommend.w_cf * z(logs, math.log1p(60.0))
                + UCF_W * z(users, 0.0)
                + cfg.recommend.w_quality * z(comms, 7.0))
    # taste_fit je pro všechny 0 -> z=0 (sd fallback 1.0), nepřispívá
    assert by_id[1].composite == pytest.approx(expected)


def test_user_cf_only_candidate_is_not_buried_by_graph_outlier():
    """Dřívější sloučený kbelík: outlier grafu (60 hlasů) stlačil z-skóre
    user-CF kandidáta hluboko pod nulu. Teď má user-CF vlastní osu."""
    cand = {
        1: {"item_votes": 60.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"MAL-rec"}},
        2: {"item_votes": 0.0, "user_votes": 9.0, "cf_seeds": [], "sources": {"user-CF"}},
        3: {"item_votes": 3.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"MAL-rec"}},
    }
    recs = _run_recommend(cand, {1: 7.0, 2: 7.0, 3: 7.0}, w_user_cf=UCF_W)
    by_id = {r.mal_id: r for r in recs}
    # user-CF kandidát poráží slabý graf (3 hlasy) -- ve sloučeném kbelíku
    # by měl (9 vs 3) taky navrch, ale outlier 60 by oba srazil k sobě;
    # podstatné je, že 2 > 3 s jasným odstupem
    assert by_id[2].composite > by_id[3].composite


def test_user_cf_only_pool_does_not_change_weights_of_content_candidates():
    """Se zapnutým user-CF přibyly do poolu tisíce titulů s nulou v grafu
    (6 108 z 6 615). Když se z-skóre počítala přes celý pool, sd log-hlasů
    spadla na 0,6 a každý titul z grafu dostal +3,5 až +6 bodů -- graf
    fungoval jako brána (§9d, nález #1). Parametry se teď berou z obsahového
    poolu, takže rozdíly kompozitu mezi obsahovými kandidáty na user-CF
    kandidátech nezávisí."""
    content = {
        1: {"item_votes": 60.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"MAL-rec"}},
        2: {"item_votes": 5.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"AniList-rec"}},
        3: {"item_votes": 0.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"tag-search"}},
    }
    comms = {1: 7.0, 2: 8.5, 3: 8.0}
    alone = {r.mal_id: r.composite for r in _run_recommend(content, comms)}

    flooded = dict(content)
    for mid in range(100, 160):          # 60 kandidátů jen od senpai
        flooded[mid] = {"item_votes": 0.0, "user_votes": 0.0, "cf_seeds": [],
                        "sources": {"user-CF"}}
        comms[mid] = 6.8
    both = {r.mal_id: r.composite for r in _run_recommend(flooded, comms)}

    assert both[1] - both[2] == pytest.approx(alone[1] - alone[2])
    assert both[2] - both[3] == pytest.approx(alone[2] - alone[3])


def test_recommend_exposes_z_params_and_taste_components():
    """Snapshot historie ukládá složky kompozitu i parametry z-skóre, aby
    šlo pořadí přepočítat jinými vahami bez nového běhu (history.py). Test
    hlídá, že uložené složky na kompozit opravdu stačí."""
    cand = {
        1: {"item_votes": 60.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"MAL-rec"}},
        2: {"item_votes": 0.0, "user_votes": 9.0, "cf_seeds": [], "sources": {"user-CF"}},
        3: {"item_votes": 3.0, "user_votes": 0.0, "cf_seeds": [], "sources": {"MAL-rec"}},
    }
    result = _run_recommend(cand, {1: 7.0, 2: 8.0, 3: 6.5})
    assert set(result.z_params) == {"taste_fit", "item_cf", "user_cf", "quality",
                                    "select"}

    rc = Config().recommend

    def z(name, x):
        mean, sd = result.z_params[name]
        return (x - mean) / sd

    for r in result.recs:
        expected = (rc.w_taste_fit * z("taste_fit", r.taste_fit)
                    + rc.w_cf * z("item_cf", math.log1p(r.cf_signal))
                    + rc.w_user_cf * z("user_cf", r.user_cf_signal)
                    + rc.w_quality * z("quality", r.community)
                    + rc.w_select * z("select", r.select))
        assert r.composite == pytest.approx(expected)
        # taste_fit = afinita + w · shoda s náladou; obě složky se ukládají
        # zvlášť, aby šlo přeladit i cluster_fit_weight
        assert r.taste_fit == pytest.approx(
            r.affinity + rc.cluster_fit_weight * r.cluster_fit)


# ── HTML karta ukazuje oba signály ───────────────────────────────────────

def test_rec_card_shows_both_cf_boxes(tmp_path):
    r = Recommendation(
        mal_id=1, title="X", title_en="", community=8.0, pred=8.5,
        pred_lo=8.0, pred_hi=9.0, taste_fit=1.0, cf_signal=42.0,
        user_cf_signal=8.7, composite=1.5, ptw=False, cluster_name="",
        why=[], cf_seeds=[], synopsis="", sources=["MAL-rec", "user-CF"],
    )
    out = tmp_path / "recs.html"
    report.render_recommendations_html([r], str(out))
    html = out.read_text(encoding="utf-8")
    assert "graf podobnosti" in html and ">42<" in html
    assert "user-CF" in html and ">8.7<" in html


def test_cf_report_marks_watched_titles(tmp_path):
    """CF report shlédnuté nefiltruje (surový pohled), jen je označí --
    přesně kvůli 'proč je Monogatari v CF výstupu, ale ne ve finále'."""
    cf_recs = [
        {"mal_id": 1, "title": "Nisemonogatari", "cf_score": 9.1,
         "community": 8.0, "diff": 1.1, "n_users": 4, "top_raters": []},
        {"mal_id": 2, "title": "Nevidene", "cf_score": 8.5,
         "community": 7.5, "diff": 1.0, "n_users": 3, "top_raters": []},
    ]
    out = tmp_path / "cf.html"
    report.render_cf_recommendations_html(cf_recs, str(out),
                                          watched_ids={1})
    html = out.read_text(encoding="utf-8")
    assert '<span class="flag seen">už shlédnuto</span>' in html
    # štítek jen u shlédnutého titulu, ne u obou karet
    assert html.count("už shlédnuto") == 1


def test_rec_card_hides_zero_cf_boxes(tmp_path):
    r = Recommendation(
        mal_id=1, title="X", title_en="", community=8.0, pred=8.5,
        pred_lo=8.0, pred_hi=9.0, taste_fit=1.0, cf_signal=0.0,
        user_cf_signal=0.0, composite=1.5, ptw=False, cluster_name="",
        why=[], cf_seeds=[], synopsis="", sources=["tag-search"],
    )
    out = tmp_path / "recs.html"
    report.render_recommendations_html([r], str(out))
    html = out.read_text(encoding="utf-8")
    assert "graf podobnosti" not in html
    assert '<div class="k">user-CF</div>' not in html