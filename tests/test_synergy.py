"""Synergie dvojic atributů (interakce) a klastrová afinita:
1. lift je smrštěný stejně jako efekty a přítomnost páru je vážená,
2. predikce škáluje interakci vahami atributů kandidáta,
3. klastr nese afinitu (vážený průměr reziduí) a _cluster_fit ji používá
   místo surové známky,
4. model.html ukazuje obě znaménka synergie bez formátovacího bugu."""
import math
import random

import pytest

from animodel.attributes import AttrValue
from animodel.config import Config
from animodel.recommend import Recommender
from animodel.taste import TasteModel, Title, Interaction, Triple, Cluster
from animodel import report


# ── fit interakcí: shrinkage + vážená přítomnost ─────────────────────────

def _pair_fixture(pair_resid_value):
    """4 tituly s párem a(w=1.0)+b(w=0.5), 2 jen s a, 2 jen s b."""
    titles, resid = [], {}

    def add(mid, attrs, r):
        titles.append(Title(mal_id=mid, title=f"T{mid}", user_score=8.0,
                            community=None, attrs=attrs))
        resid[mid] = r

    for mid in (1, 2, 3, 4):
        add(mid, {"a": AttrValue("tag", 1.0, "A"),
                  "b": AttrValue("tag", 0.5, "B")}, pair_resid_value)
    for mid in (5, 6):
        add(mid, {"a": AttrValue("tag", 1.0, "A")}, 0.0)
    for mid in (7, 8):
        add(mid, {"b": AttrValue("tag", 0.5, "B")}, 0.0)
    return titles, resid


def _fit_pair_model(pair_resid_value=1.0):
    m = TasteModel(shrinkage_k=8.0, min_attr_count=1.0,
                   interaction_min_count=2.0, interaction_min_lift=0.05)
    m.titles, m._resid = _pair_fixture(pair_resid_value)
    m._fit_effects()
    m._fit_interactions()
    return m


def test_interaction_lift_is_shrunk_and_support_is_weighted():
    m = _fit_pair_model(pair_resid_value=1.0)
    assert len(m.interactions) == 1
    it = m.interactions[0]

    # podpora páru: 4 tituly × (w_a=1.0 · w_b=0.5 · w_titulu=1.0) = 2.0
    assert it.n == pytest.approx(2.0)

    # raw lift = průměr rezidua páru − (smrštěné efekty a + b)
    eff_a = m.effects["a"].effect     # (6/14)·(2/3)
    eff_b = m.effects["b"].effect     # (3/11)·(2/3)
    raw_lift = 1.0 - (eff_a + eff_b)
    # smrštění stejným K jako efekty: n/(n+K) = 2/10
    assert it.lift == pytest.approx((2.0 / 10.0) * raw_lift)
    assert it.lift < raw_lift         # smrštění lift skutečně tlumí


def test_negative_synergy_is_kept():
    m = _fit_pair_model(pair_resid_value=-1.0)
    assert len(m.interactions) == 1
    assert m.interactions[0].lift < 0


def test_interaction_inherits_spoiler_flag():
    m = _fit_pair_model()
    # označ efekt "a" jako spoiler a přepočti interakce
    m.effects["a"].spoiler = True
    m._fit_interactions()
    assert m.interactions[0].spoiler is True


# ── predikce: škálování vahami kandidáta ─────────────────────────────────

def test_raw_resid_pred_scales_interaction_by_candidate_weights():
    m = TasteModel()
    m.effects = {}
    m.interactions = [Interaction(a="a", b="b", label="A + B", n=5.0, lift=0.4)]
    attrs = {"a": AttrValue("tag", 0.8, "A"), "b": AttrValue("tag", 0.5, "B")}
    assert m._raw_resid_pred(attrs) == pytest.approx(0.4 * 0.8 * 0.5)
    # neúplný pár nepřispívá
    assert m._raw_resid_pred({"a": AttrValue("tag", 1.0, "A")}) == 0.0


def test_predict_contrib_scales_and_carries_interaction_spoiler():
    m = TasteModel()
    m.u_mean, m.c_mean, m.beta = 8.0, 7.5, 0.0
    m.scale, m.resid_std = 1.0, 0.5
    m.effects = {}
    m.interactions = [Interaction(a="a", b="b", label="A + B", n=5.0,
                                  lift=0.4, spoiler=True)]
    attrs = {"a": AttrValue("tag", 0.8, "A"), "b": AttrValue("tag", 0.5, "B")}
    _pred, _lo, _hi, contribs = m.predict(attrs, community=8.0)
    assert contribs == [("A + B", "interakce", pytest.approx(0.4 * 0.8 * 0.5), True)]


# ── klastrová afinita ────────────────────────────────────────────────────

def _cluster_titles():
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
    return titles


def test_cluster_affinity_is_weighted_mean_of_residuals():
    m = TasteModel(shrinkage_k=8.0, min_attr_count=2.0).fit(
        _cluster_titles(), n_clusters=2)
    assert len(m.clusters) == 2
    by_id = {t.mal_id: t for t in m.titles}
    for c in m.clusters:
        ids = {mid for mid, _t, _s in c.members}
        num = sum(by_id[mid].weight * m._resid[mid] for mid in ids)
        den = sum(by_id[mid].weight for mid in ids)
        assert c.affinity == pytest.approx(num / den)


# ── _cluster_fit používá afinitu, ne surovou známku ─────────────────────

class _StubModel:
    u_mean = 8.0

    def __init__(self, clusters):
        self.clusters = clusters


class _StubEnricher:
    jikan = None
    anilist = None
    shikimori = None


def _cluster(name, sig_key, mean_score, affinity):
    return Cluster(idx=0, name=name, size=10, mean_user_score=mean_score,
                   intensity=0.0, signature=[(sig_key, sig_key.title(), "genre", 0.5, False)],
                   members=[], affinity=affinity)


def test_cluster_fit_weights_by_affinity_not_raw_score():
    # "Hi" klastr: nízké surové skóre, ale vysoká afinita (nadhodnocuju ho);
    # "Lo" klastr: vysoké surové skóre (mainstream hity), záporná afinita
    clusters = [_cluster("Hi", "comedy", mean_score=7.0, affinity=0.6),
                _cluster("Lo", "drama", mean_score=9.5, affinity=-0.2)]
    rec = Recommender(_StubModel(clusters), _StubEnricher(), Config())

    fit_hi, name_hi = rec._cluster_fit({"comedy": AttrValue("genre", 1.0, "Comedy")})
    assert name_hi == "Hi"
    assert fit_hi == pytest.approx(1.0 * (0.6 + 1.0))   # sim=1 × (aff+1)

    fit_lo, name_lo = rec._cluster_fit({"drama": AttrValue("genre", 1.0, "Drama")})
    assert name_lo == "Lo"
    # vysoké mean_user_score klastru už NEpomáhá -- rozhoduje afinita
    assert fit_lo == pytest.approx(1.0 * (-0.2 + 1.0))
    assert fit_hi > fit_lo


# ── trojice (hierarchická synergie nad klastrovými signaturami) ─────────

def _triple_fixture():
    """4 tituly se všemi třemi tagy (resid 2.0) + po dvou se singly
    (resid 0) -- efekty i všechny páry mají podporu, trojice taky."""
    titles, resid = [], {}

    def add(mid, keys, r):
        titles.append(Title(mal_id=mid, title=f"T{mid}", user_score=8.0,
                            community=None,
                            attrs={k: AttrValue("tag", 1.0, k.upper()) for k in keys}))
        resid[mid] = r

    mid = 0
    for _ in range(4):
        mid += 1; add(mid, ("a", "b", "c"), 2.0)
    for key in ("a", "b", "c"):
        for _ in range(2):
            mid += 1; add(mid, (key,), 0.0)
    return titles, resid


def _fit_triple_model():
    m = TasteModel(shrinkage_k=8.0, min_attr_count=1.0,
                   interaction_min_count=2.0, interaction_min_lift=0.05,
                   interaction_triples=True)
    m.titles, m._resid = _triple_fixture()
    m._fit_effects()
    m._fit_interactions()
    m.clusters = [Cluster(idx=0, name="X", size=4, mean_user_score=8.0,
                          intensity=0.0, members=[],
                          signature=[(k, k.upper(), "tag", 0.5, False)
                                     for k in ("a", "b", "c")])]
    m._fit_triples()
    return m


def test_triple_lift_is_hierarchical_residual_over_singles_and_pairs():
    m = _fit_triple_model()
    assert len(m.triples) == 1
    tr = m.triples[0]
    assert tr.keys == ("a", "b", "c")
    assert tr.n == pytest.approx(4.0)

    singles = sum(m.effects[k].effect for k in ("a", "b", "c"))
    pairs = sum(it.lift for it in m.interactions)   # všechny 3 páry prošly
    expected_lift = (4.0 / 12.0) * (2.0 - singles - pairs)
    assert tr.lift == pytest.approx(expected_lift)
    # v téhle konstrukci páry strukturu PŘEpočítaly (singly+páry > skutečný
    # průměr) -- hierarchická trojice musí korigovat DOLŮ
    assert tr.lift < 0


def test_triples_improve_prediction_toward_actual_mean():
    """Smysl hierarchie: singly+páry přestřelí (2.57 vs. skutečných 2.0);
    trojice zbytek koriguje zpátky směrem ke skutečnosti."""
    m = _fit_triple_model()
    attrs = {k: AttrValue("tag", 1.0, k.upper()) for k in ("a", "b", "c")}
    with_triples = m._raw_resid_pred(attrs)
    m.triples = []
    without_triples = m._raw_resid_pred(attrs)
    assert abs(with_triples - 2.0) < abs(without_triples - 2.0)


def test_triples_candidates_come_only_from_cluster_signatures():
    m = _fit_triple_model()
    # signatura jen (a, b) -> žádná trojice, i když data by ji unesla
    m.clusters[0].signature = [(k, k.upper(), "tag", 0.5, False) for k in ("a", "b")]
    m._fit_triples()
    assert m.triples == []


def test_triples_disabled_by_default_in_fit():
    titles = _cluster_titles()
    m = TasteModel(shrinkage_k=8.0, min_attr_count=2.0).fit(titles, n_clusters=2)
    assert m.use_triples is False
    assert m.triples == []


def test_fit_with_triples_enabled_runs_end_to_end():
    m = TasteModel(shrinkage_k=8.0, min_attr_count=2.0,
                   interaction_min_count=2.0, interaction_min_lift=0.0001,
                   interaction_triples=True).fit(_cluster_titles(), n_clusters=2)
    # signatury mají jen po 2 klíčích (comedy+romance / drama+psychological)
    # -> C(2,3)=0 kandidátů; podstatné je, že fit s flagem nespadne
    assert isinstance(m.triples, list)


def test_raw_resid_pred_scales_triple_by_candidate_weights():
    m = TasteModel()
    m.effects = {}
    m.interactions = []
    m.triples = [Triple(keys=("a", "b", "c"), label="A + B + C", n=5.0, lift=0.3)]
    attrs = {"a": AttrValue("tag", 1.0, "A"), "b": AttrValue("tag", 0.5, "B"),
             "c": AttrValue("tag", 0.4, "C")}
    assert m._raw_resid_pred(attrs) == pytest.approx(0.3 * 1.0 * 0.5 * 0.4)
    # neúplná trojice nepřispívá
    del attrs["c"]
    assert m._raw_resid_pred(attrs) == 0.0


# ── report: obě znaménka synergie, bez "+-" bugu ─────────────────────────

def test_model_html_shows_both_synergy_tables_and_affinity(tmp_path):
    m = TasteModel(shrinkage_k=8.0, min_attr_count=2.0).fit(
        _cluster_titles(), n_clusters=2)
    m.interactions = [
        Interaction(a="comedy", b="romance", label="Comedy + Romance",
                    n=9.0, lift=0.42),
        Interaction(a="drama", b="comedy", label="Drama + Comedy",
                    n=12.0, lift=-0.31, spoiler=True),
    ]
    m.triples = [Triple(keys=("comedy", "drama", "romance"),
                        label="Comedy + Drama + Romance", n=8.0, lift=-0.21)]
    out = tmp_path / "model.html"
    report.render_model_html(m, {"user_name": "test"},
                             {"n_rated": 32, "dist": {}}, str(out))
    html = out.read_text(encoding="utf-8")

    assert "Dělají víc než součet" in html
    assert "Nesedí si" in html
    assert "Trojice — jádra nálad" in html
    assert "-0.21" in html
    assert "+0.42" in html
    assert "-0.31" in html
    assert "+-0" not in html          # dřívější hardcoded "+" prefix bug
    # záporný lift má třídu neg (ne pos) a spoiler interakce nese spoiler-item
    assert '<td class="mono neg">-0.31</td>' in html
    assert '<tr class="spoiler-item"><td>Drama + Comedy</td>' in html
    # afinita nálad je v meta řádce klastrů
    assert "afinita" in html
