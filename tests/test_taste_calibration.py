"""Kalibrace scale (TasteModel._calibrate_scale): fold-modely se smí fitovat
jen JEDNOU na fold -- fit na `s` nezávisí, `s` vstupuje až do predikce.
Regrese na HODNOCENI_PROJEKTU.md §5.2 (dřív 115 fold-fitů místo 5)."""
import random

import pytest

from animodel.attributes import AttrValue
from animodel.taste import TasteModel, Title, Triple


def _titles(n=24):
    rng = random.Random(3)
    pool = ["romance", "comedy", "drama", "action", "school"]
    out = []
    for i in range(n):
        attrs = {g: AttrValue("genre", 1.0, g.title())
                 for g in rng.sample(pool, 2)}
        out.append(Title(mal_id=i + 1, title=f"T{i}",
                         user_score=float(7 + i % 4),
                         community=7.0 + rng.random() * 2,
                         attrs=attrs))
    return out


def test_calibration_fits_each_fold_exactly_once(monkeypatch):
    calls = {"n": 0}
    orig = TasteModel._fit_effects

    def counting(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(TasteModel, "_fit_effects", counting)
    TasteModel(shrinkage_k=8.0).fit(_titles())
    # 1 hlavní model + 5 foldů -- NE 116 (1 + 23 vyhodnocení × 5 foldů),
    # jak to dělala verze volající celou cross-validaci pro každé `s`
    assert calls["n"] == 6


def test_each_title_predicted_exactly_once_out_of_fold():
    m = TasteModel(shrinkage_k=8.0).fit(_titles())
    rows = m._cv_predictions()
    assert len(rows) == len(m.titles)


def test_grid_best_never_worse_than_baseline():
    """cv_rmse je minimum gridu, který obsahuje i s=0 (čistý baseline) --
    nikdy nesmí vyjít horší než baseline_rmse."""
    m = TasteModel(shrinkage_k=8.0).fit(_titles())
    assert 0.0 <= m.scale <= 1.0
    assert m.cv_rmse <= m.baseline_rmse + 1e-12
    assert m.resid_std == m.cv_rmse


def test_eval_scale_matches_naive_cross_val_formula():
    """_eval_scale nad předpočítanými řádky musí dávat přesně to, co
    dřívější _cross_val počítal inline (včetně ořezu predikce na 1–10).
    Bez trojic (tri=0, s_triples=0) musí vyjít historicky stejná čísla."""
    import math
    rows = [(7.5, 2.0, 0.0, 9.0), (9.8, 4.0, 0.0, 8.0), (6.0, -1.0, 0.0, 7.0)]
    s = 0.5
    # ručně: predikce clip(base + s*raw) -> 8.5, 10.0 (ořez z 11.8), 5.5
    errs = [8.5 - 9.0, 10.0 - 8.0, 5.5 - 7.0]
    exp_rmse = math.sqrt(sum(e * e for e in errs) / 3)
    exp_mae = sum(abs(e) for e in errs) / 3
    rmse, mae = TasteModel._eval_scale(rows, s)
    assert rmse == pytest.approx(exp_rmse)
    assert mae == pytest.approx(exp_mae)


def test_eval_scale_applies_separate_triple_factor():
    """Trojicová složka se škáluje `s_triples`, ne `s` -- oba faktory
    vstupují do téže predikce, každý na svou část."""
    rows = [(7.0, 2.0, 1.0, 8.0)]
    # base + s*raw + s_tri*tri = 7.0 + 0.5*2.0 + 0.25*1.0 = 8.25 → chyba 0.25
    rmse, mae = TasteModel._eval_scale(rows, 0.5, 0.25)
    assert rmse == pytest.approx(0.25)
    assert mae == pytest.approx(0.25)
    # s_triples=0 trojici úplně vypne → 7.0 + 1.0 = 8.0, přesně skutečnost
    assert TasteModel._eval_scale(rows, 0.5, 0.0)[0] == pytest.approx(0.0)
    # a `s` na trojici NEsahá: změna s_triples hýbe predikcí sama o sobě
    assert TasteModel._eval_scale(rows, 0.5, 1.0)[0] == pytest.approx(1.0)


# ── §5.1: kalibrace musí popisovat model, který skutečně predikuje ──────

def _triples_titles(n=90):
    """Dvě jasně oddělené skupiny po třech atributech, ať mají klastrové
    signatury z čeho brát kandidáty trojic (signatura kratší než 3 klíče
    žádnou trojici nevygeneruje) a ať trojice nesou signál nad rámec
    singlů a párů."""
    rng = random.Random(11)
    grp_a, grp_b = ["a", "b", "c"], ["x", "y", "z"]
    out = []
    for i in range(n):
        if i % 3 == 0:
            keys, score = grp_a, 9.0
        elif i % 3 == 1:
            keys, score = grp_b, 7.0
        else:
            keys = rng.sample(grp_a, 1) + rng.sample(grp_b, 1)
            score = 8.0
        attrs = {k: AttrValue("tag", 1.0, k.upper()) for k in keys}
        out.append(Title(mal_id=i + 1, title=f"T{i}", user_score=score,
                         community=7.5 + rng.random() * 0.5, attrs=attrs))
    return out


def _triples_model():
    return TasteModel(shrinkage_k=2.0, min_attr_count=1.0,
                      interaction_min_count=2.0, interaction_min_lift=0.01,
                      interaction_triples=True)


def test_triples_are_fitted_before_calibration_runs():
    """JÁDRO §5.1: dřív se `_calibrate_scale` volalo PŘED `_fit_triples`,
    takže se `s` hledalo na modelu bez trojic a pak aplikovalo na predikci
    s nimi. V okamžiku kalibrace už trojice MUSÍ být nafitované."""
    seen = {}
    orig = TasteModel._calibrate_scale

    def spy(self):
        seen["triples_at_calibration"] = len(self.triples)
        seen["clusters_at_calibration"] = len(self.clusters)
        return orig(self)

    m = _triples_model()
    TasteModel._calibrate_scale = spy
    try:
        m.fit(_triples_titles(), n_clusters=2)
    finally:
        TasteModel._calibrate_scale = orig

    assert seen["clusters_at_calibration"] > 0
    assert seen["triples_at_calibration"] == len(m.triples)
    assert m.triples, "fixture má produkovat aspoň jednu trojici"


def test_cv_folds_fit_their_own_clusters_and_triples():
    """Varianta (c) bez úniku informace: fold si klastruje SÁM. Kdyby
    přebíral kandidáty trojic z plného modelu, protáhla by se do foldu
    znalost jeho testovací pětiny."""
    calls = {"clusters": 0, "triples": 0}
    orig_c, orig_t = TasteModel._fit_clusters, TasteModel._fit_triples

    def spy_c(self, k=None):
        calls["clusters"] += 1
        return orig_c(self, k)

    def spy_t(self):
        calls["triples"] += 1
        return orig_t(self)

    TasteModel._fit_clusters, TasteModel._fit_triples = spy_c, spy_t
    try:
        _triples_model().fit(_triples_titles(), n_clusters=2)
    finally:
        TasteModel._fit_clusters, TasteModel._fit_triples = orig_c, orig_t

    # 1 hlavní model + 5 foldů
    assert calls["clusters"] == 6
    assert calls["triples"] == 6


def test_no_clustering_in_folds_when_triples_disabled():
    """Bez trojic se za klastrování na foldech neplatí -- CV je jen
    baseline/efekty/interakce, přesně jako dřív."""
    calls = {"n": 0}
    orig = TasteModel._fit_clusters

    def spy(self, k=None):
        calls["n"] += 1
        return orig(self, k)

    TasteModel._fit_clusters = spy
    try:
        TasteModel(shrinkage_k=8.0).fit(_titles())
    finally:
        TasteModel._fit_clusters = orig
    assert calls["n"] == 1   # jen hlavní model


def test_triples_get_their_own_calibrated_factor():
    m = _triples_model().fit(_triples_titles(), n_clusters=2)
    assert 0.0 <= m.scale <= 1.0
    assert 0.0 <= m.scale_triples <= 1.0
    # grid obsahuje s_triples=0, takže s trojicemi to nikdy nesmí být HORŠÍ
    assert m.cv_rmse <= m.cv_rmse_no_triples + 1e-12
    assert m.cv_rmse <= m.baseline_rmse + 1e-12


def test_scale_triples_is_zero_and_inert_without_triples():
    m = TasteModel(shrinkage_k=8.0).fit(_titles())
    assert m.use_triples is False
    assert m.scale_triples == 0.0
    assert m.cv_rmse == m.cv_rmse_no_triples


def test_scale_triples_zeroed_when_full_model_keeps_no_triples():
    """Fold-modely můžou trojice mít, i když plný model žádnou neudrží (jiná
    data → jiné lifty projdou prahem). Grid pak umí vrátit nenulové
    s_triples, které není co škálovat -- nesmí zůstat v atributu."""
    m = TasteModel(shrinkage_k=2.0, min_attr_count=1.0,
                   interaction_min_count=2.0,
                   interaction_min_lift=9.99,   # prahem neprojde nic
                   interaction_triples=True).fit(_triples_titles(), n_clusters=2)
    assert m.triples == []
    assert m.scale_triples == 0.0


def test_affinity_applies_both_factors_separately():
    m = TasteModel()
    m.effects, m.interactions = {}, []
    m.triples = [Triple(keys=("a", "b", "c"), label="A+B+C", n=5.0, lift=0.4)]
    m.scale, m.scale_triples = 1.0, 0.25
    attrs = {k: AttrValue("tag", 1.0, k.upper()) for k in ("a", "b", "c")}
    # _raw_resid_pred = nešálovaný součet, affinity = kalibrovaný
    assert m._raw_resid_pred(attrs) == pytest.approx(0.4)
    assert m.affinity(attrs) == pytest.approx(0.25 * 0.4)


def test_predict_contribution_uses_triple_factor_not_scale():
    """Vysvětlení v reportu musí sedět s tím, co doopravdy vstoupilo do
    predikce -- trojice se tam smí objevit jen se `scale_triples`."""
    m = TasteModel()
    m.effects, m.interactions = {}, []
    m.triples = [Triple(keys=("a", "b", "c"), label="A+B+C", n=5.0, lift=0.4)]
    m.scale, m.scale_triples, m.resid_std = 1.0, 0.5, 0.3
    m.u_mean = m.c_mean = 7.5
    m.beta = 0.0
    attrs = {k: AttrValue("tag", 1.0, k.upper()) for k in ("a", "b", "c")}
    pred, _lo, _hi, contribs = m.predict(attrs, 7.5)
    assert pred == pytest.approx(7.5 + 0.5 * 0.4)
    assert [c[2] for c in contribs] == [pytest.approx(0.5 * 0.4)]


def test_calibration_deterministic_across_runs():
    a = TasteModel(shrinkage_k=8.0).fit(_titles())
    b = TasteModel(shrinkage_k=8.0).fit(_titles())
    assert (a.scale, a.cv_rmse, a.cv_mae, a.baseline_rmse) == \
           (b.scale, b.cv_rmse, b.cv_mae, b.baseline_rmse)
