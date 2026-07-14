"""Kalibrace scale (TasteModel._calibrate_scale): fold-modely se smí fitovat
jen JEDNOU na fold -- fit na `s` nezávisí, `s` vstupuje až do predikce.
Regrese na HODNOCENI_PROJEKTU.md §5.2 (dřív 115 fold-fitů místo 5)."""
import random

import pytest

from animodel.attributes import AttrValue
from animodel.taste import TasteModel, Title


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
    """_eval_scale nad předpočítanými trojicemi musí dávat přesně to, co
    dřívější _cross_val počítal inline (včetně ořezu predikce na 1–10)."""
    import math
    rows = [(7.5, 2.0, 9.0), (9.8, 4.0, 8.0), (6.0, -1.0, 7.0)]
    s = 0.5
    # ručně: predikce clip(base + s*raw) -> 8.5, 10.0 (ořez z 11.8), 5.5
    errs = [8.5 - 9.0, 10.0 - 8.0, 5.5 - 7.0]
    exp_rmse = math.sqrt(sum(e * e for e in errs) / 3)
    exp_mae = sum(abs(e) for e in errs) / 3
    rmse, mae = TasteModel._eval_scale(rows, s)
    assert rmse == pytest.approx(exp_rmse)
    assert mae == pytest.approx(exp_mae)


def test_calibration_deterministic_across_runs():
    a = TasteModel(shrinkage_k=8.0).fit(_titles())
    b = TasteModel(shrinkage_k=8.0).fit(_titles())
    assert (a.scale, a.cv_rmse, a.cv_mae, a.baseline_rmse) == \
           (b.scale, b.cv_rmse, b.cv_mae, b.baseline_rmse)
