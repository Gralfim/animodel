"""Ridge režim efektů (HODNOCENI_PROJEKTU.md §9d, nález #3).

Marginální efekt je průměr rezidua titulů, které atribut MAJÍ -- vychází
≈ (1 − p)·Δ a titulu bez atributu se nic neubere. Ridge na centrované
atributy dává kontrast celý a absence oblíbeného atributu sráží."""
import random

import pytest

from animodel.attributes import AttrValue
from animodel.taste import Title, TasteModel


def _romance_titles(n=120, p=0.6, delta=1.0, seed=7):
    """Podíl `p` titulů má Romance a známku o `delta` vyšší; komunita
    konstantní, takže reziduum = známka − průměr."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rom = i < int(n * p)
        attrs = {"filler": AttrValue("tag", 1.0, "Filler")}
        if rom:
            attrs["romance"] = AttrValue("genre", 1.0, "Romance")
        score = 7.5 + (delta if rom else 0.0) + rng.gauss(0, 0.3)
        out.append(Title(mal_id=i + 1, title=f"T{i}", user_score=score,
                         community=7.5, attrs=attrs))
    return out


def _fit(mode, titles, **kw):
    return TasteModel(effect_model=mode, shrinkage_k=8.0, min_attr_count=1.0,
                      ridge_alpha=kw.pop("ridge_alpha", 5.0), **kw).fit(titles)


def test_ridge_recovers_full_contrast_marginal_dilutes_common_attribute():
    titles = _romance_titles()
    marg = _fit("marginal", titles)
    ridge = _fit("ridge", titles)
    # marginální efekt ≈ (1 − p)·Δ = 0,4
    assert marg.effects["romance"].effect == pytest.approx(0.4, abs=0.08)
    # ridge koeficient ≈ Δ · S/(S + α), S = Σ(x − μ)² = n·p·(1 − p)
    S = 120 * 0.6 * 0.4
    assert ridge.effects["romance"].effect == pytest.approx(S / (S + 5.0), abs=0.06)


def test_absence_of_liked_attribute_lowers_affinity_only_in_ridge():
    titles = _romance_titles()
    base = {"filler": AttrValue("tag", 1.0, "Filler")}
    with_rom = {**base, "romance": AttrValue("genre", 1.0, "Romance")}
    marg = _fit("marginal", titles)
    ridge = _fit("ridge", titles)
    # bez romantiky: marginální model nic neubere (posun dá jen centrování),
    # ridge dá zápornou afinitu -- titul je pod průměrem seznamu
    assert ridge.affinity(base) < 0 < ridge.affinity(with_rom)
    gap_ridge = ridge.affinity(with_rom) - ridge.affinity(base)
    gap_marg = marg.affinity(with_rom) - marg.affinity(base)
    assert gap_ridge > gap_marg


def test_duplicate_attributes_share_one_effect():
    """Dva tagy, které se vždy vyskytují spolu (zastupitelné), si efekt
    rozdělí -- marginální model by ho započetl dvakrát."""
    titles = _romance_titles()
    for t in titles:
        if "romance" in t.attrs:
            t.attrs["love"] = AttrValue("tag", 1.0, "Love")
    ridge = _fit("ridge", titles)
    a, b = ridge.effects["romance"].effect, ridge.effects["love"].effect
    assert a == pytest.approx(b, abs=1e-6)
    # dohromady ≈ Δ · 2S/(2S + α) -- jeden efekt, ne dva
    S = 120 * 0.6 * 0.4
    assert a + b == pytest.approx(2 * S / (2 * S + 5.0), abs=0.06)


def test_training_affinity_is_centered():
    ridge = _fit("ridge", _romance_titles())
    assert ridge.raw_center == pytest.approx(0.0, abs=1e-9)


def test_predict_explains_missing_liked_attribute():
    ridge = _fit("ridge", _romance_titles())
    _pred, _lo, _hi, contribs = ridge.predict(
        {"filler": AttrValue("tag", 1.0, "Filler")}, community=7.5)
    absent = [c for c in contribs if c[1] == "absence"]
    assert absent and absent[0][0] == "bez Romance" and absent[0][2] < 0


def test_ridge_mode_has_no_pairs_or_triples():
    titles = _romance_titles()
    m = TasteModel(effect_model="ridge", min_attr_count=1.0,
                   interaction_min_count=2.0, interaction_min_lift=0.0,
                   interaction_triples=True).fit(titles)
    assert m.interactions == [] and m.triples == [] and not m.use_triples


def test_unknown_effect_model_is_rejected():
    with pytest.raises(ValueError):
        TasteModel(effect_model="lasso")
