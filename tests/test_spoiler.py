"""Spoiler tagy (AniList isGeneralSpoiler/isMediaSpoiler): od 2026-07
vstupují do modelu normálně, jen nesou příznak, který teče přes AttrValue →
AttrEffect → signatury klastrů → predict contribs až do HTML reportu, kde
je přepínač umí skrýt. isAdult tagy zůstávají vyloučené úplně."""
import random

from animodel.attributes import build_attributes, AttrValue
from animodel.recommend import Recommendation
from animodel.taste import TasteModel, Title, AttrEffect
from animodel import report


# ── build_attributes ─────────────────────────────────────────────────────

def test_spoiler_tags_included_with_flag_adult_still_excluded():
    anilist = {"tags": [
        {"name": "Tragedy", "rank": 90, "isGeneralSpoiler": True},
        {"name": "Vampire", "rank": 60, "isMediaSpoiler": True},
        {"name": "Nudity", "rank": 90, "isAdult": True},
        {"name": "Cohabitation", "rank": 60},
    ]}
    attrs = build_attributes(None, anilist, anilist_min_rank=30)
    assert attrs["tragedy"].spoiler is True
    assert attrs["vampire"].spoiler is True      # per-media spoiler taky
    assert attrs["cohabitation"].spoiler is False
    assert "nudity" not in attrs                 # adult zůstává venku


def test_spoiler_flag_survives_key_collision():
    # MAL žánr (non-spoiler) + AniList spoiler tag se stejným klíčem:
    # opatrnější varianta (spoiler=True) vyhrává
    jikan = {"genres": [{"name": "Vampire"}], "themes": [], "demographics": [],
             "source": "", "type": "", "year": None, "studios": []}
    anilist = {"tags": [{"name": "Vampire", "rank": 90, "isMediaSpoiler": True}]}
    attrs = build_attributes(jikan, anilist, anilist_min_rank=30)
    assert attrs["vampire"].spoiler is True


# ── predict contribs ─────────────────────────────────────────────────────

def _stub_model():
    m = TasteModel()
    m.u_mean, m.c_mean, m.beta = 8.0, 7.5, 0.5
    m.scale, m.resid_std = 1.0, 0.5
    m.interactions = []
    m.effects = {
        "tragedy": AttrEffect(key="tragedy", label="Tragedy", category="tag",
                              n_eff=10, raw_mean=0.5, effect=0.4, distinct=1.0,
                              titles_pos=[], titles_neg=[], spoiler=True),
        "comedy": AttrEffect(key="comedy", label="Comedy", category="genre",
                             n_eff=20, raw_mean=-0.2, effect=-0.15, distinct=-0.5,
                             titles_pos=[], titles_neg=[], spoiler=False),
    }
    return m


def test_predict_contribs_carry_spoiler_flag_from_effect():
    m = _stub_model()
    attrs = {"tragedy": AttrValue("tag", 1.0, "Tragedy", spoiler=True),
             "comedy": AttrValue("genre", 1.0, "Comedy")}
    _pred, _lo, _hi, contribs = m.predict(attrs, community=8.0)
    flags = {lab: spoil for lab, _cat, _val, spoil in contribs}
    assert flags == {"Tragedy": True, "Comedy": False}


def test_predict_contribs_media_spoiler_wins_over_clean_effect():
    """Tag může být spoiler jen pro KONKRÉTNÍ titul (isMediaSpoiler) --
    příznak kandidátova AttrValue musí přebít non-spoiler efekt."""
    m = _stub_model()
    attrs = {"comedy": AttrValue("genre", 1.0, "Comedy", spoiler=True)}
    _pred, _lo, _hi, contribs = m.predict(attrs, community=8.0)
    assert contribs == [("Comedy", "genre", contribs[0][2], True)]


# ── HTML reporty: spoiler-item třídy + přepínač ─────────────────────────

def _fit_mini_model():
    """~24 titulů, půlka s Tragedy (spoiler), půlka s Comedy -- dost na
    fitnuté efekty obou (min_attr_count=2)."""
    rng = random.Random(1)
    titles = []
    for i in range(24):
        if i % 2 == 0:
            attrs = {"tragedy": AttrValue("tag", 1.0, "Tragedy", spoiler=True)}
        else:
            attrs = {"comedy": AttrValue("genre", 1.0, "Comedy")}
        titles.append(Title(mal_id=i + 1, title=f"T{i}", user_score=7 + (i % 4),
                            community=7.5 + rng.random(), attrs=attrs))
    return TasteModel(shrinkage_k=2.0, min_attr_count=2.0).fit(titles)


def test_model_html_marks_spoiler_rows_and_has_toggle(tmp_path):
    model = _fit_mini_model()
    out = tmp_path / "model.html"
    report.render_model_html(model, {"user_name": "test"},
                             {"n_rated": 24, "dist": {}}, str(out))
    html = out.read_text(encoding="utf-8")
    assert "spoiltoggle" in html                 # přepínač existuje
    assert 'class="spoiler-item"' in html        # řádek s Tragedy efektem
    assert "body.nospoil .spoiler-item" in html  # CSS, které skrývání dělá


def test_recommendations_html_marks_spoiler_why_items(tmp_path):
    rec = Recommendation(
        mal_id=1, title="X", title_en="", community=8.0, pred=8.5,
        pred_lo=8.0, pred_hi=9.0, taste_fit=1.0, cf_signal=2.0, composite=1.5,
        ptw=False, cluster_name="Drama",
        why=[("Tragedy", "tag", 0.3, True), ("Comedy", "genre", -0.1, False)],
        cf_seeds=["Seed"], synopsis="syn", sources=["MAL-rec"],
    )
    out = tmp_path / "recs.html"
    report.render_recommendations_html([rec], str(out))
    html = out.read_text(encoding="utf-8")
    assert "spoiltoggle" in html
    assert '<span class="pos spoiler-item">Tragedy</span>' in html
    assert '<span class="neg">Comedy</span>' in html   # non-spoiler bez třídy
