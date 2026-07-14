"""intensity.py: lexikon osy emocionální náročnosti — načítání, generování
(merge sémantika: hodnoty uživatele vždy vyhrávají), a nový výpočet
TasteModel.intensity_of jako vážený průměr lexikonových skóre."""
import pytest

from animodel.attributes import AttrValue
from animodel.intensity import (
    load_lexicon, generate_lexicon, build_universe, category_prior,
    CURATED, DEFAULT_LEXICON,
)
from animodel.taste import TasteModel


# ── load_lexicon ─────────────────────────────────────────────────────────

def test_load_lexicon_missing_file_returns_none(tmp_path):
    assert load_lexicon(tmp_path / "neexistuje.yaml") is None


def test_load_lexicon_canonizes_keys_and_resolves_aliases(tmp_path):
    p = tmp_path / "intensity.yaml"
    p.write_text("Sci-Fi: 0.2\nIyashikei: -0.9\n", encoding="utf-8")
    lex = load_lexicon(p)
    # "Sci-Fi" → canon "sci_fi" → alias "science_fiction"; "Iyashikei" → "healing"
    assert lex == {"science_fiction": 0.2, "healing": -0.9}


def test_load_lexicon_clips_out_of_range_and_skips_non_numeric(tmp_path):
    p = tmp_path / "intensity.yaml"
    p.write_text("tragedy: 5.0\ncomedy: -3\nparody: hodne\n", encoding="utf-8")
    lex = load_lexicon(p)
    assert lex == {"tragedy": 1.0, "comedy": -1.0}   # parody přeskočeno


# ── universum + generování ───────────────────────────────────────────────

class FakeAniList:
    def get_tag_collection(self):
        return [
            {"name": "Tearjerker", "description": "Sad stuff.",
             "category": "Theme-Drama", "isAdult": False, "isGeneralSpoiler": False},
            {"name": "Coming of Age", "description": "",
             "category": "Theme-Drama", "isAdult": False, "isGeneralSpoiler": False},
            {"name": "Josei Fantasy",   # není v CURATED -> kategorie prior
             "category": "Theme-Comedy", "isAdult": False, "isGeneralSpoiler": False},
            {"name": "Kuudere",         # neutrální kategorie -> 0
             "category": "Cast-Traits", "isAdult": False, "isGeneralSpoiler": False},
            {"name": "Nudity", "category": "Sexual Content", "isAdult": True},
            {"name": "Plot Twist", "category": "Theme-Drama",
             "isGeneralSpoiler": True},
        ]


class FakeJikan:
    def get_genres(self, filter=""):
        if filter == "genres":
            return [{"mal_id": 4, "name": "Comedy"}, {"mal_id": 8, "name": "Drama"}]
        if filter == "themes":
            return [{"mal_id": 40, "name": "Psychological"}]
        return []


def test_build_universe_merges_sources_and_skips_adult_and_spoiler():
    universe = build_universe(jikan=FakeJikan(), anilist=FakeAniList())
    keys = {e["key"] for e in universe}
    assert keys == {"comedy", "drama", "psychological",
                    "tearjerker", "coming_of_age", "josei_fantasy", "kuudere"}
    # isAdult (Nudity) a isGeneralSpoiler (Plot Twist) vynechány -- stejná
    # pravidla jako build_attributes(), jinak by v lexikonu nic netrefily


def test_category_prior_matches_by_prefix():
    assert category_prior("Theme-Comedy") < 0
    assert category_prior("Theme-Drama") > 0
    assert category_prior("Cast-Traits") == 0.0
    assert category_prior(None) == 0.0


def test_generate_prefills_curated_then_prior_then_zero(tmp_path):
    out = tmp_path / "intensity.yaml"
    stats = generate_lexicon(out, jikan=FakeJikan(), anilist=FakeAniList())

    lex = load_lexicon(out)
    assert lex["tearjerker"] == CURATED["tearjerker"]      # curated vyhrává
    assert lex["comedy"] == CURATED["comedy"]
    assert lex["josei_fantasy"] == category_prior("Theme-Comedy")  # prior
    assert lex["kuudere"] == 0.0                            # nic -> neutrální
    assert stats["total"] == 7
    assert stats["from_curated"] >= 4    # comedy, drama, psychological, tearjerker, coming_of_age
    assert stats["from_prior"] == 1
    assert stats["zero"] == 1


def test_generate_preserves_existing_user_values_on_regeneration(tmp_path):
    out = tmp_path / "intensity.yaml"
    generate_lexicon(out, jikan=FakeJikan(), anilist=FakeAniList())

    # Uživatel si přepíše hodnotu a přidá vlastní klíč mimo universum
    text = load_lexicon(out)
    assert text["tearjerker"] != 0.15
    content = out.read_text(encoding="utf-8")
    content = content.replace(f"tearjerker: {CURATED['tearjerker']:g}", "tearjerker: 0.15")
    content += "\nmuj_vlastni_tag: -0.4\n"
    out.write_text(content, encoding="utf-8")

    stats = generate_lexicon(out, jikan=FakeJikan(), anilist=FakeAniList())
    lex = load_lexicon(out)
    assert lex["tearjerker"] == 0.15          # úprava uživatele přežila regeneraci
    assert lex["muj_vlastni_tag"] == -0.4     # vlastní klíč zachován
    assert stats["custom_kept"] == 1


def test_generated_file_orders_by_observed_frequency(tmp_path):
    out = tmp_path / "intensity.yaml"
    generate_lexicon(out, jikan=FakeJikan(), anilist=FakeAniList(),
                     observed_counts={"coming_of_age": 42, "tearjerker": 3})
    content = out.read_text(encoding="utf-8")
    # V AniList Theme-Drama sekci je častější coming_of_age před tearjerker
    assert content.index("coming_of_age:") < content.index("tearjerker:")
    assert "42×" in content


# ── TasteModel.intensity_of ──────────────────────────────────────────────

def _attrs(**kv):
    """{klíč: váha} → attrs dict pro intensity_of."""
    return {k: AttrValue(category="tag", weight=w, label=k) for k, w in kv.items()}


def test_intensity_of_weighted_mean_of_lexicon_scores():
    m = TasteModel(intensity={"drama": 0.5, "comedy": -1.0})
    # (1.0·0.5 + 1.0·(−1.0)) / (1.0 + 1.0) = −0.25
    assert m.intensity_of(_attrs(drama=1.0, comedy=1.0)) == pytest.approx(-0.25)


def test_intensity_of_respects_attribute_weights():
    m = TasteModel(intensity={"drama": 1.0, "comedy": -1.0})
    # drama váha 0.9, comedy 0.3 → (0.9 − 0.3) / 1.2 = 0.5
    assert m.intensity_of(_attrs(drama=0.9, comedy=0.3)) == pytest.approx(0.5)


def test_intensity_of_skips_neutral_and_unknown_keys():
    m = TasteModel(intensity={"drama": 0.8, "school": 0.0})
    # school (0.0) ani action (mimo lexikon) neředí výsledek
    assert m.intensity_of(_attrs(drama=1.0, school=1.0, action=1.0)) == pytest.approx(0.8)


def test_intensity_of_empty_or_all_neutral_returns_zero():
    m = TasteModel(intensity={"school": 0.0})
    assert m.intensity_of({}) == 0.0
    assert m.intensity_of(_attrs(school=1.0)) == 0.0


def test_intensity_of_binary_lexicon_reproduces_old_heavy_light_formula():
    """S hodnotami ±1 dává nový vzorec přesně staré (h−l)/(h+l)."""
    m = TasteModel(intensity={"tragedy": 1.0, "war": 1.0, "comedy": -1.0})
    attrs = _attrs(tragedy=1.0, war=0.5, comedy=1.0)
    h, l = 1.0 + 0.5, 1.0
    assert m.intensity_of(attrs) == pytest.approx((h - l) / (h + l))


def test_default_lexicon_used_when_intensity_not_given():
    m = TasteModel()
    assert m.intensity is DEFAULT_LEXICON
    assert m.intensity_of(_attrs(comedy=1.0)) < 0
    assert m.intensity_of(_attrs(tragedy=1.0)) > 0


# ── diagnostika unrated_intensity_attrs ──────────────────────────────────

def test_unrated_intensity_attrs_reports_observed_keys_missing_from_lexicon():
    m = TasteModel(intensity={"drama": 0.5, "school": 0.0})
    # Simulace stavu po fit(): pozorované atributy + jejich četnosti
    m.all_attrs = {
        "drama":  AttrValue("genre", 1.0, "Drama"),
        "school": AttrValue("theme", 1.0, "School"),      # explicitní 0.0 = ohodnoceno
        "novy_tag": AttrValue("tag", 1.0, "Novy Tag"),    # chybí v lexikonu
        "jcstaff": AttrValue("studio", 1.0, "J.C.Staff"), # studio se nehlásí
    }
    m.attr_counts = {"drama": 40.0, "school": 30.0, "novy_tag": 7.0, "jcstaff": 12.0}

    out = m.unrated_intensity_attrs()
    assert out == [("novy_tag", "Novy Tag", 7.0)]
