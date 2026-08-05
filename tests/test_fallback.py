"""Nouzový AniList-only režim (--no-jikan / enrich.use_jikan: false):
atributy, synopse, dekáda a franšízové vazby musí fungovat čistě z AniList
dat; MAL-rec větev doporučení se musí bez Jikanu tiše přeskočit."""
import math

import pytest

from animodel.attributes import build_attributes
from animodel.config import Config
from animodel.enrich import Enricher, _relations_from_anilist, _clean_anilist_description
from animodel.mal import MalEntry
from animodel.recommend import Recommender
from animodel.taste import Title


# Reálný tvar AniList Media po rozšíření dotazu (genres/description/
# seasonYear/startDate/relations)
def _media(mal_id, anilist_id, relations=None, year=2018):
    return {
        "id": anilist_id, "idMal": mal_id,
        "title": {"romaji": f"Anime {mal_id}", "english": f"Anime {mal_id} EN"},
        "genres": ["Comedy", "Drama"],
        "tags": [{"name": "Tearjerker", "rank": 80, "isAdult": False,
                  "isGeneralSpoiler": False, "isMediaSpoiler": False,
                  "category": "Theme-Drama"}],
        "studios": {"nodes": [{"name": "Kyoto Animation", "isAnimationStudio": True}]},
        "format": "TV", "source": "MANGA",
        "averageScore": 84, "popularity": 50000,
        "seasonYear": year, "startDate": {"year": year},
        "description": "První řádek.<br><i>Kurzíva</i> ~!spoiler!~ &amp; konec.",
        "relations": {"edges": relations or []},
    }


# ── build_attributes: AniList-only ──────────────────────────────────────

def test_attributes_from_anilist_only_cover_jikan_categories():
    attrs = build_attributes(None, _media(1, 100), anilist_min_rank=30)
    assert attrs["comedy"].category == "genre"
    assert attrs["drama"].category == "genre"
    assert attrs["tearjerker"].category == "tag"
    assert attrs["kyoto_animation"].category == "studio"
    assert attrs["manga"].category == "source"
    assert attrs["tv"].category == "format"
    assert attrs["2010s"].category == "decade"


def test_attributes_jikan_wins_format_and_decade_when_both_present():
    jikan = {"genres": [{"name": "Comedy"}], "themes": [], "demographics": [],
             "source": "Manga", "type": "TV", "year": 2013, "studios": []}
    # AniList tvrdí jiný rok (2020) -- guard musí zabránit druhé dekádě
    attrs = build_attributes(jikan, _media(1, 100, year=2020), anilist_min_rank=30)
    assert "2010s" in attrs
    assert "2020s" not in attrs


def test_anilist_format_labels_keep_acronyms_and_source_other_is_filtered():
    m = _media(1, 100)
    m["format"] = "OVA"
    m["source"] = "OTHER"
    attrs = build_attributes(None, m, anilist_min_rank=30)
    assert attrs["ova"].label == "OVA"          # ne "Ova" (dřívější .title() bug)
    assert "other" not in attrs                  # source OTHER = žádná informace,
                                                 # jen by strašil ve vysvětleních
    m2 = _media(2, 200)
    m2["format"] = "TV_SHORT"
    attrs2 = build_attributes(None, m2, anilist_min_rank=30)
    assert attrs2["tv_short"].label == "TV Short"


def test_attributes_anilist_genres_merge_with_mal_genres_without_duplication():
    jikan = {"genres": [{"name": "Comedy"}], "themes": [], "demographics": [],
             "source": "", "type": "", "year": None, "studios": []}
    attrs = build_attributes(jikan, _media(1, 100), anilist_min_rank=30)
    # Comedy z obou zdrojů = jeden atribut; Drama jen z AniListu se doplní
    assert attrs["comedy"].weight == 1.0
    assert "drama" in attrs


# ── relations adaptér ────────────────────────────────────────────────────

def test_relations_adapter_maps_series_types_and_filters_non_anime():
    media = _media(1, 100, relations=[
        {"relationType": "SEQUEL", "node": {"idMal": 2, "type": "ANIME"}},
        {"relationType": "SIDE_STORY", "node": {"idMal": 3, "type": "ANIME"}},
        {"relationType": "ADAPTATION", "node": {"idMal": 99, "type": "MANGA"}},
        {"relationType": "CHARACTER", "node": {"idMal": 4, "type": "ANIME"}},
        {"relationType": "SEQUEL", "node": {"idMal": None, "type": "ANIME"}},
    ])
    rel = _relations_from_anilist(media)
    got = {(r["relation"], r["entry"][0]["mal_id"]) for r in rel["relations"]}
    # ADAPTATION/CHARACTER nejsou sériové typy, manga uzel a uzel bez MAL ID pryč
    assert got == {("sequel", 2), ("side story", 3)}


def test_relations_adapter_returns_none_when_nothing_useful():
    assert _relations_from_anilist(_media(1, 100)) is None
    assert _relations_from_anilist(None) is None


# ── čištění AniList description ─────────────────────────────────────────

def test_clean_description_strips_html_spoilers_and_entities():
    raw = "První řádek.<br><i>Kurzíva</i> ~!spoiler!~ &amp; konec."
    assert _clean_anilist_description(raw) == "První řádek.\nKurzíva & konec."


def test_clean_description_empty_input():
    assert _clean_anilist_description("") == ""
    assert _clean_anilist_description(None) == ""


# ── Enricher v AniList-only režimu ──────────────────────────────────────

class FakeAniListClient:
    def __init__(self, media_by_id):
        self.media = media_by_id

    def get_anime_batch(self, mal_ids, show_progress=True):
        return {mid: self.media[mid] for mid in mal_ids if mid in self.media}


def _no_jikan_cfg():
    cfg = Config()
    cfg.enrich.use_jikan = False
    return cfg


def _entry(mal_id, score):
    return MalEntry(mal_id=mal_id, title=f"T{mal_id}", type="TV", episodes=12,
                    watched_episodes=12, score=score, status="Completed",
                    start_date="", finish_date="", rewatched=0)


def test_enricher_without_jikan_builds_full_titles_from_anilist():
    m1 = _media(1, 100, relations=[
        {"relationType": "SEQUEL", "node": {"idMal": 2, "type": "ANIME"}}])
    m2 = _media(2, 200, relations=[
        {"relationType": "PREQUEL", "node": {"idMal": 1, "type": "ANIME"}}])
    enricher = Enricher(_no_jikan_cfg(), anilist=FakeAniListClient({1: m1, 2: m2}))
    assert enricher.jikan is None   # use_jikan=False -> klient se nevytvoří

    titles = enricher.build_titles([_entry(1, 9), _entry(2, 8)], show_progress=False)
    by_id = {t.mal_id: t for t in titles}

    assert len(titles) == 2
    assert by_id[1].community == pytest.approx(8.4)          # averageScore/10
    assert "comedy" in by_id[1].attrs and "2010s" in by_id[1].attrs
    # franšíza přes AniList relations: oba členové váha 1/sqrt(2)
    assert by_id[1].weight == pytest.approx(1 / math.sqrt(2))
    assert by_id[2].weight == pytest.approx(1 / math.sqrt(2))


def test_enricher_without_jikan_uses_cleaned_anilist_synopsis():
    enricher = Enricher(_no_jikan_cfg(), anilist=FakeAniListClient({1: _media(1, 100)}))
    enriched = enricher.enrich_ids([1], show_progress=False)
    assert enriched[1].synopsis == "První řádek.\nKurzíva & konec."
    assert enriched[1].title_en == "Anime 1 EN"


# ── Recommender: MAL-rec větev se bez Jikanu přeskočí ───────────────────

class _StubModel:
    u_mean = 8.0
    clusters = []

    def top_effects(self, n=40, sign=1):
        return []


class _StubEnricher:
    jikan = None
    anilist = None
    shikimori = None


def test_gather_candidates_skips_mal_rec_without_jikan():
    cfg = Config()
    rec = Recommender(_StubModel(), _StubEnricher(), cfg)
    seed = Title(mal_id=1, title="Seed", user_score=9.0, community=8.0, attrs={})
    # Nesmí spadnout na self.enr.jikan.get_recommendations (jikan je None);
    # bez zdrojů prostě nevrátí žádné kandidáty.
    assert rec._gather_candidates([seed], seen_ids=set()) == {}


# ── staff: jeden zdroj na běh, klíč nezávislý na formátu jména ───────────

def test_person_key_is_source_independent():
    """Jikan píše "Mizushima, Tsutomu", AniList "Tsutomu Mizushima" -- tentýž
    člověk. Ze ~159 jmen se doslovně shodovalo 9, po seřazení slov 100."""
    from animodel.attributes import person_key
    assert person_key("Mizushima, Tsutomu") == person_key("Tsutomu Mizushima")
    assert person_key("Ōkawa, Nanase") == person_key("Nanase Okawa")
    assert person_key("") == ""


def test_staff_attribute_key_ignores_name_order_but_label_stays_readable():
    from animodel.attributes import build_attributes

    jikan_shape = [{"person": {"name": "Mizushima, Tsutomu"},
                    "positions": ["Director"]}]
    anilist_shape = [{"person": {"name": "Tsutomu Mizushima"},
                      "positions": ["Director"]}]
    a = build_attributes(None, None, staff=jikan_shape)
    b = build_attributes(None, None, staff=anilist_shape)
    assert set(a) == set(b), "klíč musí být stejný pro oba zdroje"
    # popisek ale zůstává tak, jak ho zdroj dodal
    assert list(a.values())[0].label == "Director: Mizushima, Tsutomu"
    assert list(b.values())[0].label == "Director: Tsutomu Mizushima"


def test_anilist_staff_adapter_maps_roles_to_jikan_shape():
    from animodel.enrich import _staff_from_anilist

    media = {"staff": {"edges": [
        {"role": "Director", "node": {"name": {"full": "Shinichirou Watanabe"}}},
        {"role": "Series Composition", "node": {"name": {"full": "Keiko Nobumoto"}}},
        {"role": "", "node": {"name": {"full": "Bez role"}}},        # vynechá se
    ]}}
    out = _staff_from_anilist(media)
    assert out == [
        {"person": {"name": "Shinichirou Watanabe"}, "positions": ["Director"]},
        {"person": {"name": "Keiko Nobumoto"}, "positions": ["Series Composition"]},
    ]
    assert _staff_from_anilist(None) == []
    assert _staff_from_anilist({}) == []


def test_staff_sources_are_never_mixed(tmp_path):
    """Míchání per titul by tomutéž člověku dalo dva klíče podle toho, který
    zdroj titul pokryl."""
    from animodel.config import Config
    from animodel.enrich import Enricher

    class FakeJikan:
        def __init__(self):
            self.staff_calls = 0

        def get_anime_batch(self, ids, show_progress=True):
            return {i: {"title": f"T{i}"} for i in ids}

        def get_staff_batch(self, ids, show_progress=True):
            self.staff_calls += 1
            return {i: [{"person": {"name": "Jikan Person"},
                         "positions": ["Director"]}] for i in ids}

    class FakeAniList:
        def get_anime_batch(self, ids, show_progress=True):
            # titul 2 staff NEMÁ -- při míchání by spadl na Jikan
            return {1: {"idMal": 1, "staff": {"edges": [
                        {"role": "Director",
                         "node": {"name": {"full": "AniList Person"}}}]}},
                    2: {"idMal": 2}}

    cfg = Config()
    cfg.cache_dir = str(tmp_path)
    cfg.enrich.include_staff = True
    cfg.enrich.staff_source = "anilist"
    jik = FakeJikan()
    enr = Enricher(cfg, jikan=jik, anilist=FakeAniList())
    out = enr.enrich_ids([1, 2])

    assert jik.staff_calls == 0, "s anilist zdrojem se Jikan /staff nesmí volat"
    assert any(av.category == "director" for av in out[1].attrs.values())
    assert not any(av.category == "director" for av in out[2].attrs.values())


def test_staff_falls_back_to_anilist_without_jikan(tmp_path):
    """--no-jikan režim: staff musí jít z AniListu, jinak by signál zmizel."""
    from animodel.config import Config
    from animodel.enrich import Enricher

    class FakeAniList:
        def get_anime_batch(self, ids, show_progress=True):
            return {1: {"idMal": 1, "staff": {"edges": [
                {"role": "Director", "node": {"name": {"full": "A B"}}}]}}}

    cfg = Config()
    cfg.cache_dir = str(tmp_path)
    cfg.enrich.include_staff = True
    cfg.enrich.staff_source = "jikan"      # i tak musí spadnout na AniList
    cfg.enrich.use_jikan = False
    enr = Enricher(cfg, jikan=None, anilist=FakeAniList())
    out = enr.enrich_ids([1])
    assert any(av.category == "director" for av in out[1].attrs.values())


def test_country_of_origin_added_only_when_not_japanese():
    """Japonsko tvoří ~95 % seznamu -- jako atribut by to byla konstanta
    s nulovým efektem. Informace je v tom, když titul japonský NENÍ."""
    from animodel.attributes import build_attributes

    jp = build_attributes(None, {"countryOfOrigin": "JP"})
    cn = build_attributes(None, {"countryOfOrigin": "CN"})
    assert not [av for av in jp.values() if av.category == "origin"]
    assert [av.label for av in cn.values() if av.category == "origin"] \
        == ["Čínský původ"]
    assert not build_attributes(None, {"countryOfOrigin": None})
