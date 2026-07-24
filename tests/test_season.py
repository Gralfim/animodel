"""Sezónní doporučení (season.py): detekce/parsování sezóny, výpočet data
finále, rozdělení na pokračování mých sérií vs. nové tituly, vyloučení
shlédnutých, řazení, a get_season stránkování + necachovaný airing batch."""
import datetime as dt

import pytest

from animodel.attributes import AttrValue
from animodel.config import RecommendCfg
from animodel.enrich import Enriched
from animodel.season import (
    current_season, parse_season_arg, finale_date, build_season_view,
)


# ── detekce a parsování sezóny ───────────────────────────────────────────

def test_current_season_month_boundaries():
    assert current_season(dt.date(2026, 1, 5)) == (2026, "winter")
    assert current_season(dt.date(2026, 4, 1)) == (2026, "spring")
    assert current_season(dt.date(2026, 7, 15)) == (2026, "summer")
    assert current_season(dt.date(2026, 10, 31)) == (2026, "fall")
    assert current_season(dt.date(2026, 12, 20)) == (2026, "fall")


def test_parse_season_arg_orders_and_fallback():
    assert parse_season_arg(["2026", "summer"]) == (2026, "summer")
    assert parse_season_arg(["spring", "2025"]) == (2025, "spring")   # opačné pořadí
    assert parse_season_arg([]) == current_season()
    assert parse_season_arg(["nesmysl"]) == current_season()          # fallback


# ── výpočet data finále ──────────────────────────────────────────────────

def _ts(y, m, d):
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


def test_finale_releasing_extrapolates_from_next_episode():
    # 5. díl 2026-08-02, celkem 14 -> +9 týdnů = 2026-10-04
    fin = finale_date({"status": "RELEASING", "episodes": 14,
                       "next_ep": 5, "next_airing_at": _ts(2026, 8, 2)})
    assert fin == "2026-10-04"


def test_finale_unknown_when_episode_count_missing():
    assert finale_date({"status": "RELEASING", "episodes": None,
                        "next_ep": 5, "next_airing_at": _ts(2026, 8, 2)}) is None
    assert finale_date(None) is None


def test_finale_finished_uses_end_date():
    assert finale_date({"status": "FINISHED", "end_date": (2026, 9, 20)}) == "2026-09-20"


# ── fake enricher / model pro build_season_view ─────────────────────────

class FakeModel:
    """taste_fit = součet vah 'good' atributů (deterministické řazení)."""
    def _raw_resid_pred(self, attrs):
        return sum(av.weight for k, av in attrs.items() if k == "good")

    def predict(self, attrs, community):
        tf = self._raw_resid_pred(attrs)
        return 7.0 + tf, 6.5, 8.5, [(k, "genre", av.weight, False)
                                    for k, av in attrs.items()]


class FakeJikan:
    def __init__(self, season_list):
        self._season = season_list

    def get_season(self, year, season):
        return self._season


class FakeAniList:
    def __init__(self, airing):
        self._airing = airing

    def get_airing_batch(self, mal_ids):
        return {mid: self._airing[mid] for mid in mid_list(mal_ids, self._airing)}


def mid_list(mal_ids, d):
    return [m for m in mal_ids if m in d]


class FakeEnricher:
    """Řídí enrich_ids (Enriched objekty) a relations_data (franšízy)."""
    def __init__(self, enriched, relations, jikan, anilist):
        self._enr = enriched          # mal_id -> Enriched
        self._rel = relations         # mal_id -> jikan-shape {"relations":[...]}
        self.jikan = jikan
        self.anilist = anilist

    def enrich_ids(self, ids, show_progress=True):
        return {m: self._enr[m] for m in ids if m in self._enr}

    def relations_data(self, enriched_map):
        return {m: self._rel[m] for m in enriched_map if m in self._rel}


def _en(mid, title, good=0.0, community=7.5):
    attrs = {"good": AttrValue("genre", good, "Good")} if good else {}
    return Enriched(mal_id=mid, title=title, title_en="", community=community,
                    attrs=attrs, synopsis="syn")


def _rel(*mal_ids):
    """Jikan-shape relations spojující do jedné série."""
    return {"relations": [{"relation": "Sequel",
                           "entry": [{"mal_id": m, "type": "anime"} for m in mal_ids]}]}


def _season_entry(mid, typ="TV", broadcast="Mondays"):
    return {"mal_id": mid, "type": typ, "broadcast": {"day": broadcast}}


def _rc(season_min_prequel_score=7.0, season_top_new=30):
    return RecommendCfg(season_min_prequel_score=season_min_prequel_score,
                        season_top_new=season_top_new)


def test_sequel_of_liked_series_goes_to_sequels_section():
    # můj titul 1 (známka 9) -> sezónní pokračování 100 (stejná franšíza)
    my_scores = {1: 9.0}
    enriched = {1: _en(1, "Oblíbená S1"), 100: _en(100, "Oblíbená S2", good=1.0)}
    relations = {1: _rel(100), 100: _rel(1)}
    enr = FakeEnricher(enriched, relations,
                       FakeJikan([_season_entry(100)]),
                       FakeAniList({100: {"status": "RELEASING", "episodes": 12,
                                          "next_ep": 3, "next_airing_at": _ts(2026, 7, 20)}}))
    sequels, new = build_season_view(FakeModel(), enr, my_scores, watched_ids={1},
                                     ptw_ids=set(), year=2026, season="summer",
                                     rc=_rc(), show_progress=False)
    assert [r.mal_id for r in sequels] == [100]
    assert new == []
    assert "Oblíbená S1" in sequels[0].season_note and "9" in sequels[0].season_note
    assert sequels[0].finale_date is not None   # dopočteno


def test_sequel_below_threshold_goes_to_new_with_note():
    my_scores = {1: 6.0}   # pod prahem 7
    enriched = {1: _en(1, "Vlažná S1"), 100: _en(100, "Vlažná S2", good=0.5)}
    relations = {1: _rel(100), 100: _rel(1)}
    enr = FakeEnricher(enriched, relations, FakeJikan([_season_entry(100)]),
                       FakeAniList({}))
    sequels, new = build_season_view(FakeModel(), enr, my_scores, watched_ids={1},
                                     ptw_ids=set(), year=2026, season="summer",
                                     rc=_rc(), show_progress=False)
    assert sequels == []
    assert [r.mal_id for r in new] == [100]
    assert "pod prahem" in new[0].season_note


def test_sequel_of_unwatched_series_marked_prerequisites():
    # sezónní titul 100 je pokračování 99, které vůbec nemám
    my_scores = {1: 9.0}
    enriched = {1: _en(1, "Nesouvisející"),
                100: _en(100, "Cizí S3", good=0.9), 99: _en(99, "Cizí S1")}
    relations = {100: _rel(99), 99: _rel(100)}
    enr = FakeEnricher(enriched, relations, FakeJikan([_season_entry(100)]),
                       FakeAniList({}))
    sequels, new = build_season_view(FakeModel(), enr, my_scores, watched_ids=set(),
                                     ptw_ids=set(), year=2026, season="summer",
                                     rc=_rc(), show_progress=False)
    assert sequels == []
    assert new[0].mal_id == 100
    assert "neviděny" in new[0].season_note


def test_watched_season_titles_excluded_and_music_filtered():
    my_scores = {1: 9.0}
    enriched = {m: _en(m, f"T{m}", good=0.1) for m in (100, 101, 102)}
    enr = FakeEnricher(enriched, {}, FakeJikan([
        _season_entry(100),                 # nový
        _season_entry(101),                 # už shlédnutý -> ven
        _season_entry(102, typ="Music"),    # hudba -> ven
    ]), FakeAniList({}))
    sequels, new = build_season_view(FakeModel(), enr, my_scores,
                                     watched_ids={101}, ptw_ids=set(),
                                     year=2026, season="summer", rc=_rc(),
                                     show_progress=False)
    assert [r.mal_id for r in new] == [100]


def test_new_titles_sorted_by_taste_fit_and_capped():
    my_scores = {1: 9.0}
    # taste_fit = váha 'good': tituly s vyšší vahou první
    enriched = {10: _en(10, "A", good=0.2), 11: _en(11, "B", good=0.9),
                12: _en(12, "C", good=0.5)}
    enr = FakeEnricher(enriched, {}, FakeJikan([_season_entry(m) for m in (10, 11, 12)]),
                       FakeAniList({}))
    _seq, new = build_season_view(FakeModel(), enr, my_scores, watched_ids=set(),
                                  ptw_ids=set(), year=2026, season="summer",
                                  rc=_rc(season_top_new=2), show_progress=False)
    assert [r.mal_id for r in new] == [11, 12]   # top 2 dle taste_fit, D vypadlo


def test_ptw_season_title_is_flagged():
    my_scores = {1: 9.0}
    enriched = {100: _en(100, "Na frontě", good=0.5)}
    enr = FakeEnricher(enriched, {}, FakeJikan([_season_entry(100)]), FakeAniList({}))
    _seq, new = build_season_view(FakeModel(), enr, my_scores, watched_ids=set(),
                                  ptw_ids={100}, year=2026, season="summer",
                                  rc=_rc(), show_progress=False)
    assert new[0].ptw is True
