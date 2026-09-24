"""Franšízové pohledy nad poolem (Recommender._franchise_views): jedna karta
na franšízu, pokračování rozjetých sérií mimo „nové objevy", PTW zvlášť a
štítek „začni od" u pokračování s neviděným prequelem.

Motivace (HODNOCENI_PROJEKTU.md §9c, nález #6): doporučení #1 ve dvou
snapshotech bylo „3-gatsu no Lion 2nd Season", zatímco první řadu uživatel
neviděl (byla #3), a PTW tituly zabíraly 15-16 ze 40 míst přehledu."""
import pytest

from animodel.config import Config
from animodel.enrich import Enriched
from animodel.recommend import Recommender


class _Model:
    u_mean, c_mean = 8.0, 7.5
    clusters, cluster_feat_keys = [], set()
    effects, interactions, triples = {}, [], []

    def top_effects(self, n=40, sign=1):
        return []

    def affinity(self, attrs):
        return 0.0

    def predict(self, attrs, community):
        return 8.0, 7.5, 8.5, []

    def residuals(self):
        return {}


def _rel(kind, *ids):
    return {"relation": kind, "entry": [{"type": "anime", "mal_id": i} for i in ids]}


def _en(mid, title, fmt="TV", relations=None, community=7.5):
    return Enriched(mal_id=mid, title=title, title_en="", community=community,
                    attrs={}, jikan={"type": fmt, "relations": relations or []})


class _Enr:
    def __init__(self, enriched):
        self.jikan = self.anilist = self.shikimori = None
        self._enr = enriched

    def enrich_ids(self, ids, show_progress=True):
        return {m: self._enr[m] for m in ids if m in self._enr}

    def relations_data(self, enriched):
        return {m: e.jikan for m, e in enriched.items() if e.jikan}


def _run(votes: dict, enriched: dict, watched=(), ptw=()):
    meta = {mid: {"item_votes": v, "user_votes": 0.0, "cf_seeds": [],
                  "sources": {"MAL-rec"}} for mid, v in votes.items()}
    rec = Recommender(_Model(), _Enr(enriched), Config())
    rec._gather_candidates = lambda titles, seen_ids: meta
    return rec.recommend([], ptw_ids=set(ptw), watched_ids=set(watched),
                         show_progress=False, limit=None)


def test_one_card_per_franchise_and_sections_are_split():
    enriched = {
        1: _en(1, "Moje série S1", relations=[_rel("Sequel", 2)]),
        2: _en(2, "Moje série S2", relations=[_rel("Prequel", 1)]),
        3: _en(3, "Nový hit"),
        4: _en(4, "Nový hit 2", relations=[_rel("Prequel", 3)]),
        5: _en(5, "Z mého PTW"),
    }
    res = _run({2: 50.0, 3: 40.0, 4: 60.0, 5: 30.0}, enriched,
               watched=[1], ptw=[5])

    # pool zůstává úplný -- pohledy jsou jen jiné řezy týchž dat
    assert len(res.recs) == 4
    # franšíza 3+4: jedna karta (vyšší kompozit), druhý díl jen vyjmenovaný
    assert [r.mal_id for r in res.discovery] == [4]
    assert res.discovery[0].franchise_members == ["Nový hit"]
    # pokračování rozjeté série ven z objevů, PTW zvlášť
    assert [r.mal_id for r in res.continuations] == [2]
    assert [r.mal_id for r in res.ptw_ranked] == [5]
    assert res.roots[1] == res.roots[2] and res.roots[3] == res.roots[4]


def test_mid_franchise_sequel_gets_entry_note():
    enriched = {
        20: _en(20, "Třetí řada", relations=[_rel("Prequel", 21)]),
        21: _en(21, "Druhá řada", relations=[_rel("Prequel", 22), _rel("Sequel", 20)]),
        22: _en(22, "První řada", relations=[_rel("Sequel", 21)]),
    }
    res = _run({20: 60.0, 21: 40.0, 22: 10.0}, enriched)
    card = res.discovery[0]
    assert card.mal_id == 20
    assert card.entry_note == "začni od: První řada"      # nejstarší neviděný díl


def test_prologue_ova_is_not_an_entry_part():
    """Bez kontroly formátu se za „začátek série" označí i prologová OVA --
    2 z 8 nálezů na reálném poolu (§9c, nález #6)."""
    enriched = {
        10: _en(10, "Hlavní série", relations=[_rel("Prequel", 11)]),
        11: _en(11, "Prolog OVA", fmt="OVA", relations=[_rel("Sequel", 10)]),
    }
    res = _run({10: 50.0, 11: 5.0}, enriched)
    card = res.discovery[0]
    assert card.mal_id == 10 and card.entry_note is None
    assert card.franchise_members == ["Prolog OVA"]


def test_watched_franchise_member_keeps_title_out_of_discovery():
    """Remake ani paralelní adaptace pokračování není -- „alternative
    version" franšízy nespojuje (§9c)."""
    enriched = {
        30: _en(30, "Originál (viděl jsem)", relations=[_rel("Alternative version", 31)]),
        31: _en(31, "Remake"),
    }
    res = _run({31: 40.0}, enriched, watched=[30])
    assert [r.mal_id for r in res.discovery] == [31]
    assert res.continuations == []


def _pop(mid, title, popularity, relations=None):
    return Enriched(mal_id=mid, title=title, title_en="", community=7.5, attrs={},
                    jikan={"type": "TV", "relations": relations or [],
                           "popularity": popularity})


def test_popular_titles_outside_list_go_to_known_section():
    """Populární titul, který nemám ani na PTW, skoro jistě znám a vědomě
    přeskakuju (§9d.4): nesráží se, jen jde z objevů do vlastní sekce.
    Rozhoduje nejpopulárnější díl franšízy -- karta druhé řady s horším
    pořadím popularity patří do sekce taky."""
    enriched = {
        40: _pop(40, "Slavný thriller", 12),
        41: _pop(41, "Neznámá romance", 2400),
        42: _pop(42, "Slavná série S1", 120, [_rel("Sequel", 43)]),
        43: _pop(43, "Slavná série S2", 900, [_rel("Prequel", 42)]),
        44: _pop(44, "Slavné z mého PTW", 30),
    }
    res = _run({40: 50.0, 41: 20.0, 42: 5.0, 43: 60.0, 44: 40.0}, enriched,
               ptw=[44])
    assert [r.mal_id for r in res.discovery] == [41]
    assert [r.mal_id for r in res.known] == [43, 40]
    assert [r.mal_id for r in res.ptw_ranked] == [44]     # PTW zůstává PTW


def test_known_section_can_be_disabled():
    enriched = {40: _pop(40, "Slavný thriller", 12)}
    cfg_off = Config()
    cfg_off.recommend.known_popularity = 0
    meta = {40: {"item_votes": 5.0, "user_votes": 0.0, "cf_seeds": [],
                 "sources": {"MAL-rec"}}}
    rec = Recommender(_Model(), _Enr(enriched), cfg_off)
    rec._gather_candidates = lambda titles, seen_ids: meta
    res = rec.recommend([], ptw_ids=set(), watched_ids=set(),
                        show_progress=False, limit=None)
    assert [r.mal_id for r in res.discovery] == [40] and res.known == []
