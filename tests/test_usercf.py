"""Senpai pipeline (usercf.py): discovery → plné seznamy → podobnost na
plném překryvu → výběr senpai → doporučení. Vše nad fake klientem, bez sítě."""
import math

import pytest

from animodel.config import Config, RecommendCfg
from animodel.usercf import (
    Senpai, _norm_score, _pearson, discover_candidates, evaluate_candidate,
    evaluate_candidates, select_senpai, recommend_from_senpai,
    find_senpai_recommendations,
)


class FakeClient:
    """Úzký kontrakt, který usercf.py potřebuje: _cached_media,
    _media_popularity, get_watcher_entries, get_user_animelist."""

    def __init__(self, media=None, watchers=None, userlists=None):
        self.media = media or {}           # mal_id -> {"id": aid, "popularity": p}
        self.watchers = watchers or {}     # anilist_id -> [[uid, name, raw], ...]
        self.userlists = userlists or {}   # uid -> {"fmt", "entries"} | None
        self.list_calls = 0

    def _cached_media(self, mal_id):
        return self.media.get(mal_id)

    def _media_popularity(self, anilist_id, fallback_mal_id):
        return self.media.get(fallback_mal_id, {}).get("popularity", 0)

    def get_watcher_entries(self, anilist_id, users_per_seed, per_page=50):
        return self.watchers.get(anilist_id, [])[:users_per_seed]

    def get_user_animelist(self, uid):
        self.list_calls += 1
        return self.userlists.get(uid)


def _entry(mid, raw, avg=75, title=""):
    return [mid, raw, avg, title or f"A{mid}"]


# ── pomocné funkce ───────────────────────────────────────────────────────

def test_norm_score_uses_format_with_heuristic_fallback():
    assert _norm_score(85, "POINT_100") == pytest.approx(0.85)
    assert _norm_score(4, "POINT_5") == pytest.approx(0.8)
    assert _norm_score(85, None) == pytest.approx(0.85)   # heuristika >10
    assert _norm_score(8, None) == pytest.approx(0.8)
    assert _norm_score(0, "POINT_10") == 0.0


def test_pearson_matches_manual_computation():
    xs, ys = [1.0, 2.0, 3.0], [2.0, 4.0, 6.1]
    assert _pearson(xs, ys) == pytest.approx(0.9999, abs=1e-3)
    assert _pearson([1.0, 2.0], [5.0, 3.0]) == pytest.approx(-1.0)
    assert _pearson([1.0], [1.0]) == 0.0          # málo bodů
    assert _pearson([1.0, 1.0], [2.0, 3.0]) == 0.0  # nulová variance


# ── fáze 1: discovery ────────────────────────────────────────────────────

def _discovery_client():
    # tituly 1 (pop 500, nišový) a 2 (pop 50000, populární)
    media = {1: {"id": 10, "popularity": 500},
             2: {"id": 20, "popularity": 50000}}
    watchers = {
        10: [[100, "alfa", 90], [101, "beta", 80]],
        20: [[100, "alfa", 85], [102, "gama", 70]],
    }
    return FakeClient(media=media, watchers=watchers)


def test_discovery_qualification_and_idf_priority():
    client = _discovery_client()
    cands = discover_candidates(client, {1: 9.0, 2: 8.0},
                                seed_count=2, users_per_seed=50,
                                min_sample_overlap=2)
    # jen alfa sdílí oba seedy; beta a gama mají po jednom -> nekvalifikují se
    assert [(c[0], c[1]) for c in cands] == [(100, "alfa")]


def test_discovery_excludes_own_account_case_insensitively():
    """Vlastní AniList účet (import MAL seznamu) má podobnost 1.00 a jako
    senpai je k ničemu -- musí vypadnout hned v discovery, ať se jeho plný
    seznam ani nestahuje."""
    client = _discovery_client()
    cands = discover_candidates(client, {1: 9.0, 2: 8.0},
                                seed_count=2, users_per_seed=50,
                                min_sample_overlap=1,
                                exclude_users={"ALFA"})   # jiná velikost písmen
    assert 100 not in [c[0] for c in cands]
    assert {c[0] for c in cands} == {101, 102}


def test_discovery_exclude_users_empty_or_none_is_noop():
    client = _discovery_client()
    base = discover_candidates(client, {1: 9.0, 2: 8.0}, seed_count=2,
                               users_per_seed=50, min_sample_overlap=1)
    for excl in (None, set(), {"", "  "}):
        out = discover_candidates(client, {1: 9.0, 2: 8.0}, seed_count=2,
                                  users_per_seed=50, min_sample_overlap=1,
                                  exclude_users=excl)
        assert [c[0] for c in out] == [c[0] for c in base]


def test_find_senpai_respects_exclude_users_from_config():
    media = {1: {"id": 10, "popularity": 300}}
    watchers = {10: [[100, "Gralfim", 90], [101, "kamarad", 85]]}
    my = {1: 9.0, 2: 8.0, 3: 7.0, 4: 10.0}
    # oba mají identický vkus; jen "Gralfim" je můj vlastní import
    entries = [_entry(m, my[m]) for m in my] + [_entry(50, 9.0)]
    userlists = {100: {"fmt": "POINT_10", "entries": entries},
                 101: {"fmt": "POINT_10", "entries": entries}}
    client = FakeClient(media=media, watchers=watchers, userlists=userlists)

    rc = RecommendCfg(user_cf_seed_count=1, user_cf_users_per_seed=10,
                      user_cf_min_sample_overlap=1, user_cf_candidate_pool=10,
                      user_cf_senpai_count=5, user_cf_min_full_overlap=3,
                      user_cf_shrink_k=1.0, user_cf_exclude_users=["gralfim"])
    senpai, _recs = find_senpai_recommendations(client, my, watched_ids=set(my), rc=rc)
    assert [s.name for s in senpai] == ["kamarad"]
    assert client.list_calls == 1   # vlastní účet se ani nestahoval


def test_discovery_min_sample_overlap_one_lets_singles_in_rarity_order():
    client = _discovery_client()
    cands = discover_candidates(client, {1: 9.0, 2: 8.0},
                                seed_count=2, users_per_seed=50,
                                min_sample_overlap=1)
    ids = [c[0] for c in cands]
    assert ids[0] == 100                    # sdílí oba -> nejvyšší váha
    assert ids.index(101) < ids.index(102)  # nišový seed (pop 500) váží víc


# ── fáze 3: vyhodnocení jednoho kandidáta ────────────────────────────────

def test_evaluate_candidate_pearson_on_full_overlap_and_shrinkage():
    # překryv 4 tituly; jeho odchylky od komunity = 2x moje -> Pearson 1.0
    my_scores = {1: 9.0, 2: 8.0, 3: 7.0, 4: 10.0}
    # komunita avg_raw=75 (0.75); moje diffy: +0.15, +0.05, -0.05, +0.25
    # jeho POINT_10 skóre: diffy 2x -> 10.5(!)->10, ...: použij mírnější:
    # jeho norm = 0.75 + 2*moje_diff kde to jde do 1.0: 1.05 nejde ->
    # vezmi jeho = 0.75 + moje_diff + 0.05 (afinní transformace, Pearson=1)
    their = {1: 9.5, 2: 8.5, 3: 7.5, 4: 10.0}   # 0.95, 0.85, 0.75, 1.0
    entries = [_entry(m, their[m]) for m in their]
    entries.append(_entry(99, 8.0))   # titul, který já nemám -> novelty
    s = evaluate_candidate(7, "senpai-kun", {"fmt": "POINT_10", "entries": entries},
                           my_scores, watched_ids=set(my_scores), shrink_k=50.0)
    assert s.overlap == 4
    assert s.n_rated == 5
    assert s.n_novel == 1
    # afinní vztah diffů -> Pearson téměř 1 (titul 4: 1.0 místo 1.05, mírný ořez)
    assert s.similarity > 0.95
    assert s.score == pytest.approx(s.similarity * 4 / 54)
    assert 0 < s.personal_avg <= 1


def test_dropped_with_score_counts_as_a_real_rating():
    """Dropnutý titul SE známkou je platný (a výmluvný) datový bod --
    'zkusil a dal 3' o shodě vkusu říká hodně. Klient ho teď posílá v
    entries; tady jen ověřujeme, že vstupuje do Pearsona a sráží podobnost."""
    my_scores = {1: 10.0, 2: 9.0, 3: 8.0}
    # senpai mou desítku dropnul a dal jí 2 -> antikorelace
    entries = [_entry(1, 2.0, avg=80), _entry(2, 9.5, avg=75), _entry(3, 8.5, avg=75)]
    s = evaluate_candidate(7, "dropper", {"fmt": "POINT_10", "entries": entries},
                           my_scores, watched_ids=set(my_scores), shrink_k=1.0)
    assert s.overlap == 3
    assert s.similarity < 0    # dropnutá desítka správně zabíjí podobnost


def test_favorites_on_planning_are_not_penalized():
    """Mít můj oblíbený titul ve frontě (PTW) není důvod k penalizaci --
    na rozdíl od 'nikdy o tom neslyšel'."""
    my_scores = {1: 10.0, 2: 10.0, 3: 8.0}
    favs = {1, 2}
    entries = [_entry(1, 9.0), _entry(3, 8.0)]     # oblíbený 2 nemá ohodnocený
    base = {"fmt": "POINT_10", "entries": entries}

    without = evaluate_candidate(7, "a", dict(base, planning=[]),
                                 my_scores, set(my_scores), 1.0,
                                 favorites=favs, fav_miss_penalty=0.3)
    with_ptw = evaluate_candidate(7, "b", dict(base, planning=[2]),
                                  my_scores, set(my_scores), 1.0,
                                  favorites=favs, fav_miss_penalty=0.3)

    assert without.fav_covered == 1 and without.penalty == pytest.approx(1 - 0.3 * 0.5)
    assert with_ptw.fav_covered == 2 and with_ptw.penalty == 1.0
    assert with_ptw.score > without.score
    # podobnost zůstává nedotčená -- penalizace je až nad ní
    assert with_ptw.similarity == without.similarity


def test_fav_miss_penalty_scales_with_missing_fraction():
    my_scores = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 7.0}
    favs = {1, 2, 3, 4}
    # zná jen jeden ze čtyř oblíbených -> chybí 75 %
    entries = [_entry(1, 9.0), _entry(5, 7.0)]
    s = evaluate_candidate(7, "a", {"fmt": "POINT_10", "entries": entries,
                                    "planning": []},
                           my_scores, set(my_scores), 1.0,
                           favorites=favs, fav_miss_penalty=0.4)
    assert s.fav_covered == 1 and s.fav_total == 4
    assert s.fav_coverage == pytest.approx(0.25)
    assert s.penalty == pytest.approx(1 - 0.4 * 0.75)   # 0.70


def test_fav_miss_penalty_zero_disables_feature():
    my_scores = {1: 10.0, 2: 10.0}
    s = evaluate_candidate(7, "a", {"fmt": "POINT_10",
                                    "entries": [_entry(1, 9.0)], "planning": []},
                           my_scores, set(my_scores), 1.0,
                           favorites={1, 2}, fav_miss_penalty=0.0)
    assert s.penalty == 1.0


def test_evaluate_candidate_no_overlap_gives_zero_score():
    s = evaluate_candidate(7, "cizinec", {"fmt": "POINT_10",
                                          "entries": [_entry(99, 8.0)]},
                           {1: 9.0}, watched_ids={1}, shrink_k=50.0)
    assert s.overlap == 0 and s.score == 0.0


# ── fáze 2: pool a přeskočení privátních ─────────────────────────────────

def test_private_lists_skipped_without_losing_pool_slot():
    userlists = {
        100: None,                                     # privátní
        101: {"fmt": "POINT_10", "entries": [_entry(1, 8.0)]},
        102: {"fmt": "POINT_10", "entries": [_entry(1, 9.0)]},
    }
    client = FakeClient(userlists=userlists)
    cands = [(100, "priv", 3.0, 2), (101, "a", 2.0, 2), (102, "b", 1.0, 2)]
    out = evaluate_candidates(client, cands, {1: 9.0}, {1},
                              candidate_pool=2, shrink_k=50.0)
    # pool=2: privátní 100 nezabral slot, vyhodnotili se 101 i 102
    assert [s.uid for s in out] == [101, 102]


def test_scan_budget_caps_attempts():
    client = FakeClient(userlists={})   # všichni "privátní"
    cands = [(uid, f"u{uid}", 1.0, 2) for uid in range(100)]
    out = evaluate_candidates(client, cands, {1: 9.0}, {1},
                              candidate_pool=10, shrink_k=50.0,
                              scan_budget_factor=2.0)
    assert out == []
    assert client.list_calls == 20   # 10 * 2.0, ne všech 100


# ── výběr senpai ─────────────────────────────────────────────────────────

def _senpai(uid, score, overlap, sim=None):
    return Senpai(uid=uid, name=f"u{uid}", similarity=sim if sim is not None else score,
                  score=score, overlap=overlap, n_rated=500, n_novel=300,
                  personal_avg=0.7)


def test_select_senpai_applies_thresholds_and_count():
    evaluated = [
        _senpai(1, 0.50, 100),
        _senpai(2, 0.40, 39),     # málo překryvu
        _senpai(3, -0.20, 200),   # záporná podobnost
        _senpai(4, 0.30, 60),
        _senpai(5, 0.20, 45),
    ]
    out = select_senpai(evaluated, senpai_count=2, min_full_overlap=40)
    assert [s.uid for s in out] == [1, 4]   # top 2 dle skóre, prahy splněny


# ── fáze 4: doporučení od senpai ─────────────────────────────────────────

def test_recommend_from_senpai_differential_and_min_raters():
    s1 = Senpai(uid=1, name="a", similarity=0.6, score=0.5, overlap=50,
                n_rated=3, n_novel=2, personal_avg=0.8, fmt="POINT_10",
                entries=[_entry(10, 10.0, avg=80), _entry(11, 9.0, avg=70),
                         _entry(5, 9.0)])
    s2 = Senpai(uid=2, name="b", similarity=0.4, score=0.25, overlap=45,
                n_rated=2, n_novel=1, personal_avg=0.7, fmt="POINT_10",
                entries=[_entry(10, 9.0, avg=80)])
    out = recommend_from_senpai([s1, s2], rated_ids={5}, min_raters=2)

    # titul 5 mám ohodnocený -> pryč; titul 11 má jen 1 hodnotitele -> pryč
    assert [r["mal_id"] for r in out] == [10]
    r = out[0]
    # ruční výpočet: diff1 = 1.0-0.8=0.2, diff2 = 0.9-0.7=0.2
    # w_diff = (0.5*0.2 + 0.25*0.2)/(0.75) = 0.2; cf = 0.8 + 0.2 + 0 = 1.0
    assert r["cf_score"] == pytest.approx(10.0)
    assert r["n_users"] == 2
    assert r["top_raters"][0][0] == "a"


# ── orchestrátor end-to-end ──────────────────────────────────────────────

def test_find_senpai_recommendations_end_to_end():
    media = {1: {"id": 10, "popularity": 300}, 2: {"id": 20, "popularity": 800}}
    watchers = {
        10: [[100, "senpai-kun", 90], [101, "random", 60]],
        20: [[100, "senpai-kun", 85], [101, "random", 50]],
    }
    # senpai-kun: velký překryv, konzistentně korelovaný; random: antikorelace
    my = {1: 9.0, 2: 8.0, 3: 7.0, 4: 10.0, 5: 6.0}
    good_entries = [_entry(m, my[m] + 0.5 if my[m] < 10 else 10.0) for m in my]
    good_entries += [_entry(50, 9.5, avg=80, title="Skrytý klenot"),
                     _entry(51, 9.0, avg=75, title="Druhý klenot")]
    bad_entries = [_entry(m, 17 - my[m]) for m in my]   # inverzní vkus
    userlists = {
        100: {"fmt": "POINT_10", "entries": good_entries},
        101: {"fmt": "POINT_10", "entries": bad_entries},
    }
    client = FakeClient(media=media, watchers=watchers, userlists=userlists)

    rc = RecommendCfg(user_cf_seed_count=2, user_cf_users_per_seed=10,
                      user_cf_min_sample_overlap=2, user_cf_candidate_pool=10,
                      user_cf_senpai_count=5, user_cf_min_full_overlap=3,
                      user_cf_shrink_k=5.0)
    senpai, recs = find_senpai_recommendations(client, my, watched_ids=set(my), rc=rc)

    assert [s.name for s in senpai] == ["senpai-kun"]   # random má zápornou korelaci
    assert senpai[0].n_novel == 2
    # doporučení od jediného senpaie neprojdou min_raters=2 -> tituly 50/51
    # se objeví, jen když snížíme práh; default chování = prázdno
    assert recs == []
    recs1 = recommend_from_senpai(senpai, rated_ids=set(my), min_raters=1)
    assert {r["mal_id"] for r in recs1} == {50, 51}


# ── config: varování na neznámé (staré) klíče ────────────────────────────

def test_config_load_warns_on_unknown_keys(tmp_path, caplog):
    p = tmp_path / "config.yaml"
    p.write_text(
        "recommend:\n  user_cf_top_users: 200\n  user_cf_senpai_count: 10\n"
        "neznamy_top_klic: 1\n",
        encoding="utf-8",
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="animodel.config"):
        cfg = Config.load(str(p))
    assert cfg.recommend.user_cf_senpai_count == 10       # známý klíč se aplikuje
    warnings = " ".join(r.message for r in caplog.records)
    assert "user_cf_top_users" in warnings                 # starý klíč nahlášen
    assert "neznamy_top_klic" in warnings