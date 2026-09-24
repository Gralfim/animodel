"""history.py: záznam běhů klíčovaný OTISKEM STAVU SEZNAMU (ne datem),
snapshot v2 (stav seznamu, pool, log predikcí) a vyhodnocení ledgerem
událostí nad všemi snapshoty.

Jádro návrhu: ladění parametrů znamená desítky běhů nad týmž exportem;
datové klíčování by z nich udělalo desítky skoro identických snapshotů a
evaluaci by to ředilo. Otisk se počítá z (mal_id, status, score), takže
nezměněný seznam vždy padne na tentýž klíč."""
import datetime as _dt
import json
import statistics

import pytest

from animodel.config import Config
from animodel.history import (
    EXTRA_COLUMNS, POOL_COLUMNS, SCHEMA, Snapshot, build_snapshot,
    evaluate_history, export_fingerprint, format_report,
    franchise_neighbourhood, franchise_roots, load_predictions, load_snapshots,
    prediction_row, save_snapshot,
)
from animodel.mal import MalEntry
from animodel.recommend import Recommendation

T1, T2 = "2026-01-01T10:00:00", "2026-02-01T10:00:00"


def _entry(mal_id, status="Completed", score=8, episodes=12, watched=12,
           finish="2025-01-01"):
    return MalEntry(mal_id=mal_id, title=f"T{mal_id}", type="TV",
                    episodes=episodes, watched_episodes=watched, score=score,
                    status=status, start_date="", finish_date=finish,
                    rewatched=0)


def _rec(mal_id, rank_composite=1.0, ptw=False, pred=8.0):
    return Recommendation(
        mal_id=mal_id, title=f"T{mal_id}", title_en="", community=7.5,
        pred=pred, pred_lo=pred - 0.5, pred_hi=pred + 0.5, taste_fit=0.5,
        cf_signal=1.0, composite=rank_composite, ptw=ptw,
        cluster_name="Nálada", why=[], cf_seeds=[], sources=["MAL-rec"])


class _Model:
    scale, scale_triples, cv_rmse, baseline_rmse = 0.3, 0.15, 0.9, 0.95
    beta, u_mean, c_mean = 0.5, 8.0, 7.0
    clusters = []


def _snap(entries, recs, when=T1, **kw):
    return build_snapshot(entries, recs, _Model(), Config(), top=100,
                          now=_dt.datetime.fromisoformat(when), **kw)


def _history(*states):
    """states: (entries, recs, when) -> snapshoty"""
    return [_snap(e, r, when=w) for e, r, w in states]


def _main_files(d):
    """Hlavní soubory snapshotů (vedlejší `*.pool.json` nepočítat)."""
    return [p for p in d.glob("*.json") if not p.name.endswith(".pool.json")]


# ── otisk ────────────────────────────────────────────────────────────────

def test_fingerprint_is_stable_for_same_list_state():
    a = [_entry(1), _entry(2, "Plan to Watch", 0)]
    b = [_entry(2, "Plan to Watch", 0), _entry(1)]      # jiné pořadí
    assert export_fingerprint(a) == export_fingerprint(b)


def test_fingerprint_ignores_progress_and_dates():
    """Odsledovaný díl ani datum dokončení nemění ground truth -- hash
    SOUBORU by se změnil, otisk stavu seznamu ne. To je celý důvod, proč
    se nehashují bajty exportu."""
    a = [_entry(1, watched=3, finish="0000-00-00")]
    b = [_entry(1, watched=11, finish="2026-07-26")]
    assert export_fingerprint(a) == export_fingerprint(b)


@pytest.mark.parametrize("changed", [
    _entry(1, score=9),                       # jiná známka
    _entry(1, status="Dropped"),              # jiný status
])
def test_fingerprint_changes_on_rating_or_status(changed):
    base = [_entry(1)]
    assert export_fingerprint(base) != export_fingerprint([changed])


def test_fingerprint_changes_when_title_added():
    base = [_entry(1)]
    assert export_fingerprint(base) != export_fingerprint(base + [_entry(2)])


# ── ukládání: jeden snapshot na stav seznamu, ne na běh ──────────────────

def test_repeated_runs_on_same_export_keep_one_snapshot(tmp_path):
    """JÁDRO: ladění parametrů nad týmž exportem nesmí hromadit snapshoty."""
    entries = [_entry(i) for i in range(1, 6)]
    for i in range(5):                     # pět „ladicích" běhů
        save_snapshot(_snap(entries, [_rec(100 + i)]), tmp_path)
    files = _main_files(tmp_path)
    assert len(files) == 1, [f.name for f in files]
    got = load_snapshots(tmp_path)         # poslední běh vyhrává
    assert [r["mal_id"] for r in got[0].recommendations] == [104]


def test_changed_export_creates_new_snapshot(tmp_path):
    entries = [_entry(i) for i in range(1, 6)]
    save_snapshot(_snap(entries, [_rec(100)]), tmp_path)
    save_snapshot(_snap(entries + [_entry(6)], [_rec(200)]), tmp_path)
    assert len(_main_files(tmp_path)) == 2


def test_filename_has_rated_count_prefix_for_chronological_listing(tmp_path):
    small = [_entry(i) for i in range(1, 4)]
    big = [_entry(i) for i in range(1, 21)]
    p1 = save_snapshot(_snap(small, [_rec(1)]), tmp_path)
    p2 = save_snapshot(_snap(big, [_rec(1)]), tmp_path)
    assert p1.name.startswith("00003_") and p2.name.startswith("00020_")
    assert sorted([p1.name, p2.name]) == [p1.name, p2.name]   # abecedně = růst


def test_load_skips_foreign_and_corrupt_files(tmp_path):
    save_snapshot(_snap([_entry(1)], [_rec(1)]), tmp_path)
    (tmp_path / "poznamky.txt").write_text("nic", encoding="utf-8")
    (tmp_path / "00009_deadbeef1234.json").write_text("{ rozbité", encoding="utf-8")
    assert len(load_snapshots(tmp_path)) == 1


def test_load_missing_dir_is_empty(tmp_path):
    assert load_snapshots(tmp_path / "neexistuje") == []


def test_roundtrip_preserves_payload(tmp_path):
    snap = _snap([_entry(1), _entry(2, "Plan to Watch", 0)], [_rec(50, 2.5)])
    save_snapshot(snap, tmp_path)
    back = load_snapshots(tmp_path)[0]
    assert back.fingerprint == snap.fingerprint
    assert back.recommendations == snap.recommendations
    assert back.model["scale"] == 0.3
    assert back.config["w_cf"] == Config().recommend.w_cf


def test_load_sorts_by_time_not_by_rated_count(tmp_path):
    """n_rated klesne, když titul odhodnotíš nebo smažeš -- okna ledgeru
    potřebují chronologii, takže se řadí podle času běhu."""
    save_snapshot(_snap([_entry(1)], [_rec(10)], when=T2), tmp_path)
    save_snapshot(_snap([_entry(1), _entry(2)], [_rec(10)], when=T1), tmp_path)
    assert [s.saved_at for s in load_snapshots(tmp_path)] == [T1, T2]


# ── snapshot v2: stav seznamu, pool a log predikcí ───────────────────────

def test_snapshot_v2_stores_list_state_export_date_and_baseline(tmp_path):
    entries = [_entry(1, score=9, finish="2026-07-19"),
               _entry(2, "Plan to Watch", 0, finish="")]
    snap = _snap(entries, [_rec(10)], export_date="2026-09-15T15:32:00",
                 z_params={"taste_fit": [0.1, 1.2]})
    save_snapshot(snap, tmp_path)
    raw = json.loads(_main_files(tmp_path)[0].read_text(encoding="utf-8"))

    assert raw["schema"] == SCHEMA == 2
    assert raw["export_date"] == "2026-09-15T15:32:00"
    assert raw["list"] == [[1, "Completed", 9, "2026-07-19"],
                           [2, "Plan to Watch", 0, ""]]
    # baseline modelu: bez něj nejde spočítat známku vůči očekávání z komunity
    assert (raw["model"]["u_mean"], raw["model"]["c_mean"]) == (8.0, 7.0)
    assert raw["model"]["z_params"] == {"taste_fit": [0.1, 1.2]}
    assert "pool" not in raw                  # pool patří do vedlejšího souboru

    back = load_snapshots(tmp_path)[0]
    assert back.schema == 2 and back.date == "2026-09-15"
    assert back.list_state == raw["list"]


def test_pool_and_prediction_log_go_to_compact_side_file(tmp_path):
    extra = [prediction_row(30, 6.5, 6.0, 7.0, -0.2, 7.1, "ptw")]
    snap = _snap([_entry(1)], [_rec(10, 2.0, pred=9.0), _rec(11, 1.0, pred=7.5)],
                 extra=extra)
    save_snapshot(snap, tmp_path)

    text = (tmp_path / snap.pool_filename).read_text(encoding="utf-8")
    assert "\n" not in text                   # bez odsazení (~300 kB místo ~470)
    side = json.loads(text)
    assert side["pool_columns"] == list(POOL_COLUMNS) and len(side["pool"]) == 2
    assert side["extra_columns"] == list(EXTRA_COLUMNS)
    assert side["extra"] == [[30, 6.5, 6.0, 7.0, -0.2, 7.1, "ptw"]]

    assert len(load_snapshots(tmp_path)) == 1  # vedlejší soubor není snapshot
    back = load_snapshots(tmp_path)[0]
    assert load_predictions(tmp_path, back) == {10: 9.0, 11: 7.5, 30: 6.5}


def test_load_predictions_without_side_file_uses_top_n(tmp_path):
    snap = _snap([_entry(1)], [_rec(10, pred=9.0)])
    save_snapshot(snap, tmp_path)
    (tmp_path / snap.pool_filename).unlink()
    assert load_predictions(tmp_path, snap) == {10: 9.0}


def test_v1_snapshot_loads_with_defaults():
    """Starší snapshoty se musí dál číst -- jen bez stavu seznamu a poolu."""
    snap = Snapshot.from_dict({
        "fingerprint": "abc", "saved_at": "2026-07-28T14:26:42",
        "watched_ids": [1], "ptw_ids": [], "recommendations": []})
    assert snap.schema == 1 and snap.list_state == []
    assert snap.date == "2026-07-28"
    assert snap.shown == {}


def test_snapshot_stores_what_the_report_showed(tmp_path):
    """`recommendations` je top-N celého poolu (vč. PTW a pokračování), ne to,
    co bylo v reportu vidět -- „ukázáno a do PTW nepřidáno" potřebuje sekce
    (§9d.8)."""
    shown = {"discovery": [10, 11], "known": [12], "ptw": [], "continuations": []}
    save_snapshot(_snap([_entry(1)], [_rec(10)], shown=shown), tmp_path)
    assert load_snapshots(tmp_path)[0].shown == shown


# ── franšízy ─────────────────────────────────────────────────────────────

def _rel(*pairs):
    return {"relations": [{"relation": typ,
                           "entry": [{"type": "anime", "mal_id": mid}]}
                          for typ, mid in pairs]}


def test_franchise_neighbourhood_follows_series_relations_up_to_hops():
    """Log predikcí musí dosáhnout na pokračování rozjetých franšíz -- ta
    v poolu doporučení nejsou (pool + PTW pokryl 5 z 23 nových shlédnutí)."""
    rel = {1: _rel(("Sequel", 2), ("Other", 9)), 2: _rel(("Sequel", 3)),
           3: _rel(("Sequel", 4))}
    calls = []

    def relations_of(ids):
        calls.append(list(ids))
        return {m: rel[m] for m in ids if m in rel}

    assert franchise_neighbourhood({1}, relations_of, hops=2) == {2, 3}
    assert calls == [[1], [2]]          # jeden dotaz na krok, ne na titul


def test_franchise_roots_join_through_unseen_part_but_not_remakes():
    rel = {1: _rel(("Sequel", 2)), 3: _rel(("Prequel", 2)),
           5: _rel(("Alternative version", 6))}
    roots = franchise_roots([1, 3, 5, 6], rel)
    assert roots[1] == roots[3]         # spojeno přes neviděný díl 2
    assert roots[5] != roots[6]         # remake není pokračování


# ── vyhodnocení: ledger událostí ─────────────────────────────────────────

def test_evaluate_history_is_none_without_snapshots():
    assert evaluate_history([], [_entry(1)]) is None
    assert format_report(None) == []


def test_each_new_watch_is_one_event_across_snapshots():
    """v1 vyhodnocoval každý snapshot zvlášť: titul doporučený ve dvou
    snapshotech se započetl dvakrát. Ledger ho má jednou, s první expozicí."""
    base = [_entry(1)]
    snaps = _history((base, [_rec(10)], T1), (base + [_entry(2)], [_rec(10)], T2))
    now = base + [_entry(2), _entry(10, score=9)]

    res = evaluate_history(list(reversed(snaps)), now)   # pořadí vstupu nehraje roli
    rec = [ev for ev in res["events"] if ev["rec"]]
    assert [ev["mal_id"] for ev in rec] == [10]
    assert rec[0]["exposed_in"] == 0 and rec[0]["window"] == 1
    assert [w["n_new"] for w in res["windows"]] == [1, 1]


def test_recommendation_that_went_through_ptw_is_credited_to_the_model():
    """3-gatsu no Lion: doporučeno mimo PTW → v dalším snapshotu na PTW →
    dokoukáno. Přechod do PTW je vidět jen přes mezisnapshot a zásluha
    patří doporučení, ne vlastnímu plánu."""
    base = [_entry(1)]
    mid_state = base + [_entry(10, "Plan to Watch", 0)]
    snaps = _history((base, [_rec(10, pred=9.08)], T1),
                     (mid_state, [_rec(10, ptw=True, pred=9.07)], T2))

    res = evaluate_history(snaps, base + [_entry(10, score=7)])
    assert res["windows"][0]["ptw_added_from_recs"] == [(1, "T10")]
    ev = next(ev for ev in res["events"] if ev["mal_id"] == 10)
    assert ev["rec"] and ev["ptw_at_exposure"] is False
    assert res["cohorts"]["rec_not_ptw"] == (1, 1)
    assert res["cohorts"]["rec_ptw"] == (0, 0)


def test_cohorts_split_by_ptw_status_at_first_exposure():
    before = [_entry(1), _entry(5, "Plan to Watch", 0), _entry(6, "Plan to Watch", 0)]
    snaps = _history((before, [_rec(5, ptw=True), _rec(7)], T1))
    now = [_entry(1), _entry(5, score=8), _entry(6, score=7)]
    assert evaluate_history(snaps, now)["cohorts"] == {
        "rec_not_ptw": (0, 1), "rec_ptw": (1, 1), "ptw_only": (1, 1)}


def test_control_reported_without_continuations_of_started_franchises():
    """O pokračování rozjeté franšízy doporučovač nesoutěží -- od snapshotu
    2026-07-28 jich bylo 11 z 18 kontrolních titulů."""
    before = [_entry(1, score=8)]
    snaps = _history((before, [_rec(10)], T1))
    now = before + [_entry(10, score=9), _entry(2, score=6), _entry(3, score=8)]

    res = evaluate_history(snaps, now, roots={1: 1, 2: 1})
    g = res["scores"]["groups"]
    assert g["control_new"]["raw"]["mean"] == pytest.approx(8.0)   # jen titul 3
    assert g["control_all"]["raw"]["mean"] == pytest.approx(7.0)   # 6 a 8
    assert res["scores"]["deltas"]["control_new"]["raw"]["delta"] == pytest.approx(1.0)


def test_scores_are_averaged_per_franchise_not_per_title():
    before = [_entry(1)]
    snaps = _history((before, [_rec(10), _rec(11)], T1))
    now = before + [_entry(10, score=9), _entry(11, score=7), _entry(20, score=6)]
    res = evaluate_history(snaps, now, roots={10: 10, 11: 10})
    assert res["scores"]["groups"]["rec"]["raw"] == {
        "mean": pytest.approx(8.0), "n_groups": 1, "n_titles": 2}


def test_residual_delta_uses_baseline_stored_in_snapshot():
    """Doporučené tituly mají vyšší komunitní skóre z konstrukce; teprve
    známka vůči baseline ukazuje vkus, ne výběr kvality."""
    before = [_entry(1)]
    snaps = _history((before, [_rec(10)], T1))          # _Model: ū 8, c̄ 7, β 0.5
    now = before + [_entry(10, score=9), _entry(20, score=8)]

    res = evaluate_history(snaps, now, community={10: 9.0, 20: 7.0},
                           baseline=(0.0, 0.0, 0.0))    # fallback se nepoužije
    d = res["scores"]["deltas"]["control_new"]
    assert d["raw"]["delta"] == pytest.approx(1.0)
    assert d["resid"]["delta"] == pytest.approx(0.0)    # 9−9 vs. 8−8


def test_delta_interval_is_franchise_level_and_small_samples_are_inconclusive():
    before = [_entry(i, score=s) for i, s in ((1, 6), (2, 8), (3, 10))]
    snaps = _history((before, [_rec(10)], T1))
    now = before + [_entry(10, score=9), _entry(20, score=8)]

    res = evaluate_history(snaps, now)
    d = res["scores"]["deltas"]["control_new"]["raw"]
    sd = statistics.stdev([6, 8, 10, 9, 8])
    assert d["ci95"] == pytest.approx(1.96 * sd * (1 / 1 + 1 / 1) ** 0.5)
    assert d["conclusive"] is False
    assert "zatím neprůkazné" in "\n".join(format_report(res))


def test_rank_buckets_need_enough_titles():
    before = [_entry(1)]
    snaps = _history((before, [_rec(100 + i) for i in range(25)], T1))
    now = before + [_entry(100, score=9), _entry(101, score=8),
                    _entry(102, score=7), _entry(115, score=6)]

    res = evaluate_history(snaps, now)
    assert res["by_rank"] == [(1, 10, 3, pytest.approx(8.0))]
    text = "\n".join(format_report(res))
    assert "pořadí 1–10" in text and "11–20" not in text


def test_prediction_check_covers_titles_beyond_recommendations():
    """Nejhustší zpětná vazba: ~13 nových hodnocení měsíčně proti ~1 zásahu
    doporučení -- predikce se porovná u všech, pro které je uložená."""
    before = [_entry(1)]
    snaps = _history((before, [_rec(10, pred=9.0)], T1))
    now = before + [_entry(10, score=7), _entry(20, score=8), _entry(30, score=8)]
    preds = {snaps[0].fingerprint: {20: 6.0}}          # 30 predikci nemá

    p = evaluate_history(snaps, now, predictions=preds)["prediction"]
    assert p["n_scored"] == 3
    assert p["all"]["n"] == 2                          # 10 z top-N, 20 z logu
    assert p["all"]["bias"] == pytest.approx(0.0)      # (7−9 + 8−6) / 2
    assert p["all"]["rmse"] == pytest.approx(2.0)
    assert p["all"]["spearman"] is None                # pod MIN_SPEARMAN_N


def test_prediction_check_separates_continuations_from_new_franchises():
    before = [_entry(1)]
    snaps = _history((before, [], T1))
    now = before + [_entry(2, score=7), _entry(3, score=9)]
    preds = {snaps[0].fingerprint: {2: 8.0, 3: 8.0}}

    p = evaluate_history(snaps, now, roots={1: 1, 2: 1},
                         predictions=preds)["prediction"]
    assert p["cont"]["n"] == 1 and p["cont"]["bias"] == pytest.approx(-1.0)
    assert p["new"]["n"] == 1 and p["new"]["bias"] == pytest.approx(1.0)


def test_format_report_summarises_windows_cohorts_and_scores():
    base = [_entry(1)]
    snaps = _history((base, [_rec(10)], T1))
    res = evaluate_history(snaps, base + [_entry(10, score=9),
                                          _entry(11, "Watching", 7)])
    text = "\n".join(format_report(res))
    assert "2026-01-01 → dnes" in text and "nově shlédnuto 2" in text
    assert "#1 T10 (9)" in text
    assert "vyzkoušeno" in text and "doporučené" in text
    assert "predikce vs. skutečná známka" in text
