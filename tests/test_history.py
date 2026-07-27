"""history.py: záznam běhů klíčovaný OTISKEM STAVU SEZNAMU (ne datem) a
zpětná vazba z pozdějšího exportu.

Jádro návrhu: ladění parametrů znamená desítky běhů nad týmž exportem;
datové klíčování by z nich udělalo desítky skoro identických snapshotů a
evaluaci by to ředilo. Otisk se počítá z (mal_id, status, score), takže
nezměněný seznam vždy padne na tentýž klíč."""
import datetime as _dt

import pytest

from animodel.config import Config
from animodel.history import (
    Snapshot, build_snapshot, evaluate, export_fingerprint, format_report,
    load_snapshots, save_snapshot,
)
from animodel.mal import MalEntry
from animodel.recommend import Recommendation


def _entry(mal_id, status="Completed", score=8, episodes=12, watched=12,
           finish="2025-01-01"):
    return MalEntry(mal_id=mal_id, title=f"T{mal_id}", type="TV",
                    episodes=episodes, watched_episodes=watched, score=score,
                    status=status, start_date="", finish_date=finish,
                    rewatched=0)


def _rec(mal_id, rank_composite=1.0, ptw=False):
    return Recommendation(
        mal_id=mal_id, title=f"T{mal_id}", title_en="", community=7.5,
        pred=8.0, pred_lo=7.5, pred_hi=8.5, taste_fit=0.5, cf_signal=1.0,
        composite=rank_composite, ptw=ptw, cluster_name="Nálada", why=[],
        cf_seeds=[])


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

def _snap(entries, recs, when="2026-01-01T10:00:00"):
    cfg = Config()

    class M:
        scale, scale_triples, cv_rmse, baseline_rmse, beta = 0.3, 0.15, 0.9, 0.95, 0.5
        clusters = []
    return build_snapshot(entries, recs, M(), cfg, top=100,
                          now=_dt.datetime.fromisoformat(when))


def test_repeated_runs_on_same_export_keep_one_snapshot(tmp_path):
    """JÁDRO: ladění parametrů nad týmž exportem nesmí hromadit snapshoty."""
    entries = [_entry(i) for i in range(1, 6)]
    for i in range(5):                     # pět „ladicích" běhů
        save_snapshot(_snap(entries, [_rec(100 + i)]), tmp_path)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1, [f.name for f in files]
    # poslední běh vyhrává
    got = load_snapshots(tmp_path)
    assert [r["mal_id"] for r in got[0].recommendations] == [104]


def test_changed_export_creates_new_snapshot(tmp_path):
    entries = [_entry(i) for i in range(1, 6)]
    save_snapshot(_snap(entries, [_rec(100)]), tmp_path)
    entries2 = entries + [_entry(6)]
    save_snapshot(_snap(entries2, [_rec(200)]), tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2


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


# ── vyhodnocení proti pozdějšímu exportu ─────────────────────────────────

def test_evaluate_returns_none_when_nothing_changed():
    entries = [_entry(1)]
    snap = _snap(entries, [_rec(2)])
    assert evaluate(snap, entries) is None


def test_evaluate_counts_completed_recommendations():
    before = [_entry(1), _entry(2, "Plan to Watch", 0)]
    snap = _snap(before, [_rec(10), _rec(11), _rec(12)])
    # 10 dokoukáno s 9, 11 rozkoukáno, 12 nic
    after = before + [_entry(10, score=9), _entry(11, "Watching", 0)]
    r = evaluate(snap, after)
    assert r["n_completed"] == 1 and r["n_started"] == 1
    assert r["hit_rate"] == pytest.approx(1 / 3)
    assert r["mean_score_recommended"] == pytest.approx(9.0)


def test_evaluate_compares_against_control_group():
    """Klíčová metrika: doporučené vs. ostatní nově dokoukané. Bez kontroly
    by hit rate říkal jen „něco jsem viděl", ne „bylo to lepší"."""
    before = [_entry(1)]
    snap = _snap(before, [_rec(10), _rec(11)])
    after = before + [
        _entry(10, score=9), _entry(11, score=9),     # doporučené
        _entry(20, score=6), _entry(21, score=6),     # nedoporučené novinky
    ]
    r = evaluate(snap, after)
    assert r["mean_score_recommended"] == pytest.approx(9.0)
    assert r["mean_score_control"] == pytest.approx(6.0)
    assert r["n_control"] == 2
    assert r["delta"] == pytest.approx(3.0)


def test_control_excludes_titles_already_watched_before():
    """Kontrola musí být „co přibylo OD snapshotu", ne celý seznam --
    jinak by se do ní počítalo i to, co bylo dokoukané dávno předtím."""
    before = [_entry(1, score=5)]                  # dávno dokoukané, známka 5
    snap = _snap(before, [_rec(10)])
    after = before + [_entry(10, score=8), _entry(20, score=7)]
    r = evaluate(snap, after)
    assert r["n_control"] == 1                     # jen titul 20
    assert r["mean_score_control"] == pytest.approx(7.0)


def test_newly_planned_is_tracked_separately_from_prior_ptw():
    before = [_entry(1), _entry(5, "Plan to Watch", 0)]
    snap = _snap(before, [_rec(5, ptw=True), _rec(6)])
    after = [_entry(1), _entry(5, "Plan to Watch", 0), _entry(6, "Plan to Watch", 0)]
    r = evaluate(snap, after)
    assert r["n_newly_planned"] == 1               # jen 6; 5 na PTW bylo už dřív


def test_evaluate_reports_mean_score_by_rank_bucket():
    before = [_entry(1)]
    recs = [_rec(100 + i) for i in range(25)]
    snap = _snap(before, recs)
    after = before + [_entry(100, score=10), _entry(115, score=6)]
    r = evaluate(snap, after)
    buckets = {lo: (hi, n, m) for lo, hi, n, m in r["by_rank"]}
    assert buckets[1] == (10, 1, pytest.approx(10.0))     # pořadí 1–10
    assert buckets[11] == (20, 1, pytest.approx(6.0))     # pořadí 11–20
    # hranice se hlásí NOMINÁLNÍ (kbelík 11–20), ne podle nejvyššího
    # pozorovaného pořadí -- "11–15" by mátlo
    assert "11–20" in "\n".join(format_report([r]))


def test_rank_buckets_report_open_ended_tail():
    before = [_entry(1)]
    snap = _snap(before, [_rec(100 + i) for i in range(40)])
    after = before + [_entry(130, score=7)]        # pořadí 31
    r = evaluate(snap, after)
    assert r["by_rank"] == [(21, None, 1, pytest.approx(7.0))]
    assert "21+" in "\n".join(format_report([r]))


def test_format_report_is_empty_without_results():
    assert format_report([]) == []


def test_format_report_mentions_hit_rate_and_control():
    before = [_entry(1)]
    snap = _snap(before, [_rec(10)])
    after = before + [_entry(10, score=9), _entry(20, score=6)]
    text = "\n".join(format_report([evaluate(snap, after)]))
    assert "dokoukáno" in text and "doporučené" in text and "9.00" in text
