"""Časová validace (backtest.py): disjunktní okna podle `my_finish_date`,
klastrovaný bootstrap a oddělené metriky pro nové franšízy a pokračování.

Proč vůbec: pořadí konfigurací podle OOF metrik je proti pořadí mimo čas
antikorelované (HODNOCENI_PROJEKTU.md §9c), takže cross-validace na otázku
„která varianta líp seřadí to, co uvidím příště" odpovídat neumí."""
import pytest

from animodel.backtest import (
    bootstrap_spearman, default_cuts, format_report, metrics, run, split,
)
from animodel.mal import MalEntry
from animodel.taste import Title


def _e(mal_id, finish, score=8):
    return MalEntry(mal_id=mal_id, title=f"T{mal_id}", type="TV", episodes=12,
                    watched_episodes=12, score=score, status="Completed",
                    start_date="", finish_date=finish, rewatched=0)


# ── okna ─────────────────────────────────────────────────────────────────

def test_split_makes_disjoint_windows():
    """Vnořené řezy (každý test obsahuje ten následující) dělaly z jednoho
    měření čtyři „nezávislá potvrzení" -- okna musí být disjunktní."""
    entries = [_e(1, "2025-01-01"), _e(2, "2025-08-01"),
               _e(3, "2026-02-01"), _e(4, "2026-06-01")]
    w = split(entries, ["2025-06-01", "2026-01-01"])
    assert [x["start"] for x in w] == ["2025-06-01", "2026-01-01"]
    assert [[e.mal_id for e in x["test"]] for x in w] == [[2], [3, 4]]
    assert [[e.mal_id for e in x["train"]] for x in w] == [[1], [1, 2]]


def test_undated_titles_train_unless_they_premiered_after_the_cut():
    """Titul bez data dokončení se bere jako starší -- to ale neplatí, když
    v době řezu ještě neměl premiéru."""
    entries = [_e(1, "0000-00-00"), _e(2, "0000-00-00"), _e(3, "2026-06-01")]
    w = split(entries, ["2026-01-01"], premieres={2: "2026-04-01"})
    assert [e.mal_id for e in w[0]["train"]] == [1]
    assert [e.mal_id for e in w[0]["test"]] == [3]


def test_default_cuts_need_enough_dated_titles():
    assert default_cuts([_e(i, "2025-01-01") for i in range(10)]) == []
    cuts = default_cuts([_e(i, f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}")
                         for i in range(80)])
    assert cuts == sorted(cuts) and len(cuts) >= 2


# ── metriky ──────────────────────────────────────────────────────────────

def test_metrics_separate_ordering_from_level():
    """Posun hladiny a kvalita řazení jsou dvě různé věci; jejich míchání
    vedlo k závěru „model je horší než baseline"."""
    rows = [(1.0, 7.0, 7.0, 1), (2.0, 7.0, 8.0, 2), (3.0, 7.0, 9.0, 3)]
    m = metrics(rows, interval=1.0)
    assert m["spearman"] == pytest.approx(1.0)      # pořadí je dokonalé
    assert m["bias"] == pytest.approx(-1.0)         # ...ale predikce je o bod výš
    assert m["rmse_model"] == pytest.approx(1.0)
    assert m["rmse_model_debiased"] == pytest.approx(0.0)
    assert m["coverage"] == pytest.approx(1.0)


def test_clustered_bootstrap_resamples_franchises():
    """Deset dílů jedné série je jeden důkaz, ne deset. V krajním případě,
    kdy jsou všechny řádky z jedné franšízy, je každý resample tentýž
    seznam -- iid bootstrap by tu předstíral nejistotu z ničeho."""
    rows = [(float(i), 7.0, 7.0 + i % 3, 42) for i in range(12)]
    lo, hi = bootstrap_spearman(rows, n_boot=50, seed=1)
    assert lo == hi


# ── běh ──────────────────────────────────────────────────────────────────

class _FakeModel:
    """Afinita = předpočítaná hodnota v atributech titulu."""
    resid_std, scale, cv_rmse = 1.0, 0.3, 0.9

    def affinity(self, attrs):
        return attrs.get("aff", 0.0)

    def _baseline_pred(self, community):
        return 7.5


def _fixture():
    train = [_e(i, f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}", score=7 + i % 4)
             for i in range(1, 41)]
    test = [_e(100 + j, f"2026-06-{1 + j:02d}", score=8 + j % 2) for j in range(6)]
    return train + test


def _build_titles(subset):
    out = []
    for e in subset:
        # titul 100 je pokračování franšízy, jejíž první díl (1) je v tréninku
        root = 1 if e.mal_id in (1, 100) else None
        out.append(Title(mal_id=e.mal_id, title=e.title,
                         user_score=float(e.score), community=7.0,
                         attrs={"aff": (e.mal_id % 5) * 0.1}, series_root=root))
    return out


def test_run_reports_new_franchises_and_continuations_separately():
    res = run(_fixture(), build_titles=_build_titles,
              fit=lambda titles: _FakeModel(), cuts=["2026-01-01"], n_boot=20)
    assert [w["start"] for w in res["windows"]] == ["2026-01-01"]
    w = res["windows"][0]
    assert w["n_train"] == 40 and w["n_test"] == 6
    assert w["groups"]["cont"]["n"] == 1        # jen titul 100
    assert w["groups"]["new"]["n"] == 5
    assert res["pooled"]["all"]["n"] == 6


def test_run_skips_windows_without_enough_training_data():
    res = run(_fixture(), build_titles=_build_titles,
              fit=lambda titles: _FakeModel(), cuts=["2025-01-05"], n_boot=10)
    assert res["windows"] == []                 # trénink by měl jen 4 tituly


def test_format_report_mentions_windows_and_pooled_rows():
    res = run(_fixture(), build_titles=_build_titles,
              fit=lambda titles: _FakeModel(), cuts=["2026-01-01"], n_boot=20)
    text = "\n".join(format_report(res))
    assert "Časová validace" in text and "2026-01-01" in text
    assert "nové franšízy" in text and "pokračování" in text


def test_format_report_says_when_there_is_too_little_data():
    res = run([_e(i, "0000-00-00") for i in range(5)], build_titles=_build_titles,
              fit=lambda titles: _FakeModel())
    assert "málo datovaných titulů" in "\n".join(format_report(res))
