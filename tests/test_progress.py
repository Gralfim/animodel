"""Souhra \\r progress řádky (stdout) s log záznamy (stderr):
ProgressAwareLogHandler musí před vypsáním záznamu progress řádku smazat
a po něm ji překreslit, ať série warningů nerozbije viditelnost progressu.
"""
import logging

import pytest

import animodel.sources as src
from animodel.sources import (
    progress, progress_done, status, ProgressAwareLogHandler,
)


@pytest.fixture
def tty(monkeypatch):
    """capsys streamy nejsou TTY -> clear/redraw by byly no-op; testy je
    zapnou vynuceně (přesně tenhle hook je důvod, proč je _stdout_isatty
    samostatná funkce)."""
    monkeypatch.setattr(src, "_stdout_isatty", lambda: True)


@pytest.fixture(autouse=True)
def reset_progress_state():
    """Progress stav je modulový -- ať testy nezávisí na pořadí."""
    yield
    src._LAST_WIDTH = 0
    src._LAST_MSG = None
    src._DISPLAYED_WIDTH = 0


def make_handler(level=logging.WARNING):
    logger = logging.getLogger("test_progress_aware")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level)
    handler = ProgressAwareLogHandler()   # default stream = stderr
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def test_log_emit_clears_and_redraws_active_progress_line(tty, capsys):
    logger = make_handler()
    progress("Stahuji data: 40/411 (10%)…")
    logger.warning("Jikan: HTTP 504 …")

    out, err = capsys.readouterr()
    msg = "Stahuji data: 40/411 (10%)…"
    # 1) progress vykreslen, 2) smazán mezerami na plnou šířku, 3) překreslen
    assert out.startswith(f"\r{msg}")
    assert "\r" + " " * len(msg) + "\r" in out
    assert out.endswith(f"\r{msg}")
    # log záznam šel na stderr, celý a na vlastní řádce
    assert err == "WARNING Jikan: HTTP 504 …\n"


def test_log_emit_without_active_progress_touches_nothing(tty, capsys):
    logger = make_handler()
    logger.warning("hláška bez progressu")
    out, err = capsys.readouterr()
    assert out == ""                      # žádné mazání/překreslování
    assert err == "WARNING hláška bez progressu\n"


def test_progress_done_deactivates_line_for_handler(tty, capsys):
    logger = make_handler()
    progress("krok 1…")
    progress_done("hotovo.")
    logger.warning("po dokončení")
    out, err = capsys.readouterr()
    # po progress_done se už nic nemaže ani nepřekresluje
    assert out.endswith("hotovo.\n")
    assert err == "WARNING po dokončení\n"


def test_status_clears_prints_own_line_and_redraws(tty, capsys):
    progress("průběh 5/10…")
    status("  AniList cache: 3 hit, 2 ke stažení…")
    out, _ = capsys.readouterr()
    # status řádka je celá (s \n) a progress je za ní znovu vykreslený
    assert "  AniList cache: 3 hit, 2 ke stažení…\n" in out
    assert out.endswith("\rprůběh 5/10…")


def test_clear_and_redraw_are_noop_outside_tty(capsys):
    # BEZ tty fixtures -- _stdout_isatty vrací False (capsys není TTY):
    # do přesměrovaného výstupu nesmí přibýt mazací sekvence
    logger = make_handler()
    progress("data: 1/2…")
    logger.warning("chyba")
    out, err = capsys.readouterr()
    assert out == "\rdata: 1/2…"          # jen samotný progress, žádný clear
    assert err == "WARNING chyba\n"
