"""Sdílené fixtures pro tests/ -- žádné skutečné síťové volání ani spánek."""
import pytest


@pytest.fixture
def no_sleep():
    """Injektovatelná no-op náhrada za time.sleep -- předává se přímo
    klientům (JikanClient(..., sleep=no_sleep)), ne přes monkeypatch
    globálního time.sleep, aby testy zůstaly nezávislé na sobě."""
    calls = []

    def _sleep(seconds):
        calls.append(seconds)

    _sleep.calls = calls
    return _sleep
