"""FileCache + cached_fetch: cache primitiv sdílený všemi klienty v sources/.

Klíčová věc, kterou tahle sada hlídá jako regresi: JikanClient dřív ukládal
trvalé selhání jako `None` (JSON `null`), což při čtení kolidovalo s "cache
miss" (taky `None`) -- trvale nenalezené tituly se tak ve skutečnosti
necachovaly vůbec. Nový envelope (`{"found": bool, "data": ...}`) tuhle
kolizi strukturálně vylučuje.
"""
from animodel.sources import Result
from animodel.sources.cache import FileCache, cached_fetch


def test_filecache_miss_returns_none(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.get("nic") is None
    assert cache.has("nic") is False


def test_filecache_set_then_get_roundtrip(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("klic", {"found": True, "data": {"a": 1}})
    assert cache.get("klic") == {"found": True, "data": {"a": 1}}
    assert cache.has("klic") is True


def test_filecache_sanitizes_weird_keys_to_valid_filenames(tmp_path):
    cache = FileCache(tmp_path)
    key = "anime/123/full?x=1&y=2"
    cache.set(key, {"found": True, "data": "ok"})
    assert cache.get(key) == {"found": True, "data": "ok"}
    # soubor skutečně vznikl jako jeden JSON soubor, ne rozsypaný přes '/'
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_cached_fetch_hit_does_not_call_fetch(tmp_path):
    cache = FileCache(tmp_path)
    cache.set("x", {"found": True, "data": 42})
    calls = []

    def fetch():
        calls.append(1)
        return Result.success(999)

    assert cached_fetch(cache, "x", fetch) == 42
    assert calls == []


def test_cached_fetch_success_writes_and_returns_data(tmp_path):
    cache = FileCache(tmp_path)

    def fetch():
        return Result.success({"hello": "world"})

    assert cached_fetch(cache, "x", fetch) == {"hello": "world"}
    assert cache.get("x") == {"found": True, "data": {"hello": "world"}}


def test_cached_fetch_success_with_empty_data_is_still_a_hit(tmp_path):
    """Prázdný, ale ÚSPĚŠNÝ výsledek (např. 'žádná doporučení') se musí
    cachovat jako nalezený -- ne zaměnit za 'nezkoušeno'."""
    cache = FileCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return Result.success([])

    assert cached_fetch(cache, "x", fetch) == []
    assert cache.get("x") == {"found": True, "data": []}
    # druhé volání je z cache, fetch se nevolá znovu
    assert cached_fetch(cache, "x", fetch) == []
    assert calls == [1]


def test_cached_fetch_permanent_failure_writes_sentinel_and_short_circuits(tmp_path):
    cache = FileCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return Result.failure(permanent=True)

    assert cached_fetch(cache, "x", fetch) is None
    assert cache.get("x") == {"found": False, "data": None}

    # REGRESNÍ TEST na dřívější jikan.py bug: druhé volání NESMÍ zavolat
    # fetch znovu -- cache soubor existuje a jeho obsah (found=False) se
    # nesmí zaměnit s "cache miss".
    assert cached_fetch(cache, "x", fetch) is None
    assert calls == [1]


def test_cached_fetch_transient_failure_writes_nothing(tmp_path):
    cache = FileCache(tmp_path)

    def fetch():
        return Result.failure(permanent=False)

    assert cached_fetch(cache, "x", fetch) is None
    assert cache.has("x") is False


def test_cached_fetch_transient_failure_retries_on_next_call(tmp_path):
    cache = FileCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return Result.failure(permanent=False)

    cached_fetch(cache, "x", fetch)
    cached_fetch(cache, "x", fetch)
    assert calls == [1, 1]
    assert cache.has("x") is False
