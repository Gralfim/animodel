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


# ── rozvržení cache: každý klient má vlastní podsložku (§9.8) ────────────

def test_each_client_uses_its_own_subdirectory(tmp_path):
    """Všichni tři klienti berou KOŘEN cache a podsložku si doplní sami.
    Dřív psal MAL klient rovnou do kořene (~10 tis. souborů vedle podsložek
    ostatních zdrojů) a Shikimori jako jediný očekával už hotovou cestu."""
    from animodel.sources.anilist import AniListClient
    from animodel.sources.jikan import JikanClient
    from animodel.sources.shikimori import ShikimoriClient

    root = str(tmp_path)
    assert JikanClient(root)._cache.root == tmp_path / "mal"
    assert ShikimoriClient(root)._cache.root == tmp_path / "shikimori"
    al = AniListClient(root)
    assert al._cache.root == tmp_path / "anilist"
    assert al._cf_cache.root == tmp_path / "cf_al"


def test_cache_root_holds_no_loose_files(tmp_path):
    """Kořen cache má obsahovat jen podsložky -- díky tomu jde invalidovat
    jeden zdroj přes `rm -r`, ne globem."""
    from animodel.sources.jikan import JikanClient

    client = JikanClient(str(tmp_path))
    client._cache.set("anime/1/full", {"found": True, "data": {"x": 1}})
    assert [p.name for p in tmp_path.iterdir()] == ["mal"]
    assert client._cache.get("anime/1/full")["data"] == {"x": 1}


def test_enricher_gives_all_clients_the_same_root(tmp_path):
    from animodel.config import Config
    from animodel.enrich import Enricher

    cfg = Config()
    cfg.cache_dir = str(tmp_path)
    cfg.enrich.use_shikimori = True
    enr = Enricher(cfg)
    assert enr.jikan._cache.root == tmp_path / "mal"
    assert enr.anilist._cache.root == tmp_path / "anilist"
    assert enr.shikimori._cache.root == tmp_path / "shikimori"
