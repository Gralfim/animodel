"""
cache.py — sdílený, čistě funkční cache primitiv pro sources/.

Jeden klíč = jeden soubor = jeden request. Nahrazuje tři nezávislé ruční
implementace (jikan.py, anilist.py, shikimori.py), které se lišily v
detailech a v jednom případě (jikan.py) i chybovaly (viz níž).

Konkrétní bug, který tenhle modul strukturálně odstraňuje: staré
`JikanClient._load_cache`/`_save_cache` používaly Python `None` jako
hodnotu pro "trvale nenalezeno" (zapsáno jako JSON `null`). Při čtení
`json.loads("null")` vrátí `None` — stejnou hodnotu jako "soubor vůbec
neexistuje". Obě situace tak kolidovaly a trvalé selhání se ve
skutečnosti necachovalo. Tady je "cache miss" vždy Python `None` a
"cachovaná hodnota" je vždy `dict` (envelope), takže kolize není možná.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from . import Result


def _sanitize_key(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return safe.strip("_") or "_"


class FileCache:
    """Čisté get/set přes filesystem, jeden JSON soubor na klíč.

    Žádný sdílený mutable stav mimo filesystem -- zápisy na RŮZNÉ klíče
    se nikdy nekolidují, takže je bezpečné je volat i souběžně (kolize
    hrozí jen při dvou současných zápisech na TENTÝŽ klíč, což se v
    dnešním sekvenčním použití nestává; do budoucna by to řešil zámek
    per-klíč, ne přepis celého cache mechanismu).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{_sanitize_key(key)}.json"

    def get(self, key: str) -> dict | None:
        """`None` = cache miss (soubor neexistuje). Jinak vrátí uložený
        envelope dict beze změny (interpretace je na volajícím/`cached_fetch`)."""
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def set(self, key: str, envelope: dict) -> None:
        p = self._path(key)
        p.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")

    def has(self, key: str) -> bool:
        return self._path(key).exists()


def cached_fetch(cache: FileCache, key: str, fetch: Callable[[], Result]):
    """
    Cache-aside obálka kolem libovolného `fetch()`, které vrací `Result`.
    Jednotné chování pro všechny klienty v sources/:

        cache hit                         -> vrať uloženou hodnotu, `fetch` se NEVOLÁ
        fetch() -> Result(ok=True)        -> ulož {"found": true, "data": ...}, vrať data
        fetch() -> Result(permanent=True) -> ulož {"found": false, "data": None}, vrať None
        fetch() -> Result(permanent=False)-> NIC neukládej, vrať None (zkusí se příště)

    `fetch` je zavolané jen při cache miss (líné vyhodnocení) -- volající
    nemusí sám rozhodovat, kdy zapsat/nezapsat cache, takže se tahle
    tříbodová logika nemůže na jednotlivých místech volání rozjet.
    """
    envelope = cache.get(key)
    if envelope is not None:
        return envelope["data"] if envelope["found"] else None

    result = fetch()
    if result.ok:
        cache.set(key, {"found": True, "data": result.data})
        return result.data
    if result.permanent:
        cache.set(key, {"found": False, "data": None})
        return None
    return None
