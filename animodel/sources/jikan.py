"""
jikan.py — klient pro Jikan-KOMPATIBILNÍ MAL API (Jikan v4 / Tenrai).

Původní zdroj byl Jikan (https://docs.api.jikan.moe/), ale od 2026-07 má
trvalé 504 výpadky. Default base URL je proto Tenrai
(https://api.tenrai.org/v1) -- 1:1 mirror Jikan v4 schématu za Cloudflare,
ověřeno na všech endpointech, které tenhle klient používá. Base URL je
konfigurovatelná (enrich.anime_api_base_url), takže přepnutí zpět na Jikan
je jen změna configu -- schéma i cache klíče (dle endpointu, ne hostu) jsou
společné. Název souboru/třídy zůstává "Jikan" kvůli zpětné kompatibilitě
importů; jde o klienta pro *jikanovské schéma*, ne nutně jikan.moe.

Rate limit: klient čeká REQUEST_DELAY mezi requesty (bezpečný interval).
"""

import time
import logging
from pathlib import Path
from typing import Callable

import requests

from . import progress, progress_done, is_permanent_status
from .cache import FileCache, cached_fetch
from .http import (
    FixedRateLimiter, Attempt, attempt_success, attempt_permanent,
    attempt_rate_limited, attempt_retryable, request_with_retry,
)

log = logging.getLogger(__name__)

# Default zdroj MAL dat. Historicky "https://api.jikan.moe/v4"; přepíná se
# přes EnrichCfg.anime_api_base_url (viz config.example.yaml pro obě URL).
BASE_URL = "https://api.tenrai.org/v1"
REQUEST_DELAY = 0.4          # sekundy mezi requesty (bezpečný interval)
RETRY_DELAYS = [2, 5, 10, 30]  # exponenciální backoff; 429 = len()+1 pokusů,
                                # ostatní dočasné chyby = len() pokusů (poslední
                                # delay slouží jen jako 429 floor -- viz http.py)


class JikanClient:
    def __init__(self, cache_dir: str = "cache", sleep: Callable[[float], None] = time.sleep,
                 base_url: str = BASE_URL):
        # `cache_dir` je KOŘEN cache; podsložku si klient doplní sám -- stejně
        # jako AniList (anilist/, cf_al/) a Shikimori (shikimori/). Dřív psal
        # MAL data rovnou do kořene, takže tam leželo ~10 tis. souborů vedle
        # podsložek ostatních zdrojů a selektivní invalidace znamenala glob
        # místo `rm -r` (HODNOCENI_PROJEKTU.md §2).
        self._cache = FileCache(Path(cache_dir) / "mal")
        self._rate_limiter = FixedRateLimiter(REQUEST_DELAY)
        self._sleep = sleep
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "anime-taste-model/1.0"

    # ── Interní helpers ────────────────────────────────────────────

    def _classify(self, resp: requests.Response, url: str) -> Attempt:
        if resp.status_code == 404:
            # Anime neexistuje nebo je NSFW — potvrzeně natrvalo.
            return attempt_permanent(f"HTTP 404 {url}")
        if resp.status_code == 429:
            return attempt_rate_limited()
        if is_permanent_status(resp.status_code):
            # 400/422 -- request samotný je špatně (neplatné ID apod.),
            # retry se stejnou URL by dopadl stejně.
            return attempt_permanent(f"HTTP {resp.status_code} {url}")
        if not resp.ok:
            # Typicky 5xx -- zkusit znovu má smysl.
            return attempt_retryable(f"HTTP {resp.status_code} {url}")
        return attempt_success(resp.json())

    def _request(self, endpoint: str):
        """Čistě síťová vrstva -- BEZ cache. Vrací Result(ok, data, permanent)."""
        url = f"{self.base_url}/{endpoint}"
        return request_with_retry(
            perform=lambda: self.session.get(url, timeout=15),
            classify=lambda resp: self._classify(resp, url),
            rate_limiter=self._rate_limiter,
            retry_delays=RETRY_DELAYS,
            label="MAL-API",
            sleep=self._sleep,
        )

    def _get(self, endpoint: str):
        """Cache-aware wrapper kolem `_request()` přes sdílený `cached_fetch`
        primitiv -- jedno místo, které rozhoduje o zápisu do cache pro
        VŠECHNY klienty v sources/, ne tři nezávislé kopie stejné logiky."""
        return cached_fetch(self._cache, endpoint, lambda: self._request(endpoint))

    # ── Veřejné metody ─────────────────────────────────────────────

    def get_anime(self, mal_id: int) -> dict | None:
        """
        Vrátí detailní informace o anime dle MAL ID.

        Vrací klíčové pole 'data' s atributy:
            title, type, source, episodes, score, year,
            genres, themes, demographics, studios
        """
        result = self._get(f"anime/{mal_id}/full")
        if result and "data" in result:
            return result["data"]
        return None

    def get_anime_staff(self, mal_id: int) -> list[dict]:
        """
        Vrátí staff pro anime dle MAL ID (samostatný endpoint /anime/{id}/staff).

        Vrací list objektů:
            [{"person": {"mal_id": ..., "name": ...}, "positions": [...]}, ...]

        Cachováno odděleně od /full dat pod klíčem "anime/{id}/staff".
        """
        result = self._get(f"anime/{mal_id}/staff")
        if result and "data" in result:
            return result["data"]
        return []

    def get_staff_batch(
        self,
        mal_ids: list[int],
        show_progress: bool = True,
    ) -> dict[int, list[dict]]:
        """
        Stáhne staff data pro seznam MAL ID.
        Vrací dict {mal_id: [staff_entries]}.
        """
        results = {}
        total   = len(mal_ids)

        for i, mal_id in enumerate(mal_ids):
            if show_progress and i % 10 == 0:
                progress(f"  Staff data: {i}/{total}…")
            staff = self.get_anime_staff(int(mal_id))
            results[mal_id] = staff  # může být prázdný list

        if show_progress:
            non_empty = sum(1 for v in results.values() if v)
            progress_done(f"  Staff stažen: {non_empty}/{total} titulů s daty.")

        return results

    def get_season(self, year: int, season: str) -> list[dict]:
        """
        Všechny tituly dané vysílané sezóny (Jikan/Tenrai /seasons/{y}/{s}).
        `season` ∈ {winter, spring, summer, fall}. Stránkuje přes
        has_next_page. Vrací list anime dicts (Jikan schéma: mal_id, title,
        type, source, status, episodes, aired, broadcast, genres, …).

        Cachuje se (přes _get) -- sezónní členství je stabilní; airing
        detaily (datum finále) se berou zvlášť z AniListu, který se
        NEcachuje (mění se týdně).
        """
        results = []
        page = 1
        while True:
            data = self._get(f"seasons/{year}/{season.lower()}?page={page}")
            if not data or "data" not in data:
                break
            results.extend(data["data"])
            if not data.get("pagination", {}).get("has_next_page"):
                break
            page += 1
            self._sleep(REQUEST_DELAY)
        return results

    def get_anime_batch(
        self,
        mal_ids: list[int],
        show_progress: bool = True
    ) -> dict[int, dict]:
        """
        Stáhne informace pro seznam MAL ID.
        Vrací dict {mal_id: anime_data}.
        Přeskočí ID, která nelze stáhnout.
        """
        results = {}
        total = len(mal_ids)

        for i, mal_id in enumerate(mal_ids):
            if show_progress and i % 10 == 0:
                progress(f"  Stahuji data: {i}/{total} ({i/total*100:.0f}%)…")

            data = self.get_anime(int(mal_id))
            if data:
                results[mal_id] = data

        if show_progress:
            progress_done(f"  Staženo: {len(results)}/{total} titulů.")

        return results

    def get_genres(self, filter: str = "") -> list[dict]:
        """
        Kompletní seznam MAL žánrů/témat přes /genres/anime.
        `filter`: "genres" | "explicit_genres" | "themes" | "demographics"
        (prázdné = všechno dohromady). Vrací [{mal_id, name, url, count}, ...].
        Universum pro intensity lexikon (viz animodel/intensity.py).
        """
        q = f"genres/anime?filter={filter}" if filter else "genres/anime"
        data = self._get(q)
        if data and "data" in data:
            return data["data"]
        return []

    # ── Doporučení (item-based CF graf) ────────────────────────────────────
    def get_recommendations(self, mal_id: int) -> list[dict]:
        """
        Vrátí MAL doporučení k danému titulu: [{mal_id, title, votes}, ...].
        Endpoint /anime/{id}/recommendations.
        """
        result = self._get(f"anime/{mal_id}/recommendations")
        out = []
        if result and "data" in result:
            for rec in result["data"]:
                entry = rec.get("entry") or {}
                if entry.get("mal_id"):
                    out.append({
                        "mal_id": entry["mal_id"],
                        "title": entry.get("title", ""),
                        "votes": rec.get("votes", 0),
                    })
        return out

