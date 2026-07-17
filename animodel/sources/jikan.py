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
        self._cache = FileCache(Path(cache_dir))
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

    def get_top_anime(self, limit: int = 100, min_score: float = 7.0) -> list[dict]:
        """
        Stáhne top anime z MAL seřazené podle skóre.
        Filtruje na min_score. Vrací list anime data objektů.
        """
        results = []
        page = 1
        per_page = 25  # Jikan maximum

        while len(results) < limit:
            data = self._get(f"top/anime?page={page}&type=tv")
            if not data or "data" not in data:
                break

            for item in data["data"]:
                if item.get("score", 0) < min_score:
                    # Top anime jsou seřazené — pod min_score už nic nepřijde
                    return results
                results.append(item)
                if len(results) >= limit:
                    break

            if not data.get("pagination", {}).get("has_next_page"):
                break
            page += 1
            self._sleep(REQUEST_DELAY)

        return results

    def list_all_staff(
        self,
        mal_ids: list[int],
        show_progress: bool = False,
    ) -> dict[str, list[tuple[int, str, str, int]]]:
        """
        Projde Jikan data pro zadaná MAL ID a vrátí frekvenční přehled
        režisérů a scenáristů ve formátu vhodném pro config.yaml.

        Vrací:
            {
              "directors": [(mal_id, name, position, count), ...],
              "writers":   [(mal_id, name, position, count), ...],
            }
        Seřazeno sestupně dle počtu titulů.
        """
        from collections import defaultdict
        from .attributes import DIRECTOR_POSITIONS, WRITER_POSITIONS  # sdílené s build_attributes()

        directors: dict[int, list] = defaultdict(lambda: ["", "", 0])
        writers:   dict[int, list] = defaultdict(lambda: ["", "", 0])

        staff_data = self.get_staff_batch(mal_ids, show_progress=show_progress)
        for staff_list in staff_data.values():
            for entry in staff_list:
                person    = entry.get("person") or {}
                person_id = person.get("mal_id")
                name      = person.get("name", "")
                if not person_id:
                    continue
                positions = {p.lower() for p in (entry.get("positions") or [])}
                if positions & DIRECTOR_POSITIONS:
                    pos_str = next(iter(positions & DIRECTOR_POSITIONS))
                    directors[person_id][0] = name
                    directors[person_id][1] = pos_str
                    directors[person_id][2] += 1
                if positions & WRITER_POSITIONS:
                    pos_str = next(iter(positions & WRITER_POSITIONS))
                    writers[person_id][0] = name
                    writers[person_id][1] = pos_str
                    writers[person_id][2] += 1

        def to_list(d):
            return sorted(
                [(pid, info[0], info[1], info[2]) for pid, info in d.items()],
                key=lambda x: -x[3]
            )

        return {"directors": to_list(directors), "writers": to_list(writers)}

    def list_mal_features(
        self,
        mal_ids: list[int],
    ) -> dict[str, dict]:
        """
        Projde Jikan data pro zadaná MAL ID a vrátí frekvenční přehledy
        pro genres, themes, demographics, sources a types.

        Vrací:
            {
              "genres":       {(mal_id, name): count},
              "themes":       {(mal_id, name): count},
              "demographics": {name: count},
              "sources":      {source: count},
              "types":        {type: count},
            }
        """
        from collections import Counter

        genres       = Counter()  # (mal_id, name) → count
        themes       = Counter()
        demographics = Counter()  # name → count
        sources      = Counter()
        types        = Counter()

        data = self.get_anime_batch(mal_ids, show_progress=False)
        for anime in data.values():
            for g in anime.get("genres", []):
                genres[(g["mal_id"], g["name"])] += 1
            for t in anime.get("themes", []):
                themes[(t["mal_id"], t["name"])] += 1
            for d in anime.get("demographics", []):
                demographics[d["name"]] += 1
            src = (anime.get("source") or "").strip()
            if src:
                sources[src] += 1
            typ = (anime.get("type") or "").strip()
            if typ:
                types[typ] += 1

        return {
            "genres":       dict(genres),
            "themes":       dict(themes),
            "demographics": dict(demographics),
            "sources":      dict(sources),
            "types":        dict(types),
        }

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

    def search_anime(self, genres: list[int] = None, min_score: float = 7.0,
                     page: int = 1, order_by: str = "score") -> list[dict]:
        """Vyhledá anime dle žánrů (MAL genre IDs), seřazené dle skóre."""
        q = f"anime?order_by={order_by}&sort=desc&min_score={min_score}&page={page}&sfw=false"
        if genres:
            q += "&genres=" + ",".join(str(g) for g in genres)
        data = self._get(q)
        return data.get("data", []) if data else []
