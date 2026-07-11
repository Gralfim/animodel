"""
jikan.py — Jikan API v4 klient s cachováním a rate limitingem

Jikan je neoficiální REST API pro MyAnimeList.
Dokumentace: https://docs.api.jikan.moe/
Rate limit: ~3 requesty/sekundu (klient automaticky čeká)
"""

import json
import time
import logging
from pathlib import Path
import requests

from . import progress, progress_done, Result, is_permanent_status

log = logging.getLogger(__name__)

BASE_URL = "https://api.jikan.moe/v4"
REQUEST_DELAY = 0.4          # sekundy mezi requesty (bezpečný interval)
MAX_RETRIES   = 4
RETRY_DELAYS  = [2, 5, 10, 30]  # exponenciální backoff při 429


class JikanClient:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(exist_ok=True)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "anime-taste-model/1.0"

    # ── Interní helpers ────────────────────────────────────────────

    def _cache_file(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("?", "_")
        return self.cache_path / f"{safe}.json"

    def _load_cache(self, key: str):
        f = self._cache_file(key)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return None

    def _save_cache(self, key: str, data) -> None:
        f = self._cache_file(key)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _request(self, endpoint: str) -> Result:
        """
        Čistě síťová vrstva -- BEZ cache. Vrací Result(ok, data, permanent).
        Cachováním se zabývá výhradně _get() níž, tohle je jediné, co dělá
        request samotný, takže se dá i samostatně testovat/mockovat.
        """
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                resp = self.session.get(url, timeout=15)
                self._last_request = time.time()

                if resp.status_code == 429:
                    wait = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 60
                    # INFO, ne WARNING: běžný, očekávaný jev u rate-limitovaného
                    # API, co se vyřeší samo -- teprve VYČERPÁNÍ pokusů (níž)
                    # je něco, co má smysl vypíchnout jako warning/error.
                    log.info(f"Rate limit 429, čekám {wait}s…")
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    # Anime neexistuje nebo je NSFW — potvrzeně natrvalo.
                    return Result.failure(permanent=True)

                if is_permanent_status(resp.status_code):
                    # 400/422 -- request samotný je špatně (neplatné ID
                    # apod.), retry se stejnou URL by dopadl stejně. Vzdej
                    # se hned, neplýtvej celým 2+5+10+30s retry schedulem.
                    log.warning(f"Jikan natrvalo selhal (nebude se opakovat): "
                                f"HTTP {resp.status_code} {url}")
                    return Result.failure(permanent=True)

                resp.raise_for_status()
                return Result.success(resp.json())

            except requests.RequestException as e:
                # Sem se dostaneme jen pro kódy MIMO is_permanent_status
                # (ty jsou zachyceny výš, ještě před raise_for_status) --
                # typicky 5xx, nebo čistě síťová chyba bez odpovědi vůbec.
                if attempt < len(RETRY_DELAYS) - 1:
                    log.warning(f"Chyba {e}, retry za {RETRY_DELAYS[attempt]}s…")
                    time.sleep(RETRY_DELAYS[attempt])
                else:
                    log.error(f"Selhalo po {MAX_RETRIES} pokusech: {url}")
                    return Result.failure(permanent=False)

        # Sem se dřív dalo dojít tiše (opakované 429 vyčerpaly všechny pokusy
        # bez jediného log.error) -- na rozdíl od větve se síťovou výjimkou
        # výš, která error loguje vždy. Titul, co se opakovaně rate-limitoval,
        # tak potichu zmizel z výsledků beze stopy proč.
        log.error(f"Selhalo po {MAX_RETRIES} pokusech (rate limit): {url}")
        return Result.failure(permanent=False)

    def _get(self, endpoint: str) -> dict:
        """
        Cache-aware wrapper kolem _request() -- JEDINÉ místo v JikanClient,
        které rozhoduje o zápisu do cache (dřív se to řešilo uvnitř
        _get() samotného na 3 různých místech; teď je to jedna zřetelná
        if/elif/else větev tady dole).
        """
        cached = self._load_cache(endpoint)
        if cached is not None:
            return cached

        result = self._request(endpoint)

        if result.ok:
            self._save_cache(endpoint, result.data)
            return result.data

        if result.permanent:
            # 400/404/422 -- retry by stejně nikdy neuspěl, bezpečné
            # zacachovat "nenalezeno" natrvalo.
            self._save_cache(endpoint, None)
            log.info(f"Jikan {endpoint}: selhalo natrvalo -- cachuju jako nenalezeno")
            return None

        # Dočasné selhání (5xx/timeout/vyčerpaný rate limit) -- NEUKLÁDÁME,
        # ať to příští běh zkusí znovu.
        log.warning(f"Jikan {endpoint}: dočasné selhání, cache beze změny")
        return None

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

        Cachováno odděleně od /full dat pod klíčem "staff_{mal_id}".
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
            time.sleep(REQUEST_DELAY)

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
