"""
anilist_client.py — AniList GraphQL klient s cachováním a rate limitingem

AniList API: https://anilist.gitbook.io/anilist-apiv2-docs/
GraphQL endpoint: https://graphql.anilist.co
Rate limit: 90 requestů/minutu (klient automaticky čeká při 429)

Klíčová výhoda oproti MAL: 500+ granulárních tagů včetně archetypů
postav (Tsundere, Kuudere), narativních vzorů (Love Triangle, Slow
Romance) a tematických kategorií (Tearjerker, Philosophy).
Každý tag má rank 0–100 (jak dominantní je v daném titulu).
"""

import json
import math
import time
import logging
from pathlib import Path

import requests

log = logging.getLogger(__name__)

GRAPHQL_URL   = "https://graphql.anilist.co"
REQUEST_DELAY = 0.7          # sekundy mezi requesty (konzervativní)
RETRY_DELAYS  = [5, 15, 60]  # backoff při 429

# GraphQL dotaz — stáhne tagy, studia a základní metadata přes MAL ID
QUERY_BY_MAL_ID = """
query ($idMal: Int) {
  Media(idMal: $idMal, type: ANIME) {
    id
    idMal
    title { romaji english }
    tags {
      name
      rank
      isAdult
      isGeneralSpoiler
      isMediaSpoiler
      category
    }
    studios {
      nodes { name isAnimationStudio }
    }
    format
    source
    averageScore
    popularity
    favourites
  }
}
"""

# GraphQL dotaz pro hromadné stažení (až 50 ID najednou)
# Anilist nemá batch endpoint pro arbitrary IDs, ale
# můžeme stránkovat přes idMal_in
QUERY_BATCH = """
query ($ids: [Int]) {
  Page(perPage: 50) {
    media(idMal_in: $ids, type: ANIME) {
      id
      idMal
      title { romaji english }
      tags {
        name
        rank
        isAdult
        isGeneralSpoiler
        isMediaSpoiler
        category
      }
      studios {
        nodes { name isAnimationStudio }
      }
      format
      source
      averageScore
      popularity
      favourites
    }
  }
}
"""


class AniListClient:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_path = Path(cache_dir) / "anilist"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
            "User-Agent":   "anime-taste-model/1.0",
        })

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_file(self, mal_id: int) -> Path:
        return self.cache_path / f"mal_{mal_id}.json"

    def _load_cache(self, mal_id: int):
        f = self._cache_file(mal_id)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        return None        # None = not cached yet; {} = cached but not found

    def _save_cache(self, mal_id: int, data) -> None:
        f = self._cache_file(mal_id)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _post(self, query: str, variables: dict) -> dict | None:
        """Provede GraphQL POST request s rate limitingem a retry logikou."""
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

        payload = {"query": query, "variables": variables}

        for attempt, delay in enumerate(RETRY_DELAYS + [None]):
            try:
                resp = self.session.post(GRAPHQL_URL, json=payload, timeout=20)
                self._last_request = time.time()

                if resp.status_code == 429:
                    # AniList vrátí Retry-After header
                    wait = int(resp.headers.get("Retry-After", delay or 60))
                    log.warning(f"Rate limit 429, čekám {wait}s…")
                    time.sleep(wait)
                    continue

                if resp.status_code == 404:
                    return None

                resp.raise_for_status()
                data = resp.json()

                # GraphQL vrátí errors pole i při HTTP 200
                if "errors" in data:
                    for err in data["errors"]:
                        if err.get("status") == 404:
                            return None
                        log.warning(f"GraphQL error: {err.get('message')}")
                    return None

                return data

            except requests.RequestException as e:
                if attempt < len(RETRY_DELAYS) - 1:
                    log.warning(f"Request chyba ({e}), retry za {delay}s…")
                    time.sleep(delay)
                else:
                    log.error(f"AniList request selhal: {e}")
                    return None

        return None

    # ── Veřejné metody ─────────────────────────────────────────────────────────

    def get_anime(self, mal_id: int) -> dict | None:
        """
        Vrátí AniList data pro anime dle MAL ID.
        Výsledek je cachován — opakované volání je okamžité.

        Vrátí None pokud anime na AniList neexistuje.
        Vrátí {} (prázdný dict) pokud byl uložen jako nenalezený.
        """
        cached = self._load_cache(mal_id)
        if cached is not None:
            return cached if cached else None   # {} → None

        result = self._post(QUERY_BY_MAL_ID, {"idMal": mal_id})
        if result and result.get("data", {}).get("Media"):
            data = result["data"]["Media"]
            self._save_cache(mal_id, data)
            return data
        else:
            # Nenalezeno — ulož prázdný dict jako sentinel
            self._save_cache(mal_id, {})
            log.debug(f"AniList: MAL ID {mal_id} nenalezeno")
            return None

    def get_anime_batch(
        self,
        mal_ids: list[int],
        show_progress: bool = True,
    ) -> dict[int, dict]:
        """
        Stáhne AniList data pro seznam MAL ID.

        Nejprve zkusí cache, nekešované stáhne po dávkách 50 kusů
        (AniList Page query je efektivnější než jednotlivé requesty).

        Vrací dict {mal_id: anilist_data}.
        """
        results   = {}
        uncached  = []

        # 1. Načti z cache
        for mal_id in mal_ids:
            cached = self._load_cache(mal_id)
            if cached is not None:
                if cached:   # neprázdný = platná data
                    results[mal_id] = cached
                # prázdný = nenalezeno, přeskočíme
            else:
                uncached.append(mal_id)

        if show_progress and uncached:
            print(f"  AniList cache: {len(results)} hit, {len(uncached)} ke stažení…")

        if not uncached:
            return results

        # 2. Stáhni nekešované po dávkách 50
        batch_size = 50
        batches    = [uncached[i:i+batch_size] for i in range(0, len(uncached), batch_size)]

        for b_idx, batch in enumerate(batches):
            if show_progress:
                done = b_idx * batch_size
                print(
                    f"  AniList stahování: {done}/{len(uncached)}…",
                    end="\r"
                )

            result = self._post(QUERY_BATCH, {"ids": batch})
            if not result:
                # Fallback: stáhni jeden po jednom
                log.warning("Batch query selhala, přepínám na jednotlivé requesty…")
                for mal_id in batch:
                    data = self.get_anime(mal_id)
                    if data:
                        results[mal_id] = data
                continue

            media_list = result.get("data", {}).get("Page", {}).get("media", [])

            # Ulož do cache i results
            fetched_mal_ids = set()
            for media in media_list:
                mid = media.get("idMal")
                if mid:
                    self._save_cache(mid, media)
                    results[mid] = media
                    fetched_mal_ids.add(mid)

            # Ulož sentinel pro nenalezená
            for mal_id in batch:
                if mal_id not in fetched_mal_ids:
                    self._save_cache(mal_id, {})

        if show_progress and uncached:
            print(f"  AniList staženo: {len(results)}/{len(mal_ids)} titulů.          ")

        return results

    # ── Utilitní metody ────────────────────────────────────────────────────────

    @staticmethod
    def extract_tags(
        anilist_data: dict,
        exclude_adult:   bool = True,
        exclude_spoiler: bool = True,
        min_rank:        int  = 0,
    ) -> dict[str, float]:
        """
        Extrahuje tagy z AniList dat jako dict {tag_name: rank_0_to_1}.

        Parametry:
            exclude_adult   — vynech adult tagy (isAdult=True)
            exclude_spoiler — vynech spoilerové tagy
            min_rank        — minimální rank pro zahrnutí (0–100)

        Vrací rank normalizovaný na 0–1.
        """
        tags = {}
        for tag in anilist_data.get("tags", []):
            if exclude_adult   and tag.get("isAdult"):
                continue
            if exclude_spoiler and (
                tag.get("isGeneralSpoiler") or tag.get("isMediaSpoiler")
            ):
                continue
            rank = tag.get("rank", 0) or 0
            if rank >= min_rank:
                tags[tag["name"]] = rank / 100.0
        return tags

    @staticmethod
    def extract_animation_studios(anilist_data: dict) -> list[str]:
        """Vrátí seznam animačních studií (isAnimationStudio=True)."""
        studios = []
        for node in anilist_data.get("studios", {}).get("nodes", []):
            if node.get("isAnimationStudio"):
                studios.append(node["name"])
        return studios

    def list_all_studios(self, mal_ids: list[int]) -> dict[str, int]:
        """
        Vrátí frekvenční slovník animačních studií pro zadaná MAL ID.
        Počítá počet unikátních titulů (ne výskytů) — ochrana před duplicitami.
        """
        from collections import Counter
        counter: Counter = Counter()

        data = self.get_anime_batch(mal_ids, show_progress=False)
        for anilist_data in data.values():
            seen: set[str] = set()
            for node in anilist_data.get("studios", {}).get("nodes", []):
                if node.get("isAnimationStudio"):
                    name = node["name"]
                    if name not in seen:
                        counter[name] += 1
                        seen.add(name)

        return dict(counter)

    def list_all_tags(self, mal_ids: list[int]) -> dict[str, int]:
        """
        Projde data pro zadaná MAL ID a vrátí frekvenční slovník
        všech tagů. Počítá počet *unikátních titulů* s daným tagem
        (ne surový počet výskytů — chrání před duplikáty v API datech).

        Použití:
            tags = client.list_all_tags(scored_mal_ids)
            for tag, count in sorted(tags.items(), key=lambda x: -x[1])[:50]:
                print(f"{count:4d}  {tag}")
        """
        from collections import Counter
        counter: Counter = Counter()

        data = self.get_anime_batch(mal_ids, show_progress=False)
        for anilist_data in data.values():
            # Deduplikuj tagy v rámci jednoho titulu před počítáním
            seen_in_this_anime: set[str] = set()
            for tag in anilist_data.get("tags", []):
                if tag.get("isAdult") or tag.get("isGeneralSpoiler"):
                    continue
                name = tag["name"]
                if name not in seen_in_this_anime:
                    counter[name] += 1
                    seen_in_this_anime.add(name)

        return dict(counter)

    # ── Doporučení a vyhledávání podle tagů ────────────────────────────────
    REC_QUERY = """
    query ($idMal: Int) {
      Media(idMal: $idMal, type: ANIME) {
        recommendations(sort: RATING_DESC, perPage: 25) {
          nodes { rating mediaRecommendation { idMal title { romaji english } averageScore } }
        }
      }
    }"""

    def get_recommendations(self, mal_id: int) -> list[dict]:
        """AniList doporučení k titulu: [{mal_id, title, rating, community}, ...]."""
        ck = self.cache_path / f"rec_{mal_id}.json"
        if ck.exists():
            return json.loads(ck.read_text(encoding="utf-8"))
        result = self._post(self.REC_QUERY, {"idMal": mal_id})
        out = []
        if result:
            nodes = (result.get("data", {}).get("Media", {}) or {}).get("recommendations", {}).get("nodes", [])
            for n in nodes:
                mr = n.get("mediaRecommendation") or {}
                if mr.get("idMal"):
                    out.append({
                        "mal_id": mr["idMal"],
                        "title": (mr.get("title") or {}).get("romaji", ""),
                        "rating": n.get("rating", 0),
                        "community": (mr.get("averageScore") or 0) / 10.0,
                    })
        ck.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    TAG_SEARCH = """
    query ($tags: [String], $page: Int) {
      Page(page: $page, perPage: 50) {
        media(tag_in: $tags, type: ANIME, sort: SCORE_DESC, format_not: MUSIC) {
          idMal title { romaji english } averageScore popularity
        }
      }
    }"""

    def search_by_tags(self, tags: list[str], pages: int = 2) -> list[dict]:
        """Discovery: tituly s danými tagy, seřazené dle skóre."""
        out = []
        for p in range(1, pages + 1):
            result = self._post(self.TAG_SEARCH, {"tags": tags, "page": p})
            if not result:
                break
            media = result.get("data", {}).get("Page", {}).get("media", [])
            for m in media:
                if m.get("idMal"):
                    out.append({
                        "mal_id": m["idMal"],
                        "title": (m.get("title") or {}).get("romaji", ""),
                        "community": (m.get("averageScore") or 0) / 10.0,
                    })
            if len(media) < 50:
                break
        return out

    # ── User-based CF ───────────────────────────────────────────────────────

    # Uživatelé, kteří dokončili dané AniList media a seřazení dle skóre.
    QUERY_USERS_BY_MEDIA = """
    query ($mediaId: Int, $page: Int) {
      Page(page: $page, perPage: 50) {
        mediaList(mediaId: $mediaId, status: COMPLETED, sort: SCORE_DESC) {
          score
          user { id }
        }
      }
    }"""

    # Dokončené anime daného uživatele, seřazené dle skóre sestupně.
    QUERY_USER_ANIMELIST = """
    query ($userId: Int) {
      MediaListCollection(userId: $userId, type: ANIME, status: COMPLETED,
                          sort: SCORE_DESC) {
        lists {
          entries {
            score
            media { idMal }
          }
        }
      }
    }"""

    def similar_users_recommendations(
        self,
        liked_mal_ids: list[int],
        min_overlap: int = 15,
        top_users: int = 50,
    ) -> list[dict]:
        """
        User-based CF: najde AniList uživatele s podobným vkusem a vrátí
        jejich oblíbené anime jako kandidáty pro doporučení.

        Postup:
          1) Přeloží MAL ID -> AniList interní ID z existující enrich cache.
          2) Pro každý seed (max 20) stáhne top 50 uživatelů, kteří ho
             dokončili (sort SCORE_DESC, filtruji nulová hodnocení).
          3) Spočítá překryv: kolik seedů sdílí každý uživatel. Uživatelé
             s překryvem >= min_overlap jsou „podobní".
          4) Pro každého podobného uživatele stáhne jeho dokončený seznam.
          5) Vrátí agregovaná doporučení: score = avg_norm × log(count),
             kde avg_norm je průměrné normalizované hodnocení uživateli
             a count počet unikátních doporučitelů.

        Poznámka ke skóre: AniList ukládá skóre ve formátu uživatele
        (POINT_10, POINT_100, POINT_5, POINT_3). Normalizujeme heuristicky:
        hodnota <= 10 -> /10, jinak -> /100. Jde o signál, ne přesné číslo.

        Vrátí list[dict] s klíči 'mal_id' a 'score'.
        Best-effort -- při API chybách vrátí co stihlo.
        """
        from collections import defaultdict, Counter

        # 1. Přeložit MAL ID -> AniList interní ID (z existující enrich cache)
        mal_to_anilist: dict[int, int] = {}
        for mal_id in liked_mal_ids:
            cached = self._load_cache(mal_id)
            if cached and isinstance(cached, dict) and cached.get("id"):
                mal_to_anilist[mal_id] = cached["id"]

        if not mal_to_anilist:
            log.warning("user-CF: žádné AniList ID v cache – enrich musí proběhnout dřív")
            return []

        # 2. Pro každý seed (max 20 API volání) stáhni top uživatele
        user_overlap: Counter = Counter()
        seeds = list(mal_to_anilist.items())[:20]
        n = len(seeds)

        for i, (mal_id, anilist_id) in enumerate(seeds):
            print(f"  user-CF: hledám podobné uživatele [{i+1}/{n}] ...", end="\r")
            result = self._post(
                self.QUERY_USERS_BY_MEDIA,
                {"mediaId": anilist_id, "page": 1},
            )
            if not result:
                continue
            entries = (
                result.get("data", {}).get("Page", {}) or {}
            ).get("mediaList", [])
            for entry in entries:
                if not (entry.get("score") or 0):
                    continue          # přeskoč neohodnocené záznamy
                uid = (entry.get("user") or {}).get("id")
                if uid:
                    user_overlap[uid] += 1

        print(f"  user-CF: hledám podobné uživatele [{n}/{n}] ... hotovo      ")

        if not user_overlap:
            log.info("user-CF: žádní uživatelé nenalezeni")
            return []

        # 3. Filtruj uživatele s dostatečným překryvem
        similar_users = [
            uid for uid, count in user_overlap.most_common(top_users * 4)
            if count >= min_overlap
        ][:top_users]

        if not similar_users:
            best = user_overlap.most_common(1)[0][1]
            print(
                f"  user-CF: žádný uživatel nesplňuje min_overlap={min_overlap} "
                f"(max dosažený překryv: {best})"
            )
            log.info(f"user-CF: max překryv byl pouze {best}")
            return []

        print(f"  user-CF: {len(similar_users)} podobných uživatelů, stahuji jejich seznamy ...")

        # 4. Stáhni anime listy podobných uživatelů
        liked_set = set(liked_mal_ids)
        anime_scores: defaultdict[int, list[float]] = defaultdict(list)

        for uid in similar_users:
            result = self._post(self.QUERY_USER_ANIMELIST, {"userId": uid})
            if not result:
                continue
            collection = (
                result.get("data", {}).get("MediaListCollection") or {}
            )
            for lst in collection.get("lists", []):
                for entry in lst.get("entries", []):
                    raw_score = entry.get("score") or 0
                    if not raw_score:
                        continue
                    media = entry.get("media") or {}
                    mid = media.get("idMal")
                    if not mid or mid in liked_set:
                        continue
                    # Heuristická normalizace: POINT_10 -> /10, POINT_100 -> /100
                    norm = raw_score / 10.0 if raw_score <= 10 else raw_score / 100.0
                    anime_scores[mid].append(norm)

        # 5. Agreguj: průměrné norm. skóre × log(počet doporučitelů)
        out = []
        for mal_id, scores in anime_scores.items():
            avg = sum(scores) / len(scores)
            weighted = avg * math.log1p(len(scores))
            out.append({"mal_id": mal_id, "score": weighted})

        out.sort(key=lambda x: -x["score"])
        log.info(f"user-CF: {len(out)} kandidátů z {len(similar_users)} uživatelů")
        return out[:300]
