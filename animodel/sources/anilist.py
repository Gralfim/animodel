"""
anilist.py — AniList GraphQL klient s cachováním a rate limitingem

AniList API: https://anilist.gitbook.io/anilist-apiv2-docs/
GraphQL endpoint: https://graphql.anilist.co
Rate limit: 90 requestů/minutu (klient automaticky čeká při 429)

Klíčová výhoda oproti MAL: 500+ granulárních tagů včetně archetypů
postav (Tsundere, Kuudere), narativních vzorů (Love Triangle, Slow
Romance) a tematických kategorií (Tearjerker, Philosophy).
Každý tag má rank 0–100 (jak dominantní je v daném titulu).
"""

import math
import time
import logging
from pathlib import Path
from typing import Callable

import requests

from . import progress, progress_done, status, is_permanent_status, Result
from .cache import FileCache, cached_fetch
from .http import (
    AdaptiveRateLimiter, Attempt, attempt_success, attempt_permanent,
    attempt_rate_limited, attempt_retryable, request_with_retry,
)

log = logging.getLogger(__name__)

GRAPHQL_URL        = "https://graphql.anilist.co"
REQUEST_DELAY_BASE = 0.7    # výchozí sekundy mezi requesty
REQUEST_DELAY_MAX  = 4.0    # strop pro adaptivní zpomalení po sérii 429
RETRY_DELAYS       = [5, 15, 40, 90]  # backoff při 429 — minimální čekání,
                                       # NIKDY nepřepsáno hlavičkou Retry-After
                                       # směrem dolů (AniList umí poslat i 0)
MAX_WATCHER_PAGE    = 5000 // 50      # AniList limit: page*perPage ≤ 5000
NO_PROGRESS_PAGE_LIMIT = 5            # kolik stránek bez nového uživatele
                                       # ještě zkusit než to vzdát

# Sdílený výčet polí pro Media — jediné místo definice, ať se single a batch
# dotaz nemohou rozjet. Pole genres/description/seasonYear/startDate/relations
# přidána kvůli nouzovému AniList-only režimu (--no-jikan): pokrývají to, co
# jinak dodává Jikan (žánry, synopse, dekáda, franšízové vazby).
_MEDIA_FIELDS = """
      id
      idMal
      title { romaji english }
      genres
      description
      seasonYear
      startDate { year }
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
      relations {
        edges {
          relationType
          node { idMal type }
        }
      }
      format
      source
      averageScore
      popularity
      favourites
"""

# GraphQL dotaz — stáhne tagy, studia a základní metadata přes MAL ID
QUERY_BY_MAL_ID = f"""
query ($idMal: Int) {{
  Media(idMal: $idMal, type: ANIME) {{
{_MEDIA_FIELDS}
  }}
}}
"""

# GraphQL dotaz pro hromadné stažení (až 50 ID najednou)
# Anilist nemá batch endpoint pro arbitrary IDs, ale
# můžeme stránkovat přes idMal_in.
# POZN.: s relations/tags pro 50 titulů se dotaz blíží AniList limitu
# složitosti -- kdyby ho někdy překročil (GraphQL chyba), get_anime_batch
# má fallback na jednotlivé requesty, které jsou hluboko pod limitem.
QUERY_BATCH = f"""
query ($ids: [Int]) {{
  Page(perPage: 50) {{
    media(idMal_in: $ids, type: ANIME) {{
{_MEDIA_FIELDS}
    }}
  }}
}}
"""


class AniListClient:
    def __init__(self, cache_dir: str = "cache", sleep: Callable[[float], None] = time.sleep):
        root = Path(cache_dir)
        self._cache = FileCache(root / "anilist")     # mal_{id}_v2 media, rec_{id} recommendations
        self._cf_cache = FileCache(root / "cf_al")     # watchers_{aid}_p{n}, userlist_{uid}
        self._rate_limiter = AdaptiveRateLimiter(REQUEST_DELAY_BASE, REQUEST_DELAY_MAX, growth=1.5)
        self._sleep = sleep
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
            "User-Agent":   "anime-taste-model/1.0",
        })

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    def _classify(self, resp: requests.Response) -> Attempt:
        if resp.status_code == 429:
            # AniList Retry-After hlavička bývá nespolehlivá — umí poslat i
            # "0". `request_with_retry` bere `max(naplánovaný_floor, wait)`,
            # takže hlavička může čekání jen PRODLOUŽIT, nikdy zkrátit.
            header_wait = int(resp.headers.get("Retry-After", 0) or 0)
            return attempt_rate_limited(wait=header_wait)

        if is_permanent_status(resp.status_code):
            body = resp.text[:300].replace(chr(10), " ")
            return attempt_permanent(f"HTTP {resp.status_code} {body!r}")

        if not resp.ok:
            body = resp.text[:300].replace(chr(10), " ")
            return attempt_retryable(f"HTTP {resp.status_code} {body!r}")

        data = resp.json()  # chyba parsování -> RequestException -> retryable (viz request_with_retry)

        # GraphQL vrátí errors pole i při HTTP 200 — typicky problém s
        # konkrétními proměnnými (neplatné ID, špatný typ) u KONKRÉTNÍHO
        # requestu, ne dočasný stav serveru, takže retry se stejnými
        # parametry by dopadl stejně.
        if "errors" in data:
            msgs = [e.get("message", "?") for e in data["errors"]]
            return attempt_permanent(f"GraphQL error(s): {'; '.join(msgs)}")

        return attempt_success(data)

    def _request(self, query: str, variables: dict) -> Result:
        """
        Čistě síťová GraphQL vrstva -- BEZ cache. Vrací Result(ok, data, permanent).
        Cachování řeší volající metody samy (get_anime, get_recommendations,
        watchers stránky, ...) přes sdílený `cached_fetch` -- každá má jiný
        tvar dat, ale VŠECHNY teď stejnou tříbodovou logiku (hit / success /
        permanent / transient), ne vlastní ruční kopii.
        """
        def perform():
            return self.session.post(
                GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=20,
            )
        return request_with_retry(
            perform=perform,
            classify=self._classify,
            rate_limiter=self._rate_limiter,
            retry_delays=RETRY_DELAYS,
            label="AniList",
            sleep=self._sleep,
        )

    # ── Cache pomocníci ──────────────────────────────────────────────────────

    @staticmethod
    def _media_key(mal_id: int) -> str:
        """Cache klíč pro Media data. Přípona _v2 = verze schématu dotazu:
        v2 přidala genres/description/seasonYear/startDate/relations (nouzový
        AniList-only režim). Starší mal_{id}.json soubory tahle pole nemají
        a chybějící pole nejde rozlišit od "titul je nemá" -- proto nový
        klíč. Staré soubory neškodně osiří; re-fetch celého seznamu jde přes
        batch (50 titulů/request), takže stojí ~10 requestů, ne stovky."""
        return f"mal_{mal_id}_v2"

    def _cached_media(self, mal_id: int) -> dict | None:
        """Poslední cachovaná AniList Media data pro `mal_id`, BEZ síťového
        volání. `None` jak pro "nikdy nezkoušeno", tak pro "potvrzeně
        nenalezeno" -- volající, kterým na rozdílu nezáleží (CF příprava,
        popularita), tak nemusí znát tvar cache envelope."""
        envelope = self._cache.get(self._media_key(mal_id))
        if envelope is None or not envelope["found"]:
            return None
        return envelope["data"]

    # ── Veřejné metody ─────────────────────────────────────────────────────────

    def get_anime(self, mal_id: int) -> dict | None:
        """
        Vrátí AniList data pro anime dle MAL ID. Výsledek je cachován —
        opakované volání je okamžité.

        Vrátí None pokud anime na AniList neexistuje NEBO pokud request
        selhal dočasně -- volajícímu na tom rozdílu nezáleží (v obou
        případech nemá co vrátit); rozdíl je jen v tom, že první případ se
        cachuje natrvalo a druhý ne (zkusí se znovu příště).
        """
        return cached_fetch(self._cache, self._media_key(mal_id), lambda: self._fetch_media(mal_id))

    def _fetch_media(self, mal_id: int) -> Result:
        result = self._request(QUERY_BY_MAL_ID, {"idMal": mal_id})
        if not result.ok:
            return result
        media = result.data.get("data", {}).get("Media")
        if media:
            return Result.success(media)
        # Request uspěl (HTTP 200), AniList potvrdil že titul neexistuje —
        # bezpečné cachovat natrvalo, i když to technicky není HTTP selhání.
        return Result.failure(permanent=True)

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

        # 1. Načti z cache. Tři možné stavy na klíč: chybí (miss -> stáhnout),
        # cachováno a nalezeno (-> results), cachováno jako potvrzeně
        # nenalezeno (-> ani results, ani uncached, definitivně vyřešeno).
        for mal_id in mal_ids:
            envelope = self._cache.get(self._media_key(mal_id))
            if envelope is not None:
                if envelope["found"]:
                    results[mal_id] = envelope["data"]
            else:
                uncached.append(mal_id)

        if show_progress and uncached:
            status(f"  AniList cache: {len(results)} hit, {len(uncached)} ke stažení…")

        if not uncached:
            return results

        # 2. Stáhni nekešované po dávkách 50
        batch_size = 50
        batches    = [uncached[i:i+batch_size] for i in range(0, len(uncached), batch_size)]

        for b_idx, batch in enumerate(batches):
            if show_progress:
                done = b_idx * batch_size
                progress(f"  AniList stahování: {done}/{len(uncached)}…")

            result = self._request(QUERY_BATCH, {"ids": batch})
            if not result.ok:
                # Batch request selhal (ne "nic se nenašlo") — fallback na
                # jednotlivé requesty. get_anime() sám správně rozlišuje
                # selhání od potvrzeného nenalezení, takže ani tady se
                # neukládá falešný sentinel.
                log.warning(
                    f"AniList batch query selhala ({len(batch)} ID), "
                    f"přepínám na jednotlivé requesty…"
                )
                batch_failed = 0
                for mal_id in batch:
                    data = self.get_anime(mal_id)
                    if data:
                        results[mal_id] = data
                    elif self._cache.get(self._media_key(mal_id)) is None:
                        batch_failed += 1  # stále nestažené (request selhal)
                if batch_failed:
                    log.warning(
                        f"  {batch_failed}/{len(batch)} titulů se nepodařilo "
                        f"stáhnout ani jednotlivě — zkusí se příště znovu"
                    )
                continue

            media_list = result.data.get("data", {}).get("Page", {}).get("media", [])

            # Ulož do cache i results
            fetched_mal_ids = set()
            for media in media_list:
                mid = media.get("idMal")
                if mid:
                    self._cache.set(self._media_key(mid), {"found": True, "data": media})
                    results[mid] = media
                    fetched_mal_ids.add(mid)

            # Ulož sentinel pro nenalezená
            for mal_id in batch:
                if mal_id not in fetched_mal_ids:
                    self._cache.set(self._media_key(mal_id), {"found": False, "data": None})

        if show_progress and uncached:
            progress_done(f"  AniList staženo: {len(results)}/{len(mal_ids)} titulů.")

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
        """AniList doporučení k titulu: [{mal_id, title, rating, community}, ...].

        Stejná opatrnost jako get_anime(): dočasné selhání se NEcachuje jako
        "žádná doporučení" -- jinak by přechodný výpadek/vyčerpané retries
        vypadaly navždy stejně jako titul, co doporučení skutečně nemá.
        Trvalé selhání (400/404/422/GraphQL chyba) se ale cachuje jako
        prázdný list -- retry by stejně nikdy nepomohl.
        """
        out = cached_fetch(self._cache, f"rec_{mal_id}", lambda: self._fetch_recommendations(mal_id))
        return out if out is not None else []

    def _fetch_recommendations(self, mal_id: int) -> Result:
        result = self._request(self.REC_QUERY, {"idMal": mal_id})
        if not result.ok:
            return result
        out = []
        nodes = (result.data.get("data", {}).get("Media", {}) or {}).get("recommendations", {}).get("nodes", [])
        for n in nodes:
            mr = n.get("mediaRecommendation") or {}
            if mr.get("idMal"):
                out.append({
                    "mal_id": mr["idMal"],
                    "title": (mr.get("title") or {}).get("romaji", ""),
                    "rating": n.get("rating", 0),
                    "community": (mr.get("averageScore") or 0) / 10.0,
                })
        # Sem se dostaneme jen když _request() skutečně uspěl (HTTP 200 + bez
        # GraphQL "errors") -- prázdný `out` teď znamená potvrzeně žádná
        # doporučení, ne nejistotu, takže je bezpečné to natrvalo zacachovat.
        return Result.success(out)

    # Kompletní seznam všech AniList tagů (name, description, category) --
    # universum pro intensity lexikon (viz animodel/intensity.py).
    TAG_COLLECTION_QUERY = """
    query {
      MediaTagCollection {
        name
        description
        category
        isAdult
        isGeneralSpoiler
      }
    }"""

    # Žánry jsou u AniListu ODDĚLENĚ od tagů (Media.genres je [String] a
    # MediaTagCollection je neobsahuje) -- bez tohohle dotazu by universum
    # v AniList-only režimu (--no-jikan) nemělo comedy/drama/horror/...,
    # tedy nejsilnější klíče celé osy náročnosti.
    GENRE_COLLECTION_QUERY = """
    query { GenreCollection }
    """

    def get_genre_collection(self) -> list[str]:
        """Úplný seznam AniList žánrů (~18 stringů). Cachováno pod klíčem
        "genre_collection" (stejná logika jako tag_collection)."""
        out = cached_fetch(self._cache, "genre_collection", self._fetch_genre_collection)
        return out if out is not None else []

    def _fetch_genre_collection(self) -> Result:
        result = self._request(self.GENRE_COLLECTION_QUERY, {})
        if not result.ok:
            return result
        genres = (result.data.get("data") or {}).get("GenreCollection") or []
        return Result.success(genres)

    def get_tag_collection(self) -> list[dict]:
        """Úplný seznam AniList tagů (~350) s popisem a kategorií. Cachováno
        pod klíčem "tag_collection" -- universum se mění jen když AniList
        přidá nový tag (vzácně); pro vynucené obnovení smaž
        cache/anilist/tag_collection.json."""
        out = cached_fetch(self._cache, "tag_collection", self._fetch_tag_collection)
        return out if out is not None else []

    def _fetch_tag_collection(self) -> Result:
        result = self._request(self.TAG_COLLECTION_QUERY, {})
        if not result.ok:
            return result
        tags = (result.data.get("data") or {}).get("MediaTagCollection") or []
        return Result.success(tags)

    TAG_SEARCH = """
    query ($tags: [String], $page: Int) {
      Page(page: $page, perPage: 50) {
        media(tag_in: $tags, type: ANIME, sort: SCORE_DESC, format_not: MUSIC) {
          idMal title { romaji english } averageScore popularity
        }
      }
    }"""

    def search_by_tags(self, tags: list[str], pages: int = 2) -> list[dict]:
        """Discovery: tituly s danými tagy, seřazené dle skóre. Necachuje se
        (čistě průběžný discovery dotaz, ne perzistentní znalost o titulu)."""
        out = []
        for p in range(1, pages + 1):
            result = self._request(self.TAG_SEARCH, {"tags": tags, "page": p})
            if not result.ok:
                break
            media = result.data.get("data", {}).get("Page", {}).get("media", [])
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

    # Popularita titulu (počet uživatelů) – kvůli vážení vzácných seedů.
    QUERY_MEDIA_POPULARITY = """
    query ($id: Int) {
      Media(id: $id, type: ANIME) {
        popularity
      }
    }"""

    # Uživatelé, kteří daný titul ohodnotili, seřazeni dle skóre.
    # Bereme COMPLETED i CURRENT/REPEATING – kdokoli s ratingem nese signál.
    QUERY_USERS_BY_MEDIA = """
    query ($mediaId: Int, $page: Int, $perPage: Int) {
      Page(page: $page, perPage: $perPage) {
        pageInfo { hasNextPage }
        mediaList(mediaId: $mediaId, sort: SCORE_DESC) {
          score
          user { id name }
        }
      }
    }"""

    # Celý seznam uživatele včetně custom listů (jinak přijdeme o entries,
    # které uživatel skryl ze status-listů – viz AniList docs).
    QUERY_USER_ANIMELIST = """
    query ($userId: Int) {
      MediaListCollection(userId: $userId, type: ANIME, sort: SCORE_DESC) {
        lists {
          entries {
            status
            score
            media { idMal popularity averageScore title { romaji } }
          }
        }
        user {
          mediaListOptions {
            scoreFormat
          }
        }
      }
    }"""

    QUERY_USER_NAMES = """
    query ($ids: [Int]) {
      Page(perPage: 50) {
        users(id_in: $ids) {
          id
          name
        }
      }
    }"""

    @staticmethod
    def _norm_score(raw: float) -> float:
        """
        Normalizuj AniList skóre na 0–1. AniList ukládá v uživatelově formátu
        (POINT_100, POINT_10, POINT_10_DECIMAL, POINT_5, POINT_3). Heuristika:
        hodnota > 10 → /100, jinak → /10. Jde o signál, ne o přesné číslo.
        """
        if raw <= 0:
            return 0.0
        return raw / 100.0 if raw > 10 else raw / 10.0

    def _media_popularity(self, anilist_id: int, fallback_mal_id: int) -> int:
        """Počet uživatelů, kteří titul mají v listu. Z cache nebo dotazem."""
        media = self._cached_media(fallback_mal_id)
        if media and media.get("popularity"):
            return int(media["popularity"])
        result = self._request(self.QUERY_MEDIA_POPULARITY, {"id": anilist_id})
        if result.ok:
            media = (result.data.get("data") or {}).get("Media") or {}
            return int(media.get("popularity") or 0)
        return 0

    # ── CF: sledující média, stránkováno a cachováno PO STRÁNKÁCH ──────────

    def _fetch_watchers_page(self, anilist_id: int, page: int, per_page: int) -> Result:
        """Jeden GraphQL request = jedna stránka sledujících média
        `anilist_id`. Na úspěchu vrací `Result.success` s daty JIŽ
        přetvarovanými do tvaru, který se cachuje pod
        `watchers_{anilist_id}_p{page}`:

            {"entries": [[uid, uname, raw_score], ...], "has_next": bool}

        (entries obsahuje jen skutečně ohodnocené záznamy — nehodnocené
        `score=0` položky se zahazují hned tady, ne až u volajícího).
        """
        result = self._request(
            self.QUERY_USERS_BY_MEDIA,
            {"mediaId": anilist_id, "page": page, "perPage": per_page},
        )
        if not result.ok:
            return result
        pg = (result.data.get("data") or {}).get("Page") or {}
        entries = []
        for entry in pg.get("mediaList") or []:
            raw = entry.get("score") or 0
            if not raw:
                continue
            user_obj = entry.get("user") or {}
            uid = user_obj.get("id")
            if not uid:
                continue
            entries.append([uid, user_obj.get("name", ""), raw])
        has_next = bool((pg.get("pageInfo") or {}).get("hasNextPage"))
        return Result.success({"entries": entries, "has_next": has_next})

    def _iter_watcher_entries(self, anilist_id: int, users_per_seed: int,
                              per_page: int = 50) -> list:
        """
        Sekvenčně skládá ohodnocené sledující média `anilist_id` -- stránku
        po stránce, z cache kde je, jinak čerstvým fetchem. Každá stránka
        je cachovaná NEZÁVISLE (`cached_fetch` na vlastní klíč), takže když
        stránka N selže dočasně, stránky 1..N-1 zůstanou na disku a příští
        volání pokračuje od N, ne od 1 -- to je hlavní rozdíl oproti dřívější
        verzi, kde celý seed byl jeden cache soubor zapsaný až na konci.

        Vrací list [[uid, uname, raw_score], ...] o délce až `users_per_seed`
        (může být kratší, když dojdou stránky/hodnotitelé nebo request selže).
        """
        collected: list = []
        no_progress_streak = 0
        page = 1
        while len(collected) < users_per_seed:
            if page > MAX_WATCHER_PAGE:
                log.warning(
                    f"user-CF watchers AL {anilist_id}: narazil na "
                    f"MAX_WATCHER_PAGE={MAX_WATCHER_PAGE} (AniList limit "
                    f"page×perPage≤5000) — {len(collected)} uživatelů použito, "
                    f"titul má pravděpodobně mnoho nehodnocených záznamů"
                )
                break

            key = f"watchers_{anilist_id}_p{page}"
            page_data = cached_fetch(
                self._cf_cache, key,
                lambda page=page: self._fetch_watchers_page(anilist_id, page, per_page),
            )
            if page_data is None:
                # Buď dočasné selhání (nezacachováno -- příští volání zkusí
                # TUHLE stránku znovu), nebo trvalé (zacachováno jako
                # nenalezeno -- stránka za koncem dat). V obou případech
                # dál stránkovat nemá smysl.
                log.debug(
                    f"user-CF watchers AL {anilist_id}: stránka {page} "
                    f"nedostupná, končím se {len(collected)} uživateli"
                )
                break

            entries = page_data.get("entries") or []
            collected.extend(entries)
            if entries:
                no_progress_streak = 0
            else:
                # Stránka bez jakéhokoliv NOVÉHO ohodnoceného uživatele --
                # sort=SCORE_DESC řadí nehodnocené záznamy na konec, takže
                # dlouhý ocas prázdných stránek je běžný, ne chyba.
                no_progress_streak += 1
                if no_progress_streak >= NO_PROGRESS_PAGE_LIMIT:
                    break

            if not page_data.get("has_next"):
                break
            page += 1

        return collected[:users_per_seed]

    def similar_users_recommendations(
        self,
        liked_mal_ids: list[int],
        min_overlap: int = 4,
        top_users: int = 120,
        seed_count: int = 25,
        users_per_seed: int = 100,
        user_scores: dict[int, float] | None = None,
        scan_budget_factor: float = 3.0,
    ) -> list[dict]:
        """
        User-based collaborative filtering přes AniList.

          1) SEEDY = MÉNĚ POPULÁRNÍ tvé oblíbené tituly. Sdílení nišového
             titulu je mnohem silnější signál podobného vkusu než sdílení
             blockbusteru — a navíc u nišového titulu pokryje vzorek
             `users_per_seed` uživatelů výrazně větší podíl celé populace,
             takže opakované výskyty (= překryv) reálně vznikají.

          2) VÁŽENÝ PŘEKRYV. Každý sdílený seed přispěje vahou
             ~ -log(popularita) — vzácné tituly váží víc (analogie IDF).
             `min_overlap` se pak vztahuje k počtu sdílených seedů (drží se
             jako tvrdý práh počtu), zatímco řazení uživatelů jde dle váhy.

          3) PODOBNOST = kosinus ratingových vektorů na překryvu, ne jen
             počet. Bere v potaz, ZDA titul hodnotíme podobně.

          4) Stahují se i CUSTOM listy (jinak by se ztratily skryté entries).

          5) SOUKROMÉ/SMAZANÉ ÚČTY se přeskočí BEZ ztráty místa v top_users.
             AniList User typ nemá žádné queryovatelné "isPrivate" pole --
             pozná se to až při pokusu o MediaListCollection, typicky jako
             404 "Private User". Kandidáti se proto neořezávají na top_users
             hned při výběru, ale prochází se CELÝ seřazený pool, dokud se
             nenajde top_users POUŽITELNÝCH. `scan_budget_factor` je pojistka
             proti neomezenému skenování.

          Sledující jednotlivých seedů se stahují a cachují PO STRÁNKÁCH
          (viz `_iter_watcher_entries`/`_fetch_watchers_page`) -- dočasné
          selhání na stránce N neztratí už stažené stránky 1..N-1.

        Parametry:
            liked_mal_ids   — tvoje oblíbené (seedy bereme z méně populárních)
            min_overlap     — minimální počet sdílených seedů (tvrdý práh)
            top_users       — kolik POUŽITELNÝCH (ne jen zkusených) uživatelů chceš
            seed_count      — kolik nejméně populárních seedů použít
            users_per_seed  — kolik uživatelů stáhnout na jeden seed
            user_scores     — {mal_id: tvé_score} pro kosinovou podobnost
                              (volitelné; bez něj se použije jen vážený překryv)
            scan_budget_factor — kolikrát víc kandidátů (než top_users) zkusit
                              projít, než se to vzdá s tím, co se sehnalo.

        Vrátí list[dict] {'mal_id', 'score'}. Best-effort.
        """
        from collections import defaultdict

        # 1. MAL ID -> AniList ID (z enrich cache, jež už proběhla)
        mal_to_anilist: dict[int, int] = {}
        for mal_id in liked_mal_ids:
            media = self._cached_media(mal_id)
            if media and media.get("id"):
                mal_to_anilist[mal_id] = media["id"]

        if not mal_to_anilist:
            log.warning("user-CF: žádné AniList ID v cache – enrich musí proběhnout dřív")
            return []

        # 2. Vyber NEJMÉNĚ POPULÁRNÍ seedy (vzácnost = silnější signál).
        #    Popularitu bereme z cache (uložená při enrichi), fallback dotazem.
        pop_by_mal:      dict[int, int]   = {}
        seed_comm_norm:  dict[int, float]  = {}  # mal_id → AniList averageScore/100
        for mal_id, aid in mal_to_anilist.items():
            pop_by_mal[mal_id] = self._media_popularity(aid, mal_id) or 10**9
            media = self._cached_media(mal_id)
            avg   = (media.get("averageScore") or 0) if media else 0
            seed_comm_norm[mal_id] = avg / 100.0 if avg else 0.75

        # seřaď vzestupně dle popularity -> nejnišovější první
        ranked = sorted(mal_to_anilist.items(), key=lambda kv: pop_by_mal[kv[0]])
        seeds = ranked[:seed_count]
        n = len(seeds)

        # IDF-like váha seedu: vzácný titul (nízká popularita) váží víc.
        # +2 ať se vyhneme log(0/1); normujeme později implicitně.
        seed_weight = {
            mal_id: math.log10((pop_by_mal[mal_id] or 1) + 10)
            for mal_id, _ in seeds
        }
        max_w = max(seed_weight.values()) if seed_weight else 1.0
        # invertuj: nízká popularita -> vysoká váha
        seed_weight = {m: (max_w - w + 0.5) for m, w in seed_weight.items()}

        # 3. Pro každý seed slož sledující (stránka po stránce, viz
        #    _iter_watcher_entries) i s jejich ratingem.
        #    user_profile[uid] = {seed_mal_id: norm_score}
        #    user_names[uid]   = username (pro výstup)
        user_profile: defaultdict[int, dict[int, float]] = defaultdict(dict)
        user_weight: defaultdict[int, float] = defaultdict(float)
        user_names: dict[int, str] = {}

        n_incomplete = 0
        for i, (mal_id, anilist_id) in enumerate(seeds):
            progress(f"  user-CF: sbírám uživatele [{i+1}/{n}] (pop≈{pop_by_mal[mal_id]}) ...")
            watchers = self._iter_watcher_entries(anilist_id, users_per_seed)
            if len(watchers) < users_per_seed:
                n_incomplete += 1
            for uid, uname, raw in watchers:
                user_profile[uid][mal_id] = self._norm_score(raw)
                user_weight[uid] += seed_weight.get(mal_id, 1.0)
                user_names[uid]   = uname

        progress_done(
            f"  user-CF: sbírám uživatele [{n}/{n}] ... hotovo"
            + (f"  ({n_incomplete} seedů má méně než {users_per_seed} sledujících "
               f"— vyčerpaný titul nebo chyba, viz log)" if n_incomplete else "")
        )

        # Doplň chybějící jména batch dotazem (záchrana pro starou cache)
        missing_ids = [uid for uid in user_profile if uid not in user_names]
        if missing_ids:
            batch_size = 50
            for i in range(0, len(missing_ids), batch_size):
                chunk  = missing_ids[i:i + batch_size]
                result = self._request(self.QUERY_USER_NAMES, {"ids": chunk})
                if result.ok:
                    for u in (result.data.get("data", {}).get("Page", {})
                              .get("users", [])):
                        if u.get("id") and u.get("name"):
                            user_names[u["id"]] = u["name"]
            log.debug(f"user-CF: doplněno {len(missing_ids)} jmen")

        if not user_profile:
            log.info("user-CF: žádní uživatelé nenalezeni")
            return []

        # 4. Podobnost uživatele = kosinus ratingových vektorů na překryvu
        def cosine(uid: int) -> float:
            """Vážená Pearsonova korelace na komunitně-relativních skóre.

            diff_mine_j  = my_norm_j  - c_norm_j   (odchylka ode mě od komunity)
            diff_their_j = their_norm_j - c_norm_j  (odchylka od nich od komunity)

            Pearson automaticky centruje oba vektory kolem jejich průměru →
            odstraní absolutní bias (kdo hodnotí výš/níž celkově) a zachová
            TVAR preference. Nízká variance (plochý hodnotitel) → blízko 0.
            """
            shared = user_profile[uid]
            if not user_scores:
                return user_weight[uid]

            my_diffs: list[float]    = []
            their_diffs: list[float] = []
            weights: list[float]     = []

            for mid, their_norm in shared.items():
                my_raw = user_scores.get(mid) or 0
                if not my_raw:
                    continue
                my_norm = self._norm_score(my_raw * 10)
                c       = seed_comm_norm.get(mid, 0.75)
                my_diffs.append(my_norm  - c)
                their_diffs.append(their_norm - c)
                weights.append(seed_weight.get(mid, 1.0))

            if len(my_diffs) < 2:
                return 0.0

            total_w  = sum(weights)
            my_mean  = sum(w * d for w, d in zip(weights, my_diffs))   / total_w
            th_mean  = sum(w * d for w, d in zip(weights, their_diffs)) / total_w
            my_c     = [d - my_mean for d in my_diffs]
            th_c     = [d - th_mean for d in their_diffs]
            num      = sum(w * a * b for w, a, b in zip(weights, my_c, th_c))
            den_m    = math.sqrt(sum(w * a * a for w, a in zip(weights, my_c)))
            den_t    = math.sqrt(sum(w * b * b for w, b in zip(weights, th_c)))
            if den_m < 1e-9 or den_t < 1e-9:
                return 0.0
            return num / (den_m * den_t)

        candidates = [
            (uid, cosine(uid))
            for uid, prof in user_profile.items()
            if len(prof) >= min_overlap
        ]

        if not candidates:
            best = max(len(p) for p in user_profile.values())
            status(f"  user-CF: nikdo nesplnil min_overlap={min_overlap} "
                   f"(max překryv: {best}) – zkus snížit min_overlap")
            log.info(f"user-CF: max překryv {best} < min_overlap {min_overlap}")
            return []

        candidates.sort(key=lambda x: -x[1])
        # Neořezávej na top_users HNED -- ponech celý seřazený pool. Soukromé
        # profily ("Private User" 404) se pozná až při pokusu o stažení
        # seznamu, ne dřív. Když by se ořezávalo hned tady, soukromý uživatel
        # by natrvalo sebral jedno z `top_users` míst.
        sim_by_uid = dict(candidates)

        status(f"  user-CF: {len(candidates)} kandidátů nad min_overlap, "
               f"cílím na {top_users} použitelných (soukromé/smazané se "
               f"přeskočí bez ztráty místa)…")

        # 5. Stáhni listy podobných uživatelů.
        #    Diferenciální agregace:
        #      diff_i = norm_score_i − community_norm_i
        #      → kladné: uživatel hodnotí výš než průměr (dobrý signál)
        #      → záporné: uživatel hodnotí níž (negativní signál, přispívá záporně)
        #    Výsledek: weighted_diff = Σ(sim_i × diff_i) / Σ(sim_i)
        #    CF skóre = community_norm + weighted_diff  (0–1 škála → *10 = 0–10)
        liked_set = set(liked_mal_ids)

        # Σ sim×diff, Σ sim, počet hodnotitelů, community avg, top raters
        agg_diff: defaultdict[int, float]   = defaultdict(float)  # Σ sim×diff
        agg_sim:  defaultdict[int, float]   = defaultdict(float)  # Σ sim
        title_store: dict[int, str]         = {}                   # mal_id → název
        rec_count: defaultdict[int, int]    = defaultdict(int)
        comm_norm: dict[int, float]         = {}                   # community 0–1
        rater_list: defaultdict[int, list]  = defaultdict(list)   # [(sim, uname)]

        _DIVISORS = {
            "POINT_100":        100.0,
            "POINT_10_DECIMAL": 10.0,
            "POINT_10":         10.0,
            "POINT_5":          5.0,
            "POINT_3":          3.0,
        }

        m = len(candidates)
        # Neskenuj neomezeně dlouho do pool -- pojistka pro patologický případ
        # (např. 80 % kandidátů soukromých by jinak mohlo projít celý pool).
        max_attempts = min(m, max(top_users, int(top_users * scan_budget_factor)))

        _ul_cache_hits     = 0
        _ul_fetched        = 0
        _ul_failed_transient = 0   # síť/5xx/timeout -- zkusí se znovu příště
        _ul_skipped_empty  = 0     # bez použitelných dat (privátní/smazaní/prázdní) -- z cache NEBO čerstvě zjištěno
        good_users = 0
        j = 0
        while good_users < top_users and j < max_attempts:
            uid, sim_val = candidates[j]
            j += 1
            sim  = max(0.0, sim_val)
            name = user_names.get(uid, str(uid))
            ck   = f"userlist_{uid}"

            cached_ul = self._cf_cache.get(ck)
            if cached_ul is not None:
                _ul_cache_hits += 1
                if not cached_ul["found"]:
                    # Známo z minula: soukromý/smazaný/natrvalo selhaný
                    # účet. NEpřičítej se do good_users -- jde se rovnou na
                    # dalšího kandidáta, ať tenhle mrtvý slot nepřijde nazmar.
                    _ul_skipped_empty += 1
                    progress(f"  user-CF: [{j}/{max_attempts}] {name} přeskočen (známo: bez dat)")
                    continue
                _fmt        = cached_ul["data"]["fmt"]
                raw_entries = cached_ul["data"]["entries"]
                progress(f"  user-CF: seznam [{j}/{max_attempts}] z cache")
            else:
                # Cache miss — stáhni a zpracuj
                progress(f"  user-CF: stahuji seznam [{j}/{max_attempts}] ...")
                result = self._request(self.QUERY_USER_ANIMELIST, {"userId": uid})
                if not result.ok:
                    if result.permanent:
                        # 400/404/422/GraphQL chyba -- typicky "Private
                        # User" (404), neplatné userId nebo smazaný účet.
                        # Retry se stejným uid nikdy neuspěje, takže je
                        # bezpečné zacachovat "žádná data" natrvalo. Na
                        # PŘÍŠTÍM běhu tenhle uid nikdy ani nezavolá síť --
                        # cache-check výš (`cached_ul is not None`) ho
                        # rovnou přeskočí.
                        self._cf_cache.set(ck, {"found": False, "data": None})
                        _ul_skipped_empty += 1
                        log.info(
                            f"user-CF userlist uid={uid} ({name}): request "
                            f"selhal natrvalo (pravděpodobně privátní/smazaný "
                            f"účet) -- cachuju jako prázdný, přeskočeno bez "
                            f"ztráty místa v top_users"
                        )
                    else:
                        # Dočasné selhání (5xx/timeout/vyčerpaný rate limit) —
                        # NEUKLÁDÁME, tento uživatel se vynechá v tomto běhu,
                        # ale příští spuštění to zkusí znovu (cache zůstává
                        # prázdná = cache miss).
                        _ul_failed_transient += 1
                        log.warning(
                            f"user-CF userlist uid={uid} ({name}): request "
                            f"dočasně selhal, cache nezměněna, uživatel vynechán"
                        )
                    continue
                _ul_fetched += 1
                collection = (result.data.get("data") or {}).get("MediaListCollection") or {}
                _fmt       = (
                    (collection.get("user") or {})
                    .get("mediaListOptions", {})
                    .get("scoreFormat", "UNKNOWN")
                )
                raw_entries: list = []
                for _lst in collection.get("lists", []):
                    for _e in _lst.get("entries", []):
                        if _e.get("status") in ("PLANNING", "DROPPED"):
                            continue
                        _raw = _e.get("score") or 0
                        if not _raw:
                            continue
                        _media   = _e.get("media") or {}
                        _mid     = _media.get("idMal")
                        _avg_raw = _media.get("averageScore") or 0
                        _title   = (_media.get("title") or {}).get("romaji", "")
                        if _mid and _avg_raw > 0:
                            raw_entries.append([_mid, _raw, _avg_raw, _title])
                self._cf_cache.set(ck, {"found": True, "data": {"fmt": _fmt, "entries": raw_entries}})
                if not raw_entries:
                    # Úspěšný request, ale doopravdy nic použitelného (prázdný
                    # seznam / nic ohodnoceného) -- taky nepočítat do good_users.
                    _ul_skipped_empty += 1
                    continue

            good_users += 1

            # Zpracuj záznamy (z cache nebo čerstvě stažené)
            _div = _DIVISORS.get(_fmt)

            # Osobní průměr — základ pro diferenciální skóre
            _all_norms = [
                max(0.0, min(1.0, raw / _div if _div else self._norm_score(raw)))
                for _, raw, _, _ in raw_entries
                if (raw / _div if _div else self._norm_score(raw)) > 0
            ]
            personal_avg = (sum(_all_norms) / len(_all_norms)) if _all_norms else 0.7

            for mid, raw, avg_raw, title in raw_entries:
                if mid in liked_set:
                    continue
                norm   = max(0.0, min(1.0,
                    raw / _div if _div else self._norm_score(raw)
                ))
                c_norm = avg_raw / 100.0
                diff   = norm - personal_avg
                agg_diff[mid]  += sim * diff
                agg_sim[mid]   += sim
                rec_count[mid] += 1
                if mid not in comm_norm and title:
                    title_store[mid] = title
                comm_norm[mid] = c_norm
                rater_list[mid].append((sim, name))

        progress_done(f"  user-CF: {good_users}/{top_users} použitelných uživatelů "
                      f"(prošel {j}/{max_attempts} kandidátů)")
        log.info(
            f"user-CF userlist: {_ul_cache_hits} z cache, "
            f"{_ul_fetched} staženo, {_ul_failed_transient} dočasně selhalo, "
            f"{_ul_skipped_empty} bez dat (privátní/smazaní/prázdní) — "
            f"{good_users} použitelných z {j} zkusených ({m} v poolu)"
        )
        if _ul_failed_transient:
            print(
                f"  user-CF: {_ul_failed_transient} uživatelů dočasně selhalo (viz log) "
                f"— zkusí se znovu při příštím spuštění"
            )
        if good_users < top_users and j >= max_attempts:
            log.warning(
                f"user-CF: dosaženo stropu {max_attempts} pokusů (scan_budget_factor="
                f"{scan_budget_factor}) a získáno jen {good_users}/{top_users} "
                f"použitelných uživatelů -- zvaž vyšší scan_budget_factor, "
                f"min_overlap, nebo větší candidate pool (podíl bez dat: "
                f"{_ul_skipped_empty}/{j})"
            )

        # 6. Finální CF skóre a filtrování.
        #    weighted_diff = Σ(sim×diff)/Σ(sim)  – vážený průměr diferenciálů
        #    cf_score = community + weighted_diff  (0–1 → *10 na vstup do bump)
        #    Práh: aspoň 2 nezávislí doporučitelé, kladné cf_score.
        out = []
        for mid in agg_diff:
            if rec_count[mid] < 2:
                continue
            c       = comm_norm.get(mid, 0.5)
            w_diff  = agg_diff[mid] / agg_sim[mid] if agg_sim[mid] else 0.0
            # Aditivní bonus za počet hodnotitelů (max ≈ +0.10 pro 10+ uživatelů).
            # Záměrně malý a aditivní — nechceme násobit diff, jen jemně upřednostnit
            # tituly doporučené více spřízněnými dušemi před těmi s jediným.
            n_bonus = 0.03 * math.log1p(max(0, rec_count[mid] - 2))
            cf_raw  = max(0.0, c + w_diff + n_bonus)
            # top raters (nejpodobnější, kteří tento titul hodnotili)
            top_r   = sorted(rater_list[mid], key=lambda x: -x[0])[:5]
            out.append({
                "mal_id":       mid,
                "score":        cf_raw * 10.0,   # pro bump() — může být > 10
                "cf_score":     cf_raw * 10.0,
                "community":    c * 10.0,
                "diff":         w_diff * 10.0,
                "n_users":      rec_count[mid],
                "top_raters":   [(name, round(sim, 3)) for sim, name in top_r],
                "title":        title_store.get(mid, ""),
            })

        # Seřaď primárně dle cf_score
        out.sort(key=lambda x: -x["cf_score"])
        log.info(f"user-CF: {len(out)} kandidátů z {good_users} použitelných uživatelů")
        return out
