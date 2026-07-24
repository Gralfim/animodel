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

    # Airing info pro výpočet data posledního dílu (sezónní doporučení).
    # nextAiringEpisode dává PŘESNÝ čas dalšího dílu; finále dopočítá
    # season.py z (episodes − nextEp)·7 dní.
    QUERY_AIRING_BATCH = """
    query ($ids: [Int]) {
      Page(perPage: 50) {
        media(idMal_in: $ids, type: ANIME) {
          idMal
          status
          episodes
          nextAiringEpisode { episode airingAt }
          endDate { year month day }
        }
      }
    }"""

    def get_airing_batch(self, mal_ids: list[int]) -> dict[int, dict]:
        """
        Airing info pro dané MAL ID: {mal_id: {status, episodes, next_ep,
        next_airing_at, end_date}}. **NEcachuje se** -- na rozdíl od
        statických media dat se airing stav (nextAiringEpisode) mění týdně,
        cache by rychle zastarala. ~1 dotaz na 50 titulů, levné.

        end_date je (year, month, day) tuple nebo None. next_airing_at je
        unix timestamp dalšího dílu nebo None.
        """
        out: dict[int, dict] = {}
        for i in range(0, len(mal_ids), 50):
            chunk = mal_ids[i:i + 50]
            result = self._request(self.QUERY_AIRING_BATCH, {"ids": chunk})
            if not result.ok:
                continue
            for m in (result.data.get("data", {}).get("Page", {}).get("media", []) or []):
                mid = m.get("idMal")
                if not mid:
                    continue
                nae = m.get("nextAiringEpisode") or {}
                ed = m.get("endDate") or {}
                end_date = ((ed.get("year"), ed.get("month"), ed.get("day"))
                            if ed.get("year") and ed.get("month") and ed.get("day")
                            else None)
                out[mid] = {
                    "status": m.get("status"),
                    "episodes": m.get("episodes"),
                    "next_ep": nae.get("episode"),
                    "next_airing_at": nae.get("airingAt"),
                    "end_date": end_date,
                }
        return out

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

    def get_watcher_entries(self, anilist_id: int, users_per_seed: int,
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

    def get_user_animelist(self, uid: int) -> dict | None:
        """
        Kompletní seznam uživatele (včetně custom listů), cache
        `userlist_{uid}_v2`. Vrací:

            {"fmt": scoreFormat,
             "entries":  [[mal_id, raw_score, avg_raw, title], ...],
             "planning": [mal_id, ...]}

        `entries` = OHODNOCENÉ záznamy, nově VČETNĚ statusu DROPPED: jeho
        známka je platný (a pro shodu vkusu velmi výmluvný) signál --
        „zkusil a dal 3" o kompatibilitě říká víc než většina desítek.
        Dřív se DROPPED zahazoval celý, čímž se ta informace ztrácela.

        `planning` = tituly na jeho plan-to-watch. Nemají známku, takže do
        `entries` nepatří, ale usercf.py je potřebuje odlišit od „nikdy o
        tom neslyšel": mít můj oblíbený titul ve frontě není důvod k
        penalizaci (viz user_cf_fav_miss_penalty).

        Cache klíč nese _v2 (schéma odpovědi) -- staré `userlist_{uid}`
        soubory pole `planning` nemají a chybějící pole nejde odlišit od
        „nic neplánuje", což by vedlo k falešné penalizaci.

        None = privátní/smazaný účet (permanent, cachováno natrvalo --
        příští běhy už síť nezkouší) NEBO dočasné selhání (necachováno,
        zkusí se příště). Volajícímu (usercf.py) na rozdílu nezáleží:
        v obou případech kandidáta přeskočí bez ztráty místa v poolu.
        """
        return cached_fetch(self._cf_cache, f"userlist_{uid}_v2",
                            lambda: self._fetch_user_animelist(uid))

    def _fetch_user_animelist(self, uid: int) -> Result:
        result = self._request(self.QUERY_USER_ANIMELIST, {"userId": uid})
        if not result.ok:
            return result
        collection = (result.data.get("data") or {}).get("MediaListCollection") or {}
        fmt = ((collection.get("user") or {})
               .get("mediaListOptions", {})
               .get("scoreFormat", "UNKNOWN"))
        entries = []
        planning = []
        for lst in collection.get("lists", []):
            for e in lst.get("entries", []):
                media = e.get("media") or {}
                mid = media.get("idMal")
                if not mid:
                    continue
                if e.get("status") == "PLANNING":
                    planning.append(mid)
                    continue
                raw = e.get("score") or 0
                if not raw:
                    # Neohodnocený záznam (včetně dropnutého bez známky) --
                    # nenese měřitelný signál; pro usercf.py to znamená
                    # „nepokryto" (viz penalizace oblíbených).
                    continue
                avg = media.get("averageScore") or 0
                if avg > 0:
                    entries.append([mid, raw, avg,
                                    (media.get("title") or {}).get("romaji", "")])
        return Result.success({"fmt": fmt, "entries": entries,
                               "planning": planning})
