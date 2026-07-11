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

import json
import math
import time
import logging
from pathlib import Path

import requests

from . import progress, progress_done, status, Result, is_permanent_status

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
        self._last_request   = 0.0
        # Adaptivní base rate: roste po sérii 429, postupně klesá zpět
        # k REQUEST_DELAY_BASE po sérii úspěšných requestů.
        self._current_delay  = REQUEST_DELAY_BASE
        self._consecutive_429 = 0
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
            "User-Agent":   "anime-taste-model/1.0",
        })

    # ── Cache helpers ──────────────────────────────────────────────────────────

    @property
    def _cf_cache_path(self) -> Path:
        p = self.cache_path.parent / "cf_al"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _cf_load(self, key: str):
        """CF cache čtení — None pokud soubor neexistuje."""
        f = self._cf_cache_path / f"{key}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def _cf_save(self, key: str, data) -> None:
        """CF cache zápis."""
        f = self._cf_cache_path / f"{key}.json"
        f.write_text(json.dumps(data, ensure_ascii=False, separators=(",",":")),
                     encoding="utf-8")

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

    def _request(self, query: str, variables: dict) -> Result:
        """
        Čistě síťová GraphQL vrstva -- BEZ cache. Vrací Result(ok, data, permanent).

        Cachování řeší volající metody samy (get_anime, get_recommendations,
        similar_users_recommendations, ...) -- každá má jiný cache formát/klíč,
        takže se nedá centralizovat na jedno místo jako u JikanClient._get.
        Ale všechny teď čtou klasifikaci ZE STEJNÉ návratové hodnoty, ne
        z odděleného mutable stavu.
        """
        elapsed = time.time() - self._last_request
        if elapsed < self._current_delay:
            time.sleep(self._current_delay - elapsed)

        payload = {"query": query, "variables": variables}

        for attempt, delay in enumerate(RETRY_DELAYS + [None]):
            try:
                resp = self.session.post(GRAPHQL_URL, json=payload, timeout=20)
                self._last_request = time.time()

                if resp.status_code == 429:
                    # AniList Retry-After hlavička bývá nespolehlivá — umí
                    # poslat i "0", což by bez floor hodnoty vedlo k
                    # okamžitému dalšímu pokusu a vyčerpání všech retry
                    # v ulomku sekundy bez skutečného čekání.
                    # floor = naplánovaný backoff (RETRY_DELAYS), hlavička
                    # ho může jen PRODLOUŽIT, nikdy zkrátit.
                    floor       = delay or RETRY_DELAYS[-1]
                    header_wait = int(resp.headers.get("Retry-After", 0) or 0)
                    wait        = max(floor, header_wait)
                    # INFO, ne WARNING: rate limit, co se sám vyřeší retry, je
                    # u tohohle API běžný a očekávaný -- vyčerpání pokusů (dole
                    # v except větvi) je to, co má smysl vypíchnout jako error.
                    log.info(
                        f"Rate limit 429 (pokus {attempt+1}/{len(RETRY_DELAYS)+1}), "
                        f"čekám {wait}s (floor={floor}s, header={header_wait}s)…"
                    )
                    time.sleep(wait)

                    # Adaptivní zpomalení base rate — po sérii 429 zvyš
                    # interval mezi VŠEMI dalšími requesty, ne jen retry
                    # tohoto jednoho. Postupně klesá zpět po úspěších.
                    self._consecutive_429 += 1
                    self._current_delay = min(
                        REQUEST_DELAY_MAX,
                        REQUEST_DELAY_BASE * (1.5 ** self._consecutive_429)
                    )
                    continue

                # Klasifikace se dělá TADY, přímo na status_code, ještě před
                # raise_for_status() -- ne v except větvi níž. Dřív se to
                # (pro 400/404/422 doručené jako HTTPError) muselo znovu
                # vytahovat z výjimky (`getattr(e, "response", None)`), což
                # byla druhá, mírně odlišná cesta ke stejnému rozhodnutí.
                # Jedna kontrola na jednom místě = nemůže se rozjet.
                if is_permanent_status(resp.status_code):
                    body_excerpt = resp.text[:300].replace(chr(10), " ")
                    log.warning(
                        f"AniList natrvalo selhal (nebude se opakovat): "
                        f"HTTP {resp.status_code} {body_excerpt!r}"
                    )
                    return Result.failure(permanent=True)

                if not resp.ok:
                    body_excerpt = resp.text[:300].replace(chr(10), " ")
                    log.warning(
                        f"AniList HTTP {resp.status_code} (pokus {attempt+1}): "
                        f"{body_excerpt!r}"
                    )

                resp.raise_for_status()
                data = resp.json()

                # Úspěšný request — postupně dekrementuj adaptivní zpomalení
                # zpět k base rate (rychlejší než lineární decay by rušil
                # ochranu příliš brzy po sérii 429).
                if self._consecutive_429 > 0:
                    self._consecutive_429 = max(0, self._consecutive_429 - 1)
                    self._current_delay = max(
                        REQUEST_DELAY_BASE,
                        REQUEST_DELAY_BASE * (1.5 ** self._consecutive_429)
                    )

                # GraphQL vrátí errors pole i při HTTP 200. Klasifikováno jako
                # "permanent" -- typicky jde o problém s konkrétními proměnnými
                # (neplatné ID, špatný typ) u KONKRÉTNÍHO requestu, ne o
                # dočasný stav serveru, takže retry se stejnými parametry by
                # dopadl stejně. (Není to jistota pro každý myslitelný GraphQL
                # error, ale je to výrazně častější případ než dočasná chyba.)
                if "errors" in data:
                    err_msgs = [e.get("message", "?") for e in data["errors"]]
                    log.warning(f"GraphQL error(s): {'; '.join(err_msgs)}")
                    return Result.failure(permanent=True)

                return Result.success(data)

            except requests.RequestException as e:
                # Sem už NIKDY nedorazí 400/404/422 -- ty jsou zachyceny výš,
                # ještě před raise_for_status(). Tahle větev vidí jen 5xx
                # (přes raise_for_status) nebo čistě síťovou chybu bez
                # odpovědi vůbec (ConnectionError, Timeout) -- obojí transient.
                detail = ""
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None:
                    try:
                        body = resp_obj.text[:300].replace(chr(10), " ")
                        detail = f" | HTTP {resp_obj.status_code}: {body!r}"
                    except Exception:
                        pass

                if attempt < len(RETRY_DELAYS) - 1:
                    log.warning(f"Request chyba ({e}){detail}, retry za {delay}s…")
                    time.sleep(delay)
                else:
                    log.error(
                        f"AniList request selhal po {len(RETRY_DELAYS)+1} pokusech: "
                        f"{e}{detail}"
                    )
                    return Result.failure(permanent=False)

        # Sem se dostaneme jen když KAŽDÝ pokus skončil na 429 (viz continue
        # výš) -- vyčerpaný rate limit je transient, příští běh (nebo i
        # jen o pár minut později) může uspět.
        return Result.failure(permanent=False)

    # ── Veřejné metody ─────────────────────────────────────────────────────────

    def get_anime(self, mal_id: int) -> dict | None:
        """
        Vrátí AniList data pro anime dle MAL ID.
        Výsledek je cachován — opakované volání je okamžité.

        Vrátí None pokud anime na AniList neexistuje NEBO pokud request
        selhal dočasně (rozdíl viz níže).
        Vrátí {} (prázdný dict) pokud byl uložen jako potvrzeně nenalezený
        -- ať už proto, že AniList explicitně řekl "Media neexistuje", nebo
        proto, že request selhal NATRVALO (400/404/422/GraphQL chyba) a
        retry by stejně nikdy nepomohl.

        Tři různé důvody pro None/{}:
          1. _request() vrátí Result(ok=False, permanent=False) kvůli
             DOČASNÉMU selhání (5xx, timeout, vyčerpaný rate limit) →
             NEUKLÁDÁME do cache, příští spuštění to zkusí znovu
          2. _request() vrátí Result(ok=False, permanent=True) kvůli
             TRVALÉMU selhání (400/404/422/GraphQL chyba) → bezpečně
             cachujeme {} -- retry by nikdy neuspěl
          3. Request uspěl, ale AniList potvrdil že Media neexistuje
             (HTTP 200, data.Media = null) → bezpečně cachujeme {} sentinel
        """
        cached = self._load_cache(mal_id)
        if cached is not None:
            return cached if cached else None   # {} → None

        result = self._request(QUERY_BY_MAL_ID, {"idMal": mal_id})

        if not result.ok:
            if result.permanent:
                # Důvod 2: retry se stejným ID by dopadl stejně.
                self._save_cache(mal_id, {})
                log.info(
                    f"AniList get_anime(MAL {mal_id}): request selhal natrvalo "
                    f"(400/404/422/GraphQL) -- cachuju jako nenalezeno"
                )
            else:
                # Důvod 1: NEUKLÁDÁME.
                log.warning(
                    f"AniList get_anime(MAL {mal_id}): request dočasně selhal, "
                    f"cache beze změny (zkusí se znovu příště)"
                )
            return None

        media = result.data.get("data", {}).get("Media")
        if media:
            self._save_cache(mal_id, media)
            return media
        else:
            # Request byl úspěšný (HTTP 200), AniList potvrdil že titul
            # neexistuje — toto JE bezpečné cachovat natrvalo.
            self._save_cache(mal_id, {})
            log.debug(f"AniList: MAL ID {mal_id} potvrzeně nenalezeno")
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
                # jednotlivé requesty. get_anime() teď sám správně rozlišuje
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
                    elif self._load_cache(mal_id) is None:
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
                    self._save_cache(mid, media)
                    results[mid] = media
                    fetched_mal_ids.add(mid)

            # Ulož sentinel pro nenalezená
            for mal_id in batch:
                if mal_id not in fetched_mal_ids:
                    self._save_cache(mal_id, {})

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
        ck = self.cache_path / f"rec_{mal_id}.json"
        if ck.exists():
            return json.loads(ck.read_text(encoding="utf-8"))
        result = self._request(self.REC_QUERY, {"idMal": mal_id})
        if not result.ok:
            if result.permanent:
                ck.write_text("[]", encoding="utf-8")
                log.info(
                    f"AniList get_recommendations(MAL {mal_id}): request selhal "
                    f"natrvalo (400/404/422/GraphQL) -- cachuju jako prázdné"
                )
            else:
                log.warning(
                    f"AniList get_recommendations(MAL {mal_id}): request dočasně "
                    f"selhal, cache beze změny (zkusí se znovu příště)"
                )
            return []
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
        cached = self._load_cache(fallback_mal_id)
        if cached and isinstance(cached, dict) and cached.get("popularity"):
            return int(cached["popularity"])
        result = self._request(self.QUERY_MEDIA_POPULARITY, {"id": anilist_id})
        if result.ok:
            media = (result.data.get("data") or {}).get("Media") or {}
            return int(media.get("popularity") or 0)
        return 0

    def similar_users_recommendations(
        self,
        liked_mal_ids: list[int],
        min_overlap: int = 4,
        top_users: int = 120,
        seed_count: int = 25,
        users_per_seed: int = 100,
        user_scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """
        User-based collaborative filtering přes AniList.

        Vylepšená logika oproti původní verzi:

          1) SEEDY = MÉNĚ POPULÁRNÍ tvé oblíbené tituly. Sdílení nišového
             titulu je mnohem silnější signál podobného vkusu než sdílení
             blockbusteru — a navíc u nišového titulu pokryje vzorek
             `users_per_seed` uživatelů výrazně větší podíl celé populace,
             takže opakované výskyty (= překryv) reálně vznikají. Tím se
             řeší hlavní problém původní verze (téměř nulový překryv).

          2) VÁŽENÝ PŘEKRYV. Každý sdílený seed přispěje vahou
             ~ -log(popularita) — vzácné tituly váží víc (analogie IDF).
             `min_overlap` se pak vztahuje k počtu sdílených seedů (drží se
             jako tvrdý práh počtu), zatímco řazení uživatelů jde dle váhy.

          3) PODOBNOST = kosinus ratingových vektorů na překryvu, ne jen
             počet. Bere v potaz, ZDA titul hodnotíme podobně.

          4) Stahují se i CUSTOM listy (jinak by se ztratily skryté entries).

        Parametry:
            liked_mal_ids   — tvoje oblíbené (seedy bereme z méně populárních)
            min_overlap     — minimální počet sdílených seedů (tvrdý práh)
            top_users       — kolik nejpodobnějších uživatelů použít
            seed_count      — kolik nejméně populárních seedů použít
            users_per_seed  — kolik uživatelů stáhnout na jeden seed
            user_scores     — {mal_id: tvé_score} pro kosinovou podobnost
                              (volitelné; bez něj se použije jen vážený překryv)

        Vrátí list[dict] {'mal_id', 'score'}. Best-effort.
        """
        from collections import defaultdict

        # 1. MAL ID -> AniList ID (z enrich cache, jež už proběhla)
        mal_to_anilist: dict[int, int] = {}
        for mal_id in liked_mal_ids:
            cached = self._load_cache(mal_id)
            if cached and isinstance(cached, dict) and cached.get("id"):
                mal_to_anilist[mal_id] = cached["id"]

        if not mal_to_anilist:
            log.warning("user-CF: žádné AniList ID v cache – enrich musí proběhnout dřív")
            return []

        # 2. Vyber NEJMÉNĚ POPULÁRNÍ seedy (vzácnost = silnější signál).
        #    Popularitu bereme z cache (uložená při enrichi), fallback dotazem.
        pop_by_mal:      dict[int, int]   = {}
        seed_comm_norm:  dict[int, float]  = {}  # mal_id → AniList averageScore/100
        for mal_id, aid in mal_to_anilist.items():
            pop_by_mal[mal_id] = self._media_popularity(aid, mal_id) or 10**9
            # Komunitní průměr pro seed titul — z hlavní AniList cache
            _cached = self._load_cache(mal_id)
            _avg    = (_cached.get("averageScore") or 0) if _cached else 0
            seed_comm_norm[mal_id] = _avg / 100.0 if _avg else 0.75

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

        # 3. Pro každý seed stáhni uživatele (stránkovaně) i s jejich ratingem.
        #    user_profile[uid] = {seed_mal_id: norm_score}
        #    user_names[uid]   = username (pro výstup)
        user_profile: defaultdict[int, dict[int, float]] = defaultdict(dict)
        user_weight: defaultdict[int, float] = defaultdict(float)
        user_names: dict[int, str] = {}

        for i, (mal_id, anilist_id) in enumerate(seeds):
            ck = f"watchers_{anilist_id}"
            cached_w = self._cf_load(ck)
            if cached_w is not None:
                # Cache hit: použij uložené watchers
                progress(f"  user-CF: seed [{i+1}/{n}] z cache ({len(cached_w)} uživatelů)")
                for uid, uname, raw in cached_w[:users_per_seed]:
                    user_profile[uid][mal_id] = self._norm_score(raw)
                    user_weight[uid] += seed_weight.get(mal_id, 1.0)
                    user_names[uid]   = uname
            else:
                # Cache miss: stáhni a ulož
                progress(f"  user-CF: sbírám uživatele [{i+1}/{n}] (pop≈{pop_by_mal[mal_id]}) ...")
                collected     = 0
                page          = 1
                per_page      = 50
                fetched_w: list = []
                fetch_failed  = False  # True = request selhal, ne legitimní konec
                fetch_failed_permanent = False  # True = 400/404/422/GraphQL -- retry nikdy nepomůže
                hit_page_cap  = False  # True = narazili jsme na MAX_WATCHER_PAGE
                no_progress_streak = 0  # počet stránek po sobě bez nového uživatele

                while collected < users_per_seed:
                    # Strop na page — AniList odmítne page*perPage > 5000
                    # bez ohledu na to, kolik dat skutečně existuje.
                    # Nejčastější příčina: sort=SCORE_DESC řadí nehodnocené
                    # záznamy (score=0) na konec; pokud titul má málo
                    # skutečně hodnotících diváků, smyčka by jinak stránkovala
                    # donekonečna a hledala hodnocení, která už nepřijdou.
                    if page > MAX_WATCHER_PAGE:
                        hit_page_cap = True
                        break

                    result = self._request(
                        self.QUERY_USERS_BY_MEDIA,
                        {"mediaId": anilist_id, "page": page, "perPage": per_page},
                    )
                    if not result.ok:
                        # request selhal (síť/400/500/timeout) — odliš od
                        # legitimního konce výsledků (prázdné entries níže).
                        # `result.permanent` je součástí TÉTO návratové
                        # hodnoty, ne odděleného stavu klienta -- nejde ho
                        # přečíst pozdě ani si ho nechat přepsat jiným
                        # mezitímním voláním.
                        fetch_failed = True
                        fetch_failed_permanent = result.permanent
                        break
                    pg      = (result.data.get("data") or {}).get("Page") or {}
                    entries = pg.get("mediaList") or []
                    if not entries:
                        break  # legitimní konec — žádná další data

                    collected_before = collected
                    for entry in entries:
                        raw      = entry.get("score") or 0
                        if not raw:
                            continue
                        user_obj = entry.get("user") or {}
                        uid      = user_obj.get("id")
                        uname    = user_obj.get("name", "")
                        if not uid:
                            continue
                        # Použij vždy — i částečné výsledky obohatí tento běh,
                        # i když je nakonec neuložíme do cache (viz níže).
                        user_profile[uid][mal_id] = self._norm_score(raw)
                        user_weight[uid] += seed_weight.get(mal_id, 1.0)
                        user_names[uid]   = uname
                        fetched_w.append([uid, uname, raw])
                        collected += 1

                    # Sleduj stránky bez jakéhokoliv nového ohodnoceného
                    # uživatele — pokud jich je moc po sobě, dál stránkovat
                    # je marné (jsme už v dlouhém ocasu nehodnocených
                    # záznamů) a jen to plýtvá requesty a riskuje page cap.
                    if collected == collected_before:
                        no_progress_streak += 1
                        if no_progress_streak >= NO_PROGRESS_PAGE_LIMIT:
                            break
                    else:
                        no_progress_streak = 0

                    if not (pg.get("pageInfo") or {}).get("hasNextPage"):
                        break
                    page += 1

                if hit_page_cap:
                    log.warning(
                        f"user-CF seed MAL {mal_id} (AL {anilist_id}): "
                        f"narazil na MAX_WATCHER_PAGE={MAX_WATCHER_PAGE} "
                        f"(AniList limit page×perPage≤5000) — "
                        f"{len(fetched_w)} uživatelů použito, titul má "
                        f"pravděpodobně mnoho nehodnocených záznamů"
                    )

                if fetch_failed and not fetch_failed_permanent:
                    # Dočasné selhání (5xx/timeout/vyčerpaný rate limit) --
                    # NEUKLÁDÁME do cache. Částečné výsledky (fetched_w) jsme
                    # už použili výše pro tento běh, ale příští spuštění má
                    # dostat šanci stáhnout zbytek znovu, ne tvářit se že seed
                    # má jen {len(fetched_w)} uživatelů natrvalo.
                    log.warning(
                        f"user-CF seed MAL {mal_id} (AL {anilist_id}): "
                        f"request dočasně selhal po {len(fetched_w)} získaných "
                        f"uživatelích (stránka {page}) — cache NEUKLÁDÁM"
                    )
                else:
                    # Buď žádné selhání, nebo selhání NATRVALO (400/404/422/
                    # GraphQL chyba) -- retry se stejnými parametry by dopadl
                    # stejně, takže i částečný/prázdný výsledek je bezpečné
                    # zacachovat jako konečný.
                    self._cf_save(ck, fetched_w)
                    if fetch_failed_permanent:
                        log.info(
                            f"user-CF seed MAL {mal_id} (AL {anilist_id}): "
                            f"request selhal natrvalo (400/404/422/GraphQL) -- "
                            f"cachuju {len(fetched_w)} částečných výsledků jako konečné"
                        )
                    log.debug(
                        f"user-CF seed MAL {mal_id}: {len(fetched_w)} "
                        f"uživatelů staženo a cachováno"
                    )

        n_failed_seeds = sum(
            1 for mal_id, aid in seeds
            if self._cf_load(f"watchers_{aid}") is None
        )
        print(
            f"  user-CF: sbírám uživatele [{n}/{n}] ... hotovo  "
            f"({n_failed_seeds} seedů bez cache kvůli chybám, zkusí se příště)"
            if n_failed_seeds else
            f"  user-CF: sbírám uživatele [{n}/{n}] ... hotovo                       "
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
        similar_users = [uid for uid, _ in candidates[:top_users]]
        sim_by_uid    = dict(candidates[:top_users])

        status(f"  user-CF: {len(similar_users)} podobných uživatelů "
               f"(z {len(user_profile)} kandidátů), stahuji jejich seznamy ...")

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

        m = len(similar_users)
        _ul_cache_hits   = 0
        _ul_fetched      = 0
        _ul_failed       = 0
        for j, uid in enumerate(similar_users):
            sim  = max(0.0, sim_by_uid.get(uid, 0.0))
            name = user_names.get(uid, str(uid))
            ck   = f"userlist_{uid}"

            cached_ul = self._cf_load(ck)
            if cached_ul is not None:
                # Cache hit — [fmt, [[mid, raw, avg_raw, title], ...]]
                progress(f"  user-CF: seznam [{j+1}/{m}] z cache")
                _fmt, raw_entries = cached_ul[0], cached_ul[1]
                _ul_cache_hits += 1
            else:
                # Cache miss — stáhni a zpracuj
                progress(f"  user-CF: stahuji seznam [{j+1}/{m}] ...")
                result     = self._request(self.QUERY_USER_ANIMELIST, {"userId": uid})
                if not result.ok:
                    _ul_failed += 1
                    if result.permanent:
                        # 400/404/422/GraphQL chyba (např. neplatné userId,
                        # smazaný účet) -- retry se stejným uid nikdy neuspěje,
                        # takže je bezpečné zacachovat "žádná data" natrvalo.
                        self._cf_save(ck, ["UNKNOWN", []])
                        log.info(
                            f"user-CF userlist uid={uid} ({name}): request "
                            f"selhal natrvalo (400/404/422/GraphQL) -- "
                            f"cachuju jako prázdný, uživatel vynechán"
                        )
                    else:
                        # Dočasné selhání (5xx/timeout/vyčerpaný rate limit) —
                        # NEUKLÁDÁME, tento uživatel se vynechá v tomto běhu,
                        # ale příští spuštění to zkusí znovu (cache zůstává
                        # prázdná = cache miss).
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
                self._cf_save(ck, [_fmt, raw_entries])

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

        progress_done(f"  user-CF: stahuji seznamy [{m}/{m}] ... hotovo")
        log.info(
            f"user-CF userlist: {_ul_cache_hits} z cache, "
            f"{_ul_fetched} staženo, {_ul_failed} selhalo "
            f"(z {m} podobných uživatelů)"
        )
        if _ul_failed:
            print(
                f"  user-CF: {_ul_failed} uživatelů selhalo (viz log) "
                f"— zkusí se znovu při příštím spuštění"
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
            import math as _math
            n_bonus = 0.03 * _math.log1p(max(0, rec_count[mid] - 2))
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
        log.info(f"user-CF: {len(out)} kandidátů z {len(similar_users)} uživatelů")
        return out[:300]
