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
          user { id }
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
            media { idMal popularity }
          }
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
        result = self._post(self.QUERY_MEDIA_POPULARITY, {"id": anilist_id})
        if result:
            media = (result.get("data") or {}).get("Media") or {}
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
        pop_by_mal: dict[int, int] = {}
        for mal_id, aid in mal_to_anilist.items():
            pop_by_mal[mal_id] = self._media_popularity(aid, mal_id) or 10**9

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
        user_profile: defaultdict[int, dict[int, float]] = defaultdict(dict)
        user_weight: defaultdict[int, float] = defaultdict(float)

        for i, (mal_id, anilist_id) in enumerate(seeds):
            print(f"  user-CF: sbírám uživatele [{i+1}/{n}] (pop≈{pop_by_mal[mal_id]}) ...",
                  end="\r")
            collected = 0
            page = 1
            per_page = 50
            while collected < users_per_seed:
                result = self._post(
                    self.QUERY_USERS_BY_MEDIA,
                    {"mediaId": anilist_id, "page": page, "perPage": per_page},
                )
                if not result:
                    break
                pg = (result.get("data") or {}).get("Page") or {}
                entries = pg.get("mediaList") or []
                if not entries:
                    break
                for entry in entries:
                    raw = entry.get("score") or 0
                    if not raw:
                        continue
                    uid = (entry.get("user") or {}).get("id")
                    if not uid:
                        continue
                    user_profile[uid][mal_id] = self._norm_score(raw)
                    user_weight[uid] += seed_weight.get(mal_id, 1.0)
                    collected += 1
                if not (pg.get("pageInfo") or {}).get("hasNextPage"):
                    break
                page += 1

        print(f"  user-CF: sbírám uživatele [{n}/{n}] ... hotovo                       ")

        if not user_profile:
            log.info("user-CF: žádní uživatelé nenalezeni")
            return []

        # 4. Podobnost uživatele = kosinus ratingových vektorů na překryvu
        #    (pokud máme user_scores), váženo `seed_weight`. Tvrdý práh na
        #    počet sdílených seedů zůstává `min_overlap`.
        def cosine(uid: int) -> float:
            shared = user_profile[uid]
            if user_scores:
                num = den_u = den_o = 0.0
                for mid, their in shared.items():
                    mine = self._norm_score((user_scores.get(mid) or 0) * 10)
                    w = seed_weight.get(mid, 1.0)
                    num += w * mine * their
                    den_u += w * mine * mine
                    den_o += w * their * their
                if den_u <= 0 or den_o <= 0:
                    return 0.0
                return num / math.sqrt(den_u * den_o)
            # bez mých skóre: použij váženou velikost překryvu
            return user_weight[uid]

        candidates = [
            (uid, cosine(uid))
            for uid, prof in user_profile.items()
            if len(prof) >= min_overlap
        ]

        if not candidates:
            best = max(len(p) for p in user_profile.values())
            print(f"  user-CF: nikdo nesplnil min_overlap={min_overlap} "
                  f"(max překryv: {best}) – zkus snížit min_overlap")
            log.info(f"user-CF: max překryv {best} < min_overlap {min_overlap}")
            return []

        candidates.sort(key=lambda x: -x[1])
        similar_users = [uid for uid, _ in candidates[:top_users]]
        sim_by_uid = dict(candidates[:top_users])

        print(f"  user-CF: {len(similar_users)} podobných uživatelů "
              f"(z {len(user_profile)} kandidátů), stahuji jejich seznamy ...")

        # 5. Stáhni listy podobných uživatelů a agreguj kandidáty.
        #    Příspěvek = similarita_uživatele × jeho_norm_rating, a navíc
        #    vážíme vzácnost (méně populární doporučený titul = zajímavější).
        liked_set = set(liked_mal_ids)
        agg_num: defaultdict[int, float] = defaultdict(float)   # Σ sim×rating
        agg_sim: defaultdict[int, float] = defaultdict(float)   # Σ sim
        rec_count: defaultdict[int, int] = defaultdict(int)
        pop_seen: dict[int, int] = {}

        m = len(similar_users)
        for j, uid in enumerate(similar_users):
            print(f"  user-CF: stahuji seznamy [{j+1}/{m}] ...", end="\r")
            result = self._post(self.QUERY_USER_ANIMELIST, {"userId": uid})
            if not result:
                continue
            collection = (result.get("data") or {}).get("MediaListCollection") or {}
            sim = max(0.0, sim_by_uid.get(uid, 0.0))
            for lst in collection.get("lists", []):
                for entry in lst.get("entries", []):
                    status = entry.get("status")
                    if status in ("PLANNING", "DROPPED"):
                        continue
                    raw = entry.get("score") or 0
                    if not raw:
                        continue
                    media = entry.get("media") or {}
                    mid = media.get("idMal")
                    if not mid or mid in liked_set:
                        continue
                    norm = self._norm_score(raw)
                    if norm < 0.6:           # ignoruj jimi spíš odmítnuté
                        continue
                    agg_num[mid] += sim * norm
                    agg_sim[mid] += sim
                    rec_count[mid] += 1
                    if media.get("popularity"):
                        pop_seen[mid] = int(media["popularity"])

        print(f"  user-CF: stahuji seznamy [{m}/{m}] ... hotovo            ")

        # 6. Skóre kandidáta: vážený průměr ratingu (sim jako váha) × důvěra
        #    v počet doporučitelů. Bez populárního zkreslení – netlačíme
        #    blockbustery, ty stejně přijdou přes item-based větev.
        out = []
        for mid in agg_num:
            if rec_count[mid] < 2:           # aspoň 2 nezávislí doporučitelé
                continue
            weighted_rating = agg_num[mid] / agg_sim[mid] if agg_sim[mid] else 0.0
            confidence = math.log1p(rec_count[mid])
            score = weighted_rating * confidence
            out.append({"mal_id": mid, "score": score, "n_users": rec_count[mid]})

        out.sort(key=lambda x: -x["score"])
        log.info(f"user-CF: {len(out)} kandidátů z {len(similar_users)} uživatelů")
        return out[:300]
