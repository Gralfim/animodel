"""
usercf.py — user-based CF: hledání „senpai" přes plné seznamy.

Cíl (původní očekávání modulu): najít PÁR (max desítky) uživatelů se silně
podobným vkusem, kteří viděli a ohodnotili VÍC anime než já — a doporučení
brát od nich. Ne statisticky průměrovat stovky slabě ověřených anonymů.

Čtyři fáze:

  1. DISCOVERY (osvědčené, zůstává): z nejméně populárních mých ohodnocených
     titulů posbírat kandidátní uživatele přes watchers stránky. Sdílení
     nišového titulu je silný signál; vzorek ale slouží JEN k prioritizaci,
     ne k měření podobnosti (to byla hlavní slabina staré verze).
  2. PLNÉ SEZNAMY: pro kandidáty v pořadí priority stáhnout kompletní
     seznamy (cache `userlist_{uid}`; privátní/smazané účty jsou trvale
     zacachované jako prázdné a přeskakují se bez ztráty místa v poolu).
  3. SENPAI SKÓRE na plném překryvu: Pearson na komunitně-relativních
     odchylkách přes VŠECHNY tituly, které máme ohodnocené oba, smrštěný
     `n/(n+K)` — málo překryvu = málo důvěry. Stejná filozofie jako zbytek
     modelu (baseline/efekty/interakce), žádné ad-hoc váhy.
  4. DOPORUČENÍ: diferenciální agregace (komunita + vážená odchylka senpaie
     od jeho osobního průměru) jen přes vybrané senpai; práh ≥ 2 nezávislí.

Modul je čistá logika nad úzkým klientským kontraktem (testovatelné bez
sítě): `_cached_media`, `_media_popularity`, `get_watcher_entries`,
`get_user_animelist`.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from .sources import progress, progress_done, status

log = logging.getLogger(__name__)

# AniList ukládá skóre v uživatelově formátu; bez známého formátu heuristika
# (>10 → stupnice 0-100, jinak 0-10).
_DIVISORS = {
    "POINT_100":        100.0,
    "POINT_10_DECIMAL": 10.0,
    "POINT_10":         10.0,
    "POINT_5":          5.0,
    "POINT_3":          3.0,
}


def _norm_score(raw: float, fmt: str | None = None) -> float:
    """Normalizace skóre na 0–1 dle formátu uživatele (fallback heuristika)."""
    if raw <= 0:
        return 0.0
    div = _DIVISORS.get(fmt or "")
    if div:
        return max(0.0, min(1.0, raw / div))
    return max(0.0, min(1.0, raw / 100.0 if raw > 10 else raw / 10.0))


@dataclass
class Senpai:
    """Uživatel s ověřeně podobným vkusem (na plném překryvu seznamů)."""
    uid: int
    name: str
    similarity: float      # Pearson na komunitně-relativních odchylkách
    score: float           # similarity · n/(n+K) · penalizace_pokrytí
    overlap: int           # kolik titulů máme ohodnocených OBA
    n_rated: int           # kolik má ohodnoceno celkem
    n_novel: int           # kolik z jeho titulů já nemám shlédnutých
    personal_avg: float    # jeho osobní průměr (0–1) pro diferenciální skóre
    fav_covered: int = 0   # kolik mých oblíbených má ohodnocených nebo na PTW
    fav_total: int = 0     # kolik oblíbených mám celkem
    penalty: float = 1.0   # faktor za nepokryté oblíbené (1.0 = bez penalizace)
    fmt: str = "UNKNOWN"
    entries: list = field(default_factory=list, repr=False)  # [[mid, raw, avg_raw, title], ...]

    @property
    def fav_coverage(self) -> float:
        """Podíl mých oblíbených titulů, které senpai zná (0–1)."""
        return self.fav_covered / self.fav_total if self.fav_total else 1.0


# ── Fáze 1: discovery ────────────────────────────────────────────────────

def discover_candidates(client, user_scores: dict[int, float], *,
                        seed_count: int, users_per_seed: int,
                        min_sample_overlap: int,
                        exclude_users: set[str] | None = None,
                        ) -> list[tuple[int, str, float, int]]:
    """
    Kandidátní uživatelé z watchers stránek nejméně populárních mých titulů.

    Vrací [(uid, jméno, IDF_váha, počet_sdílených_seedů), ...] seřazené
    sestupně podle váhy (vzácnější sdílené tituly váží víc). Kvalifikace:
    aspoň `min_sample_overlap` sdílených seedů -- jediný společný nišový
    titul může být náhoda, dva už jsou vzorec.

    `exclude_users` = AniList jména (case-insensitive), která se přeskočí
    už tady -- typicky TVŮJ vlastní účet: import vlastního MAL seznamu má
    podobnost 1.00 a jako senpai je k ničemu (doporučil by ti jen to, co
    už máš). Filtruje se v discovery, ať se jeho plný seznam ani nestahuje.
    """
    excluded = {n.strip().lower() for n in (exclude_users or set()) if n and n.strip()}
    mal_to_anilist: dict[int, int] = {}
    for mal_id in user_scores:
        media = client._cached_media(mal_id)
        if media and media.get("id"):
            mal_to_anilist[mal_id] = media["id"]
    if not mal_to_anilist:
        log.warning("user-CF: žádné AniList ID v cache – enrich musí proběhnout dřív")
        return []

    pop_by_mal = {mal_id: (client._media_popularity(aid, mal_id) or 10**9)
                  for mal_id, aid in mal_to_anilist.items()}
    ranked = sorted(mal_to_anilist.items(), key=lambda kv: pop_by_mal[kv[0]])
    seeds = ranked[:seed_count]

    # IDF-like váha seedu: vzácný titul váží víc (log-inverze popularity)
    raw_w = {mal_id: math.log10((pop_by_mal[mal_id] or 1) + 10) for mal_id, _ in seeds}
    max_w = max(raw_w.values()) if raw_w else 1.0
    seed_weight = {m: (max_w - w + 0.5) for m, w in raw_w.items()}

    weight: defaultdict[int, float] = defaultdict(float)
    shared: defaultdict[int, int] = defaultdict(int)
    names: dict[int, str] = {}
    for i, (mal_id, anilist_id) in enumerate(seeds):
        progress(f"  user-CF discovery: seed [{i+1}/{len(seeds)}] (pop≈{pop_by_mal[mal_id]}) …")
        seen_here: set[int] = set()
        for uid, uname, _raw in client.get_watcher_entries(anilist_id, users_per_seed):
            if uid in seen_here or (uname or "").strip().lower() in excluded:
                continue
            seen_here.add(uid)
            weight[uid] += seed_weight.get(mal_id, 1.0)
            shared[uid] += 1
            names[uid] = uname
    progress_done(f"  user-CF discovery: {len(weight)} kandidátů z {len(seeds)} seedů"
                  + (f" (vyloučeno: {', '.join(sorted(excluded))})" if excluded else ""))

    out = [(uid, names.get(uid, str(uid)), weight[uid], shared[uid])
           for uid in weight if shared[uid] >= min_sample_overlap]
    out.sort(key=lambda x: -x[2])
    return out


# ── Fáze 2+3: plné seznamy a senpai skóre ────────────────────────────────

def evaluate_candidate(uid: int, name: str, userlist: dict,
                       my_scores: dict[int, float], watched_ids: set[int],
                       shrink_k: float, favorites: set[int] | None = None,
                       fav_miss_penalty: float = 0.0) -> Senpai:
    """
    Senpai metriky JEDNOHO kandidáta z jeho plného seznamu.

    `favorites` = mé nejoblíbenější tituly (mal_id). Senpai, který je nemá
    ohodnocené ani na PTW, dostane lehkou srážku skóre úměrnou podílu
    nepokrytých (`fav_miss_penalty` = srážka při nulovém pokrytí):
    „senpai", co neviděl většinu toho, co miluju, je slabší průvodce, i
    když na společném průniku koreluje pěkně. Podobnost (Pearson) zůstává
    nedotčená -- penalizace je vědomě až nad ní, ať jde v reportu vidět
    obojí zvlášť.
    """
    fmt = userlist.get("fmt") or "UNKNOWN"
    entries = userlist.get("entries") or []
    planning = set(userlist.get("planning") or [])

    my_diffs: list[float] = []
    their_diffs: list[float] = []
    novel = 0
    norms: list[float] = []
    rated_ids: set[int] = set()
    for mid, raw, avg_raw, _title in entries:
        norm = _norm_score(raw, fmt)
        norms.append(norm)
        rated_ids.add(mid)
        if mid not in watched_ids:
            novel += 1
        my_raw = my_scores.get(mid)
        if not my_raw:
            continue
        c_norm = avg_raw / 100.0
        my_diffs.append(my_raw / 10.0 - c_norm)
        their_diffs.append(norm - c_norm)

    n = len(my_diffs)
    similarity = _pearson(my_diffs, their_diffs)
    base = similarity * (n / (n + shrink_k)) if n else 0.0

    favs = favorites or set()
    covered = sum(1 for mid in favs if mid in rated_ids or mid in planning)
    penalty = 1.0
    if favs and fav_miss_penalty:
        penalty = 1.0 - fav_miss_penalty * (1.0 - covered / len(favs))

    return Senpai(
        uid=uid, name=name,
        similarity=similarity,
        score=base * penalty,
        overlap=n,
        n_rated=len(entries),
        n_novel=novel,
        personal_avg=(sum(norms) / len(norms)) if norms else 0.7,
        fav_covered=covered, fav_total=len(favs), penalty=penalty,
        fmt=fmt, entries=entries,
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx, my_ = sum(xs) / n, sum(ys) / n
    xc = [x - mx for x in xs]
    yc = [y - my_ for y in ys]
    num = sum(a * b for a, b in zip(xc, yc))
    den = math.sqrt(sum(a * a for a in xc)) * math.sqrt(sum(b * b for b in yc))
    return num / den if den > 1e-9 else 0.0


def evaluate_candidates(client, candidates: list[tuple[int, str, float, int]],
                        my_scores: dict[int, float], watched_ids: set[int], *,
                        candidate_pool: int, shrink_k: float,
                        favorites: set[int] | None = None,
                        fav_miss_penalty: float = 0.0,
                        scan_budget_factor: float = 3.0) -> list[Senpai]:
    """
    Projde kandidáty v pořadí priority a vyhodnotí `candidate_pool`
    POUŽITELNÝCH plných seznamů (privátní/smazané/dočasně selhané se
    přeskočí bez ztráty místa; scan budget je pojistka proti extrémnímu
    podílu nepoužitelných).
    """
    evaluated: list[Senpai] = []
    max_attempts = min(len(candidates),
                       max(candidate_pool, int(candidate_pool * scan_budget_factor)))
    tried = 0
    for uid, name, _w, _shared in candidates:
        if len(evaluated) >= candidate_pool or tried >= max_attempts:
            break
        tried += 1
        progress(f"  user-CF: plný seznam [{len(evaluated)}/{candidate_pool}] "
                 f"(zkuseno {tried}/{max_attempts}) …")
        userlist = client.get_user_animelist(uid)
        if userlist is None:
            continue   # privátní/smazaný (trvale) nebo dočasné selhání
        evaluated.append(evaluate_candidate(
            uid, name, userlist, my_scores, watched_ids, shrink_k,
            favorites=favorites, fav_miss_penalty=fav_miss_penalty))
    progress_done(f"  user-CF: vyhodnoceno {len(evaluated)} plných seznamů "
                  f"({tried} kandidátů zkuseno)")
    if evaluated and len(evaluated) < candidate_pool and tried >= max_attempts:
        log.warning(
            f"user-CF: dosažen scan budget ({max_attempts}) s "
            f"{len(evaluated)}/{candidate_pool} použitelnými seznamy"
        )
    return evaluated


def select_senpai(evaluated: list[Senpai], *, senpai_count: int,
                  min_full_overlap: int) -> list[Senpai]:
    """Top `senpai_count` podle smrštěného skóre; požadavky: dostatečný
    plný překryv a kladná podobnost (záporně korelovaný „anti-senpai"
    není doporučovatel)."""
    ok = [s for s in evaluated
          if s.overlap >= min_full_overlap and s.score > 0]
    ok.sort(key=lambda s: -s.score)
    return ok[:senpai_count]


# ── Fáze 4: doporučení od senpai ─────────────────────────────────────────

def recommend_from_senpai(senpai: list[Senpai], rated_ids: set[int],
                          *, min_raters: int = 2) -> list[dict]:
    """
    Diferenciální agregace přes vybrané senpai:
        diff  = jeho_norm − jeho_osobní_průměr
        cf    = komunita + Σ(score·diff)/Σ(score) + malý bonus za počet
    Vyloučeny jen MNOU OHODNOCENÉ tituly -- shlédnuté-neohodnocené v surovém
    výstupu zůstávají (CF report je označí štítkem „už shlédnuto"; z
    finálních žebříčků je vyřadí bump() přes seen_ids).
    """
    agg_diff: defaultdict[int, float] = defaultdict(float)
    agg_score: defaultdict[int, float] = defaultdict(float)
    rec_count: defaultdict[int, int] = defaultdict(int)
    comm_norm: dict[int, float] = {}
    title_store: dict[int, str] = {}
    raters: defaultdict[int, list] = defaultdict(list)

    for s in senpai:
        for mid, raw, avg_raw, title in s.entries:
            if mid in rated_ids:
                continue
            norm = _norm_score(raw, s.fmt)
            agg_diff[mid] += s.score * (norm - s.personal_avg)
            agg_score[mid] += s.score
            rec_count[mid] += 1
            comm_norm[mid] = avg_raw / 100.0
            if title and mid not in title_store:
                title_store[mid] = title
            raters[mid].append((s.score, s.name))

    out = []
    for mid, total in agg_diff.items():
        if rec_count[mid] < min_raters:
            continue
        c = comm_norm.get(mid, 0.5)
        w_diff = total / agg_score[mid] if agg_score[mid] else 0.0
        n_bonus = 0.03 * math.log1p(max(0, rec_count[mid] - 2))
        cf_raw = max(0.0, c + w_diff + n_bonus)
        top_r = sorted(raters[mid], key=lambda x: -x[0])[:5]
        out.append({
            "mal_id":     mid,
            "score":      cf_raw * 10.0,   # pro bump()
            "cf_score":   cf_raw * 10.0,
            "community":  c * 10.0,
            "diff":       w_diff * 10.0,
            "n_users":    rec_count[mid],
            "top_raters": [(name, round(sc, 3)) for sc, name in top_r],
            "title":      title_store.get(mid, ""),
        })
    out.sort(key=lambda x: -x["cf_score"])
    return out


# ── Orchestrátor ─────────────────────────────────────────────────────────

def find_senpai_recommendations(client, user_scores: dict[int, float],
                                watched_ids: set[int], rc) -> tuple[list[Senpai], list[dict]]:
    """
    Celý senpai pipeline. `user_scores` = {mal_id: moje_známka 1-10},
    `watched_ids` = vše shlédnuté (pro novelty metriku), `rc` = RecommendCfg.
    """
    candidates = discover_candidates(
        client, user_scores,
        seed_count=rc.user_cf_seed_count,
        users_per_seed=rc.user_cf_users_per_seed,
        min_sample_overlap=rc.user_cf_min_sample_overlap,
        exclude_users=set(rc.user_cf_exclude_users or []),
    )
    status(f"  user-CF: {len(candidates)} kvalifikovaných kandidátů "
           f"(≥{rc.user_cf_min_sample_overlap} sdílené nišové tituly)")
    if not candidates:
        return [], []

    favorites = {mid for mid, sc in user_scores.items()
                 if sc >= rc.user_cf_fav_score}
    if favorites and rc.user_cf_fav_miss_penalty:
        status(f"  user-CF: {len(favorites)} oblíbených titulů "
               f"(známka ≥{rc.user_cf_fav_score:g}) -- kdo je nezná ani z PTW, "
               f"ztrácí až {rc.user_cf_fav_miss_penalty:.0%} skóre")
    evaluated = evaluate_candidates(
        client, candidates, user_scores, watched_ids,
        candidate_pool=rc.user_cf_candidate_pool,
        shrink_k=rc.user_cf_shrink_k,
        favorites=favorites,
        fav_miss_penalty=rc.user_cf_fav_miss_penalty,
    )
    senpai = select_senpai(
        evaluated,
        senpai_count=rc.user_cf_senpai_count,
        min_full_overlap=rc.user_cf_min_full_overlap,
    )
    if not senpai:
        best = max((s.overlap for s in evaluated), default=0)
        status(f"  user-CF: žádný senpai nesplnil požadavky "
               f"(max plný překryv {best} < {rc.user_cf_min_full_overlap}?) "
               f"— zvaž nižší user_cf_min_full_overlap")
        return [], []

    status(f"  user-CF: vybráno {len(senpai)} senpai "
           f"(skóre {senpai[0].score:.2f}–{senpai[-1].score:.2f}, "
           f"překryv {min(s.overlap for s in senpai)}–{max(s.overlap for s in senpai)})")
    recs = recommend_from_senpai(senpai, rated_ids=set(user_scores))
    return senpai, recs
