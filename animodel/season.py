"""
season.py — doporučení pro aktuální vysílanou sezónu.

Dvě sekce:
  1. POKRAČOVÁNÍ tvých sérií -- sezónní tituly, jejichž franšíza obsahuje
     titul, který hodnotíš ≥ season_min_prequel_score. Řazeno podle mé
     známky předchozí řady (chceš vědět, že oblíbená série pokračuje).
  2. NOVÉ tituly pro tebe -- zbytek sezóny řazený podle taste_fit (shoda
     obsahu s mým modelem vkusu). CF složky se ZÁMĚRNĚ nepoužívají: čerstvě
     vysílané série nemají graf podobnosti ani hodnocení od senpai, takže
     jediný spolehlivý signál od 1. dílu je obsahová shoda. Sezónní titul,
     který je pokračováním série mimo můj seznam (nebo <7), sem patří a
     označí se poznámkou "předchozí díly neviděny".

U každého titulu se zobrazí datum posledního dílu (dopočítané z AniList
nextAiringEpisode + episodes), pokud je k dispozici.

Čistá orchestrace nad Enricher/model/klienty -- testovatelné bez sítě.
"""
from __future__ import annotations

import datetime as _dt
import logging

from .recommend import Recommendation
from .series import build_series_groups

log = logging.getLogger(__name__)

_SEASONS = ["winter", "spring", "summer", "fall"]
# měsíc → sezóna (MAL konvence: zima = leden-březen, …)
_MONTH_SEASON = {1: "winter", 2: "winter", 3: "winter",
                 4: "spring", 5: "spring", 6: "spring",
                 7: "summer", 8: "summer", 9: "summer",
                 10: "fall", 11: "fall", 12: "fall"}


def current_season(today: _dt.date | None = None) -> tuple[int, str]:
    """(rok, sezóna) pro dané datum; default dnes."""
    d = today or _dt.date.today()
    return d.year, _MONTH_SEASON[d.month]


def parse_season_arg(args: list[str] | None) -> tuple[int, str]:
    """
    Parsuje CLI argument sezóny: ["2026", "summer"] nebo ["summer", "2026"]
    nebo prázdné (→ aktuální). Nerozpoznané → aktuální sezóna s warningem.
    """
    if not args:
        return current_season()
    year, season = None, None
    for tok in args:
        t = tok.strip().lower()
        if t.isdigit() and len(t) == 4:
            year = int(t)
        elif t in _SEASONS:
            season = t
    if year and season:
        return year, season
    log.warning(f"nerozpoznaný --season argument {args!r}, používám aktuální sezónu")
    return current_season()


def finale_date(airing: dict | None) -> str | None:
    """
    ISO datum posledního dílu z AniList airing dat, nebo None.

    - RELEASING + známý počet epizod + nextAiringEpisode → dopočítá:
      poslední = airing(next_ep) + (episodes − next_ep)·7 dní.
    - FINISHED → end_date (už doběhlo).
    - jinak (neznámý počet epizod, film bez rozvrhu) → None.
    """
    if not airing:
        return None
    status = airing.get("status")
    if status == "FINISHED" and airing.get("end_date"):
        y, m, d = airing["end_date"]
        return f"{y:04d}-{m:02d}-{d:02d}"
    eps = airing.get("episodes")
    next_ep = airing.get("next_ep")
    next_at = airing.get("next_airing_at")
    if eps and next_ep and next_at:
        remaining = eps - next_ep
        if remaining < 0:
            remaining = 0
        ts = next_at + remaining * 7 * 86400
        return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat()
    return None


def _make_rec(model, en, ptw: bool, airing: dict | None,
              season_note: str | None) -> Recommendation:
    """Enriched titul → Recommendation s predikcí, 'proč' a airing údaji."""
    pred, lo, hi, contribs = model.predict(en.attrs, en.community)
    return Recommendation(
        mal_id=en.mal_id, title=en.title, title_en=en.title_en,
        community=en.community, pred=pred, pred_lo=lo, pred_hi=hi,
        taste_fit=model._raw_resid_pred(en.attrs), cf_signal=0.0,
        composite=0.0, ptw=ptw, cluster_name="",
        why=contribs[:6], cf_seeds=[], synopsis=en.synopsis,
        sources=["season"],
        finale_date=finale_date(airing),
        broadcast=(airing or {}).get("broadcast"),
        airing_status=(airing or {}).get("status"),
        season_note=season_note,
    )


def build_season_view(model, enricher, my_scores: dict[int, float],
                      watched_ids: set[int], ptw_ids: set[int],
                      year: int, season: str, rc,
                      show_progress: bool = True
                      ) -> tuple[list[Recommendation], list[Recommendation]]:
    """
    Vrátí (pokračování_mých_sérií, nové_tituly) pro danou sezónu.

    my_scores   = {mal_id: moje známka} (ohodnocené tituly)
    watched_ids = vše shlédnuté (Completed/Watching/… -- vyloučí se ze sezóny)
    ptw_ids     = plan-to-watch (jen se označí)
    """
    jikan = enricher.jikan
    if jikan is None:
        log.warning("--season potřebuje MAL API (Jikan/Tenrai); je vypnuté (--no-jikan)")
        return [], []

    raw = jikan.get_season(year, season)
    # jen skutečné série/filmy, ne hudební klipy; vyluč už shlédnuté
    season_ids = [a["mal_id"] for a in raw
                  if a.get("mal_id") and (a.get("type") or "") != "Music"
                  and a["mal_id"] not in watched_ids]
    if not season_ids:
        return [], []
    broadcast_by_id = {a["mal_id"]: ((a.get("broadcast") or {}).get("day"))
                       for a in raw if a.get("mal_id")}

    # obohať sezónní tituly (atributy/komunita/relations) + airing data
    enr = enricher.enrich_ids(season_ids, show_progress=show_progress)
    airing = {}
    if enricher.anilist:
        airing = enricher.anilist.get_airing_batch(list(enr.keys()))
    for mid, a in airing.items():
        if mid in broadcast_by_id and broadcast_by_id[mid]:
            a["broadcast"] = broadcast_by_id[mid]

    # franšízové skupiny nad (mé ohodnocené ∪ sezóna ∪ tituly REFERENCOVANÉ
    # z relací sezónních titulů). Ty referencované řady sám nemusím mít v
    # seznamu -- ale bez nich by union-find nepoznal, že je sezónní titul
    # pokračováním něčeho, co jsem nikdy neviděl.
    my_enr = enricher.enrich_ids(list(my_scores), show_progress=False)
    rel_data = enricher.relations_data({**my_enr, **enr})
    referenced = set()
    for rd in rel_data.values():
        for rel in rd.get("relations", []):
            for e in rel.get("entry", []):
                if e.get("type") == "anime" and e.get("mal_id"):
                    referenced.add(e["mal_id"])
    all_ids = set(my_enr) | set(enr) | referenced
    groups = build_series_groups(list(all_ids), rel_data)
    root_of: dict[int, int] = {}
    for root, members in groups.items():
        for m in members:
            root_of[m] = root
    group_size = {root: len(members) for root, members in groups.items()}
    # pro každý kořen: můj nejlépe hodnocený titul v té franšíze
    best_in_root: dict[int, tuple[int, float]] = {}
    for mid, sc in my_scores.items():
        r = root_of.get(mid, mid)
        if r not in best_in_root or sc > best_in_root[r][1]:
            best_in_root[r] = (mid, sc)

    sequels: list[Recommendation] = []
    new_titles: list[Recommendation] = []
    for mid, en in enr.items():
        root = root_of.get(mid, mid)
        linked = best_in_root.get(root)   # (můj_mal_id, má_známka) nebo None
        ptw = mid in ptw_ids
        if linked and linked[1] >= rc.season_min_prequel_score:
            prequel = my_enr.get(linked[0])
            pname = prequel.title if prequel else f"#{linked[0]}"
            note = f"pokračování: {pname} (tvá známka {linked[1]:.0f})"
            rec = _make_rec(model, en, ptw, airing.get(mid), note)
            rec._prequel_score = linked[1]   # jen pro řazení
            sequels.append(rec)
        else:
            note = None
            if linked:   # franšíza, kterou znám, ale hodnotil jsem < práh
                note = "pokračování série (tvá známka pod prahem)"
            elif group_size.get(root, 1) > 1:  # pokračování série, kterou nemám
                note = "pokračování — předchozí díly neviděny"
            new_titles.append(_make_rec(model, en, ptw, airing.get(mid), note))

    # pokračování: podle mé známky předchozí řady, pak taste_fit
    sequels.sort(key=lambda r: (-getattr(r, "_prequel_score", 0), -r.taste_fit))
    # nové: čistě podle taste_fit (obsahová shoda), ořez na top N
    new_titles.sort(key=lambda r: -r.taste_fit)
    return sequels, new_titles[: rc.season_top_new]
