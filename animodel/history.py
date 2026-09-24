"""
history.py — záznam běhů a zpětná vazba z pozdějších exportů.

═══════════════════════════════════════════════════════════════════════════
PROČ
═══════════════════════════════════════════════════════════════════════════
Model se dnes validuje CV RMSE — a metodika sama říká, že to je ŠPATNÁ
metrika: atributy nepomáhají hádat číslo, ale řadit. Na to řazení ale
neexistuje žádná měřená zpětná vazba.

Přitom vstup pro ni je zadarmo: **každý nový MAL export je ground truth pro
předchozí doporučení.** Když se doporučený titul objeví o pár měsíců později
v Completed s devítkou, model fungoval; když ho tam nenajdeme nikdy, nefungoval.
Tenhle modul si proto každý běh zapíše a při dalším exportu ho vyhodnotí.

═══════════════════════════════════════════════════════════════════════════
KLÍČOVÁNÍ: OTISK SEZNAMU, NE DATUM BĚHU
═══════════════════════════════════════════════════════════════════════════
Snapshot se NEjmenuje podle data, ale podle otisku stavu seznamu. Ladění
parametrů znamená desítky běhů nad TÝMŽ exportem a datové klíčování by z nich
udělalo desítky skoro identických snapshotů, které by evaluaci jen ředily
(tentýž stav seznamu započítaný mnohokrát).

Otisk se počítá z trojic `(mal_id, status, score)` — tedy přesně z toho, co
model konzumuje a co zároveň tvoří ground truth. Záměrně NE z bajtů souboru:
`my_watched_episodes` a `my_finish_date` se v exportu mění průběžně, takže
hash souboru by se lišil po každém odsledovaném dílu, aniž by se změnilo
cokoli podstatného. Naopak přesun titulu z PTW do Completed nebo změna známky
otisk změní — a to je správně, to je nový stav světa.

Stejný otisk = přepis (poslední běh vyhrává). Při ladění tak vzniká jeden
snapshot na stav seznamu, ne na běh.

═══════════════════════════════════════════════════════════════════════════
SCHÉMA 2 (2026-09): CO SE UKLÁDÁ A PROČ
═══════════════════════════════════════════════════════════════════════════
Revize 2026-09 (HODNOCENI_PROJEKTU.md §9c) ukázala, že v1 snapshot neunese
většinu toho, co se z historie dá vytěžit — a chybějící data nejdou
dopočítat zpětně:

  * stav CELÉHO seznamu (status, známka, datum dokončení): v1 měl jen
    množiny ID, takže nešlo odlišit rozkoukané od dokoukaného ani poznat
    přehodnocení starého titulu;
  * datum exportu (mtime souboru): `saved_at` je čas POSLEDNÍHO běhu nad
    otiskem (ladicí běhy ho posouvají), ne datum exportu;
  * celý pool kandidátů se složkami kompozitu a z-parametry: jen tak jde
    kompozit později přeskládat jinými vahami nebo změřit, kde v poolu ležel
    titul, který sis nakonec vybral. Top-100 se 4 desetinnými místy nestačí
    (přepočtené pořadí celého poolu se liší u 374 kandidátů), proto 6 míst --
    s nimi se na reálném poolu (4030 kandidátů) posune 6 pozic, a to jen mezi
    remízami s rozdílem kompozitu ~2e-6;
  * log predikcí i MIMO pool (PTW + neviděné díly franšíz): nová hodnocení
    přicházejí ~13 měsíčně, zásahy doporučení ~1 — nejhustší zpětná vazba
    je porovnání uložené predikce se skutečnou známkou u VŠECH nových titulů.

Pool a log leží ve vedlejším souboru `{n}_{otisk}.pool.json` bez odsazení
(~300 kB; s indent=1 by to bylo ~470 kB). load_snapshots ho nečte, načítá
se jen když je potřeba (load_predictions).

═══════════════════════════════════════════════════════════════════════════
VYHODNOCENÍ: LEDGER UDÁLOSTÍ, NE BLOK ZA SNAPSHOT
═══════════════════════════════════════════════════════════════════════════
v1 vyhodnocoval každý starší snapshot proti dnešku zvlášť: tentýž titul
doporučený ve dvou snapshotech se započetl dvakrát, kontroly se překrývaly
a výpis s každým exportem rostl. evaluate_history() místo toho staví ledger:
každé nové shlédnutí je JEDNA událost v okně mezi sousedními snapshoty,
s první expozicí (nejstarší snapshot, kde bylo doporučené).

Statistiky nad ledgerem se drží toho, co je při dnešním objemu dat
(~1–2 zásahy doporučení za 7 týdnů) interpretovatelné:
  * počty po oknech a kohortách místo sazeb — hit rate neměl smysluplný
    jmenovatel a sazby za čas by stály na `saved_at`;
  * známky PO FRANŠÍZÁCH (díly jedné franšízy nejsou nezávislé body), surově
    i vůči baseline z komunity, s 95% intervalem ze sd celého seznamu — bez
    intervalu vypadalo „Δ −0.56" z jednoho titulu jako výsledek;
  * kontrola i bez pokračování franšíz rozjetých před oknem: o ně
    doporučovač nesoutěží, a přitom v kontrole převažovaly;
  * kbelíky pořadí až od MIN_BUCKET_N titulů;
  * predikce vs. skutečná známka, zvlášť nové franšízy a pokračování.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .series import (
    CONTINUATION_RELATION_TYPES, SERIES_RELATION_TYPES, build_roots,
    related_ids,
)

log = logging.getLogger(__name__)

FINGERPRINT_LEN = 12
SCHEMA = 2
# stavy, které znamenají "titul jsem si nějak vyzkoušel"
WATCHED_STATUSES = ("Completed", "Watching", "On-Hold", "Dropped")
_FILENAME_RE = re.compile(r"^(\d+)_([0-9a-f]+)\.json$")
POOL_SUFFIX = ".pool.json"

#: sloupce řádku poolu ve vedlejším souboru
POOL_COLUMNS = ("mal_id", "composite", "taste_fit", "affinity", "cluster_fit",
                "cf_signal", "user_cf_signal", "community", "pred", "pred_lo",
                "pred_hi", "ptw", "sources", "select")
# "select" (model výběru, §9d.4) přibyl 2026-09-24 na KONCI: pool se čte podle
# jmen sloupců uložených v souboru, starší snapshoty ho prostě nemají
#: sloupce logu predikcí mimo pool; origin = "ptw" | "franchise"
EXTRA_COLUMNS = ("mal_id", "pred", "pred_lo", "pred_hi", "affinity",
                 "community", "origin")
_DIGITS = 6
#: kbelík pořadí se vypisuje až od tolika titulů -- průměr jednoho titulu
#: vydávaný za „validaci řazení" byl zavádějící
MIN_BUCKET_N = 3
#: Spearman predikce se počítá až od tolika bodů
MIN_SPEARMAN_N = 5


def export_fingerprint(entries) -> str:
    """
    Otisk STAVU SEZNAMU (ne souboru): sha256 nad seřazenými trojicemi
    `mal_id:status:score`, zkrácený na FINGERPRINT_LEN znaků.

    Ignoruje vše, co model ani ground truth nezajímá (počet odsledovaných
    dílů, data, komentáře), takže běh nad nezměněným seznamem vždy padne
    na tentýž klíč.
    """
    payload = "\n".join(sorted(
        f"{e.mal_id}:{e.status}:{e.score}" for e in entries))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]


def _round(x):
    return None if x is None else round(float(x), _DIGITS)


@dataclass
class Snapshot:
    """Jeden zaznamenaný běh (= jeden stav seznamu)."""
    fingerprint: str
    saved_at: str
    n_rated: int
    n_ptw: int
    watched_ids: list        # co bylo shlédnuté V DOBĚ běhu -- bez toho by
                             # nešlo spočítat, co přibylo AŽ POTOM
    ptw_ids: list
    recommendations: list    # top-N [{mal_id, title, rank, composite,
                             #   taste_fit, affinity, cluster_fit, pred, ptw,
                             #   cluster}, ...]
    model: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    schema: int = 1
    export_date: str = ""    # mtime exportu (v2); u v1 prázdné
    list_state: list = field(default_factory=list)
                             # [[mal_id, status, score, finish_date], ...] (v2)
    pool: list = field(default_factory=list)
                             # řádky POOL_COLUMNS -- jen ve vedlejším souboru
    extra: list = field(default_factory=list)
                             # řádky EXTRA_COLUMNS -- jen ve vedlejším souboru

    @property
    def filename(self) -> str:
        """`{počet_hodnocených}_{otisk}.json` -- prefix dělá výpis složky
        chronologicky čitelný (seznam v čase roste), otisk drží identitu."""
        return f"{self.n_rated:05d}_{self.fingerprint}.json"

    @property
    def pool_filename(self) -> str:
        return f"{self.n_rated:05d}_{self.fingerprint}{POOL_SUFFIX}"

    @property
    def date(self) -> str:
        """Datum pro výpis: export (v2), jinak čas běhu (v1)."""
        return (self.export_date or self.saved_at)[:10]

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint,
            "saved_at": self.saved_at,
            "export_date": self.export_date,
            "n_rated": self.n_rated,
            "n_ptw": self.n_ptw,
            "watched_ids": sorted(self.watched_ids),
            "ptw_ids": sorted(self.ptw_ids),
            "model": self.model,
            "config": self.config,
            "list": self.list_state,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(
            fingerprint=d["fingerprint"], saved_at=d.get("saved_at", ""),
            n_rated=d.get("n_rated", 0), n_ptw=d.get("n_ptw", 0),
            watched_ids=d.get("watched_ids", []),
            ptw_ids=d.get("ptw_ids", []),
            recommendations=d.get("recommendations", []),
            model=d.get("model", {}), config=d.get("config", {}),
            schema=d.get("schema", 1), export_date=d.get("export_date", ""),
            list_state=d.get("list", []),
        )


def _pool_row(r) -> list:
    return [r.mal_id, _round(r.composite), _round(r.taste_fit),
            _round(r.affinity), _round(r.cluster_fit), _round(r.cf_signal),
            _round(r.user_cf_signal), _round(r.community), _round(r.pred),
            _round(r.pred_lo), _round(r.pred_hi), int(bool(r.ptw)),
            ",".join(r.sources), _round(r.select)]


def prediction_row(mal_id, pred, pred_lo, pred_hi, affinity, community,
                   origin: str) -> list:
    """Řádek logu predikcí mimo pool (EXTRA_COLUMNS)."""
    return [mal_id, _round(pred), _round(pred_lo), _round(pred_hi),
            _round(affinity), _round(community), origin]


def build_snapshot(entries, recs, model, cfg, *, top: int = 100,
                   now: _dt.datetime | None = None, export_date: str = "",
                   z_params: dict | None = None,
                   extra: list | None = None) -> Snapshot:
    """
    Sestaví snapshot z právě doběhlého běhu.

    `recs` = CELÝ seřazený pool (list Recommendation): prvních `top` jde do
    hlavního souboru, všechno do vedlejšího. `z_params` = parametry z-skóre
    složek kompozitu (RecommendResult.z_params), `extra` = log predikcí mimo
    pool (řádky z prediction_row), `export_date` = mtime exportu.
    """
    watched = [e.mal_id for e in entries if e.status in WATCHED_STATUSES]
    ptw = [e.mal_id for e in entries if e.status == "Plan to Watch"]
    rated = [e for e in entries if e.status == "Completed" and e.score]
    stamp = (now or _dt.datetime.now()).isoformat(timespec="seconds")
    return Snapshot(
        fingerprint=export_fingerprint(entries),
        saved_at=stamp,
        n_rated=len(rated),
        n_ptw=len(ptw),
        watched_ids=watched,
        ptw_ids=ptw,
        recommendations=[
            {
                "mal_id": r.mal_id,
                "title": r.title,
                "rank": i + 1,
                "composite": _round(r.composite),
                "taste_fit": _round(r.taste_fit),
                "affinity": _round(r.affinity),
                "cluster_fit": _round(r.cluster_fit),
                "pred": round(r.pred, 3),
                "ptw": bool(r.ptw),
                "cluster": r.cluster_name,
            }
            for i, r in enumerate(recs[:top])
        ],
        model={
            "scale": getattr(model, "scale", None),
            "scale_triples": getattr(model, "scale_triples", None),
            "cv_rmse": getattr(model, "cv_rmse", None),
            "baseline_rmse": getattr(model, "baseline_rmse", None),
            "beta": getattr(model, "beta", None),
            "u_mean": getattr(model, "u_mean", None),
            "c_mean": getattr(model, "c_mean", None),
            "raw_center": getattr(model, "raw_center", None),
            "cv_scheme": getattr(model, "cv_scheme", None),   # random/grouped:
                                                              # cv_rmse ze dvou
                                                              # schémat se nesmí
                                                              # porovnávat naslepo
            "n_clusters": len(getattr(model, "clusters", [])),
            "z_params": z_params or {},
        },
        config={
            "shrinkage_k": cfg.model.shrinkage_k,
            "effect_model": cfg.model.effect_model,
            "ridge_alpha": cfg.model.ridge_alpha,
            "min_attr_count": cfg.model.min_attr_count,
            "interaction_triples": cfg.model.interaction_triples,
            "cluster_fit_weight": cfg.recommend.cluster_fit_weight,
            "w_taste_fit": cfg.recommend.w_taste_fit,
            "w_cf": cfg.recommend.w_cf,
            "w_user_cf": cfg.recommend.w_user_cf,
            "w_quality": cfg.recommend.w_quality,
            "w_select": cfg.recommend.w_select,
            "known_popularity": cfg.recommend.known_popularity,
            "use_user_cf": cfg.recommend.use_user_cf,
        },
        schema=SCHEMA,
        export_date=export_date,
        list_state=[[e.mal_id, e.status, e.score, e.finish_date or ""]
                    for e in sorted(entries, key=lambda e: e.mal_id)],
        pool=[_pool_row(r) for r in recs],
        extra=list(extra or []),
    )


def save_snapshot(snap: Snapshot, history_dir: str | Path) -> Path:
    """Zapíše snapshot (+ vedlejší soubor s poolem a logem predikcí); stejný
    otisk = přepis (poslední běh vyhrává). Vrací cestu hlavního souboru."""
    d = Path(history_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / snap.filename
    path.write_text(json.dumps(snap.to_dict(), ensure_ascii=False, indent=1),
                    encoding="utf-8")
    if snap.pool or snap.extra:
        side = {"schema": snap.schema,
                "pool_columns": list(POOL_COLUMNS), "pool": snap.pool,
                "extra_columns": list(EXTRA_COLUMNS), "extra": snap.extra}
        (d / snap.pool_filename).write_text(
            json.dumps(side, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
    return path


def load_snapshots(history_dir: str | Path) -> list[Snapshot]:
    """Načte všechny snapshoty seřazené podle času běhu. Dřív podle počtu
    hodnocených -- ten ale klesne, když titul odhodnotíš nebo smažeš, a okna
    ledgeru potřebují čas. Poškozené soubory přeskočí s warningem --
    historie je diagnostika, nesmí shodit běh."""
    d = Path(history_dir)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        if not _FILENAME_RE.match(p.name):
            continue
        try:
            out.append(Snapshot.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception as exc:
            log.warning(f"historie: {p.name} se nepodařilo načíst ({exc}) -- přeskakuji")
    out.sort(key=lambda s: s.saved_at)
    return out


def load_predictions(history_dir: str | Path, snap: Snapshot) -> dict[int, float]:
    """{mal_id: pred} uložené se snapshotem: top-N z hlavního souboru, u v2
    navíc celý pool a log mimo pool z vedlejšího souboru."""
    preds = {r["mal_id"]: r["pred"] for r in snap.recommendations
             if r.get("pred") is not None}
    p = Path(history_dir) / snap.pool_filename
    if not p.exists():
        return preds
    try:
        side = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"historie: {p.name} se nepodařilo načíst ({exc}) -- jen top-N")
        return preds
    for cols_key, rows_key in (("pool_columns", "pool"), ("extra_columns", "extra")):
        cols = side.get(cols_key) or []
        i_id, i_pred = cols.index("mal_id"), cols.index("pred")
        for row in side.get(rows_key) or []:
            preds.setdefault(row[i_id], row[i_pred])
    return preds


# ── franšízy ────────────────────────────────────────────────────────────────

def franchise_neighbourhood(seed_ids, relations_of, *, hops: int = 2,
                            relation_types=SERIES_RELATION_TYPES) -> set[int]:
    """
    Tituly franšíz do `hops` kroků od `seed_ids` po sériových relacích (bez
    samotných seedů). `relations_of(ids)` vrací {mal_id: data s "relations"}
    v Jikan tvaru (Enricher.relations_data) -- volá se jednou na krok.

    Pro log predikcí: noví shlédnutí jsou většinou pokračování a vedlejší
    obsah rozjetých nebo naplánovaných franšíz, které v poolu nejsou. Změřeno
    na 23 nových shlédnutích od snapshotu 2026-07-28: pool + PTW pokryje
    5/23, +1 krok 14/23, +2 kroky 17/23.
    """
    known = set(seed_ids)
    frontier = sorted(known)
    found: set[int] = set()
    for _ in range(hops):
        nxt = set()
        for data in relations_of(frontier).values():
            nxt |= related_ids(data, relation_types) - known
        if not nxt:
            break
        found |= nxt
        known |= nxt
        frontier = sorted(nxt)
    return found


def franchise_roots(ids, relations: dict) -> dict[int, int]:
    """{mal_id: kořen franšízy} pro `ids` (series.build_roots přes
    CONTINUATION_RELATION_TYPES -- remake není pokračování)."""
    return build_roots(ids, relations, CONTINUATION_RELATION_TYPES)


# ── vyhodnocení ─────────────────────────────────────────────────────────────

def _baseline_of(snap: Snapshot, fallback):
    """(u_mean, c_mean, beta) ze snapshotu v2 -- očekávání v době doporučení;
    u v1 fallback (aktuální model, rozdíl vyšel < 0.01)."""
    m = snap.model or {}
    if all(m.get(k) is not None for k in ("u_mean", "c_mean", "beta")):
        return m["u_mean"], m["c_mean"], m["beta"]
    return fallback


def _resid(score, community, baseline):
    """Známka − baseline(komunita); stejný tvar jako TasteModel._baseline_pred."""
    if not score or baseline is None:
        return None
    u, c_mean, beta = baseline
    expected = u if community is None else u + beta * (community - c_mean)
    return score - expected


def _sd(xs) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _ranks(xs) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(xs, ys) -> float | None:
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 1e-12 else None


def _by_franchise(events, root, key) -> dict | None:
    """Průměr po franšízách: díly jedné franšízy nejsou nezávislé body (od
    snapshotu 2026-07-28 tvořilo 8 z 18 kontrolních titulů Shinmai Maou)."""
    groups = defaultdict(list)
    for ev in events:
        if ev[key] is not None:
            groups[root(ev["mal_id"])].append(ev[key])
    if not groups:
        return None
    means = [sum(v) / len(v) for v in groups.values()]
    return {"mean": sum(means) / len(means), "n_groups": len(groups),
            "n_titles": sum(len(v) for v in groups.values())}


def _delta(a, b, sd) -> dict | None:
    """
    Rozdíl průměrů s 95% intervalem; jednotka = franšíza, rozptyl = sd
    celého seznamu. V simulaci revize dával tenhle interval pokrytí ~96 %;
    cv_rmse jako rozptyl při franšízově klastrovaných událostech podpokrýval.
    Záměrně bez smrštění n/(n+K): smrštěná delta by „málo dat" ukázala jako
    „žádný efekt".
    """
    if not a or not b or not sd:
        return None
    d = a["mean"] - b["mean"]
    half = 1.96 * sd * math.sqrt(1 / a["n_groups"] + 1 / b["n_groups"])
    return {"delta": d, "ci95": half, "conclusive": abs(d) > half}


def _prediction_stats(events) -> dict:
    pts = [(ev["pred"], ev["score"]) for ev in events if ev["pred"] is not None]
    if not pts:
        return {"n": 0}
    errs = [y - p for p, y in pts]
    n = len(pts)
    return {
        "n": n,
        "bias": sum(errs) / n,
        "rmse": math.sqrt(sum(e * e for e in errs) / n),
        "spearman": (_spearman([p for p, _ in pts], [y for _, y in pts])
                     if n >= MIN_SPEARMAN_N else None),
    }


def evaluate_history(snaps, entries, *, roots: dict | None = None,
                     community: dict | None = None,
                     baseline: tuple | None = None,
                     predictions: dict | None = None) -> dict | None:
    """
    Ledger událostí nad STARŠÍMI snapshoty (ne tím s dnešním otiskem)
    proti aktuálnímu stavu seznamu. Vrací None bez snapshotů.

      roots        {mal_id: kořen franšízy} (franchise_roots); bez něj je
                   každý titul samostatná franšíza a nic není pokračování
      community    {mal_id: komunitní skóre} pro známku vůči baseline
      baseline     (u_mean, c_mean, beta) aktuálního modelu; snapshot v2 má
                   vlastní a ten má přednost
      predictions  {otisk: {mal_id: pred}} (load_predictions); bez něj jen
                   predikce top-N doporučení z hlavního souboru

    Událost = titul dnes ve WATCHED_STATUSES, který v nejstarším snapshotu
    shlédnutý nebyl. Okno = mezi posledním snapshotem, kde ještě shlédnutý
    nebyl, a dalším (poslední okno končí dneškem). Doporučená je, když byla
    v top-N některého snapshotu do začátku okna (první expozice).
    """
    if not snaps:
        return None
    snaps = sorted(snaps, key=lambda s: s.saved_at)
    roots = roots or {}
    community = community or {}
    predictions = predictions or {}

    def root(mid):
        return roots.get(mid, mid)

    k = len(snaps)
    now = {e.mal_id: e for e in entries}
    now_watched = {m for m, e in now.items() if e.status in WATCHED_STATUSES}
    now_ptw = {m for m, e in now.items() if e.status == "Plan to Watch"}
    watched_at = [set(s.watched_ids) for s in snaps]
    ptw_at = [set(s.ptw_ids) for s in snaps]
    recs_at = [{r["mal_id"]: r for r in s.recommendations} for s in snaps]
    started_at = [{root(m) for m in w} for w in watched_at]
    base_at = [_baseline_of(s, baseline) for s in snaps]

    events = []
    for mid in sorted(now_watched - watched_at[0]):
        w = next((j - 1 for j in range(1, k) if mid in watched_at[j]), k - 1)
        exp = next(((i, recs_at[i][mid]) for i in range(w + 1)
                    if mid in recs_at[i]), None)
        pred = predictions.get(snaps[w].fingerprint, {}).get(mid)
        if pred is None and mid in recs_at[w]:
            pred = recs_at[w][mid].get("pred")
        e = now[mid]
        events.append({
            "mal_id": mid, "title": e.title, "status": e.status,
            "score": e.score, "window": w, "rec": exp is not None,
            "exposed_in": exp[0] if exp else None,
            "rank": exp[1]["rank"] if exp else None,
            "ptw_at_exposure": bool(exp[1].get("ptw")) if exp else None,
            "continuation": root(mid) in started_at[w],
            "pred": pred,
            "resid": _resid(e.score, community.get(mid), base_at[w]),
        })

    windows = []
    for w in range(k):
        end_ptw = ptw_at[w + 1] if w + 1 < k else now_ptw
        added = end_ptw - ptw_at[w]
        in_window = [ev for ev in events if ev["window"] == w]
        windows.append({
            "start": snaps[w].date,
            "end": snaps[w + 1].date if w + 1 < k else None,
            "n_new": len(in_window),
            "rec_events": sorted((ev["rank"], ev["title"], ev["status"], ev["score"])
                                 for ev in in_window if ev["rec"]),
            "ptw_added": len(added),
            "ptw_added_from_recs": sorted((recs_at[w][m]["rank"], recs_at[w][m]["title"])
                                          for m in added if m in recs_at[w]),
        })

    # kohorty podle stavu při PRVNÍ expozici; vyzkoušeno = dnes shlédnuté
    # (Dropped taky -- vyzkoušel a nechal být je výsledek, ne „nic")
    first_exp: dict[int, dict] = {}
    for recs in recs_at:
        for mid, r in recs.items():
            first_exp.setdefault(mid, r)
    rec_free = {m for m, r in first_exp.items() if not r.get("ptw")}
    rec_ptw = {m for m, r in first_exp.items() if r.get("ptw")}
    ptw_only = set().union(*ptw_at) - set(first_exp)
    cohorts = {name: (len(ids & now_watched), len(ids)) for name, ids in (
        ("rec_not_ptw", rec_free), ("rec_ptw", rec_ptw), ("ptw_only", ptw_only))}

    scored = [ev for ev in events if ev["status"] == "Completed" and ev["score"]]
    rec = [ev for ev in scored if ev["rec"]]
    control_all = [ev for ev in scored if not ev["rec"]]
    control_new = [ev for ev in control_all if not ev["continuation"]]
    rated = [e for e in entries if e.status == "Completed" and e.score]
    sd_raw = _sd([e.score for e in rated])
    current_base = baseline or base_at[-1]
    sd_resid = _sd([r for r in (_resid(e.score, community[e.mal_id], current_base)
                                for e in rated if e.mal_id in community)
                    if r is not None])
    groups = {name: {"raw": _by_franchise(evs, root, "score"),
                     "resid": _by_franchise(evs, root, "resid")}
              for name, evs in (("rec", rec), ("control_new", control_new),
                                ("control_all", control_all))}
    deltas = {name: {"raw": _delta(groups["rec"]["raw"], groups[name]["raw"], sd_raw),
                     "resid": _delta(groups["rec"]["resid"], groups[name]["resid"],
                                     sd_resid)}
              for name in ("control_new", "control_all")}

    by_rank = []
    for lo, hi in ((1, 10), (11, 20), (21, None)):
        sub = [ev["score"] for ev in rec
               if ev["rank"] >= lo and (hi is None or ev["rank"] <= hi)]
        if len(sub) >= MIN_BUCKET_N:
            by_rank.append((lo, hi, len(sub), sum(sub) / len(sub)))

    return {
        "n_snapshots": k,
        "first_date": snaps[0].date,
        "windows": windows,
        "events": events,
        "cohorts": cohorts,
        "scores": {"groups": groups, "deltas": deltas},
        "by_rank": by_rank,
        "prediction": {
            "n_scored": len(scored),
            "all": _prediction_stats(scored),
            "new": _prediction_stats([ev for ev in scored if not ev["continuation"]]),
            "cont": _prediction_stats([ev for ev in scored if ev["continuation"]]),
        },
    }


# ── výpis ───────────────────────────────────────────────────────────────────

def _cz(n: int, one: str, few: str, many: str) -> str:
    return one if n == 1 else few if 2 <= n <= 4 else many


def _event_label(rank, title, status, score) -> str:
    if status == "Completed":
        tail = f" ({score})" if score else ""
    elif status == "Dropped":
        tail = " (dropnuto)"
    else:
        tail = f" (rozkoukáno, {score})" if score else " (rozkoukáno)"
    return f"#{rank} {title}{tail}"


def format_report(res: dict | None) -> list[str]:
    """Řádky pro CLI výpis. Prázdný list = není co hlásit."""
    if not res:
        return []
    n = res["n_snapshots"]
    lines = ["", f"── Zpětná vazba z historie ({n} "
                 f"{_cz(n, 'snapshot', 'snapshoty', 'snapshotů')} od "
                 f"{res['first_date']}) ─────────────"]

    lines.append("  okna mezi snapshoty:")
    for w in res["windows"]:
        rec = w["rec_events"]
        rec_txt = (f"z doporučení {len(rec)}: "
                   + ", ".join(_event_label(*ev) for ev in rec)) if rec else "z doporučení 0"
        ptw_txt = f"do PTW přidáno {w['ptw_added']}"
        if w["ptw_added_from_recs"]:
            ptw_txt += " (z doporučení: " + ", ".join(
                f"#{rank} {title}" for rank, title in w["ptw_added_from_recs"]) + ")"
        lines.append(f"    {w['start']} → {w['end'] or 'dnes'}: nově shlédnuto "
                     f"{w['n_new']} · {rec_txt} · {ptw_txt}")

    c = res["cohorts"]
    lines.append(
        f"  vyzkoušeno: z doporučení mimo PTW {c['rec_not_ptw'][0]}/{c['rec_not_ptw'][1]}"
        f" · z doporučení na PTW {c['rec_ptw'][0]}/{c['rec_ptw'][1]}"
        f" · z PTW mimo doporučení {c['ptw_only'][0]}/{c['ptw_only'][1]}")

    groups, deltas = res["scores"]["groups"], res["scores"]["deltas"]
    if any(groups[g]["raw"] for g in groups):
        lines.append("  známky dokoukaných (průměr po franšízách; vůči očekávání "
                     "= známka − baseline z komunity):")
        for key, label in (("rec", "doporučené"),
                           ("control_new", "ostatní, nové franšízy"),
                           ("control_all", "ostatní vč. pokračování")):
            g, r = groups[key]["raw"], groups[key]["resid"]
            if not g:
                lines.append(f"    {label:24s} —")
                continue
            exp_txt = f" · vůči očekávání {r['mean']:+.2f}" if r else ""
            lines.append(
                f"    {label:24s} {g['mean']:.2f}{exp_txt}  ({g['n_groups']} "
                f"{_cz(g['n_groups'], 'franšíza', 'franšízy', 'franšíz')}, "
                f"{g['n_titles']} {_cz(g['n_titles'], 'titul', 'tituly', 'titulů')})")
        for key, label in (("control_new", "novým franšízám"),
                           ("control_all", "všem ostatním")):
            d_raw, d_res = deltas[key]["raw"], deltas[key]["resid"]
            if not d_raw:
                continue
            txt = (f"    Δ proti {label}: {d_raw['delta']:+.2f} "
                   f"(95% ±{d_raw['ci95']:.2f})")
            if d_res:
                txt += f" · vůči očekávání {d_res['delta']:+.2f} (±{d_res['ci95']:.2f})"
            if not (d_raw["conclusive"] or (d_res and d_res["conclusive"])):
                txt += " — zatím neprůkazné"
            lines.append(txt)
        for lo, hi, cnt, mean in res["by_rank"]:
            rng = f"{lo}–{hi}" if hi else f"{lo}+"
            lines.append(f"    pořadí {rng}: {cnt}× průměr {mean:.2f}")

    p = res["prediction"]
    if p["n_scored"]:
        lines.append(f"  predikce vs. skutečná známka (uložená predikce u "
                     f"{p['all']['n']}/{p['n_scored']} dokoukaných; záporný "
                     f"bias = predikce nadsazené):")
        for key, label in (("new", "nové franšízy"), ("cont", "pokračování")):
            st = p[key]
            if not st["n"]:
                continue
            txt = (f"    {label:14s} n={st['n']} · bias {st['bias']:+.2f} · "
                   f"RMSE {st['rmse']:.2f}")
            if st["spearman"] is not None:
                txt += f" · Spearman {st['spearman']:.2f}"
            lines.append(txt)
    return lines
