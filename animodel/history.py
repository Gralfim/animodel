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
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

FINGERPRINT_LEN = 12
# stavy, které znamenají "titul jsem si nějak vyzkoušel"
WATCHED_STATUSES = ("Completed", "Watching", "On-Hold", "Dropped")
_FILENAME_RE = re.compile(r"^(\d+)_([0-9a-f]+)\.json$")


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
    recommendations: list    # [{mal_id, title, rank, composite, taste_fit,
                             #   pred, ptw, cluster}, ...]
    model: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        """`{počet_hodnocených}_{otisk}.json` -- prefix dělá výpis složky
        chronologicky čitelný (seznam v čase roste), otisk drží identitu."""
        return f"{self.n_rated:05d}_{self.fingerprint}.json"

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "saved_at": self.saved_at,
            "n_rated": self.n_rated,
            "n_ptw": self.n_ptw,
            "watched_ids": sorted(self.watched_ids),
            "ptw_ids": sorted(self.ptw_ids),
            "model": self.model,
            "config": self.config,
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
        )


def build_snapshot(entries, recs, model, cfg, *, top: int = 100,
                   now: _dt.datetime | None = None) -> Snapshot:
    """Sestaví snapshot z právě doběhlého běhu. `recs` = seřazený list
    Recommendation (bere se prvních `top`)."""
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
                "composite": round(r.composite, 4),
                "taste_fit": round(r.taste_fit, 4),
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
            "n_clusters": len(getattr(model, "clusters", [])),
        },
        config={
            "shrinkage_k": cfg.model.shrinkage_k,
            "interaction_triples": cfg.model.interaction_triples,
            "cluster_fit_weight": cfg.recommend.cluster_fit_weight,
            "w_taste_fit": cfg.recommend.w_taste_fit,
            "w_cf": cfg.recommend.w_cf,
            "w_user_cf": cfg.recommend.w_user_cf,
            "w_quality": cfg.recommend.w_quality,
            "use_user_cf": cfg.recommend.use_user_cf,
        },
    )


def save_snapshot(snap: Snapshot, history_dir: str | Path) -> Path:
    """Zapíše snapshot; stejný otisk = přepis (poslední běh vyhrává)."""
    d = Path(history_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / snap.filename
    path.write_text(json.dumps(snap.to_dict(), ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def load_snapshots(history_dir: str | Path) -> list[Snapshot]:
    """Načte všechny snapshoty, seřazené podle počtu hodnocených (tedy
    zhruba chronologicky). Poškozené soubory přeskočí s warningem --
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
    out.sort(key=lambda s: (s.n_rated, s.saved_at))
    return out


def evaluate(snap: Snapshot, entries) -> dict | None:
    """
    Porovná doporučení ze `snap` s AKTUÁLNÍM stavem seznamu.

    Vrací None, když se od snapshotu nic nezměnilo (stejný otisk) nebo když
    snapshot žádná doporučení nemá.

    Klíčová metrika není hit rate samotný, ale **srovnání s kontrolou**:
    průměrná známka doporučených titulů, které jsi mezitím dokoukal, proti
    průměrné známce ostatních titulů dokoukaných za tutéž dobu. Teprve to
    odpovídá na otázku „jsou doporučení lepší než to, co bych si vybral sám?".

    `hit_rate` je vždycky jen SPODNÍ ODHAD: jmenovatel je celý snapshot
    (default 100 titulů), ale za pár měsíců stihneš dokoukat jednotky.
    Nízké číslo tedy neznamená špatný model; smysl dává až jeho vývoj v čase
    a hlavně `delta` proti kontrole, která na počtu odsledovaných nezávisí.
    """
    if not snap.recommendations:
        return None
    if export_fingerprint(entries) == snap.fingerprint:
        return None

    now = {e.mal_id: e for e in entries}
    was_watched = set(snap.watched_ids)
    was_ptw = set(snap.ptw_ids)
    rec_ids = {r["mal_id"] for r in snap.recommendations}

    buckets = {"completed": [], "started": [], "dropped": [],
               "newly_planned": [], "no_action": []}
    for r in snap.recommendations:
        e = now.get(r["mal_id"])
        st = e.status if e else None
        if st == "Completed":
            buckets["completed"].append((r, e.score))
        elif st in ("Watching", "On-Hold"):
            buckets["started"].append((r, e.score))
        elif st == "Dropped":
            buckets["dropped"].append((r, e.score))
        elif st == "Plan to Watch" and r["mal_id"] not in was_ptw:
            buckets["newly_planned"].append((r, 0))
        else:
            buckets["no_action"].append((r, 0))

    # kontrola: co jsi dokoukal od snapshotu, ale doporučené to NEBYLO
    control = [e.score for e in entries
               if e.status == "Completed" and e.score
               and e.mal_id not in was_watched and e.mal_id not in rec_ids]
    rec_scores = [sc for _r, sc in buckets["completed"] if sc]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    # Drží se pořadí? Průměrná známka po desítkách. `hi=None` = otevřený
    # kbelík (21 a dál). Hranice se hlásí NOMINÁLNÍ, ne podle nejvyššího
    # pozorovaného pořadí -- "11–15" by čtenáře mátlo, kbelík je 11–20.
    by_rank = []
    for lo, hi in ((1, 10), (11, 20), (21, None)):
        sub = [sc for r, sc in buckets["completed"]
               if sc and r["rank"] >= lo and (hi is None or r["rank"] <= hi)]
        if sub:
            by_rank.append((lo, hi, len(sub), mean(sub)))

    n_recs = len(snap.recommendations)
    return {
        "fingerprint": snap.fingerprint,
        "saved_at": snap.saved_at,
        "n_recs": n_recs,
        "n_completed": len(buckets["completed"]),
        "n_started": len(buckets["started"]),
        "n_dropped": len(buckets["dropped"]),
        "n_newly_planned": len(buckets["newly_planned"]),
        "hit_rate": len(buckets["completed"]) / n_recs if n_recs else 0.0,
        "mean_score_recommended": mean(rec_scores),
        "n_scored": len(rec_scores),
        "mean_score_control": mean(control),
        "n_control": len(control),
        "delta": (mean(rec_scores) - mean(control))
                 if rec_scores and control else None,
        "by_rank": by_rank,
        "titles": [(r["rank"], r["title"], sc)
                   for r, sc in sorted(buckets["completed"],
                                       key=lambda x: x[0]["rank"])],
    }


def format_report(results: list[dict]) -> list[str]:
    """Řádky pro CLI výpis. Prázdný list = není co hlásit."""
    if not results:
        return []
    lines = ["", "── Zpětná vazba z historie ─────────────────────────────"]
    for r in results:
        lines.append(
            f"  snapshot {r['fingerprint']} ({r['saved_at'][:10]}, "
            f"{r['n_recs']} doporučení):")
        lines.append(
            f"    dokoukáno {r['n_completed']} ({r['hit_rate']:.0%}) · "
            f"rozkoukáno {r['n_started']} · dropnuto {r['n_dropped']} · "
            f"nově v PTW {r['n_newly_planned']}")
        if r["mean_score_recommended"] is not None:
            ctrl = (f"{r['mean_score_control']:.2f} (n={r['n_control']})"
                    if r["mean_score_control"] is not None else "—")
            delta = (f"  Δ {r['delta']:+.2f}" if r["delta"] is not None else "")
            lines.append(
                f"    tvá známka: doporučené {r['mean_score_recommended']:.2f} "
                f"(n={r['n_scored']}) vs. ostatní nové {ctrl}{delta}")
        for lo, hi, n, m in r["by_rank"]:
            rng = f"{lo}–{hi}" if hi else f"{lo}+"
            lines.append(f"      pořadí {rng}: {n}× průměr {m:.2f}")
        if r["titles"]:
            top = ", ".join(f"#{rank} {t} ({sc:g})"
                            for rank, t, sc in r["titles"][:5])
            lines.append(f"    např.: {top}")
    return lines
