"""
backtest.py — časová (out-of-time) validace nad `my_finish_date`.

═══════════════════════════════════════════════════════════════════════════
PROČ NESTAČÍ CROSS-VALIDACE
═══════════════════════════════════════════════════════════════════════════
CV (i ta franšízově seskupená) odpovídá na otázku „jak dobře model doplní
chybějící titul mezi ostatními z TÉHOŽ období". Otázka, na které záleží, je
jiná: jak seřadí tituly, které uvidím PŘÍŠTĚ.

Revize 2026-09 (HODNOCENI_PROJEKTU.md §9c) to změřila: pořadí konfigurací
podle OOF metrik je proti pořadí mimo čas ANTIKORELOVANÉ (Kendall tau −0,47
až −0,87), a to jak pro RMSE, tak pro Spearman. O strukturálních volbách
(K, min_lift, páry, váha nálad) tedy CV rozhodovat neumí a tenhle modul je
jediný nástroj, který na to v projektu je.

Není to diagnostika běhu, ale měřicí přístroj: `--backtest` ho spustí,
běžný běh se ho nedotkne (žádný snapshot, žádné HTML).

═══════════════════════════════════════════════════════════════════════════
METODIKA (poučená z chyb, které revize našla ve vlastních měřeních)
═══════════════════════════════════════════════════════════════════════════
* DISJUNKTNÍ okna [T_i, T_i+1), ne vnořené řezy. Vnořené testy sdílejí
  většinu titulů (161 ⊃ 128 ⊃ 115 ⊃ 76), takže „potvrzeno na 3 ze 4 řezů"
  je fakticky jedno měření započítané třikrát.
* Bootstrap KLASTROVANÝ po franšízách: díly jedné série nejsou nezávislé
  body a iid bootstrap dává o 30–50 % užší interval.
* NOVÉ FRANŠÍZY a POKRAČOVÁNÍ zvlášť. Efekty, které vypadají velké, sedí
  obvykle jen v pokračováních (u párů interakcí +0,21 proti −0,01), a
  doporučení se točí kolem nových franšíz.
* Tituly BEZ data dokončení patří do tréninku (jsou starší), ale ty
  s premiérou po řezu se vyřadí -- ty uživatel tehdy vidět nemohl.
* RMSE se hlásí i PO odečtení biasu. Posun hladiny a kvalita řazení jsou
  dvě různé věci; jejich míchání vedlo k závěru „model je horší než
  baseline", který ve skutečnosti popisoval chybějící centrování afinity.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

#: hodnoty `my_finish_date`, které znamenají „bez data"
UNDATED = ("", "0000-00-00")
#: default: okna zhruba po 10-15 % nejnovějších titulů (poslední otevřené)
_DEFAULT_QUANTILES = (0.50, 0.65, 0.80, 0.90)


def dated(entries) -> list:
    return [e for e in entries if e.finish_date not in UNDATED]


def default_cuts(entries, quantiles=_DEFAULT_QUANTILES) -> list[str]:
    """Řezy podle kvantilů dat dokončení, ať mají okna srovnatelnou velikost."""
    dates = sorted(e.finish_date for e in dated(entries))
    if len(dates) < 40:
        return []
    cuts = []
    for q in quantiles:
        d = dates[min(len(dates) - 1, int(len(dates) * q))]
        if d not in cuts:
            cuts.append(d)
    return cuts


def split(entries, cuts: list[str], premieres: dict | None = None) -> list[dict]:
    """
    Rozdělí ohodnocené tituly na okna [cuts[i], cuts[i+1]) (poslední otevřené).

    Vrací [{"start", "end", "train": [entry], "test": [entry]}, ...]:
    trénink = vše dokončené před řezem plus nedatované (bez těch, které v tu
    dobu ještě neměly premiéru), test = tituly dokončené uvnitř okna.
    """
    premieres = premieres or {}
    out = []
    for i, start in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else None
        train, test = [], []
        for e in entries:
            if e.finish_date in UNDATED:
                prem = premieres.get(e.mal_id)
                if not prem or prem < start:
                    train.append(e)
                continue
            if e.finish_date < start:
                train.append(e)
            elif end is None or e.finish_date < end:
                test.append(e)
        out.append({"start": start, "end": end, "train": train, "test": test})
    return out


# ── metriky ────────────────────────────────────────────────────────────────

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


def spearman(xs, ys) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 1e-12 else None


def _rmse(errs) -> float:
    return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else 0.0


def metrics(rows, interval: float | None = None) -> dict:
    """
    `rows` = [(afinita, baseline, známka, kořen_franšízy), ...].

    Hlásí zvlášť ŘAZENÍ (Spearman afinity vůči reziduu) a HLADINU (bias,
    RMSE před i po odečtení biasu) -- viz metodika v docstringu modulu.
    """
    if not rows:
        return {"n": 0}
    resid = [y - b for _a, b, y, _r in rows]
    preds = [max(1.0, min(10.0, b + a)) for a, b, _y, _r in rows]
    errs = [y - p for (_a, _b, y, _r), p in zip(rows, preds)]
    base_errs = [y - max(1.0, min(10.0, b)) for _a, b, y, _r in rows]
    bias = sum(errs) / len(errs)
    out = {
        "n": len(rows),
        "spearman": spearman([a for a, _b, _y, _r in rows], resid),
        "rmse_baseline": _rmse(base_errs),
        "rmse_model": _rmse(errs),
        "rmse_model_debiased": _rmse([e - bias for e in errs]),
        "bias": bias,
    }
    if interval:
        out["coverage"] = sum(1 for e in errs if abs(e) <= interval) / len(errs)
    return out


def bootstrap_spearman(rows, n_boot: int = 1000, seed: int = 42,
                       alpha: float = 0.05) -> tuple | None:
    """95% interval Spearmanu, resampluje se po FRANŠÍZÁCH (klastrovaný
    bootstrap) -- díly jedné série jsou jeden důkaz, ne pět."""
    if len(rows) < 5:
        return None
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[3]].append(r)
    keys = list(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = []
        for _ in range(len(keys)):
            sample.extend(groups[keys[rng.randrange(len(keys))]])
        s = spearman([a for a, _b, _y, _r in sample],
                     [y - b for _a, b, y, _r in sample])
        if s is not None:
            vals.append(s)
    if not vals:
        return None
    vals.sort()
    lo = vals[max(0, int(alpha / 2 * len(vals)))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


# ── běh ────────────────────────────────────────────────────────────────────

def run(entries, *, build_titles, fit, cuts: list[str] | None = None,
        premieres: dict | None = None, n_boot: int = 1000, seed: int = 42,
        progress=None) -> dict:
    """
    entries      ohodnocené MalEntry (Completed se známkou)
    build_titles callable(list[MalEntry]) -> list[Title]  (obohacení z cache)
    fit          callable(list[Title]) -> model s `affinity()` a `_baseline_pred()`
    cuts         řezy YYYY-MM-DD; None = default_cuts()
    premieres    {mal_id: 'YYYY-MM-DD'} pro vyřazení titulů, které v době
                 řezu ještě neexistovaly (nedatované se jinak berou jako starší)

    Fit běží nad tréninkem okna, metriky nad jeho testem. Title objekty se
    staví nad SJEDNOCENÍM (trénink i test) -- jinak by test neměl franšízové
    kořeny, podle kterých se dělí nové franšízy od pokračování.
    """
    cuts = cuts or default_cuts(entries)
    if not cuts:
        return {"windows": [], "pooled": {}, "cuts": []}
    windows = []
    pooled: dict = {"all": [], "new": [], "cont": []}
    for w in split(entries, cuts, premieres):
        if not w["test"] or len(w["train"]) < 30:
            continue
        if progress:
            progress(f"  backtest: okno {w['start']} → {w['end'] or 'konec'} "
                     f"(trénink {len(w['train'])}, test {len(w['test'])}) …")
        titles = {t.mal_id: t for t in build_titles(w["train"] + w["test"])}
        train_titles = [titles[e.mal_id] for e in w["train"] if e.mal_id in titles]
        if len(train_titles) < 30:
            continue
        model = fit(train_titles)
        train_roots = {t.series_root for t in train_titles if t.series_root is not None}
        rows = {"all": [], "new": [], "cont": []}
        for e in w["test"]:
            t = titles.get(e.mal_id)
            if t is None:
                continue
            root = t.series_root if t.series_root is not None else ("solo", t.mal_id)
            row = (model.affinity(t.attrs), model._baseline_pred(t.community),
                   t.user_score, root)
            rows["all"].append(row)
            rows["cont" if t.series_root in train_roots else "new"].append(row)
        interval = getattr(model, "resid_std", None)
        windows.append({
            "start": w["start"], "end": w["end"],
            "n_train": len(train_titles), "n_test": len(rows["all"]),
            "scale": getattr(model, "scale", None),
            "cv_rmse": getattr(model, "cv_rmse", None),
            "groups": {k: metrics(v, interval) for k, v in rows.items()},
        })
        for k in pooled:
            pooled[k].extend(rows[k])
    return {
        "cuts": cuts,
        "windows": windows,
        "pooled": {k: {**metrics(v),
                       "ci95": bootstrap_spearman(v, n_boot=n_boot, seed=seed)}
                   for k, v in pooled.items()},
    }


def format_report(res: dict) -> list[str]:
    """Řádky pro CLI výpis."""
    if not res.get("windows"):
        return ["", "── Časová validace ──────────", "  málo datovaných titulů "
                "(my_finish_date) na smysluplná okna"]
    n = len(res["windows"])
    okna = ("disjunktní okno" if n == 1 else
            "disjunktní okna" if n < 5 else "disjunktních oken")
    lines = ["", f"── Časová validace ({n} {okna}, "
                 f"řezy {', '.join(res['cuts'])}) ──────────",
             "  okno                     trénink  test   Spearman   RMSE baseline→model "
             "(bez biasu)   bias",
             ]
    for w in res["windows"]:
        g = w["groups"]["all"]
        sp = f"{g['spearman']:+.3f}" if g["spearman"] is not None else "   —  "
        lines.append(
            f"  {w['start']} → {(w['end'] or 'dnes'):10s} {w['n_train']:7d} "
            f"{w['n_test']:5d}   {sp}    {g['rmse_baseline']:.3f} → "
            f"{g['rmse_model']:.3f} ({g['rmse_model_debiased']:.3f})   "
            f"{g['bias']:+.2f}")
    lines.append("  dohromady (disjunktní okna, 95% interval klastrovaný po franšízách):")
    for key, label in (("all", "vše"), ("new", "nové franšízy"),
                       ("cont", "pokračování")):
        m = res["pooled"][key]
        if not m.get("n"):
            continue
        sp = f"{m['spearman']:+.3f}" if m["spearman"] is not None else "—"
        ci = m.get("ci95")
        ci_txt = f" [{ci[0]:+.2f}; {ci[1]:+.2f}]" if ci else ""
        cov = f" · pokrytí intervalu {m['coverage']:.0%}" if m.get("coverage") else ""
        lines.append(f"    {label:16s} n={m['n']:4d} · Spearman {sp}{ci_txt} · "
                     f"RMSE {m['rmse_baseline']:.3f} → {m['rmse_model']:.3f} "
                     f"({m['rmse_model_debiased']:.3f}) · bias {m['bias']:+.2f}{cov}")
    lines.append("  (Spearman = afinita vs. reziduum; RMSE v závorce = po "
                 "odečtení systematického posunu)")
    return lines
