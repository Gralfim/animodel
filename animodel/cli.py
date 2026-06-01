"""
cli.py — Orchestrace celého běhu.

  python -m animodel --export animelist.xml [--config config.yaml] [--out output]
  python -m animodel --export animelist.xml --no-recommend     # jen model
  python -m animodel --export animelist.xml --no-anilist       # jen Jikan

Jediný vstup = MAL XML export. Žádný ruční mezikrok: stáhne metadata, postaví
model, vygeneruje model.html a recommendations.html.
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import Config
from .mal import parse_export, split_by_status
from .enrich import Enricher
from .taste import TasteModel
from .recommend import Recommender
from . import report


def _build_stats(model, by_status, titles) -> dict:
    dist = {}
    rated = [e for e in by_status.get("Completed", []) if e.score and e.score > 0]
    for e in rated:
        dist[e.score] = dist.get(e.score, 0) + 1
    return {
        "n_rated": len(rated),
        "n_ptw": len(by_status.get("Plan to Watch", [])),
        "n_completed": len(by_status.get("Completed", [])),
        "dist": dist,
        "baseline_rmse": getattr(model, "baseline_rmse", model.cv_rmse),
    }


def run(args) -> int:
    cfg = Config.load(args.config)
    if args.export:
        cfg.mal_export = args.export
    if args.out:
        cfg.out_dir = args.out
    if args.cache:
        cfg.cache_dir = args.cache
    if args.no_anilist:
        cfg.enrich.use_anilist = False
    if args.shrinkage is not None:
        cfg.model.shrinkage_k = args.shrinkage
    if args.user_cf:
        cfg.recommend.use_user_cf = True

    if not os.path.exists(cfg.mal_export):
        print(f"[chyba] MAL export nenalezen: {cfg.mal_export}", file=sys.stderr)
        return 2
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    print(f"[1/5] Parsuju {cfg.mal_export} …")
    entries, userinfo = parse_export(cfg.mal_export)
    by_status = split_by_status(entries)
    completed = [e for e in by_status.get("Completed", []) if e.score and e.score > 0]
    print(f"      {len(entries)} záznamů · {len(completed)} ohodnocených · "
          f"{len(by_status.get('Plan to Watch', []))} PTW")

    print(f"[2/5] Stahuju metadata (Jikan{' + AniList' if cfg.enrich.use_anilist else ''}) …")
    enr = Enricher(cfg)
    titles = enr.build_titles(completed, show_progress=True)
    print(f"      obohaceno {len(titles)} titulů")

    print(f"[3/5] Stavím model vkusu (shrinkage K={cfg.model.shrinkage_k:g}) …")
    model = TasteModel(
        shrinkage_k=cfg.model.shrinkage_k,
        min_attr_count=cfg.model.min_attr_count,
        interaction_min_count=cfg.model.interaction_min_count,
        interaction_min_lift=cfg.model.interaction_min_lift,
    )
    model.fit(titles)
    model._fit_clusters(cfg.model.n_clusters)
    print(f"      β={model.beta:+.2f} · CV RMSE {model.cv_rmse:.3f} "
          f"(baseline {model.baseline_rmse:.3f}) · {len(model.clusters)} nálad")

    stats = _build_stats(model, by_status, titles)
    model_html = os.path.join(cfg.out_dir, "model.html")
    report.render_model_html(model, userinfo, stats, model_html)
    print(f"      → {model_html}")

    if args.no_recommend:
        print("[hotovo] (doporučení přeskočena)")
        return 0

    print(f"[4/5] Hledám doporučení …")
    watched_ids = {e.mal_id for e in entries
                   if e.status in ("Completed", "Watching", "On-Hold", "Dropped")}
    ptw_ids = {e.mal_id for e in by_status.get("Plan to Watch", [])}
    rec = Recommender(model, enr, cfg)
    recs = rec.recommend(titles, ptw_ids=ptw_ids, watched_ids=watched_ids, show_progress=True)
    print(f"      {len(recs)} doporučení")

    print(f"[5/5] Generuji HTML …")
    rec_html = os.path.join(cfg.out_dir, "recommendations.html")
    report.render_recommendations_html(recs, rec_html, userinfo)
    print(f"      → {rec_html}")
    print("[hotovo]")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="animodel",
                                description="Model anime vkusu + doporučení z MAL exportu.")
    p.add_argument("--export", "-e", help="cesta k MAL XML exportu")
    p.add_argument("--config", "-c", help="volitelný config.yaml s laděním parametrů")
    p.add_argument("--out", "-o", help="výstupní složka (default: output)")
    p.add_argument("--cache", help="složka cache (default: cache)")
    p.add_argument("--shrinkage", type=float, help="přepiš shrinkage K")
    p.add_argument("--no-anilist", action="store_true", help="použij jen Jikan/MAL")
    p.add_argument("--no-recommend", action="store_true", help="jen model, bez doporučení")
    p.add_argument("--user-cf", action="store_true", help="zapni user-based CF (pomalé)")
    args = p.parse_args(argv)
    if not args.export:
        p.error("chybí --export (cesta k MAL XML exportu)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
