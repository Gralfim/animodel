"""
cli.py — Orchestrace celého běhu.

  python -m animodel --export animelist.xml [--config config.yaml] [--out output]
  python -m animodel --export animelist.xml --no-recommend     # jen model
  python -m animodel --export animelist.xml --no-anilist       # jen Jikan
  python -m animodel --export animelist.xml --analyze          # jen přehled franšíz
  python -m animodel --export animelist.xml --gen-intensity    # jen (re)generace intensity.yaml

Jediný vstup = MAL XML export. Žádný ruční mezikrok: stáhne metadata, postaví
model, vygeneruje model.html a recommendations.html.
"""
from __future__ import annotations

import argparse
import logging
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

    if args.analyze:
        from .series import print_series_groups
        enr = Enricher(cfg)
        # print_series_groups potřebuje jen Jikan data (relations), ne AniList --
        # cachované z předchozích běhů, takže tohle typicky nic nestahuje naživo
        jdata = enr.jikan.get_anime_batch([e.mal_id for e in completed], show_progress=True)
        titles_map = {mid: (j or {}).get("title", str(mid)) for mid, j in jdata.items()}
        print_series_groups(completed, jdata, titles_map)
        return 0

    if args.gen_intensity:
        from .intensity import generate_lexicon
        enr = Enricher(cfg)
        if not enr.anilist:
            print("[pozn.] AniList vypnutý (--no-anilist / use_anilist: false) — "
                  "universum bude jen z MAL žánrů/témat, bez AniList tagů.")
        # Frekvence atributů z tvého seznamu — jen k prioritizaci revize
        # (nejčastější první). Po normálním běhu jde vše z cache; na studené
        # cache tohle stáhne metadata stejně jako běžný běh.
        print("[1/2] Počítám frekvence atributů z tvého seznamu (z cache) …")
        enriched = enr.enrich_ids([e.mal_id for e in completed], show_progress=True)
        counts: dict[str, int] = {}
        for en in enriched.values():
            for key, av in en.attrs.items():
                if av.category in ("genre", "theme", "tag"):
                    counts[key] = counts.get(key, 0) + 1
        print("[2/2] Stahuji universum tagů (AniList MediaTagCollection + MAL žánry) …")
        stats = generate_lexicon(
            cfg.model.intensity_lexicon,
            jikan=enr.jikan, anilist=enr.anilist, observed_counts=counts,
        )
        print(f"      → {cfg.model.intensity_lexicon}: {stats['total']} klíčů "
              f"({stats['from_existing']} tvých zachováno, "
              f"{stats['from_curated']} prefill, {stats['from_prior']} kategorie-prior, "
              f"{stats['zero']} neutrálních k případné revizi"
              + (f", {stats['custom_kept']} vlastních mimo universum" if stats['custom_kept'] else "")
              + ")")
        print("      Zreviduj hodnoty (nejčastější atributy jsou v sekcích nahoře) a "
              "spusť normální běh.")
        return 0

    print(f"[2/5] Stahuju metadata (Jikan{' + AniList' if cfg.enrich.use_anilist else ''}) …")
    enr = Enricher(cfg)
    titles = enr.build_titles(completed, show_progress=True)
    print(f"      obohaceno {len(titles)} titulů")

    print(f"[3/5] Stavím model vkusu (shrinkage K={cfg.model.shrinkage_k:g}) …")
    from .intensity import load_lexicon
    lexicon = load_lexicon(cfg.model.intensity_lexicon)
    if lexicon is None:
        print(f"      [pozn.] {cfg.model.intensity_lexicon} neexistuje — osa náročnosti "
              f"jede na vestavěném defaultu; vygeneruj vlastní přes --gen-intensity")
    model = TasteModel(
        shrinkage_k=cfg.model.shrinkage_k,
        min_attr_count=cfg.model.min_attr_count,
        interaction_min_count=cfg.model.interaction_min_count,
        interaction_min_lift=cfg.model.interaction_min_lift,
        intensity=lexicon,
    )
    model.fit(titles, n_clusters=cfg.model.n_clusters)
    print(f"      β={model.beta:+.2f} · CV RMSE {model.cv_rmse:.3f} "
          f"(baseline {model.baseline_rmse:.3f}) · {len(model.clusters)} nálad")

    unrated = model.unrated_intensity_attrs(top=12)
    if unrated:
        listed = ", ".join(f"{label} ({n:.0f}×)" for _key, label, n in unrated)
        print(f"      [pozn.] osa náročnosti: {len(model.unrated_intensity_attrs())} "
              f"pozorovaných atributů bez záznamu v lexikonu, nejčastější: {listed} "
              f"— doplň regenerací (--gen-intensity, tvé hodnoty se zachovají)")

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
    # Stáhneme celý ohodnocený pool (bez ořezu) — globální view ho ořízne na top_n,
    # per-klastr view pak pro každou náladu ukáže vlastních top_per_cluster.
    recs_all = rec.recommend(titles, ptw_ids=ptw_ids, watched_ids=watched_ids,
                             show_progress=True, limit=None)
    recs = recs_all[: cfg.recommend.top_n]
    print(f"      {len(recs_all)} kandidátů celkem, top {len(recs)} do globálního přehledu")

    print(f"[5/5] Generuji HTML …")
    rec_html = os.path.join(cfg.out_dir, "recommendations.html")
    report.render_recommendations_html(recs, rec_html, userinfo)
    print(f"      → {rec_html}")
    mood_html = os.path.join(cfg.out_dir, "recommendations_by_mood.html")
    report.render_cluster_recommendations_html(
        recs_all, model, mood_html, userinfo,
        top_per_cluster=cfg.recommend.top_per_cluster,
    )
    print(f"      → {mood_html}")

    # CF standalone report — generuje se jen pokud use_user_cf a jsou výsledky
    cf_recs = getattr(rec, "_cf_raw_results", [])
    if cfg.recommend.use_user_cf and cf_recs:
        cf_html = os.path.join(cfg.out_dir, "cf_recommendations.html")

        # Primární zdroj titulů: enriched Recommendation objekty z recs_all
        enr_data = {r.mal_id: r for r in recs_all if r.mal_id}

        # Doplňující zdroj: AniList batch query pro CF tituly chybějící v enr_data
        # (tituly které CF našlo, ale enricher vyloučil — např. nehodnocené sledované)
        missing_ids = [
            r["mal_id"] for r in cf_recs
            if r.get("mal_id") and r["mal_id"] not in enr_data
            and not r.get("title")   # title_store ho nemá — potřebujeme lookup
        ]
        al_titles: dict[int, str] = {}
        al_titles_en: dict[int, str] = {}
        if missing_ids and rec.enr.anilist:
            al_batch = rec.enr.anilist.get_anime_batch(
                missing_ids, show_progress=False
            )
            for mid, adata in al_batch.items():
                if adata:
                    t = adata.get("title") or {}
                    al_titles[mid]    = t.get("romaji") or t.get("english") or ""
                    al_titles_en[mid] = t.get("english") or ""

        # Přidej al_titles do cf_recs jako fallback (in-place, jen pro chybějící)
        for r in cf_recs:
            mid = r.get("mal_id")
            if mid and not r.get("title") and mid in al_titles:
                r["title"]    = al_titles[mid]
                r["title_en"] = al_titles_en.get(mid, "")

        report.render_cf_recommendations_html(cf_recs, cf_html, userinfo, enr_data)
        print(f"      → {cf_html}  ({len(cf_recs)} CF titulů)")
    elif cfg.recommend.use_user_cf:
        print("      [CF report přeskočen — žádné výsledky]")

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
    p.add_argument("--analyze", action="store_true",
                   help="vypiš přehled nalezených franšízových skupin a skonči "
                        "(print_series_groups byla v kódu, ale bez cesty ven z CLI)")
    p.add_argument("--gen-intensity", action="store_true",
                   help="vygeneruj/aktualizuj intensity.yaml (osa emocionální "
                        "náročnosti) z úplného universa AniList tagů + MAL "
                        "žánrů/témat a skonči; existující hodnoty se zachovají, "
                        "řazení podle frekvence ve tvém seznamu")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="ukaž i běžné retry/rate-limit hlášky (INFO), ne jen "
                        "skutečné chyby -- default je jen WARNING a výš, ať "
                        "log neutopí progress v routinních 429 retry zprávách")
    args = p.parse_args(argv)
    if not args.export:
        p.error("chybí --export (cesta k MAL XML exportu)")

    # Bez tohohle žádný handler nikdy nebyl explicitně nastavený -- Python
    # spadl na `logging.lastResort`, který ukazuje jen WARNING+ BEZ formátu
    # (žádné "WARNING:", žádný čas, jen holá zpráva) -- warningy tak vypadaly
    # identicky jako běžný print() text, jen se navíc chovaly jinak
    # (stderr, nebufferované) než progress výpisy (stdout, bufferované), což
    # dělalo dojem, že log je plný "chyb" a progress info chybí.
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
