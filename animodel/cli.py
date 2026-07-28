"""
cli.py — Orchestrace celého běhu.

  python -m animodel --export animelist.xml [--config config.yaml] [--out output]
  python -m animodel --export animelist.xml --no-recommend     # jen model
  python -m animodel --export animelist.xml --no-anilist       # jen Jikan
  python -m animodel --export animelist.xml --analyze          # jen přehled franšíz
  python -m animodel --export animelist.xml --analyze-attrs    # diagnostika kanonizace
  python -m animodel --export animelist.xml --gen-intensity    # jen (re)generace intensity.yaml
  python -m animodel --export animelist.xml --season           # sezónní doporučení

Jediný vstup = MAL XML export. Žádný ruční mezikrok: stáhne metadata, postaví
model, vygeneruje model.html a recommendations.html.

STRUKTURA: `run()` jen připraví kontext a vybere režim; každý režim je vlastní
`run_*` funkce nad sdíleným `RunContext`. Dřív to byla jedna 215řádková funkce
se čtyřmi early-return větvemi a napevno psaným číslováním kroků, které se už
stihlo rozejít (`[4/4]` v sezónním režimu vs. `[1/5]`…`[5/5]` jinde). Číslování
teď drží `Steps` -- každý režim si řekne, kolik jich má, a pořadí se dopočítá
samo (HODNOCENI_PROJEKTU.md §2, §9.7).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

from .config import Config
from .mal import parse_export, split_by_status
from .enrich import Enricher
from .taste import TasteModel
from .recommend import Recommender
from . import report

log = logging.getLogger(__name__)


class Steps:
    """Číslovač kroků `[n/celkem]`. Existuje proto, aby se čísla nedala
    rozjet ručním přepisováním -- stačí říct, kolik kroků režim má."""

    def __init__(self, total: int):
        self.total = total
        self.n = 0

    def __call__(self, msg: str) -> None:
        self.n += 1
        if self.n > self.total:
            # Programátorská chyba (režim hlásí míň kroků, než dělá) -- dřív
            # se projevila jen tím, že se vytisklo "[3/2]" a nikdo si nevšiml.
            log.warning(f"kroků víc než ohlášeno ({self.n} > {self.total}) -- "
                        f"zkontroluj _mode_steps()")
        print(f"[{self.n}/{self.total}] {msg}")

    @staticmethod
    def detail(msg: str) -> None:
        """Odsazený řádek pod krokem (výsledky, cesty k souborům)."""
        print(f"      {msg}")


@dataclass
class RunContext:
    """Co potřebuje každý režim: načtený seznam + připravený enricher."""
    cfg: Config
    entries: list
    userinfo: dict
    by_status: dict
    completed: list          # Completed SE ZNÁMKOU -- vstup modelu
    enricher: Enricher
    steps: Steps

    @property
    def watched_ids(self) -> set:
        return {e.mal_id for e in self.entries
                if e.status in ("Completed", "Watching", "On-Hold", "Dropped")}

    @property
    def ptw_ids(self) -> set:
        return {e.mal_id for e in self.by_status.get("Plan to Watch", [])}

    def enrich_completed(self):
        return self.enricher.enrich_ids([e.mal_id for e in self.completed],
                                        show_progress=True)


# ── příprava ────────────────────────────────────────────────────────────────

def _apply_overrides(cfg: Config, args) -> None:
    """CLI přepíná nad configem (CLI vyhrává)."""
    if args.export:
        cfg.mal_export = args.export
    if args.out:
        cfg.out_dir = args.out
    if args.cache:
        cfg.cache_dir = args.cache
    if args.no_anilist:
        cfg.enrich.use_anilist = False
    if args.no_jikan:
        cfg.enrich.use_jikan = False
    if args.shrinkage is not None:
        cfg.model.shrinkage_k = args.shrinkage
    if args.user_cf:
        cfg.recommend.use_user_cf = True


def _check(cfg: Config) -> str | None:
    """Vrátí chybovou hlášku, nebo None když je vše v pořádku."""
    if not cfg.enrich.use_jikan and not cfg.enrich.use_anilist:
        return "--no-jikan a --no-anilist zároveň = žádný zdroj metadat"
    if not os.path.exists(cfg.mal_export):
        return f"MAL export nenalezen: {cfg.mal_export}"
    return None


def _exclude_own_account(cfg: Config, userinfo: dict) -> None:
    """
    Vlastní účet nesmí vyjít jako senpai: import vlastního MAL seznamu na
    AniList má podobnost 1.00 a doporučil by ti jen to, co už máš. Jméno z
    MAL exportu se vyloučí automaticky -- funguje, jen když máš na AniListu
    stejnou přezdívku; jinak (nebo pro alt účty) přidej ručně do
    recommend.user_cf_exclude_users.
    """
    mal_user = (userinfo.get("user_name") or "").strip()
    if mal_user and mal_user.lower() not in {
        n.strip().lower() for n in cfg.recommend.user_cf_exclude_users
    }:
        cfg.recommend.user_cf_exclude_users = list(
            cfg.recommend.user_cf_exclude_users) + [mal_user]


def prepare(cfg: Config, steps: Steps) -> RunContext:
    """Naparsuje export a postaví enricher -- společný začátek všech režimů."""
    steps(f"Parsuju {cfg.mal_export} …")
    entries, userinfo = parse_export(cfg.mal_export)
    by_status = split_by_status(entries)
    completed = [e for e in by_status.get("Completed", []) if e.score and e.score > 0]
    Steps.detail(f"{len(entries)} záznamů · {len(completed)} ohodnocených · "
                 f"{len(by_status.get('Plan to Watch', []))} PTW")
    _exclude_own_account(cfg, userinfo)
    return RunContext(cfg=cfg, entries=entries, userinfo=userinfo,
                      by_status=by_status, completed=completed,
                      enricher=Enricher(cfg), steps=steps)


def _build_stats(model, by_status) -> dict:
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


def fit_model(ctx: RunContext):
    """Obohatí seznam, postaví model vkusu a vyrenderuje model.html."""
    cfg = ctx.cfg
    ctx.steps(f"Stahuju metadata (Jikan"
              f"{' + AniList' if cfg.enrich.use_anilist else ''}) …")
    titles = ctx.enricher.build_titles(ctx.completed, show_progress=True)
    Steps.detail(f"obohaceno {len(titles)} titulů")

    ctx.steps(f"Stavím model vkusu (shrinkage K={cfg.model.shrinkage_k:g}) …")
    from .intensity import load_lexicon
    lexicon = load_lexicon(cfg.model.intensity_lexicon)
    if lexicon is None:
        Steps.detail(f"[pozn.] {cfg.model.intensity_lexicon} neexistuje — osa "
                     f"náročnosti jede na vestavěném defaultu; vygeneruj "
                     f"vlastní přes --gen-intensity")
    model = TasteModel(
        shrinkage_k=cfg.model.shrinkage_k,
        min_attr_count=cfg.model.min_attr_count,
        interaction_min_count=cfg.model.interaction_min_count,
        interaction_min_lift=cfg.model.interaction_min_lift,
        interaction_triples=cfg.model.interaction_triples,
        intensity=lexicon,
    )
    model.fit(titles, n_clusters=cfg.model.n_clusters)
    Steps.detail(f"β={model.beta:+.2f} · scale {model.scale:.2f} · "
                 f"CV RMSE {model.cv_rmse:.3f} "
                 f"(baseline {model.baseline_rmse:.3f}) · "
                 f"{len(model.clusters)} nálad")
    if model.triples:
        # Kalibrace běží AŽ PO fitu trojic, takže cv_rmse popisuje model, který
        # doopravdy predikuje; rozdíl proti cv_rmse_no_triples je poctivá
        # odpověď, jestli se experiment s trojicemi vyplácí.
        delta = model.cv_rmse_no_triples - model.cv_rmse
        Steps.detail(f"{len(model.triples)} trojic · scale_triples "
                     f"{model.scale_triples:.2f} · bez trojic by CV RMSE bylo "
                     f"{model.cv_rmse_no_triples:.3f} ({delta:+.4f})")

    unrated = model.unrated_intensity_attrs(top=12)
    if unrated:
        listed = ", ".join(f"{label} ({n:.0f}×)" for _key, label, n in unrated)
        Steps.detail(f"[pozn.] osa náročnosti: "
                     f"{len(model.unrated_intensity_attrs())} pozorovaných "
                     f"atributů bez záznamu v lexikonu, nejčastější: {listed} "
                     f"— doplň regenerací (--gen-intensity, tvé hodnoty se "
                     f"zachovají)")

    model_html = os.path.join(cfg.out_dir, "model.html")
    report.render_model_html(model, ctx.userinfo,
                             _build_stats(model, ctx.by_status), model_html)
    Steps.detail(f"→ {model_html}")
    return model, titles


# ── režimy ──────────────────────────────────────────────────────────────────

def run_analyze(ctx: RunContext) -> int:
    """--analyze: přehled franšízových skupin."""
    from .series import print_series_groups
    ctx.steps("Načítám franšízové vazby (z cache) …")
    # Relations přes Enricher.relations_data -- primárně Jikan, per-titul
    # fallback AniList, takže --analyze funguje i v --no-jikan režimu.
    enriched = ctx.enrich_completed()
    rel_data = ctx.enricher.relations_data(enriched)
    titles_map = {mid: en.title for mid, en in enriched.items()}
    print_series_groups(ctx.completed, rel_data, titles_map)
    return 0


def run_analyze_attrs(ctx: RunContext) -> int:
    """
    --analyze-attrs: hledá dvojice atributů, které vypadají jako týž koncept
    zapsaný dvakrát a ALIAS je nespojuje.

    Tichý selhací mód `attributes.py`: nikde to nespadne, jen se evidence
    rozdělí na dvě poloviny, obě se silněji smrští k nule a efekt zmizí --
    přesně to, čemu má kanonizace bránit. Z běžného výstupu to poznat nejde.
    """
    from .attributes import find_near_duplicate_keys
    ctx.steps("Sbírám pozorované atributy (z cache) …")
    enriched = ctx.enrich_completed()
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    cats: dict[str, str] = {}
    for en in enriched.values():
        for key, av in en.attrs.items():
            counts[key] = counts.get(key, 0) + 1
            labels.setdefault(key, av.label)
            cats.setdefault(key, av.category)
    Steps.detail(f"{len(counts)} unikátních atributů z {len(enriched)} titulů")

    ctx.steps("Hledám možné duplicity kanonizace …")
    pairs = find_near_duplicate_keys(counts)
    if not pairs:
        Steps.detail("nic podezřelého — žádné dva klíče si nejsou blízké")
        return 0

    # řadíme podle DOPADU: dvojice, kterou nese hodně titulů, stojí za revizi
    # dřív než dvojice se dvěma výskyty
    pairs.sort(key=lambda p: -(counts.get(p[0], 0) + counts.get(p[1], 0)))
    Steps.detail(f"{len(pairs)} dvojic k revizi "
                 f"(seřazeno podle četnosti = podle dopadu):")
    print()
    for a, b, why, _d in pairs:
        na, nb = counts.get(a, 0), counts.get(b, 0)
        print(f"  {labels.get(a, a)} ({cats.get(a, '?')}, {na}×)"
              f"  ↔  {labels.get(b, b)} ({cats.get(b, '?')}, {nb}×)")
        print(f"      {a} / {b} — {why}")
    print()
    print("  Pokud jde o TÝŽ koncept, přidej do attributes.ALIAS mapování na")
    print("  jeden klíč. Pokud ne (např. 'josei' vs. 'josei_fantasy'), nedělej")
    print("  nic — nástroj hlásí podezření, ne chybu.")
    return 0


def run_gen_intensity(ctx: RunContext) -> int:
    """--gen-intensity: (re)generace intensity.yaml z universa atributů."""
    from .intensity import generate_lexicon
    cfg = ctx.cfg
    if not ctx.enricher.anilist:
        Steps.detail("[pozn.] AniList vypnutý (--no-anilist / use_anilist: "
                     "false) — universum bude jen z MAL žánrů/témat, bez "
                     "AniList tagů.")
    # Frekvence atributů z tvého seznamu — jen k prioritizaci revize
    # (nejčastější první). Po normálním běhu jde vše z cache.
    ctx.steps("Počítám frekvence atributů z tvého seznamu (z cache) …")
    enriched = ctx.enrich_completed()
    counts: dict[str, int] = {}
    for en in enriched.values():
        for key, av in en.attrs.items():
            if av.category in ("genre", "theme", "tag"):
                counts[key] = counts.get(key, 0) + 1

    ctx.steps("Stahuji universum tagů (AniList MediaTagCollection + MAL žánry) …")
    stats = generate_lexicon(
        cfg.model.intensity_lexicon,
        jikan=ctx.enricher.jikan, anilist=ctx.enricher.anilist,
        observed_counts=counts,
    )
    Steps.detail(
        f"→ {cfg.model.intensity_lexicon}: {stats['total']} klíčů "
        f"({stats['from_existing']} tvých zachováno, "
        f"{stats['from_curated']} prefill, {stats['from_prior']} kategorie-prior, "
        f"{stats['zero']} neutrálních k případné revizi"
        + (f", {stats['custom_kept']} vlastních mimo universum"
           if stats['custom_kept'] else "")
        + ")")
    Steps.detail("Zreviduj hodnoty (nejčastější atributy jsou v sekcích "
                 "nahoře) a spusť normální běh.")
    return 0


def run_season(ctx: RunContext, model, season_arg) -> int:
    """--season: pokračování mých sérií + nové tituly vysílané sezóny."""
    from .season import build_season_view, parse_season_arg
    cfg = ctx.cfg
    year, season = parse_season_arg(season_arg)
    ctx.steps(f"Sezónní doporučení: {season} {year} …")
    my_scores = {e.mal_id: float(e.score) for e in ctx.completed}
    sequels, new_titles = build_season_view(
        model, ctx.enricher, my_scores, ctx.watched_ids, ctx.ptw_ids,
        year, season, cfg.recommend, show_progress=True)
    Steps.detail(f"{len(sequels)} pokračování tvých sérií · "
                 f"{len(new_titles)} nových titulů")
    season_html = os.path.join(cfg.out_dir, "recommendations_season.html")
    report.render_season_html(sequels, new_titles, season_html,
                              year, season, ctx.userinfo)
    Steps.detail(f"→ {season_html}")
    return 0


def _render_cf_report(ctx: RunContext, result, recs_all) -> None:
    """Standalone CF report (jen když user-CF něco našlo)."""
    cfg = ctx.cfg
    cf_html = os.path.join(cfg.out_dir, "cf_recommendations.html")
    # Primární zdroj titulů: enriched Recommendation objekty z recs_all
    enr_data = {r.mal_id: r for r in recs_all if r.mal_id}
    # Doplňující zdroj: AniList batch pro CF tituly chybějící v enr_data
    # (tituly, které CF našlo, ale enricher vyloučil).
    missing_ids = [r["mal_id"] for r in result.cf_raw
                   if r.get("mal_id") and r["mal_id"] not in enr_data
                   and not r.get("title")]
    if missing_ids and ctx.enricher.anilist:
        al_batch = ctx.enricher.anilist.get_anime_batch(missing_ids,
                                                        show_progress=False)
        for r in result.cf_raw:
            adata = al_batch.get(r.get("mal_id")) if not r.get("title") else None
            if adata:
                t = adata.get("title") or {}
                r["title"] = t.get("romaji") or t.get("english") or ""
                r["title_en"] = t.get("english") or ""

    report.render_cf_recommendations_html(
        result.cf_raw, cf_html, ctx.userinfo, enr_data,
        watched_ids=ctx.watched_ids, senpai=result.senpai,
        top=cfg.recommend.user_cf_report_top)
    cap = cfg.recommend.user_cf_report_top
    shown = min(len(result.cf_raw), cap) if cap else len(result.cf_raw)
    Steps.detail(f"→ {cf_html}  ({shown} z {len(result.cf_raw)} CF titulů)")


def _record_history(ctx: RunContext, recs_all, model) -> None:
    """
    Zapíše běh do historie a vyhodnotí starší snapshoty.

    Klíčem je OTISK STAVU SEZNAMU, ne datum -- ladicí běhy nad týmž exportem
    přepíšou jeden snapshot místo aby jich nasypaly desítky (viz history.py).
    Diagnostika nikdy nesmí shodit běh.
    """
    try:
        from . import history
        cfg = ctx.cfg
        snap = history.build_snapshot(ctx.entries, recs_all, model, cfg,
                                      top=cfg.recommend.history_top)
        older = [s for s in history.load_snapshots(cfg.history_dir)
                 if s.fingerprint != snap.fingerprint]
        path = history.save_snapshot(snap, cfg.history_dir)
        Steps.detail(f"→ {path}  (otisk seznamu {snap.fingerprint})")
        results = [r for r in (history.evaluate(s, ctx.entries) for s in older) if r]
        for line in history.format_report(results):
            print(line)
        if older and not results:
            Steps.detail("[historie] starší snapshoty zatím bez měřitelného "
                         "výsledku (žádné doporučení jsi mezitím nedokoukal)")
    except Exception:
        log.exception("historie: záznam/vyhodnocení selhalo -- běh pokračuje")


def run_recommend(ctx: RunContext, model, titles, save_history: bool) -> int:
    """Plný běh: doporučení + HTML výstupy + historie."""
    cfg = ctx.cfg
    ctx.steps("Hledám doporučení …")
    rec = Recommender(model, ctx.enricher, cfg)
    # Celý ohodnocený pool (bez ořezu) -- globální view ho ořízne na top_n,
    # per-klastr view pak pro každou náladu ukáže vlastních top_per_cluster.
    result = rec.recommend(titles, ptw_ids=ctx.ptw_ids,
                           watched_ids=ctx.watched_ids,
                           show_progress=True, limit=None)
    recs_all = result.recs
    recs = recs_all[: cfg.recommend.top_n]
    Steps.detail(f"{len(recs_all)} kandidátů celkem, top {len(recs)} "
                 f"do globálního přehledu")

    ctx.steps("Generuji HTML …")
    rec_html = os.path.join(cfg.out_dir, "recommendations.html")
    report.render_recommendations_html(recs, rec_html, ctx.userinfo)
    Steps.detail(f"→ {rec_html}")
    mood_html = os.path.join(cfg.out_dir, "recommendations_by_mood.html")
    report.render_cluster_recommendations_html(
        recs_all, model, mood_html, ctx.userinfo,
        top_per_cluster=cfg.recommend.top_per_cluster)
    Steps.detail(f"→ {mood_html}")

    if cfg.recommend.use_user_cf:
        if result.cf_raw:
            _render_cf_report(ctx, result, recs_all)
        else:
            Steps.detail("[CF report přeskočen — žádné výsledky]")

    if cfg.recommend.save_history and save_history:
        _record_history(ctx, recs_all, model)
    return 0


# ── vstupní bod ─────────────────────────────────────────────────────────────

#: kolik kroků vypíše `prepare()` -- společný začátek všech režimů
PREPARE_STEPS = 1


def _mode_steps(args) -> int:
    """
    Kolik kroků vypíše zvolený režim POTOM, co doběhne `prepare()`.
    Celkem = PREPARE_STEPS + tohle (viz `run()`).

    Záměrně se počítají jen kroky režimu: dřív tahle funkce vracela rovnou
    součet a hned se spletla (`--analyze-attrs` má dva vlastní kroky, ne
    jeden, takže se tisklo `[3/2]`). Když je `prepare` mimo, drží se to samo.
    """
    if args.analyze:
        return 1                        # franšízové vazby
    if args.analyze_attrs:
        return 2                        # sběr atributů + hledání duplicit
    if args.gen_intensity:
        return 2                        # frekvence + universum
    if args.no_recommend:
        return 2                        # metadata + model
    if args.season is not None:
        return 3                        # + sezóna
    return 4                            # + doporučení + HTML


def run(args) -> int:
    cfg = Config.load(args.config)
    _apply_overrides(cfg, args)
    problem = _check(cfg)
    if problem:
        print(f"[chyba] {problem}", file=sys.stderr)
        return 2
    if not cfg.enrich.use_jikan:
        print("[pozn.] Nouzový režim bez Jikanu: žánry/synopse/dekáda/franšízy "
              "z AniListu, bez MAL rec grafu (slabší CF signál).")
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    ctx = prepare(cfg, Steps(PREPARE_STEPS + _mode_steps(args)))

    # režimy, které model nepotřebují
    if args.analyze:
        return run_analyze(ctx)
    if args.analyze_attrs:
        return run_analyze_attrs(ctx)
    if args.gen_intensity:
        return run_gen_intensity(ctx)

    model, titles = fit_model(ctx)
    if args.no_recommend:
        print("[hotovo] (doporučení přeskočena)")
        return 0
    if args.season is not None:
        rc = run_season(ctx, model, args.season)
    else:
        rc = run_recommend(ctx, model, titles, save_history=not args.no_history)
    print("[hotovo]")
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="animodel",
                                description="Model anime vkusu + doporučení z MAL exportu.")
    p.add_argument("--export", "-e", help="cesta k MAL XML exportu")
    p.add_argument("--config", "-c", help="volitelný config.yaml s laděním parametrů")
    p.add_argument("--out", "-o", help="výstupní složka (default: output)")
    p.add_argument("--cache", help="složka cache (default: cache)")
    p.add_argument("--shrinkage", type=float, help="přepiš shrinkage K")
    p.add_argument("--no-anilist", action="store_true", help="použij jen Jikan/MAL")
    p.add_argument("--no-jikan", action="store_true",
                   help="nouzový AniList-only režim (např. při výpadku Jikan "
                        "API): žánry/synopse/dekáda/franšízy z AniListu, "
                        "MAL rec graf se přeskočí")
    p.add_argument("--no-recommend", action="store_true", help="jen model, bez doporučení")
    p.add_argument("--user-cf", action="store_true", help="zapni user-based CF (pomalé)")
    p.add_argument("--no-history", action="store_true",
                   help="nezapisuj tenhle běh do historie (viz history.py); "
                        "běžně není potřeba -- snapshoty se klíčují otiskem "
                        "seznamu, takže ladicí běhy nad týmž exportem "
                        "přepisují jeden záznam, nehromadí se")
    p.add_argument("--season", nargs="*", metavar="ROK SEZÓNA",
                   help="doporučení pro vysílanou sezónu (pokračování tvých sérií "
                        "+ nové tituly, s datem posledního dílu). Bez argumentu = "
                        "aktuální sezóna; nebo napevno např. '--season 2026 summer'")
    p.add_argument("--analyze", action="store_true",
                   help="vypiš přehled nalezených franšízových skupin a skonči")
    p.add_argument("--analyze-attrs", action="store_true",
                   help="diagnostika kanonizace: vypiš dvojice atributů, které "
                        "vypadají jako týž koncept zapsaný dvakrát a ALIAS je "
                        "nespojuje (tichý selhací mód -- efekt se rozdělí na "
                        "dvě poloviny a obě se smrští k nule)")
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
    #
    # ProgressAwareLogHandler navíc spolupracuje s \r progress řádkou:
    # před vypsáním záznamu ji smaže a po něm překreslí, takže série
    # warningů (např. při výpadku Jikanu) roluje NAD progress řádkem,
    # místo aby ji rozbila a schovala aktuální stav stahování.
    from .sources import ProgressAwareLogHandler
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        handlers=[ProgressAwareLogHandler(sys.stderr)],
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
