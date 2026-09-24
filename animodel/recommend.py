"""
recommend.py — Generování a řazení doporučení dosud neshlédnutých anime.

Strategie (dvě nezávislé větve, sjednocené a deduplikované):

  A) ATRIBUTOVÁ / CONTENT větev
     - vezmi tvoje vysoce hodnocené tituly (seed = score >= high_score)
     - pro každý seed stáhni MAL + AniList "recommendations" (item-based CF graf)
     - navíc discovery: AniList tag-search na tvé nejcharakterističtější tagy
     => kandidáti, kteří jsou buď podobní oblíbeným, nebo nesou tvé silné atributy

  B) COLLABORATIVE / USER větev (volitelná, vypnutá defaultně)
     - „senpai" pipeline (usercf.py): pár uživatelů s ověřeně podobným
       vkusem na PLNÉM překryvu seznamů; doporučení = co hodnotí nad svůj
       osobní průměr (drahé přes AniList; zapíná se recommend.use_user_cf=True)

Skórování každého kandidáta (4 oddělené složky):
    composite = w_taste_fit * z(taste_fit)
              + w_cf        * z(log1p(item_votes))
              + w_user_cf   * z(user_votes)
              + w_quality   * z(community)
  kde
    taste_fit   = model predikuje afinitu (rezid. část) + shoda s nejbližším klastrem
    item_votes  = kolik seedů ho doporučilo + jejich hlasy/rating (graf podobnosti);
                  log1p tlumí šikmé rozdělení, jinak z-skóre outlierů přebije zbytek
    user_votes  = skóre z user-based CF (podobní uživatelé) -- vlastní složka,
                  ve sdíleném kbelíku s grafem se dřív utopilo
    community   = komunitní skóre (mírná preference kvality)
  Slabé hrany grafu (pod min_mal_rec_votes / min_anilist_rec_rating) se
  zahazují už při sběru -- jednotky hlasů jsou šum, ne podobnost.

Řadíme podle composite, NE podle predikované známky (ta se lepí na komunitní
průměr kvůli restrikci rozsahu — viz metodika). Predikovaná známka + interval
se počítá zvlášť jen pro zobrazení.

PTW tituly se z vyhledávání NEvyřazují, jen se označí příznakem `ptw`.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from .taste import TasteModel, Title
from .enrich import Enricher, Enriched, _is_side_content
from .attributes import AttrValue
from .series import build_roots, related_ids

log = logging.getLogger(__name__)

#: kolik příspěvků rozpadu predikce si karta nese (seřazené podle absolutní
#: hodnoty). Report z nich ukazuje zvlášť kladné („Pro") a záporné („Proti")
#: -- při dřívějších 6 často jedna strana úplně chyběla.
WHY_KEEP = 12


@dataclass
class Recommendation:
    mal_id: int
    title: str
    title_en: str
    community: float | None
    pred: float
    pred_lo: float
    pred_hi: float
    taste_fit: float
    cf_signal: float         # hlasy z grafu podobnosti (item-CF, po prazích)
    composite: float
    ptw: bool
    cluster_name: str
    why: list                # [(label, category, contribution, spoiler), ...] seřazeno
    cf_seeds: list           # názvy seedů, které tenhle titul "doporučily"
    synopsis: str = ""
    sources: list = field(default_factory=list)   # ['MAL-rec', 'AniList-rec', 'tag-search']
    user_cf_signal: float = 0.0   # skóre z user-based CF (oddělená složka)
    affinity: float = 0.0         # kalibrovaná afinita (model.affinity) a
    cluster_fit: float = 0.0      # shoda s náladou PŘED cluster_fit_weight --
                                  # obě složky taste_fit zvlášť, aby šla váha
                                  # nálad později přeladit nad poolem uloženým
                                  # v historii (taste_fit je jen jejich součet)
    # -- sezónní doporučení (season.py); u ostatních žebříčků zůstávají None --
    finale_date: str | None = None    # datum posledního dílu (ISO), pokud známé
    broadcast: str | None = None      # den vysílání (např. "Mondays")
    airing_status: str | None = None  # RELEASING / FINISHED / NOT_YET_RELEASED
    season_note: str | None = None    # např. "pokračování: X (tvá známka 9)"
    franchise_members: list = field(default_factory=list)
                                      # názvy dalších dílů téže franšízy, které
                                      # karta zastupuje (jedna karta = jedna
                                      # franšíza, viz Recommender._franchise_views)
    entry_note: str | None = None     # „začni od: X" u pokračování, jehož
                                      # předchozí díl jsi neviděl
    select: float = 0.0               # model výběru (selection.py): atributová
                                      # část logitu „sáhnu po tom vůbec?"
    avoided: list = field(default_factory=list)
                                      # labely atributů, kterým se obvykle
                                      # vyhýbám -- jen když výběr titul sráží
    popularity: int | None = None     # pořadí na MAL podle počtu členů
                                      # (sekce „znáš, ale nemáš v plánu")
    prequel_score: float = 0.0        # má známka předchozí řady (řazení sekce
                                      # „pokračování tvých sérií"). Dřív se to
                                      # na dataclass přišpendlovalo dynamicky
                                      # (`rec._prequel_score = …`) a četlo přes
                                      # getattr -- fungovalo, ale rozbilo by se
                                      # tiše při `slots=True` (HODNOCENI §2).


@dataclass
class RecommendResult:
    """
    Výstup jednoho běhu doporučování.

    Dřív se vracel jen `list[Recommendation]` a CF část se předávala privátními
    atributy (`rec._cf_raw_results`, `rec._cf_senpai`), které si volající tahal
    přes `getattr(..., [])`. Byl to implicitní kontrakt mezi dvěma moduly:
    přejmenování atributu by nic nerozbilo, jen by tiše zmizel CF report
    (HODNOCENI_PROJEKTU.md §2).
    """
    recs: list                      # seřazené Recommendation (CELÝ pool)
    senpai: list = field(default_factory=list)      # usercf.Senpai, pro CF report
    cf_raw: list = field(default_factory=list)      # syrové CF dicty, pro CF report
    z_params: dict = field(default_factory=dict)    # {složka: [mean, sd]} z-skóre
                                                    # kompozitu -- do snapshotu historie
    # -- pohledy nad týmž poolem (jedna karta na franšízu, viz _franchise_views) --
    discovery: list = field(default_factory=list)   # nové objevy: bez PTW a bez
                                                    # pokračování rozjetých sérií
    ptw_ranked: list = field(default_factory=list)  # „z tvého PTW" podle kompozitu
    continuations: list = field(default_factory=list)  # pokračování mých franšíz
    known: list = field(default_factory=list)       # populární, mimo seznam i PTW:
                                                    # „znáš, ale nemáš v plánu"
    selection: object = None                        # selection.SelectionModel
                                                    # (nebo None) -- pro výpis
    roots: dict = field(default_factory=dict)       # {mal_id: kořen franšízy}

    def __iter__(self):
        """Pohodlí volajícího: `for r in result` iteruje doporučení."""
        return iter(self.recs)

    def __len__(self):
        return len(self.recs)


class _ZScore:
    """z-skóre podle rozdělení `values` (robustní na konstantu), volatelné
    jako funkce. `mean`/`sd` jsou vidět zvenku: ukládají se do snapshotu
    historie, aby šel kompozit přesně přepočítat nad uloženými složkami
    poolu i s jinými vahami (history.py, HODNOCENI_PROJEKTU.md §9c)."""

    def __init__(self, values: list[float]):
        self.n = len(values)
        self.mean, self.sd = 0.0, 1.0
        if values:
            self.mean = sum(values) / self.n
            var = sum((v - self.mean) ** 2 for v in values) / self.n
            self.sd = math.sqrt(var) if var > 1e-9 else 1.0

    def __call__(self, x: float) -> float:
        return (x - self.mean) / self.sd if self.n else 0.0


def _z(values: list[float]) -> _ZScore:
    """Vrátí funkci pro z-skóre dle rozdělení values (robustní na konstantu)."""
    return _ZScore(values)


class Recommender:
    def __init__(self, model: TasteModel, enricher: Enricher, cfg):
        self.model = model
        self.enr = enricher
        self.cfg = cfg
        self.rc = cfg.recommend

    # ── Kandidáti ────────────────────────────────────────────────────────────

    def _seeds(self, titles: list[Title]) -> list[Title]:
        """
        Vysoce hodnocené tituly jako seedy, s limitem na franšízu
        (`seeds_per_franchise`): bez něj pětiřadá oblíbená franšíza sebere
        5 z max_seeds slotů a její (vzájemně skoro identické) rec grafy
        hlasují 5x -- kandidáti podobní franšíze pak dostávají násobný CF
        signál na úkor rozmanitosti.

        Pořadí = REZIDUUM (o kolik víc se mi titul líbil, než čeká baseline
        z komunity), pak známka, pak mal_id. Dřív se řadilo jen podle
        známky a remízy rozhodovalo pořadí exportu, tedy ABECEDA: ze 134
        devítek se seedem stalo prvních 10 od „2.5-jigen" po „Boku no
        Kokoro" a dva ze čtyř seedů, které přivedly tehdejší top-3, tam byly
        jen díky písmenu B (HODNOCENI_PROJEKTU.md §9d, nález #2). Reziduum
        navíc odliší osobní oblibu od obecně uznávané kvality: Steins;Gate
        (komunita 9,07) má při stejné desítce reziduum +1,19, Domestic na
        Kanojo +2,40.
        """
        resid = self.model.residuals()
        cand = [t for t in titles if t.user_score >= self.rc.high_score]
        cand.sort(key=lambda t: (-resid.get(t.mal_id, 0.0), -t.user_score,
                                 t.mal_id))
        cap = self.rc.seeds_per_franchise
        if not cap:
            return cand[: self.rc.max_seeds]
        per_franchise: dict[int, int] = {}
        seeds = []
        for t in cand:
            group = t.series_root if t.series_root is not None else t.mal_id
            if per_franchise.get(group, 0) >= cap:
                continue
            per_franchise[group] = per_franchise.get(group, 0) + 1
            seeds.append(t)
            if len(seeds) >= self.rc.max_seeds:
                break
        return seeds

    def _gather_candidates(self, titles: list[Title], seen_ids: set[int]):
        """
        Vrátí:
            cand_meta: {mal_id: {'item_votes': float, 'user_votes': float,
                                 'cf_seeds': [titul,...], 'sources': set}}

        item_votes = graf podobnosti (MAL/AniList/Shikimori), user_votes =
        user-based CF -- oddělené kbelíky, každý dostane vlastní z-skóre.
        """
        cand: dict[int, dict] = {}

        def bump(mid, votes, seed_title, source):
            if mid in seen_ids:
                return
            d = cand.setdefault(mid, {"item_votes": 0.0, "user_votes": 0.0,
                                      "cf_seeds": [], "sources": set()})
            d["user_votes" if source == "user-CF" else "item_votes"] += votes
            if seed_title and seed_title not in d["cf_seeds"]:
                d["cf_seeds"].append(seed_title)
            d["sources"].add(source)

        seeds = self._seeds(titles)
        seed_title_by_id = {t.mal_id: t.title for t in seeds}

        # Circuit breaker: klienti (jikan/anilist/shikimori) už sami zkoušej
        # retry+backoff PRO JEDEN request -- ale když je celá služba dole
        # (ne jen rate-limited), tohle by se opakovalo pro KAŽDÝ další seed
        # zvlášť a natáhlo běh o desítky minut zbytečného čekání.
        #
        # DŮLEŽITÉ (ověřeno živě, viz diskuze): try/except kolem volání NIC
        # nechytí, protože JikanClient/AniListClient/ShikimoriClient interní
        # selhání sami pohlcují a vrací prázdný list/`None` -- nikdy
        # nevyhodí výjimku ven (viz jejich _get()/_post(), poslední řádek je
        # vždy `return None`/`return []`, ne `raise`). První verze tohohle
        # breakeru byla postavená na except Exception a byla to fakticky
        # mrtvá větev -- vypadalo to opraveně, ale nedělalo to nic (ověřeno
        # instrumentovaným testem: 3 seedy pořád běžely celých ~85s KAŽDÝ,
        # ne jen první). Měř místo toho ELAPSED TIME bez ohledu na to, jestli
        # něco spadlo -- pomalá odpověď (protože klient interně vyčerpal
        # retry) je jediný spolehlivý signál, co k dispozici je.
        CIRCUIT_BREAKER_TIME_BUDGET = 20.0   # sekund promarněných na zdroj, než se zbytek dávky přeskočí
        SLOW_CALL_THRESHOLD = 5.0            # rychlá odpověď (i "nic nenalezeno") netrvá takhle dlouho
        fail_time = {"MAL-rec": 0.0, "AniList-rec": 0.0, "Shikimori": 0.0}
        tripped: set[str] = set()

        def call_source(name, fn):
            if name in tripped:
                return []
            t0 = time.time()
            try:
                result = fn() or []
            except Exception as exc:
                # klienti podle designu nevyhazují (viz pozn. výš), ale kdyby
                # se sem přece jen něco nečekaného dostalo (programátorská
                # chyba apod.), ať to nespadne celé -- jen zaloguj a pokračuj.
                log.warning(f"{name}: neočekávaná výjimka pro seed: {exc}")
                result = []
            elapsed = time.time() - t0
            if elapsed > SLOW_CALL_THRESHOLD:
                fail_time[name] += elapsed
                log.warning(
                    f"{name}: pomalá odpověď ({elapsed:.0f}s, pravděpodobně vyčerpané "
                    f"interní retry) -- promarněno celkem "
                    f"{fail_time[name]:.0f}/{CIRCUIT_BREAKER_TIME_BUDGET:.0f}s"
                )
                if fail_time[name] >= CIRCUIT_BREAKER_TIME_BUDGET:
                    tripped.add(name)
                    log.error(
                        f"{name}: {fail_time[name]:.0f}s promarněno na pomalých odpovědích -- "
                        f"vynechávám zbytek dávky (vypadá to na nedostupnou službu, ne jen rate limit)"
                    )
            return result

        # A1) item-based CF graf z MAL + AniList + Shikimori recommendations.
        # Slabé hrany (pod min_*_rec prahy) se zahazují ještě před ořezem na
        # candidates_per_seed -- jednotky hlasů / záporný rating jsou šum,
        # skutečně podobné série mívají desítky hlasů (empirie uživatele
        # potvrzená rozdělením: ~12 % AniList hran má rating 1-2).
        for s in seeds:
            # Jikan může být vypnutý (--no-jikan, nouzový AniList-only režim)
            if self.enr.jikan:
                recs = call_source("MAL-rec", lambda s=s: self.enr.jikan.get_recommendations(s.mal_id))
                # candidates_per_seed byl definovaný v configu, ale nikde se
                # nečetl -- změna hodnoty v config.yaml neměla žádný efekt.
                strong = [r for r in recs
                          if r.get("votes", 0) >= self.rc.min_mal_rec_votes]
                for r in strong[: self.rc.candidates_per_seed]:
                    # váž hlasy podle toho, jak moc seed miluju (user_score nad průměr)
                    w = max(0.1, s.user_score - self.model.u_mean + 1.0)
                    bump(r["mal_id"], (1 + math.log1p(r.get("votes", 0))) * w,
                         seed_title_by_id.get(s.mal_id), "MAL-rec")

            if self.enr.anilist:
                recs = call_source("AniList-rec", lambda s=s: self.enr.anilist.get_recommendations(s.mal_id))
                strong = [r for r in recs
                          if r.get("rating", 0) >= self.rc.min_anilist_rec_rating]
                for r in strong[: self.rc.candidates_per_seed]:
                    w = max(0.1, s.user_score - self.model.u_mean + 1.0)
                    bump(r["mal_id"], (1 + math.log1p(max(0, r.get("rating", 0)))) * w,
                         seed_title_by_id.get(s.mal_id), "AniList-rec")

            if self.enr.shikimori:
                recs = call_source("Shikimori", lambda s=s: self.enr.shikimori.get_similar(s.mal_id))
                # rank_hint = pozice v seznamu, ne potvrzené skóre podobnosti
                # (viz sources/shikimori.py docstring) -- proto tu není
                # log1p(votes)-style váhování jako u MAL/AniList-rec, jen
                # přímo rank_hint (0-1) × seed-love váha.
                for r in recs[: self.rc.candidates_per_seed]:
                    w = max(0.1, s.user_score - self.model.u_mean + 1.0)
                    bump(r["mal_id"], r.get("rank_hint", 0.5) * w,
                         seed_title_by_id.get(s.mal_id), "Shikimori")

        # A2) discovery přes tag-search na nejcharakterističtější atributy.
        # Sdílí circuit breaker s "AniList-rec" výš (stejná služba) -- pokud
        # AniList v A1 smyčce už spustil breaker, tenhle call se automaticky
        # přeskočí taky, místo aby visel na svém vlastním internim retry.
        if self.enr.anilist:
            top_tags = [e.label for e in self.model.top_effects(n=40, sign=1)
                        if e.category in ("tag", "theme", "genre")][:8]
            if top_tags:
                # pages=1 na tag: search_by_tags se od opravy ptá na KAŽDÝ tag
                # zvlášť (AniList `tag_in` je AND -- viz jeho docstring), takže
                # 5 tagů × 1 stránka = 5 requestů a ~150 neshlédnutých titulů.
                # Změřeno: druhá stránka přidá 160 kandidátů k obohacení, ale
                # do top-40 se z nich nedostane ani jeden -- stejný přínos za
                # dvojnásobnou cenu.
                matches = call_source(
                    "AniList-rec",
                    lambda: self.enr.anilist.search_by_tags(top_tags[:5], pages=1),
                )
                for m in matches:
                    bump(m["mal_id"], 0.0, None, "tag-search")

        # B) user-based CF (volitelné). Výsledky si nese `self._cf` jen po dobu
        # jednoho běhu `recommend()`, který je hned zabalí do RecommendResult --
        # ven z třídy se privátní stav nedostane.
        self._cf = ([], [])       # (senpai, syrové CF dicty)
        if self.rc.use_user_cf and self.enr.anilist:
            self._user_cf(titles, seen_ids, bump)

        return cand

    def _user_cf(self, titles, seen_ids, bump):
        """User-based CF: senpai pipeline (viz usercf.py). Best-effort."""
        from .usercf import find_senpai_recommendations
        from .sources import status
        rated = [t for t in titles if t.user_score and t.user_score > 0]
        user_scores = {t.mal_id: t.user_score for t in rated}
        # status() místo print() -- spolupracuje s \r progress řádkou
        # (uklidí ji, vypíše se, překreslí). Holý print() ji rozsekal.
        status(f"  user-CF: {len(user_scores)} ohodnocených titulů na vstupu, "
               f"hledám {self.rc.user_cf_senpai_count} senpai "
               f"z poolu {self.rc.user_cf_candidate_pool} kandidátů")
        try:
            senpai, recs = find_senpai_recommendations(
                self.enr.anilist, user_scores, watched_ids=seen_ids, rc=self.rc,
                my_resid=self.model.residuals(),
            )
            self._cf = (senpai, recs)        # pro CF HTML report
            for r in recs:
                # `signal` = smrštěná odchylka od baseline senpaie, BEZ
                # komunity (ta má v kompozitu vlastní složku) -- viz
                # usercf.recommend_from_senpai
                bump(r["mal_id"], r.get("signal", 0.0), None, "user-CF")
            status(f"  user-CF: {len(senpai)} senpai, {len(recs)} kandidátů přidáno")
        except Exception:
            # log.exception, ne print: CF fáze běží klidně hodiny a tohle je
            # jediné místo, kde se o jejím pádu dozvíš. Holý print(exc) zahodil
            # traceback -- programátorská chyba (KeyError apod.) pak vypadala
            # identicky jako "senpai se nenašli" (HODNOCENI_PROJEKTU.md §5.6).
            log.exception(
                "user-CF selhalo -- doporučení pokračují bez user-CF složky"
            )

    # ── Skórování ──────────────────────────────────────────────────────────────

    def _cluster_fit(self, attrs: dict[str, AttrValue]) -> tuple[float, str]:
        """Shoda s náladou -- viz modulová funkce `cluster_fit` (sdílí ji
        i sezónní pohled, aby „shoda s vkusem" znamenala v obou reportech
        totéž)."""
        return cluster_fit(self.model, attrs)

    # ── franšízové pohledy ──────────────────────────────────────────────────

    def _entry_note(self, rec, rel: dict, enriched_all: dict,
                    watched_ids: set[int]) -> str | None:
        """
        „začni od: X" pro pokračování, jehož předchozí díl uživatel neviděl.

        Jde po řetězci PREQUEL vazeb a hledá nejstarší neviděný díl HLAVNÍHO
        formátu. Kontrola formátu je nutná: bez ní se za prequel označí i
        prologová OVA nebo speciál (2 z 8 nálezů na reálném poolu --
        HODNOCENI_PROJEKTU.md §9c, nález #6). Když prequel v cache není,
        řetězec skončí -- žádné requesty navíc.
        """
        seen = {rec.mal_id}
        current, entry = rec.mal_id, None
        while True:
            nxt = None
            for mid in sorted(related_ids(rel.get(current) or {}, {"prequel"})):
                en = enriched_all.get(mid)
                if mid in seen or mid in watched_ids or en is None:
                    continue
                if _is_side_content(en):
                    continue
                nxt = mid
                break
            if nxt is None:
                return f"začni od: {entry.title}" if entry else None
            seen.add(nxt)
            entry, current = enriched_all[nxt], nxt

    def _franchise_views(self, recs: list, enriched: dict,
                         watched_ids: set[int], watched_enr: dict) -> dict:
        """
        Jeden průchod seskupením franšíz nad hotovým poolem:

          * **jedna karta na franšízu** -- zástupce je díl s nejvyšším
            kompozitem, ostatní se sbalí do `franchise_members`. Skóre se
            nepřepočítává: měřený rozdíl proti prostému sbalení byl v šumu
            (§9c, návrh P12);
          * **pokračování rozjetých sérií** ven z „nových objevů": uživatel si
            je hlídá sám (všechna 4 v dnešním top-100 měl v PTW), v globálním
            žebříčku jen zabírala místa objevům;
          * **„z tvého PTW" zvlášť** -- PTW tvořilo 15-16 ze 40 míst přehledu;
          * u mid-franchise sequelů štítek „začni od";
          * **„znáš, ale nemáš v plánu"** -- franšízy s dílem mezi
            `known_popularity` nejpopulárnějšími na MAL. Mimo seznam i PTW je
            skoro jistě znám a vědomě je přeskakuju (§9d.4), takže mezi
            objevy jen zabírají místo; nesrážejí se, jen mají vlastní sekci.

        Relace se berou z cache (kandidáti + shlédnuté tituly); referencované
        tituly vstupují jen jako uzly union-findu, neobohacují se.
        """
        watched = sorted(watched_ids)
        enriched_all = {**watched_enr, **enriched}
        rel = self.enr.relations_data(enriched_all)
        roots = build_roots([r.mal_id for r in recs] + watched, rel)
        watched_roots = {roots.get(m, m) for m in watched}

        by_root: dict = {}
        collapsed: list = []
        best_pop: dict = {}                 # {kořen: nejlepší pořadí popularity}
        for r in recs:                      # pool je seřazený podle kompozitu
            root = roots.get(r.mal_id, r.mal_id)
            if r.popularity:
                best_pop[root] = min(best_pop.get(root, r.popularity), r.popularity)
            keeper = by_root.get(root)
            if keeper is not None:
                keeper.franchise_members.append(r.title)
                continue
            by_root[root] = r
            r.entry_note = self._entry_note(r, rel, enriched_all, watched_ids)
            collapsed.append(r)

        cont_ids = {r.mal_id for r in collapsed
                    if roots.get(r.mal_id, r.mal_id) in watched_roots}
        rest = [r for r in collapsed if not r.ptw and r.mal_id not in cont_ids]
        limit = self.rc.known_popularity
        known_ids = {r.mal_id for r in rest
                     if limit and best_pop.get(roots.get(r.mal_id, r.mal_id),
                                               limit + 1) <= limit}
        return {
            "roots": roots,
            "continuations": [r for r in collapsed if r.mal_id in cont_ids],
            "ptw_ranked": [r for r in collapsed if r.ptw][: self.rc.ptw_top],
            "known": [r for r in rest if r.mal_id in known_ids][: self.rc.known_top],
            "discovery": [r for r in rest if r.mal_id not in known_ids][: self.rc.top_n],
        }

    def _selection(self, enriched: dict, watched_enr: dict, list_ids: set[int],
                   show_progress: bool):
        """
        Model výběru (selection.py) nad `select_universe` nejpopulárnějšími
        tituly MAL. Tituly, které už jsou obohacené (kandidáti, shlédnuté), se
        znovu nenačítají; zbytek se obohatí BEZ staff (model ho nepoužívá a
        Jikan staff by stál request na titul). Potřebuje Jikan (seznam
        popularity + počet členů) -- v --no-jikan režimu se přeskočí.
        """
        if not (self.rc.w_select and self.rc.select_universe and self.enr.jikan):
            return None
        from .selection import MAIN_FORMATS, fit_selection
        top = self.enr.jikan.get_top_popular(self.rc.select_universe)
        ids = [d["mal_id"] for d in top
               if d.get("mal_id") and d.get("type") in MAIN_FORMATS]
        have = {**watched_enr, **enriched}
        missing = [m for m in ids if m not in have]
        extra = (self.enr.enrich_ids(missing, show_progress=show_progress,
                                     with_staff=False) if missing else {})
        universe = {m: have.get(m) or extra[m] for m in ids
                    if m in have or m in extra}
        rel = self.enr.relations_data({**watched_enr, **universe})
        try:
            return fit_selection(universe, list_ids, rel)
        except Exception:
            # doplňková složka nesmí shodit doporučení
            log.exception("model výběru selhal -- doporučení pokračují bez něj")
            return None

    def recommend(self, all_titles: list[Title], ptw_ids: set[int],
                  watched_ids: set[int], show_progress=True,
                  limit: int | None = -1) -> RecommendResult:
        """
        Vrátí `RecommendResult`: seřazená doporučení + (volitelně) senpai a
        syrové CF výsledky pro CF report. Iterovat/měřit délku jde přímo přes
        výsledek, seznam je v `.recs`.

        limit=-1  → ořízni na self.rc.top_n (výchozí chování, globální přehled)
        limit=None → vrať celý ohodnocený pool (pro per-klastr pohled)
        limit=N   → vrať prvních N
        """
        # 1) kandidáti (vše co jsem viděl je "seen"; PTW NEvylučujeme)
        cand_meta = self._gather_candidates(all_titles, seen_ids=watched_ids)
        senpai, cf_raw = getattr(self, "_cf", ([], []))
        if not cand_meta:
            return RecommendResult(recs=[], senpai=senpai, cf_raw=cf_raw)

        # 2) obohať kandidáty (atributy + komunitní skóre + synopse)
        cand_ids = list(cand_meta.keys())
        enriched = self.enr.enrich_ids(cand_ids, show_progress=show_progress)
        # shlédnuté tituly jednou -- potřebují je franšízové pohledy i model výběru
        watched_enr = (self.enr.enrich_ids(sorted(watched_ids), show_progress=False)
                       if watched_ids else {})
        # 2b) model výběru: sáhnu po tom vůbec? (selection.py, §9d.4)
        selection = self._selection(enriched, watched_enr,
                                    watched_ids | ptw_ids, show_progress)

        # 3) spočti surové metriky
        rows = []
        for mid, meta in cand_meta.items():
            en = enriched.get(mid)
            if not en:
                continue
            if en.community is not None and en.community < self.rc.min_community:
                continue
            pred, lo, hi, contribs = self.model.predict(en.attrs, en.community)
            # affinity() = KALIBROVANÁ afinitní část (scale na singly+páry,
            # scale_triples na trojice). Dřív se tu bral neškálovaný součet;
            # s jedním faktorem to bylo jedno (z-skóre konstantu vykrátí),
            # se dvěma ne -- poměr faktorů je právě to, co CV zjistila.
            raw_resid = self.model.affinity(en.attrs)
            cfit, cname = self._cluster_fit(en.attrs)
            taste_fit = raw_resid + self.rc.cluster_fit_weight * cfit
            sel = selection.score(en.attrs) if selection else 0.0
            rows.append((mid, en, meta, pred, lo, hi, contribs, taste_fit, cname,
                         raw_resid, cfit, sel))

        if not rows:
            return RecommendResult(recs=[], senpai=senpai, cf_raw=cf_raw)

        # 4) z-skóry pro kompozit -- item-CF přes log1p (šikmé rozdělení:
        # kandidát doporučený mnoha seedy najednou by jinak dostal z-skóre
        # 5-15 a přebil všechny ostatní složky, viz analýza 2026-07).
        #
        # Parametry (průměr, sd) vkusu, grafu a kvality se berou z OBSAHOVÉHO
        # poolu (kandidáti z grafu podobnosti nebo tag-search), ne z celého.
        # Celý pool se se zapnutým user-CF nafoukne o tisíce titulů, které
        # přinesli jen senpai a které mají v grafu nulu: na reálných datech
        # 6 108 z 6 615, takže log-hlasy grafu měly průměr 0,13 a sd 0,60 a
        # KAŽDÝ titul z grafu dostal +3,5 až +6 bodů. Graf tak fungoval jako
        # brána -- 40/40 nových objevů z grafu, Kanon (2006) s nejlepší shodou
        # s vkusem až #110 (HODNOCENI_PROJEKTU.md §9d, nález #1). Obsahový
        # pool existuje vždy a jeho složení na user-CF nezávisí, takže váhy
        # w_* znamenají totéž se zapnutým i vypnutým user-CF.
        #
        # User-CF složka se normalizuje přes celý pool: je to jediná složka,
        # jejíž hodnoty nese hlavně user-CF větev, a nad obsahovým poolem by
        # mohla vyjít konstantní (senpai nemusí znát žádného kandidáta z grafu).
        #
        # Ořez |z| není potřeba: nad obsahovým poolem vychází graf v rozsahu
        # −1,3 až +2,1, vkus ±3,3 a kvalita ±2,4.
        content = [r for r in rows if r[2]["sources"] - {"user-CF"}] or rows
        z_taste = _z([r[7] for r in content])
        z_item = _z([math.log1p(r[2]["item_votes"]) for r in content])
        z_user = _z([r[2]["user_votes"] for r in rows])
        z_q = _z([(r[1].community or self.model.c_mean) for r in content])
        # model výběru: stejné pravidlo jako vkus (obsahový pool); bez modelu
        # jsou všechny hodnoty 0 a složka se vykrátí
        z_sel = _z([r[11] for r in content])

        recs = []
        for (mid, en, meta, pred, lo, hi, contribs, taste_fit, cname,
             raw_resid, cfit, sel) in rows:
            comp = (self.rc.w_taste_fit * z_taste(taste_fit)
                    + self.rc.w_cf * z_item(math.log1p(meta["item_votes"]))
                    + self.rc.w_user_cf * z_user(meta["user_votes"])
                    + self.rc.w_quality * z_q(en.community or self.model.c_mean)
                    + self.rc.w_select * z_sel(sel))
            recs.append(Recommendation(
                mal_id=mid, title=en.title, title_en=en.title_en,
                community=en.community, pred=pred, pred_lo=lo, pred_hi=hi,
                taste_fit=taste_fit, cf_signal=meta["item_votes"],
                user_cf_signal=meta["user_votes"], composite=comp,
                affinity=raw_resid, cluster_fit=cfit,
                ptw=(mid in ptw_ids), cluster_name=cname,
                why=contribs[:WHY_KEEP], cf_seeds=meta["cf_seeds"][:5],
                synopsis=en.synopsis, sources=sorted(meta["sources"]),
                select=sel,
                avoided=(selection.avoided(en.attrs)
                         if selection and z_sel(sel) < 0 else []),
                popularity=(en.jikan or {}).get("popularity") or None,
            ))

        recs.sort(key=lambda r: -r.composite)
        top = self.rc.top_n if limit == -1 else limit
        z_params = {name: [z.mean, z.sd] for name, z in (
            ("taste_fit", z_taste), ("item_cf", z_item),
            ("user_cf", z_user), ("quality", z_q), ("select", z_sel))}
        # pohledy se počítají nad CELÝM poolem, ne nad ořezem
        views = self._franchise_views(recs, enriched, watched_ids, watched_enr)
        return RecommendResult(recs=recs if top is None else recs[:top],
                               senpai=senpai, cf_raw=cf_raw, z_params=z_params,
                               selection=selection, **views)


def cluster_fit(model, attrs: dict[str, AttrValue]) -> tuple[float, str]:
    """
    Vážený kosinus k nejbližšímu JÁDRU nálady × jeho (smrštěná) AFINITA.

    Počítá se proti plnému těžišti klastru (`Cluster.centroid`) a jen
    v prostoru nálady (`model.cluster_feat_keys`). Dřívější verze měla
    dvě zkreslení (HODNOCENI_PROJEKTU.md §5.4):

      1. jmenovatel bral VŠECHNY atributy kandidáta, včetně studia,
         formátu, dekády a zdroje -- kategorií, které v klastrovém
         prostoru vůbec nejsou. Bohatě otagovaný titul tak dostal nižší
         podobnost bez ohledu na skutečnou shodu s náladou: při týchž
         třech shodách 0,387 (10 atributů) vs. 0,183 (45). A protože
         počet tagů nad prahem roste s popularitou, byl to systematický
         posun proti dobře zdokumentovaným titulům.
      2. podobnost se měřila jen proti šesti nejvýraznějším osám
         (zobrazovací signatuře) a binárně -- váhy atributů (AniList
         rank) se zahazovaly, takže okrajový tag vážil jako hlavní žánr.

    Afinita nálady je SMRŠTĚNÁ `n_eff/(n_eff+K)` (taste.py) -- byla to jediná
    veličina modelu bez smrštění, takže malá nálada (Σ vah 11) mluvila stejně
    silně jako velká. Dřívější tvar `(aff + 1.0)` navíc dělal z cluster_fit
    hlavně TYPIČNOST (kosinus), protože +1 přebilo afinitu v rozsahu
    −0,5…+0,2; teď nese to, co má -- o kolik nad baseline tu náladu hodnotím
    (§9c, návrh P21).
    """
    if not model.clusters:
        return 0.0, ""
    feat = model.cluster_feat_keys
    if not feat:
        return 0.0, ""
    # kandidátský vektor v prostoru nálady, s vahami jako při fitu
    vec = {k: av.weight for k, av in attrs.items()
           if k in feat and av.weight}
    if not vec:
        return 0.0, ""
    v_norm = math.sqrt(sum(w * w for w in vec.values()))

    best_sim, best_name, best_aff = 0.0, "", 0.0
    for c in model.clusters:
        if not c.centroid_norm:
            continue
        dot = 0.0
        for k, w in vec.items():
            coord = c.centroid.get(k)
            if coord:
                dot += w * coord
        if dot <= 0:
            continue
        sim = dot / (v_norm * c.centroid_norm)
        if sim > best_sim:
            best_sim, best_name = sim, c.name
            best_aff = getattr(c, "affinity_shrunk", c.affinity)
    return best_sim * best_aff, best_name
