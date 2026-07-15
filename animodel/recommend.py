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
from .enrich import Enricher, Enriched
from .attributes import AttrValue

log = logging.getLogger(__name__)


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


def _z(values: list[float]) -> dict:
    """Vrátí funkci pro z-skóre dle rozdělení values (robustní na konstantu)."""
    if not values:
        return lambda x: 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var) if var > 1e-9 else 1.0
    return lambda x: (x - mean) / sd


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
        signál na úkor rozmanitosti. Nejlépe hodnocené řady mají přednost.
        """
        cand = [t for t in titles if t.user_score >= self.rc.high_score]
        cand.sort(key=lambda t: -t.user_score)
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
                matches = call_source(
                    "AniList-rec",
                    lambda: self.enr.anilist.search_by_tags(top_tags[:5], pages=2),
                )
                for m in matches:
                    bump(m["mal_id"], 0.0, None, "tag-search")

        # B) user-based CF (volitelné)
        self._cf_raw_results = []  # reset před každým spuštěním
        self._cf_senpai = []
        if self.rc.use_user_cf and self.enr.anilist:
            self._user_cf(titles, seen_ids, bump)

        return cand

    def _user_cf(self, titles, seen_ids, bump):
        """User-based CF: senpai pipeline (viz usercf.py). Best-effort."""
        from .usercf import find_senpai_recommendations
        rated = [t for t in titles if t.user_score and t.user_score > 0]
        user_scores = {t.mal_id: t.user_score for t in rated}
        print(f"  user-CF: {len(user_scores)} ohodnocených titulů na vstupu, "
              f"hledám {self.rc.user_cf_senpai_count} senpai "
              f"z poolu {self.rc.user_cf_candidate_pool} kandidátů")
        try:
            senpai, recs = find_senpai_recommendations(
                self.enr.anilist, user_scores, watched_ids=seen_ids, rc=self.rc,
            )
            self._cf_senpai = senpai         # pro CF HTML report
            self._cf_raw_results = recs      # uloženo pro CF HTML report
            for r in recs:
                bump(r["mal_id"], r.get("score", 1.0), None, "user-CF")
            print(f"  user-CF: {len(senpai)} senpai, {len(recs)} kandidátů přidáno")
        except Exception as exc:
            print(f"  user-CF: selhalo ({exc})")

    # ── Skórování ──────────────────────────────────────────────────────────────

    def _cluster_fit(self, attrs: dict[str, AttrValue]) -> tuple[float, str]:
        """Kosinová podobnost k nejbližšímu klastru × jeho AFINITA."""
        if not self.model.clusters:
            return 0.0, ""
        present = set(attrs)
        best_sim, best_name, best_aff = 0.0, "", 0.0
        for c in self.model.clusters:
            # klíč je teď přímo v signature (viz taste.py::_fit_clusters) --
            # dřív se tady dělal lineární průchod přes VŠECHNY self.model.effects
            # pro každou položku signatury, pro každý klastr, pro každého
            # kandidáta v recommend() -- na stovkách efektů × stovkách/tisících
            # kandidátů zbytečně drahé, a ke všemu křehké (shoda podle
            # label+category by teoreticky mohla trefit jiný klíč).
            sig_keys = {sig[0] for sig in c.signature}
            if not sig_keys:
                continue
            inter = len(present & sig_keys)
            sim = inter / math.sqrt(len(sig_keys) * max(1, len(present)))
            if sim > best_sim:
                best_sim, best_name, best_aff = sim, c.name, c.affinity
        # Váž podobnost klastrovou AFINITOU (vážený průměr reziduí členů) --
        # dřívější `mean_user_score − u_mean` používalo surovou známku, čímž
        # znovu zanášelo komunitní kvalitu, kterou si model jinde pečlivě
        # odečítá (klastr mainstreamových hitů vypadal "oblíbeněji", než
        # odpovídalo skutečnému osobnímu vkladu). Tvar (aff + 1.0) zachovává
        # původní strukturu: neutrální klastr přispívá ~1×, oblíbený víc.
        return best_sim * (best_aff + 1.0), best_name

    def recommend(self, all_titles: list[Title], ptw_ids: set[int],
                  watched_ids: set[int], show_progress=True,
                  limit: int | None = -1) -> list[Recommendation]:
        """
        Vrátí seřazený list Recommendation.

        limit=-1  → ořízni na self.rc.top_n (výchozí chování, globální přehled)
        limit=None → vrať celý ohodnocený pool (pro per-klastr pohled)
        limit=N   → vrať prvních N
        """
        # 1) kandidáti (vše co jsem viděl je "seen"; PTW NEvylučujeme)
        cand_meta = self._gather_candidates(all_titles, seen_ids=watched_ids)
        if not cand_meta:
            return []

        # 2) obohať kandidáty (atributy + komunitní skóre + synopse)
        cand_ids = list(cand_meta.keys())
        enriched = self.enr.enrich_ids(cand_ids, show_progress=show_progress)

        # 3) spočti surové metriky
        rows = []
        for mid, meta in cand_meta.items():
            en = enriched.get(mid)
            if not en:
                continue
            if en.community is not None and en.community < self.rc.min_community:
                continue
            pred, lo, hi, contribs = self.model.predict(en.attrs, en.community)
            raw_resid = self.model._raw_resid_pred(en.attrs)   # afinitní část
            cfit, cname = self._cluster_fit(en.attrs)
            taste_fit = raw_resid + 0.5 * cfit
            rows.append((mid, en, meta, pred, lo, hi, contribs, taste_fit, cname))

        if not rows:
            return []

        # 4) z-skóry pro kompozit -- item-CF přes log1p (šikmé rozdělení:
        # kandidát doporučený mnoha seedy najednou by jinak dostal z-skóre
        # 5-15 a přebil všechny ostatní složky, viz analýza 2026-07)
        z_taste = _z([r[7] for r in rows])
        z_item = _z([math.log1p(r[2]["item_votes"]) for r in rows])
        z_user = _z([r[2]["user_votes"] for r in rows])
        z_q = _z([(r[1].community or self.model.c_mean) for r in rows])

        recs = []
        for (mid, en, meta, pred, lo, hi, contribs, taste_fit, cname) in rows:
            comp = (self.rc.w_taste_fit * z_taste(taste_fit)
                    + self.rc.w_cf * z_item(math.log1p(meta["item_votes"]))
                    + self.rc.w_user_cf * z_user(meta["user_votes"])
                    + self.rc.w_quality * z_q(en.community or self.model.c_mean))
            recs.append(Recommendation(
                mal_id=mid, title=en.title, title_en=en.title_en,
                community=en.community, pred=pred, pred_lo=lo, pred_hi=hi,
                taste_fit=taste_fit, cf_signal=meta["item_votes"],
                user_cf_signal=meta["user_votes"], composite=comp,
                ptw=(mid in ptw_ids), cluster_name=cname,
                why=contribs[:6], cf_seeds=meta["cf_seeds"][:5],
                synopsis=en.synopsis, sources=sorted(meta["sources"]),
            ))

        recs.sort(key=lambda r: -r.composite)
        top = self.rc.top_n if limit == -1 else limit
        return recs if top is None else recs[:top]
