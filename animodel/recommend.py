"""
recommend.py — Generování a řazení doporučení dosud neshlédnutých anime.

Strategie (dvě nezávislé větve, sjednocené a deduplikované):

  A) ATRIBUTOVÁ / CONTENT větev
     - vezmi tvoje vysoce hodnocené tituly (seed = score >= high_score)
     - pro každý seed stáhni MAL + AniList "recommendations" (item-based CF graf)
     - navíc discovery: AniList tag-search na tvé nejcharakterističtější tagy
     => kandidáti, kteří jsou buď podobní oblíbeným, nebo nesou tvé silné atributy

  B) COLLABORATIVE / USER větev (volitelná, vypnutá defaultně)
     - najdi uživatele s podobným vkusem a jejich vysoko hodnocené tituly
     - (drahé přes AniList; zapíná se recommend.use_user_cf=True)

Skórování každého kandidáta:
    composite = w_taste_fit * z(taste_fit)
              + w_cf        * z(cf_signal)
              + w_quality   * z(community)
  kde
    taste_fit  = model predikuje afinitu (rezid. část) + shoda s nejbližším klastrem
    cf_signal  = kolik seedů ho doporučilo + jejich hlasy/rating (graf-based CF)
    community  = komunitní skóre (mírná preference kvality)

Řadíme podle composite, NE podle predikované známky (ta se lepí na komunitní
průměr kvůli restrikci rozsahu — viz metodika). Predikovaná známka + interval
se počítá zvlášť jen pro zobrazení.

PTW tituly se z vyhledávání NEvyřazují, jen se označí příznakem `ptw`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .taste import TasteModel, Title
from .enrich import Enricher, Enriched
from .attributes import AttrValue


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
    cf_signal: float
    composite: float
    ptw: bool
    cluster_name: str
    why: list                # [(label, category, contribution), ...] seřazeno
    cf_seeds: list           # názvy seedů, které tenhle titul "doporučily"
    synopsis: str = ""
    sources: list = field(default_factory=list)   # ['MAL-rec', 'AniList-rec', 'tag-search']


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
        seeds = [t for t in titles if t.user_score >= self.rc.high_score]
        seeds.sort(key=lambda t: -t.user_score)
        return seeds[: self.rc.max_seeds]

    def _gather_candidates(self, titles: list[Title], seen_ids: set[int]):
        """
        Vrátí:
            cand_meta: {mal_id: {'cf_votes': float, 'cf_seeds': [titul,...], 'sources': set}}
        """
        cand: dict[int, dict] = {}

        def bump(mid, votes, seed_title, source):
            if mid in seen_ids:
                return
            d = cand.setdefault(mid, {"cf_votes": 0.0, "cf_seeds": [], "sources": set()})
            d["cf_votes"] += votes
            if seed_title and seed_title not in d["cf_seeds"]:
                d["cf_seeds"].append(seed_title)
            d["sources"].add(source)

        seeds = self._seeds(titles)
        seed_title_by_id = {t.mal_id: t.title for t in seeds}

        # A1) item-based CF graf z MAL + AniList recommendations
        for s in seeds:
            try:
                for r in self.enr.jikan.get_recommendations(s.mal_id):
                    # váž hlasy podle toho, jak moc seed miluju (user_score nad průměr)
                    w = max(0.1, s.user_score - self.model.u_mean + 1.0)
                    bump(r["mal_id"], (1 + math.log1p(r.get("votes", 0))) * w,
                         seed_title_by_id.get(s.mal_id), "MAL-rec")
            except Exception:
                pass
            if self.enr.anilist:
                try:
                    for r in self.enr.anilist.get_recommendations(s.mal_id):
                        w = max(0.1, s.user_score - self.model.u_mean + 1.0)
                        bump(r["mal_id"], (1 + math.log1p(max(0, r.get("rating", 0)))) * w,
                             seed_title_by_id.get(s.mal_id), "AniList-rec")
                except Exception:
                    pass

        # A2) discovery přes tag-search na nejcharakterističtější atributy
        if self.enr.anilist:
            top_tags = [e.label for e in self.model.top_effects(n=40, sign=1)
                        if e.category in ("tag", "theme", "genre")][:8]
            if top_tags:
                try:
                    for m in self.enr.anilist.search_by_tags(top_tags[:5], pages=2):
                        bump(m["mal_id"], 0.0, None, "tag-search")
                except Exception:
                    pass

        # B) user-based CF (volitelné)
        if self.rc.use_user_cf and self.enr.anilist:
            self._user_cf(titles, seen_ids, bump)

        return cand

    def _user_cf(self, titles, seen_ids, bump):
        """Jednoduché user-based CF přes AniList (pokud klient umí). Best-effort."""
        finder = getattr(self.enr.anilist, "similar_users_recommendations", None)
        if not callable(finder):
            # Metoda chybí – AniList klient ji nepodporuje
            print("  user-CF: přeskočeno (AniList klient metodu nepodporuje)")
            return
        liked = [t.mal_id for t in titles if t.user_score >= self.rc.high_score]
        print(f"  user-CF: {len(liked)} seedů, min_overlap={self.rc.user_cf_min_overlap}, "
              f"top_users={self.rc.user_cf_top_users}")
        try:
            recs = finder(liked,
                          min_overlap=self.rc.user_cf_min_overlap,
                          top_users=self.rc.user_cf_top_users)
            for r in recs:
                bump(r["mal_id"], r.get("score", 1.0), None, "user-CF")
            print(f"  user-CF: {len(recs)} kandidátů přidáno")
        except Exception as exc:
            print(f"  user-CF: selhalo ({exc})")

    # ── Skórování ──────────────────────────────────────────────────────────────

    def _cluster_fit(self, attrs: dict[str, AttrValue]) -> tuple[float, str]:
        """Kosinová podobnost k nejbližšímu klastru × jeho průměrné hodnocení."""
        if not self.model.clusters:
            return 0.0, ""
        present = set(attrs)
        best_sim, best_name, best_score = 0.0, "", 0.0
        for c in self.model.clusters:
            sig_keys = set()
            # rekonstruuj klíče z labelů přes effects
            for label, cat, _ in c.signature:
                for k, e in self.model.effects.items():
                    if e.label == label and e.category == cat:
                        sig_keys.add(k)
                        break
            if not sig_keys:
                continue
            inter = len(present & sig_keys)
            sim = inter / math.sqrt(len(sig_keys) * max(1, len(present)))
            if sim > best_sim:
                best_sim, best_name, best_score = sim, c.name, c.mean_user_score
        # váž podobnost tím, jak vysoko ten klastr hodnotím (nad globální průměr)
        return best_sim * (best_score - self.model.u_mean + 1.0), best_name

    def recommend(self, all_titles: list[Title], ptw_ids: set[int],
                  watched_ids: set[int], show_progress=True) -> list[Recommendation]:
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

        # 4) z-skóry pro kompozit
        z_taste = _z([r[7] for r in rows])
        z_cf = _z([r[2]["cf_votes"] for r in rows])
        z_q = _z([(r[1].community or self.model.c_mean) for r in rows])

        recs = []
        for (mid, en, meta, pred, lo, hi, contribs, taste_fit, cname) in rows:
            comp = (self.rc.w_taste_fit * z_taste(taste_fit)
                    + self.rc.w_cf * z_cf(meta["cf_votes"])
                    + self.rc.w_quality * z_q(en.community or self.model.c_mean))
            recs.append(Recommendation(
                mal_id=mid, title=en.title, title_en=en.title_en,
                community=en.community, pred=pred, pred_lo=lo, pred_hi=hi,
                taste_fit=taste_fit, cf_signal=meta["cf_votes"], composite=comp,
                ptw=(mid in ptw_ids), cluster_name=cname,
                why=contribs[:6], cf_seeds=meta["cf_seeds"][:5],
                synopsis=en.synopsis, sources=sorted(meta["sources"]),
            ))

        recs.sort(key=lambda r: -r.composite)
        return recs[: self.rc.top_n]
