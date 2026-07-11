"""
enrich.py — Z MAL ID na obohacené Title objekty.

Stáhne (a cachuje) metadata z Jikan + AniList, sloučí atributy přes
attributes.build_attributes, doplní komunitní baseline a — volitelně —
zváží franšízy, aby sequel/prequel nepočítaly jako N nezávislých bodů.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .attributes import build_attributes, community_baseline, AttrValue
from .taste import Title
from .sources.jikan import JikanClient
from .sources.anilist import AniListClient
from .sources.shikimori import ShikimoriClient
from .series import build_series_groups


@dataclass
class Enriched:
    mal_id: int
    title: str
    title_en: str
    community: float | None
    attrs: dict
    synopsis: str = ""
    jikan: dict = None
    anilist: dict = None


class Enricher:
    def __init__(self, cfg, jikan: JikanClient = None, anilist: AniListClient = None,
                 shikimori: ShikimoriClient = None):
        self.cfg = cfg
        self.jikan = jikan or JikanClient(cfg.cache_dir)
        self.anilist = anilist or (AniListClient(cfg.cache_dir) if cfg.enrich.use_anilist else None)
        self.shikimori = shikimori or (
            ShikimoriClient(f"{cfg.cache_dir}/shikimori") if cfg.enrich.use_shikimori else None
        )

    def enrich_ids(self, mal_ids: list[int], show_progress=True) -> dict[int, Enriched]:
        jdata = self.jikan.get_anime_batch(mal_ids, show_progress=show_progress)
        adata = {}
        if self.anilist:
            adata = self.anilist.get_anime_batch(mal_ids, show_progress=show_progress)
        sdata = {}
        if self.cfg.enrich.include_staff:
            sdata = self.jikan.get_staff_batch(mal_ids, show_progress=show_progress)

        out = {}
        for mid in mal_ids:
            j = jdata.get(mid)
            a = adata.get(mid)
            if not j and not a:
                continue
            attrs = build_attributes(
                j, a,
                anilist_min_rank=self.cfg.enrich.anilist_min_rank,
                include_studios=self.cfg.enrich.include_studios,
                staff=sdata.get(mid),
            )
            title = (j or {}).get("title") or ((a or {}).get("title") or {}).get("romaji", "") or str(mid)
            title_en = ""
            if a and a.get("title"):
                title_en = a["title"].get("english") or ""
            if not title_en and j:
                title_en = j.get("title_english") or ""
            syn = (j or {}).get("synopsis") or ""
            out[mid] = Enriched(
                mal_id=mid, title=title, title_en=title_en,
                community=community_baseline(j, a), attrs=attrs,
                synopsis=syn, jikan=j, anilist=a,
            )
        return out

    def build_titles(self, mal_entries, show_progress=True) -> list[Title]:
        """
        mal_entries: list MalEntry (Completed, score>0).
        Vrátí Title objekty pro model, volitelně s franšízovými vahami.
        """
        ids = [e.mal_id for e in mal_entries]
        enr = self.enrich_ids(ids, show_progress=show_progress)

        weight = {mid: 1.0 for mid in ids}
        if self.cfg.model.aggregate_franchises:
            jfull = {mid: e.jikan for mid, e in enr.items() if e.jikan}
            groups = build_series_groups(list(jfull.keys()), jfull)
            for root, members in groups.items():
                k = len(members)
                if k > 1:
                    # každý člen franšízy dostane váhu 1/sqrt(k) → tlumí inflaci,
                    # ale neztrácí úplně signál vícenásobně oblíbené série
                    w = 1.0 / math.sqrt(k)
                    for m in members:
                        weight[m] = w

        titles = []
        for e in mal_entries:
            en = enr.get(e.mal_id)
            if not en:
                continue
            titles.append(Title(
                mal_id=e.mal_id, title=en.title,
                user_score=float(e.score), community=en.community,
                attrs=en.attrs, weight=weight.get(e.mal_id, 1.0),
            ))
        return titles
