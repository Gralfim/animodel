"""
enrich.py — Z MAL ID na obohacené Title objekty.

Stáhne (a cachuje) metadata z Jikan + AniList, sloučí atributy přes
attributes.build_attributes, doplní komunitní baseline a — volitelně —
zváží franšízy, aby sequel/prequel nepočítaly jako N nezávislých bodů.

Nouzový režim (--no-jikan / enrich.use_jikan: false): Jikan klient se vůbec
nevytvoří a všechno, co jinak dodává (žánry, synopse, dekáda, franšízové
vazby), se bere z AniListu -- viz build_attributes fallbacky a
_relations_from_anilist níž. Stejné fallbacky fungují i per-titul v běžném
režimu, když Jikan pro konkrétní titul dočasně selže.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass

from .attributes import build_attributes, community_baseline, AttrValue
from .taste import Title
from .sources.jikan import JikanClient
from .sources.anilist import AniListClient
from .sources.shikimori import ShikimoriClient
from .series import build_series_groups


# AniList relationType → Jikan název relace (jen typy, které series.py
# skutečně slučuje -- viz SERIES_RELATION_TYPES; zbytek nemá smysl mapovat).
_ANILIST_RELATION = {
    "SEQUEL": "sequel",
    "PREQUEL": "prequel",
    "ALTERNATIVE": "alternative version",
    "SIDE_STORY": "side story",
}


def _relations_from_anilist(media: dict | None) -> dict | None:
    """
    Adaptér: AniList `relations.edges` → Jikan tvar {"relations": [...]},
    aby series.py (union-find franšíz) zůstal beze změny a fungoval nad
    kterýmkoli zdrojem. Ne-anime uzly (manga předloha apod.) a uzly bez
    MAL ID se vynechávají.
    """
    edges = ((media or {}).get("relations") or {}).get("edges") or []
    rels = []
    for edge in edges:
        rtype = _ANILIST_RELATION.get(edge.get("relationType"))
        node = edge.get("node") or {}
        if not rtype or node.get("type") != "ANIME" or not node.get("idMal"):
            continue
        rels.append({"relation": rtype,
                     "entry": [{"mal_id": node["idMal"], "type": "anime"}]})
    return {"relations": rels} if rels else None


# Formáty, které v rámci franšízy značí vedlejší obsah (OVA/speciály/hudební
# klipy). ONA záměrně chybí -- plnohodnotné série dnes běžně vycházejí jako
# ONA (streamovací platformy), není to signál vedlejšosti. Movie taky ne
# (kinofilmy bývají plnohodnotná pokračování). Standalone titulů se tohle
# netýká vůbec -- vedlejšost se vyhodnocuje jen uvnitř skupin k>1.
SIDE_FORMATS = {"ova", "special", "tv special", "music"}


def _is_side_content(e: "Enriched") -> bool:
    """
    Je titul VEDLEJŠÍ obsah své franšízy (side story / OVA / speciál)?

    Dva nezávislé signály (stačí jeden):
      1. formát: Jikan `type` / AniList `format` v SIDE_FORMATS,
      2. relace: titul sám deklaruje rodiče -- Jikan "Parent story",
         AniList edge PARENT (obě konvence: side story ukazuje na svůj
         hlavní titul; hlavní titul má obrácený "Side story" edge).

    Čte se přímo ze surových zdrojových dat (e.jikan/e.anilist), ne přes
    relations adaptér -- ten mapuje jen typy potřebné pro SESKUPOVÁNÍ
    (series.py) a parent-story vazby do seskupování záměrně nevstupují.
    """
    fmt = ((e.jikan or {}).get("type")
           or (e.anilist or {}).get("format") or "")
    if fmt.strip().lower().replace("_", " ") in SIDE_FORMATS:
        return True
    for rel in (e.jikan or {}).get("relations") or []:
        if (rel.get("relation") or "").lower() == "parent story":
            return True
    for edge in ((e.anilist or {}).get("relations") or {}).get("edges") or []:
        node = edge.get("node") or {}
        if edge.get("relationType") == "PARENT" and node.get("type") == "ANIME":
            return True
    return False


def _clean_anilist_description(text: str) -> str:
    """AniList description je HTML-ish (<br>, <i>, &amp;) a obsahuje
    ~!spoiler!~ bloky -- pro report potřebujeme čistý text bez spoilerů."""
    if not text:
        return ""
    text = re.sub(r"~!.*?!~", "", text, flags=re.S)      # spoiler bloky pryč
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)  # <br> → nový řádek
    text = re.sub(r"<[^>]+>", "", text)                   # zbylé tagy pryč
    text = html.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        self.jikan = jikan or (JikanClient(cfg.cache_dir) if cfg.enrich.use_jikan else None)
        self.anilist = anilist or (AniListClient(cfg.cache_dir) if cfg.enrich.use_anilist else None)
        self.shikimori = shikimori or (
            ShikimoriClient(f"{cfg.cache_dir}/shikimori") if cfg.enrich.use_shikimori else None
        )

    def enrich_ids(self, mal_ids: list[int], show_progress=True) -> dict[int, Enriched]:
        jdata = {}
        if self.jikan:
            jdata = self.jikan.get_anime_batch(mal_ids, show_progress=show_progress)
        adata = {}
        if self.anilist:
            adata = self.anilist.get_anime_batch(mal_ids, show_progress=show_progress)
        sdata = {}
        if self.cfg.enrich.include_staff and self.jikan:
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
            if not syn and a:
                syn = _clean_anilist_description(a.get("description") or "")
            out[mid] = Enriched(
                mal_id=mid, title=title, title_en=title_en,
                community=community_baseline(j, a), attrs=attrs,
                synopsis=syn, jikan=j, anilist=a,
            )
        return out

    def relations_data(self, enriched: dict[int, Enriched]) -> dict[int, dict]:
        """
        Franšízové vazby v Jikan tvaru pro series.py -- primárně z Jikan dat,
        per-titul fallback na AniList relations (adaptér výš). Vrací jen
        tituly, pro které nějaké vazby známe.
        """
        out = {}
        for mid, e in enriched.items():
            if e.jikan and e.jikan.get("relations"):
                out[mid] = e.jikan
            else:
                rel = _relations_from_anilist(e.anilist)
                if rel:
                    out[mid] = rel
        return out

    def build_titles(self, mal_entries, show_progress=True) -> list[Title]:
        """
        mal_entries: list MalEntry (Completed, score>0).
        Vrátí Title objekty pro model, volitelně s franšízovými vahami.
        """
        ids = [e.mal_id for e in mal_entries]
        enr = self.enrich_ids(ids, show_progress=show_progress)

        weight = {mid: 1.0 for mid in ids}
        series_root: dict[int, int] = {}
        if self.cfg.model.aggregate_franchises:
            rel_data = self.relations_data(enr)
            # id_set = všechny obohacené tituly (ne jen ty s vlastními
            # relations) -- titul bez vazeb pořád může být cílem vazby
            # od jiného člena franšízy.
            groups = build_series_groups(list(enr.keys()), rel_data)
            side_w = self.cfg.model.side_story_weight
            for root, members in groups.items():
                if len(members) <= 1:
                    continue
                # Příspěvek člena do franšízy: hlavní řada 1.0, vedlejší
                # obsah (OVA/speciál/side story) side_story_weight. Finální
                # váha = c_i / √k_eff, kde k_eff = Σ c_i. Když jsou všichni
                # členové hlavní, dává to přesně původní 1/√k; vedlejší
                # obsah mluví úměrně tišeji a zároveň méně zvyšuje k_eff
                # (netrestá hlavní řady za existenci OVAček).
                contrib = {
                    m: (side_w if (m in enr and _is_side_content(enr[m])) else 1.0)
                    for m in members
                }
                k_eff = sum(contrib.values())
                norm = math.sqrt(k_eff) if k_eff > 0 else 1.0
                for m in members:
                    weight[m] = contrib[m] / norm
                    series_root[m] = root

        titles = []
        for e in mal_entries:
            en = enr.get(e.mal_id)
            if not en:
                continue
            titles.append(Title(
                mal_id=e.mal_id, title=en.title,
                user_score=float(e.score), community=en.community,
                attrs=en.attrs, weight=weight.get(e.mal_id, 1.0),
                series_root=series_root.get(e.mal_id),
            ))
        return titles
