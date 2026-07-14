"""
attributes.py — Kanonizace a deduplikace atributů napříč zdroji.

Problém, který řeší:
    MAL má žánr "Drama" i téma "School"; AniList má tag "School", "Drama" atd.
    Pokud bychom je počítali zvlášť, jeden a tentýž koncept by se do modelu
    promítl víckrát a uměle nafoukl svůj vliv (double-counting).

Řešení:
    Každý atribut se převede na *kanonický klíč* (lowercase, bez interpunkce)
    a zařadí do jedné kategorie. Synonyma mezi zdroji se sloučí přes ALIAS mapu.
    Když stejný koncept přijde z víc zdrojů, ponechá se JEDEN klíč s nejvyšší
    vahou (MAL binární příznak = 1.0, AniList tag = rank 0–1).

Výstupem je pro každé anime slovník:
    { kanonický_klíč: AttrValue(category, weight, label) }
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# ── Kanonizace názvu ────────────────────────────────────────────────────────

def canon(name: str) -> str:
    """Převede název atributu na kanonický klíč."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# Synonyma napříč zdroji → jediný kanonický klíč.
# Klíč = surová kanonická podoba, hodnota = cílový klíč.
ALIAS: dict[str, str] = {
    "cgdct": "cute_girls_doing_cute_things",
    "mahou_shoujo": "magical_girl",
    "shoujo_ai": "girls_love",
    "shounen_ai": "boys_love",
    "sci_fi": "science_fiction",
    "scifi": "science_fiction",
    "slice_of_life": "slice_of_life",
    "cgi": "cg_animation",
    "iyashikei": "healing",
    "isekai": "isekai",
    "love_polygon": "love_triangle",  # MAL theme ↔ AniList tag
    "harem": "harem",
    "reverse_harem": "reverse_harem",
    "primarily_female_cast": "primarily_female_cast",
    "ensemble_cast": "ensemble_cast",
}

# Kategorie, ve kterých se může objevit tentýž koncept z více zdrojů.
# Pořadí priority při slučování (dřívější vyhraje při kolizi labelu).
CATEGORY_PRIORITY = [
    "genre", "demographic", "source", "format", "decade",
    "theme", "tag", "studio", "director", "writer",
]

# Sdíleno s jikan.py::list_all_staff (dřív duplikováno na dvou místech --
# canonicalizace pozic patří sem, k ostatní logice atributů, ne do API klienta).
DIRECTOR_POSITIONS = {"director", "series director"}
WRITER_POSITIONS = {"script", "series composition", "screenplay",
                    "original creator", "original story"}


def resolve_alias(key: str) -> str:
    return ALIAS.get(key, key)


# ── Atribut ─────────────────────────────────────────────────────────────────

@dataclass
class AttrValue:
    category: str   # genre | theme | tag | studio | demographic | source | format | decade
    weight: float   # 0–1 (binární příznak = 1.0)
    label: str      # hezký název pro zobrazení
    spoiler: bool = False  # AniList isGeneralSpoiler/isMediaSpoiler -- atribut
                           # vstupuje do modelu normálně, jen se v HTML reportu
                           # dá skrýt přepínačem (viz report.py)


def _add(out: dict[str, AttrValue], raw_name: str, category: str, weight: float,
         spoiler: bool = False):
    """Přidá atribut; při kolizi klíče ponechá vyšší váhu a kategorii dle priority."""
    key = resolve_alias(canon(raw_name))
    if not key:
        return
    label = raw_name.strip()
    if key in out:
        prev = out[key]
        # vyšší váha vyhrává
        new_w = max(prev.weight, weight)
        # spoiler stačí z jednoho zdroje (opatrnější varianta vyhrává)
        new_s = prev.spoiler or spoiler
        # kategorie dle priority (nižší index = vyšší priorita)
        def pr(c):
            return CATEGORY_PRIORITY.index(c) if c in CATEGORY_PRIORITY else 99
        if pr(category) < pr(prev.category):
            out[key] = AttrValue(category, new_w, label, new_s)
        else:
            out[key] = AttrValue(prev.category, new_w, prev.label, new_s)
    else:
        out[key] = AttrValue(category, weight, label, spoiler)


def build_attributes(
    jikan: dict | None,
    anilist: dict | None,
    *,
    anilist_min_rank: int = 0,
    include_studios: bool = True,
    staff: list[dict] | None = None,
) -> dict[str, AttrValue]:
    """
    Sestaví kanonický slovník atributů pro jedno anime ze zdrojů Jikan + AniList.

    Args:
        jikan:   data z Jikan /anime/{id}/full (nebo None)
        anilist: data z AniList Media (nebo None)
        staff:   data z Jikan /anime/{id}/staff (nebo None) -- viz
                 JikanClient.get_anime_staff. Volitelné a vypnuté defaultně
                 (EnrichCfg.include_staff), protože stojí extra API volání
                 navíc k /full pro každý titul.
    """
    out: dict[str, AttrValue] = {}

    # ── MAL / Jikan ──────────────────────────────────────────────
    if jikan:
        for g in jikan.get("genres", []) or []:
            _add(out, g["name"], "genre", 1.0)
        for t in jikan.get("themes", []) or []:
            _add(out, t["name"], "theme", 1.0)
        for d in jikan.get("demographics", []) or []:
            _add(out, d["name"], "demographic", 1.0)
        src = (jikan.get("source") or "").strip()
        if src and src.lower() not in ("unknown", "other", ""):
            _add(out, src, "source", 1.0)
        fmt = (jikan.get("type") or "").strip()
        if fmt:
            _add(out, fmt, "format", 1.0)
        year = jikan.get("year")
        if year:
            _add(out, f"{(int(year)//10)*10}s", "decade", 1.0)
        if include_studios:
            for s in jikan.get("studios", []) or []:
                _add(out, s["name"], "studio", 1.0)

    # ── AniList ──────────────────────────────────────────────────
    if anilist:
        # Žánry bezpodmínečně -- kanonizace je stejně sloučí s MAL žánry
        # (stejný klíč), takže v normálním režimu nic nezdvojí a v nouzovém
        # AniList-only režimu (--no-jikan) nesou žánrový signál samy.
        for g in anilist.get("genres") or []:
            _add(out, g, "genre", 1.0)
        for tag in anilist.get("tags", []) or []:
            if tag.get("isAdult"):
                continue
            rank = (tag.get("rank") or 0)
            if rank < anilist_min_rank:
                continue
            # Spoiler tagy (Tragedy, Tearjerker, ...) se dřív zahazovaly
            # úplně -- model tak přicházel o nejsilnější signály osy
            # náročnosti. Teď vstupují normálně, jen nesou příznak, podle
            # kterého je HTML report umí skrýt (rozhodnutí uživatele).
            spoiler = bool(tag.get("isGeneralSpoiler") or tag.get("isMediaSpoiler"))
            _add(out, tag["name"], "tag", rank / 100.0, spoiler=spoiler)
        if include_studios:
            for node in anilist.get("studios", {}).get("nodes", []) or []:
                if node.get("isAnimationStudio"):
                    _add(out, node["name"], "studio", 1.0)
        # Source/format/dekáda jen jako fallback (MAL má přednost, když ho
        # máme) -- u formátu/dekády by rozdílná hodnota z obou zdrojů (např.
        # jiný rok premiéry) vyrobila dva atributy pro jeden koncept.
        if not (jikan and jikan.get("source")):
            asrc = (anilist.get("source") or "").strip()
            if asrc:
                _add(out, asrc.replace("_", " ").title(), "source", 1.0)
        if not (jikan and jikan.get("type")):
            afmt = (anilist.get("format") or "").strip()
            if afmt:
                _add(out, afmt.replace("_", " ").title(), "format", 1.0)
        if not (jikan and jikan.get("year")):
            ayear = anilist.get("seasonYear") or (anilist.get("startDate") or {}).get("year")
            if ayear:
                _add(out, f"{(int(ayear)//10)*10}s", "decade", 1.0)

    # ── Staff (režie / scénář) ─────────────────────────────────────
    # Samostatná kategorie na osobu+roli (ne jen na osobu), protože dobrý
    # režisér nemusí být dobrý scenárista a naopak -- "líbí se mi všechno od
    # X jako scenáristy" a "od X jako režiséra" jsou dva různé signály, i
    # když jde o tutéž osobu. Vyloučeno z mood-klastrování (taste.py filtruje
    # na genre/theme/tag/demographic) -- preference tvůrce není nálada.
    if staff:
        seen_directors, seen_writers = set(), set()
        for entry in staff:
            person = entry.get("person") or {}
            name = (person.get("name") or "").strip()
            if not name:
                continue
            positions = {p.lower() for p in (entry.get("positions") or [])}
            if positions & DIRECTOR_POSITIONS and name not in seen_directors:
                _add(out, f"Director: {name}", "director", 1.0)
                seen_directors.add(name)
            if positions & WRITER_POSITIONS and name not in seen_writers:
                _add(out, f"Writer: {name}", "writer", 1.0)
                seen_writers.add(name)

    return out


def community_baseline(jikan: dict | None, anilist: dict | None) -> float | None:
    """
    Vrátí komunitní skóre (0–10) jako baseline. Primárně MAL, fallback AniList.
    Záměrně NEprůměrujeme oba (jsou silně korelované → žádný přínos, riziko
    zkreslení); MAL je referenční, protože uživatelova data jsou z MAL.
    """
    if jikan and jikan.get("score"):
        return float(jikan["score"])
    if anilist and anilist.get("averageScore"):
        return float(anilist["averageScore"]) / 10.0
    return None
