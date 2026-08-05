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
    # Nalezeno přes --analyze-attrs (2026-07-26): MAL a AniList pojmenovávají
    # týž koncept jinak, takže se evidence dělila na dvě poloviny a obě se
    # silněji smršťovaly k nule.
    "video_games": "video_game",            # MAL "Video Game" ↔ AniList "Video Games"
    "anthropomorphism": "anthropomorphic",  # MAL "Anthropomorphic" ↔ AniList tag
}

# Kategorie, ve kterých se může objevit tentýž koncept z více zdrojů.
# Pořadí priority při slučování (dřívější vyhraje při kolizi labelu).
CATEGORY_PRIORITY = [
    "genre", "demographic", "source", "format", "decade", "origin",
    "theme", "tag", "studio", "director", "writer",
]

# AniList countryOfOrigin → zobrazovací label. Japonsko SCHVÁLNĚ chybí: tvoří
# ~95 % každého seznamu, takže jako atribut by byl prakticky konstanta --
# efekt by vyšel ~0 a jen by zabíral místo v interakcích. Informace je právě
# v tom, když titul japonský NENÍ (donghua, korejská tvorba), a přesně tak se
# přidává (viz build_attributes).
_ORIGIN_LABELS = {
    "CN": "Čínský původ",
    "KR": "Korejský původ",
    "TW": "Tchajwanský původ",
}

# Kanonizace staff pozic patří sem, k ostatní logice atributů, ne do API
# klienta (dřív to bylo duplikované i v jikan.py::list_all_staff -- ta
# frekvenční pomůcka se nikdy nikde nevolala a byla 2026-07-25 smazána).
DIRECTOR_POSITIONS = {"director", "series director"}
WRITER_POSITIONS = {"script", "series composition", "screenplay",
                    "original creator", "original story"}


def resolve_alias(key: str) -> str:
    return ALIAS.get(key, key)


# AniList Media.format (UPPER_SNAKE) → zobrazovací label ve stylu MAL.
# Kanonický klíč je stejný tak jako tak (canon() case ignoruje) -- tohle je
# čistě kosmetika labelů ve vysvětleních a tabulkách.
_ANILIST_FORMAT_LABELS = {
    "TV": "TV", "TV_SHORT": "TV Short", "MOVIE": "Movie", "SPECIAL": "Special",
    "OVA": "OVA", "ONA": "ONA", "MUSIC": "Music",
}


# ── Atribut ─────────────────────────────────────────────────────────────────

@dataclass
class AttrValue:
    category: str   # genre | theme | tag | studio | demographic | source | format | decade
    weight: float   # 0–1 (binární příznak = 1.0)
    label: str      # hezký název pro zobrazení
    spoiler: bool = False  # AniList isGeneralSpoiler/isMediaSpoiler -- atribut
                           # vstupuje do modelu normálně, jen se v HTML reportu
                           # dá skrýt přepínačem (viz report.py)


def person_key(name: str) -> str:
    """
    Klíč osoby nezávislý na tom, který zdroj jméno dodal.

    Jikan píše „Mizushima, Tsutomu", AniList „Tsutomu Mizushima" -- tentýž
    člověk, dva různé řetězce. Změřeno na vzorku: ze ~159 jmen se doslovně
    shodovalo **9**, po seřazení slov **100**. Bez tohohle by přepnutí zdroje
    (nebo míchání zdrojů) rozdělilo evidence o témž režisérovi na dva klíče --
    přesně ten tichý selhací mód, který hlídá `--analyze-attrs`.

    Řadí slova abecedně, takže na pořadí příjmení/jméno nezáleží. Zůstávají
    rozdíly, které to spravit neumí (romanizace Ohkawa/Ookawa, pseudonymy) --
    ty jsou skutečné rozdíly dat, ne formátu.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = sorted(p for p in re.split(r"[^a-zA-Z0-9]+", ascii_name.lower()) if p)
    return "_".join(parts)


def _add(out: dict[str, AttrValue], raw_name: str, category: str, weight: float,
         spoiler: bool = False, key: str | None = None):
    """
    Přidá atribut; při kolizi klíče ponechá vyšší váhu a kategorii dle priority.

    `key` dovoluje klíč odvodit jinak než z `raw_name` -- používá to staff,
    kde má být klíč nezávislý na formátu jména (viz person_key), ale popisek
    má zůstat čitelný tak, jak ho dodal zdroj.
    """
    key = key or resolve_alias(canon(raw_name))
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
            # stejný filtr jako u Jikan source výš -- "OTHER" není informace
            # a jako atribut by se zobrazoval ve vysvětleních doporučení
            if asrc and asrc.lower() not in ("unknown", "other"):
                _add(out, asrc.replace("_", " ").title(), "source", 1.0)
        if not (jikan and jikan.get("type")):
            afmt = (anilist.get("format") or "").strip()
            if afmt:
                # AniList formáty jsou UPPER_SNAKE; .title() dělal z "TV"
                # label "Tv" (a "Ona", "Ova") -- zkratky drž velké, ať labely
                # odpovídají MAL podobě a kanonizace je sloučí i vizuálně
                label = _ANILIST_FORMAT_LABELS.get(afmt.upper(),
                                                   afmt.replace("_", " ").title())
                _add(out, label, "format", 1.0)
        if not (jikan and jikan.get("year")):
            ayear = anilist.get("seasonYear") or (anilist.get("startDate") or {}).get("year")
            if ayear:
                _add(out, f"{(int(ayear)//10)*10}s", "decade", 1.0)
        # Země původu jen když NENÍ japonská -- viz _ORIGIN_LABELS. Model o
        # donghua/korejské tvorbě dosud nevěděl nic, přestože se stylisticky
        # i produkčně liší; jako atribut to konečně může nést vlastní efekt.
        origin = _ORIGIN_LABELS.get((anilist.get("countryOfOrigin") or "").upper())
        if origin:
            _add(out, origin, "origin", 1.0)

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
            # Klíč z person_key (nezávislý na formátu jména napříč zdroji),
            # popisek ale zůstává tak, jak ho dodal zdroj -- v reportu má být
            # čitelné jméno, ne abecedně přeházené.
            pkey = person_key(name)
            if not pkey:
                continue
            if positions & DIRECTOR_POSITIONS and pkey not in seen_directors:
                _add(out, f"Director: {name}", "director", 1.0,
                     key=f"director_{pkey}")
                seen_directors.add(pkey)
            if positions & WRITER_POSITIONS and pkey not in seen_writers:
                _add(out, f"Writer: {name}", "writer", 1.0,
                     key=f"writer_{pkey}")
                seen_writers.add(pkey)

    return out


# ── Diagnostika kanonizace ──────────────────────────────────────────────────

def _edit_distance_within(a: str, b: str, limit: int) -> int | None:
    """
    Levenshteinova vzdálenost, ale jen dokud nepřeleze `limit` (pak `None`).

    Předčasné ukončení je tu kvůli rychlosti: diagnostika porovnává stovky
    klíčů každý s každým a plný výpočet by u drtivé většiny dvojic byl
    zbytečný -- ty se liší po prvních pár znacích.
    """
    if abs(len(a) - len(b)) > limit:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        if min(cur) > limit:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= limit else None


def _stem(token: str) -> str:
    """Hrubý stemmer na množné číslo (`games` → `game`). Nechává krátká
    slova a `-ss` být (`class`, `boss`)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def find_near_duplicate_keys(keys, *, max_distance: int = 2, min_len: int = 6,
                             prefix_ratio: float = 0.7) -> list[tuple]:
    """
    Najde dvojice kanonických klíčů, které vypadají jako TÝŽ koncept zapsaný
    dvakrát, a přitom je `ALIAS` nespojuje.

    Proč to chce vlastní nástroj: tichý selhací mód `attributes.py` je právě
    „dva klíče pro jeden koncept". Nikde nespadne, jen se evidence rozdělí na
    dvě poloviny, obě se silněji smrští k nule a efekt zmizí -- tedy přesně
    to, čemu má modul zabránit. Z běžného výstupu to poznat nejde.

    Dvě kritéria (stačí jedno):

    1. **Shodná slova** po hrubém stemmingu množného čísla, nezávisle na
       pořadí: `video_game` ↔ `video_games`, `police_female` ↔ `female_police`.
    2. **Jednoslovné klíče lišící se jen koncovkou**: editační vzdálenost
       ≤ `max_distance` A společný prefix aspoň `prefix_ratio` délky kratšího
       z nich -- `anthropomorphic` ↔ `anthropomorphism` (prefix 93 %).

    Proč tak přísně: první verze porovnávala celé klíče znakovým
    Levenshteinem a na reálných datech dala **19 dvojic, z toho 2 skutečné**.
    Znaková vzdálenost totiž nerozliší „jiný token" od „jiná koncovka":
    `female_protagonist` ↔ `male_protagonist` je vzdálenost 2 stejně jako
    `video_game` ↔ `video_games`, přestože první dvojice jsou dva různé
    koncepty a druhá jeden. Porovnání po slovech to odděluje: liší-li se
    klíče v celém slově, jde o různé koncepty; liší-li se jen koncovkou
    téhož slova, jde o tvarovou variantu. Prefixová podmínka pak vyřadí
    zbytek náhodných shod (`acting`/`action` mají prefix jen 67 %).

    `min_len` drží krátké klíče mimo -- u tří-čtyřznakových je vzdálenost 2
    skoro náhoda (`war`/`wax`).

    `keys` = iterovatelné kanonických klíčů. Vrací
    `[(klíč_a, klíč_b, důvod, vzdálenost), ...]`, deterministicky seřazené.
    """
    uniq = sorted({k for k in keys if k})
    out: list[tuple] = []
    seen_pairs: set[tuple] = set()

    def add(a, b, why, dist):
        pair = (a, b) if a < b else (b, a)
        if pair in seen_pairs or resolve_alias(a) == resolve_alias(b):
            return          # ALIAS je řešení, ne nález
        seen_pairs.add(pair)
        out.append((pair[0], pair[1], why, dist))

    # 1) shodná slova (po stemmingu), nezávisle na pořadí -- O(n) přes podpis
    by_tokens: dict[tuple, list[str]] = {}
    for k in uniq:
        sig = tuple(sorted(_stem(t) for t in k.split("_")))
        by_tokens.setdefault(sig, []).append(k)
    for group in by_tokens.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                add(group[i], group[j], "shodná slova (tvar/pořadí)", 0)

    # 2) jednoslovné klíče lišící se jen koncovkou
    singles = [k for k in uniq if "_" not in k and len(k) >= min_len]
    for i, a in enumerate(singles):
        for b in singles[i + 1:]:
            d = _edit_distance_within(a, b, max_distance)
            if not d:
                continue
            if _common_prefix_len(a, b) >= prefix_ratio * min(len(a), len(b)):
                add(a, b, f"stejný kmen, jiná koncovka (vzdálenost {d})", d)

    out.sort(key=lambda x: (x[3], x[0], x[1]))
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
