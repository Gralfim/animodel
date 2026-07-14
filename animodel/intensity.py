"""
intensity.py — osa emocionální náročnosti řízená generovaným lexikonem.

Nahrazuje dřívější ručně vypsané HEAVY/LIGHT množiny v taste.py (viz
HODNOCENI_PROJEKTU.md §7.2): množina klíčů se teď generuje EXAKTNĚ z
úplného universa atributů (AniList MediaTagCollection + Jikan /genres/anime),
lidský úsudek zůstává jen v HODNOTÁCH — v jednom revidovatelném souboru
(`intensity.yaml`), ne zahrabaný v kódu.

Tři vrstvy, jak hodnota pro klíč vzniká (první vyhrává):
  1. existující hodnota z uživatelova intensity.yaml (jeho úpravy se při
     regeneraci NIKDY nepřepisují),
  2. CURATED — jednorázové ruční ohodnocení běžných tagů/žánrů (tady v kódu,
     slouží jen jako prefill při generování + jako vestavěný default, dokud
     si uživatel soubor nevygeneruje),
  3. prior podle AniList `category` (Theme-Comedy → lehké, Theme-Drama →
     těžké, ...) — AniListova vlastní kurátorovaná taxonomie, ne odhad,
  4. 0.0 = neutrální (do výpočtu intenzity nevstupuje).

Škála je spojitá: −1.0 (nejlehčí) … +1.0 (nejtěžší). Hodnota 0.0 znamená
"neutrální/neohodnoceno" a klíč se při výpočtu přeskakuje — stejná sémantika
jako dřívější nečlenství v HEAVY ani LIGHT.
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from .attributes import canon, resolve_alias

log = logging.getLogger(__name__)


def _k(name: str) -> str:
    """Kanonický klíč atributu — MUSÍ odpovídat tomu, co vyrábí
    attributes._add() (canon + alias), jinak se lexikon mine s daty."""
    return resolve_alias(canon(name))


# ── Prior podle AniList kategorie ────────────────────────────────────────
# AniList tagy nesou kurátorované pole `category` (Theme-Comedy,
# Theme-Drama, Cast-Traits, Setting-…, Technical, ...). Tyhle tři kategorie
# mají jednoznačný vztah k ose náročnosti; všechno ostatní je bez prioru
# (→ 0.0, dokud to CURATED nebo uživatel neohodnotí).
CATEGORY_PRIOR: dict[str, float] = {
    "Theme-Comedy": -0.6,
    "Theme-Slice of Life": -0.5,
    "Theme-Drama": 0.5,
}


def category_prior(category: str | None) -> float:
    if not category:
        return 0.0
    for prefix, value in CATEGORY_PRIOR.items():
        if category.startswith(prefix):
            return value
    return 0.0


# ── Jednorázové ruční ohodnocení (prefill + vestavěný default) ──────────
# Klíče v post-alias canon podobě. Jen nenulové hodnoty — nula je default.
# Položky, které v universu neexistují, se při generování prostě neuplatní
# (a jako default lexikon jsou neškodné — nikdy se nepozorují), takže je
# bezpečné být tady velkorysý. Hodnoty jsou startovní odhad k REVIZI v
# vygenerovaném intensity.yaml, ne finální pravda.
CURATED: dict[str, float] = {
    # ── MAL žánry ──
    "comedy": -0.8,
    "drama": 0.6,
    "horror": 0.8,
    "mystery": 0.2,
    "suspense": 0.6,
    "slice_of_life": -0.6,
    "gourmet": -0.5,
    "sports": -0.2,
    "ecchi": -0.3,
    "avant_garde": 0.2,

    # ── MAL témata ──
    "psychological": 0.7,
    "gore": 0.8,
    "military": 0.4,
    "survival": 0.7,
    "harem": -0.4,
    "reverse_harem": -0.4,
    "parody": -0.7,
    "gag_humor": -0.9,
    "anthropomorphic": -0.5,
    "cute_girls_doing_cute_things": -0.8,
    "childcare": -0.3,
    "healing": -0.9,            # alias cíl pro Iyashikei
    "detective": 0.2,
    "educational": -0.4,
    "high_stakes_game": 0.5,
    "strategy_game": 0.2,
    "idols_female": -0.3,
    "idols_male": -0.3,
    "kids": -0.8,
    "love_triangle": 0.1,       # alias cíl pro Love Polygon
    "medical": 0.2,
    "organized_crime": 0.4,
    "otaku_culture": -0.3,
    "pets": -0.5,
    "team_sports": -0.2,
    "time_travel": 0.1,
    "music": -0.2,

    # ── AniList tagy: těžká strana ──
    "tragedy": 0.9,
    "tearjerker": 0.8,
    "suicide": 0.9,
    "bullying": 0.6,
    "body_horror": 0.9,
    "cosmic_horror": 0.7,
    "torture": 0.9,
    "war": 0.7,
    "politics": 0.4,
    "dystopian": 0.6,
    "post_apocalyptic": 0.5,
    "philosophy": 0.6,
    "terrorism": 0.6,
    "slavery": 0.7,
    "child_abuse": 0.9,
    "domestic_abuse": 0.8,
    "terminal_illness": 0.8,
    "grief": 0.7,
    "depression": 0.7,
    "revenge": 0.5,
    "crime": 0.3,
    "yakuza": 0.4,
    "prison": 0.5,
    "noir": 0.4,
    "assassins": 0.2,
    "death_game": 0.6,
    "melodrama": 0.5,
    "disability": 0.4,
    "drugs": 0.5,
    "amnesia": 0.2,
    "coming_of_age": 0.2,

    # ── AniList tagy: lehká strana ──
    "slapstick": -0.8,
    "surreal_comedy": -0.7,
    "satire": -0.4,
    "cooking": -0.5,
    "food": -0.5,
    "episodic": -0.2,
    "family_life": -0.4,
    "found_family": -0.2,
    "school_club": -0.2,
    "band": -0.2,
    "chibi": -0.7,
    "cute_boys_doing_cute_things": -0.8,
    "idol": -0.3,
    "heartwarming": -0.5,
}

# Vestavěný default: použije se, dokud si uživatel nevygeneruje vlastní
# intensity.yaml (--gen-intensity). Nahrazuje dřívější HEAVY/LIGHT množiny —
# na rozdíl od nich má spojité hodnoty a jednu sadu klíčů s prefillem výše.
DEFAULT_LEXICON: dict[str, float] = dict(CURATED)


# ── Načtení uživatelova lexikonu ─────────────────────────────────────────

def load_lexicon(path: str | Path) -> dict[str, float] | None:
    """
    Načte intensity.yaml → {canon_klíč: hodnota ∈ [−1, 1]}.
    Vrací None, když soubor neexistuje (volající spadne na DEFAULT_LEXICON).

    Klíče se kanonizují znovu při načtení (defenzivně — uživatel může do
    souboru napsat "Sci-Fi" a trefí se na science_fiction), hodnoty mimo
    rozsah se oříznou s warningem, nečíselné se přeskočí s warningem.
    """
    p = Path(path)
    if not p.exists():
        return None
    import yaml
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        ck = _k(str(key))
        if not ck:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            log.warning(f"intensity lexikon {p}: '{key}' má nečíselnou hodnotu "
                        f"{value!r} -- přeskakuju")
            continue
        if v < -1.0 or v > 1.0:
            log.warning(f"intensity lexikon {p}: '{key}' = {v} mimo rozsah "
                        f"[-1, 1] -- ořezávám")
            v = max(-1.0, min(1.0, v))
        out[ck] = v
    return out


# ── Universum atributů + generování souboru ──────────────────────────────

def build_universe(jikan=None, anilist=None) -> list[dict]:
    """
    Úplné universum atributů, které build_attributes() může vyprodukovat
    v kategoriích genre/theme/tag:

      - Jikan /genres/anime?filter=genres|themes  (MAL žánry + témata)
      - AniList MediaTagCollection                (všechny tagy)

    Vrací list {"key","label","group","description","category"}, dedupe
    podle klíče (MAL má přednost — je menší a tvoří jádro; stejný koncept
    z AniListu by stejně dostal tentýž canon klíč).

    isAdult tagy se vynechávají — build_attributes() je zahazuje taky,
    takže by v lexikonu nikdy nic netrefily. Spoiler tagy (isGeneralSpoiler)
    se naopak ZAHRNUJÍ: build_attributes je od 2026-07 pouští do modelu
    s příznakem (report je umí skrýt) a pro osu náročnosti jsou to
    nejsilnější signály vůbec (Tragedy, Tearjerker, ...).
    """
    universe: list[dict] = []
    seen: set[str] = set()

    def add(name, group, description="", category=None):
        key = _k(name)
        if not key or key in seen:
            return
        seen.add(key)
        universe.append({
            "key": key, "label": name.strip(), "group": group,
            "description": (description or "").strip(),
            "category": category,
        })

    if jikan is not None:
        for g in jikan.get_genres("genres"):
            if g.get("name"):
                add(g["name"], "MAL: žánry")
        for t in jikan.get_genres("themes"):
            if t.get("name"):
                add(t["name"], "MAL: témata")

    if anilist is not None:
        # Žánry zvlášť -- MediaTagCollection je neobsahuje (Media.genres je
        # samostatné pole) a bez nich by v AniList-only režimu chyběly
        # comedy/drama/horror/... = nejsilnější klíče osy náročnosti.
        for g in anilist.get_genre_collection():
            if g:
                add(g, "AniList: žánry")
        for tag in anilist.get_tag_collection():
            if not tag.get("name"):
                continue
            if tag.get("isAdult"):
                continue
            cat = tag.get("category") or "ostatní"
            add(tag["name"], f"AniList: {cat}",
                description=tag.get("description") or "", category=cat)

    return universe


def generate_lexicon(out_path: str | Path, jikan=None, anilist=None,
                     observed_counts: dict[str, int] | None = None) -> dict:
    """
    Vygeneruje (nebo aktualizuje) intensity.yaml.

    Merge sémantika: existující hodnoty uživatele VŽDY vyhrávají; nové klíče
    z universa se doplní s prefillem CURATED → category prior → 0.0.
    Uživatelovy klíče mimo universum se zachovají v samostatné sekci.

    observed_counts ({canon_klíč: počet titulů v uživatelově seznamu})
    slouží jen k řazení uvnitř sekcí (nejčastější první = priorita revize)
    a ke komentáři "N×" u řádku.

    Vrací statistiky {"total", "from_existing", "from_curated",
    "from_prior", "zero", "custom_kept"}.
    """
    out_path = Path(out_path)
    counts = observed_counts or {}
    existing = load_lexicon(out_path) or {}

    universe = build_universe(jikan=jikan, anilist=anilist)
    stats = {"total": len(universe), "from_existing": 0, "from_curated": 0,
             "from_prior": 0, "zero": 0, "custom_kept": 0}

    def value_for(entry) -> float:
        key = entry["key"]
        if key in existing:
            stats["from_existing"] += 1
            return existing[key]
        if key in CURATED:
            stats["from_curated"] += 1
            return CURATED[key]
        prior = category_prior(entry["category"])
        if prior:
            stats["from_prior"] += 1
            return prior
        stats["zero"] += 1
        return 0.0

    # Skupiny ve stabilním pořadí: MAL žánry, MAL témata, AniList žánry,
    # pak AniList tag kategorie abecedně
    _fixed = {"MAL: žánry": 0, "MAL: témata": 1, "AniList: žánry": 2}
    groups: dict[str, list[dict]] = {}
    for entry in universe:
        groups.setdefault(entry["group"], []).append(entry)
    group_order = sorted(groups, key=lambda g: (_fixed.get(g, 2), g))

    lines = [
        "# intensity.yaml — osa emocionální náročnosti",
        "# Škála: -1.0 (nejlehčí: komedie, iyashikei) … +1.0 (nejtěžší: tragédie, psycho).",
        "# 0.0 = neutrální, do výpočtu nevstupuje.",
        "#",
        f"# Vygenerováno: python -m animodel --gen-intensity ({_dt.date.today().isoformat()})",
        "# Při regeneraci se tvé hodnoty ZACHOVAJÍ, doplní se jen nové klíče.",
        "# Řádky s N× = kolik titulů ve tvém seznamu ten atribut nese (priorita revize).",
        "",
    ]
    universe_keys = set()
    for group in group_order:
        entries = sorted(groups[group],
                         key=lambda e: (-counts.get(e["key"], 0), e["key"]))
        lines.append(f"# ── {group} ──")
        for entry in entries:
            universe_keys.add(entry["key"])
            v = value_for(entry)
            n = counts.get(entry["key"], 0)
            comment_bits = [entry["label"]]
            if n:
                comment_bits.append(f"{n}×")
            desc = entry["description"]
            if desc:
                comment_bits.append(desc[:70].replace("\n", " ").rstrip())
            lines.append(f"{entry['key']}: {v:g}  # {' — '.join(comment_bits)}")
        lines.append("")

    custom = {k: v for k, v in existing.items() if k not in universe_keys}
    if custom:
        stats["custom_kept"] = len(custom)
        lines.append("# ── Vlastní klíče (mimo stažené universum) ──")
        for key in sorted(custom):
            lines.append(f"{key}: {custom[key]:g}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return stats
