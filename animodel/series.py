"""
series.py — Seskupování franšíz přes sequel/prequel vazby

Problém: MAL a Jikan vidí každou řadu série jako samostatný titul.
Hodnocení jednotlivých řad ale silně koreluje navzájem — bez korekce by
oblíbená mnohadílná franšíza uměle nafoukla trénovací data.

Řešení: Union-Find (disjoint-set) přes Sequel/Prequel vazby.
Každá connected component = jedna franšíza. Členové skupiny pak dostanou
snížené váhy (enrich.py::build_titles: hlavní řady 1/√k_eff, vedlejší
obsah ještě míň dle side_story_weight) -- každá řada zůstává samostatným
datovým bodem s vlastní známkou i atributy, jen mluví tišeji.

(Dřívější alternativa `aggregate_entries` -- kolaps franšízy na jeden
záznam s MAX skóre a atributy zástupce -- byla odstraněna 2026-07:
zahazovala vnitro-franšízový signál (řady hodnocené různě), párovala
max(známky) s komunitním skóre jiné řady a byla křehká vůči chybnému
seskupení. Vážený přístup degraduje elegantně.)
"""

import logging

log = logging.getLogger(__name__)


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    """Jednoduchá union-find datová struktura s path compressionem."""

    def __init__(self):
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py

    def components(self, ids: list[int]) -> dict[int, list[int]]:
        """Vrátí dict {root: [members]} pro zadaná ID."""
        result: dict[int, list[int]] = {}
        for id_ in ids:
            root = self.find(id_)
            result.setdefault(root, []).append(id_)
        return result


# ── Hlavní funkce ─────────────────────────────────────────────────────────────

# Typy vazeb z Jikan, které spojujeme do série
SERIES_RELATION_TYPES = {
    "sequel",
    "prequel",
    "alternative version",
    "side story",        # spojuje do skupiny; vedlejšost řeší váhy (enrich.py)
}

# Typy vazeb které NESPOJUJEME (jiný příběh/vesmír)
SKIP_RELATION_TYPES = {
    "adaptation",
    "character",
    "other",
    "summary",
}


def build_series_groups(
    mal_ids:    list[int],
    jikan_data: dict,          # {mal_id: jikan_full_data}
    relation_types: set[str] = SERIES_RELATION_TYPES,
) -> dict[int, list[int]]:
    """
    Sestaví skupiny sérií přes Sequel/Prequel vazby.

    Vstup:
        mal_ids       — seznam MAL ID k seskupení
        jikan_data    — Jikan /anime/{id}/full data (musí obsahovat 'relations')
        relation_types — typy vazeb které spojují do série

    Výstup:
        dict {root_mal_id: [member_mal_ids]}
        Každá skupina = jedna série. Singleton = standalone titul.
    """
    id_set = set(mal_ids)
    uf     = UnionFind()

    # Inicializuj všechna ID
    for mid in mal_ids:
        uf.find(mid)

    # Projdi relace a spoj příbuzné tituly
    linked = 0
    for mal_id in mal_ids:
        data = jikan_data.get(mal_id)
        if not data:
            continue

        for relation in data.get("relations", []):
            rel_type = (relation.get("relation") or "").lower()
            if rel_type not in relation_types:
                continue

            for entry in relation.get("entry", []):
                if entry.get("type") != "anime":
                    continue
                related_id = entry.get("mal_id")
                # Spoj pouze pokud je related_id také v našem datasetu
                if related_id and related_id in id_set:
                    if uf.find(mal_id) != uf.find(related_id):
                        uf.union(mal_id, related_id)
                        linked += 1

    groups = uf.components(mal_ids)
    singletons = sum(1 for g in groups.values() if len(g) == 1)
    series     = len(groups) - singletons

    log.info(
        f"Série: {series} skupin z {len(mal_ids)} titulů "
        f"({linked} vazeb), {singletons} standalone titulů"
    )
    return groups


def print_series_groups(
    entries:    list,
    jikan_data: dict,
    titles_map: dict[int, str],
) -> None:
    """Vypíše přehled nalezených sériových skupin (pro --analyze)."""
    mal_ids = [e.mal_id for e in entries]
    groups  = build_series_groups(mal_ids, jikan_data)
    entry_map = {e.mal_id: e for e in entries}

    multi = [(root, members) for root, members in groups.items() if len(members) > 1]
    multi.sort(key=lambda x: -len(x[1]))

    print(f"\n{'═'*60}")
    print(f"  SÉRIOVÉ SKUPINY ({len(multi)} sérií nalezeno)")
    print(f"{'═'*60}")

    for root, members in multi:
        scores = [entry_map[m].score for m in members if m in entry_map]
        ttls   = [titles_map.get(m, str(m))[:35] for m in members]
        print(f"\n  Serie ({len(members)} řad, scores: {scores}, max: {max(scores)}):")
        for t in ttls:
            print(f"    · {t}")
