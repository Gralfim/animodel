"""
taste.py — Jádro: model vkusu postavený na ODCHYLKÁCH od komunity.

═══════════════════════════════════════════════════════════════════════════
PROČ NE OBYČEJNÁ REGRESE NA HODNOCENÍ
═══════════════════════════════════════════════════════════════════════════
Uživatel dělá silný předvýběr → skoro vše má 7–10 (restriction of range).
Regrese na surovém skóre má proto minimum signálu a snadno přefituje.

Klíčový trik: necílíme na surové skóre, ale na REZIDUUM vůči komunitě:

    e_i = (moje_skóre_i − komunitní_skóre_i) − b0

kde b0 je moje průměrná "štědrost" (offset vůči komunitě). Reziduum říká
"líbí se mi tohle VÍC nebo MÍŇ než průměrnému divákovi" — a to má bohatou
varianci i tam, kde surová skóre varianci nemají.

═══════════════════════════════════════════════════════════════════════════
JAK SE UČÍ VLIV ATRIBUTŮ (proč shrinkage místo ručního výběru featur)
═══════════════════════════════════════════════════════════════════════════
Pro každý atribut spočítáme vážený průměr reziduí titulů, které ho mají,
a SMRŠTÍME ho k nule podle počtu vzorků (empirical Bayes / James–Stein):

    effect(attr) = (n_eff / (n_eff + K)) · vážený_průměr_reziduí

Atribut viděný u 3 titulů se přitáhne k nule (málo důkazů), atribut u 80
titulů si svůj odhad podrží. Tím automaticky odpadá ruční kurátorství
seznamu featur v configu — bereme VŠECHNY atributy a data sama rozhodnou.

═══════════════════════════════════════════════════════════════════════════
PREDIKCE
═══════════════════════════════════════════════════════════════════════════
    predikce_rezidua = b0 + s · Σ effect(attr)·weight(attr) + Σ interakce
    predikce_skóre   = komunita + predikce_rezidua  (oříznuto na 1–10)

Globální faktor s ∈ [0,1] se kalibruje cross-validací (další smrštění proti
přefitování). Vše je plně interpretovatelné — predikci lze rozložit na
příspěvky jednotlivých atributů.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from .attributes import AttrValue


# ── Osa "emocionální náročnosti" (osa únavy) ────────────────────────────────
# Reziduum říká CO mám rád; tahle osa říká, jak je daný titul EMOČNĚ náročný,
# aby šlo doporučení filtrovat podle aktuální nálady / únavy.
#
# Dřívější ručně vypsané HEAVY/LIGHT množiny (binární, odhadované klíče --
# viz HODNOCENI_PROJEKTU.md §7.2) nahradil spojitý lexikon {klíč: −1..+1}
# v animodel/intensity.py: množina klíčů se generuje exaktně z AniList
# MediaTagCollection + Jikan /genres/anime (`--gen-intensity` → intensity.yaml),
# hodnoty jsou revidovatelné v jednom souboru. Dokud soubor neexistuje,
# použije se vestavěný DEFAULT_LEXICON (prefill stejných hodnot).
from .intensity import DEFAULT_LEXICON


@dataclass
class Title:
    """Jeden ohodnocený titul vstupující do modelu."""
    mal_id: int
    title: str
    user_score: float
    community: float | None
    attrs: dict[str, AttrValue]
    weight: float = 1.0           # váha vzorku (franšízové tlumení, viz enrich.py)
    series_root: int | None = None  # kořen franšízové skupiny (union-find);
                                    # None = standalone. recommend.py přes něj
                                    # omezuje počet seedů na franšízu.


@dataclass
class AttrEffect:
    key: str
    label: str
    category: str
    n_eff: float          # efektivní počet vzorků (suma vah)
    raw_mean: float       # vážený průměr cílové proměnné (afinita)
    effect: float         # po shrinkage
    distinct: float       # o kolik to hodnotím výš než komunita (user−community)
    titles_pos: list      # příklady titulů s nejvyšší afinitou
    titles_neg: list
    spoiler: bool = False # tag je na AniListu spoiler-flagged (report ho umí skrýt)


@dataclass
class Interaction:
    a: str
    b: str
    label: str
    n: float              # vážená podpora páru (Σ w_a·w_b·w_titulu)
    lift: float           # reziduum navíc oproti součtu jednotlivých efektů,
                          # PO empiricko-bayesovském smrštění (stejné K jako efekty)
    spoiler: bool = False # aspoň jeden z atributů je spoiler-flagged


@dataclass
class Triple:
    """Hierarchická synergie TROJICE atributů (experiment, viz _fit_triples).

    Lift je definovaný NAD nižšími řády: od průměrného rezidua titulů
    nesoucích všechny tři atributy se odečtou jak singl efekty, tak lifty
    párů uvnitř trojice (jen těch, které prošly prahem -- přesně to, co by
    predikce pro tu kombinaci beztak dala; jinak by se tatáž struktura
    započetla dvakrát)."""
    keys: tuple           # (a, b, c) -- seřazené kanonické klíče
    label: str
    n: float              # vážená podpora (Σ w_a·w_b·w_c·w_titulu)
    lift: float           # hierarchický lift po smrštění (stejné K jako efekty)
    spoiler: bool = False


@dataclass
class Cluster:
    idx: int
    name: str
    size: int
    mean_user_score: float
    intensity: float                 # −1 (lehké) … +1 (náročné)
    signature: list                  # [(key, label, category, distinctiveness, spoiler), ...]
                                     # -- jen ZOBRAZOVACÍ výběr top-6 os klastru
    members: list                    # [(mal_id, title, user_score), ...] seřazeno
    affinity: float = 0.0            # vážený průměr REZIDUÍ členů: o kolik
                                     # náladu hodnotím nad baseline (komunita
                                     # + můj posun) -- synergický efekt celé
                                     # nálady, ne jen součtu jejích atributů
    centroid: dict = field(default_factory=dict)  # {feat_key: souřadnice} pro
                                     # NENULOVÉ složky těžiště klastru. Plný
                                     # profil nálady, ne jen 6 slov signatury --
                                     # recommend.py._cluster_fit na něm počítá
                                     # vážený kosinus (viz HODNOCENI §5.4).
    centroid_norm: float = 0.0       # ‖centroid‖₂, předpočítané (jmenovatel
                                     # kosinu; kandidátů jsou tisíce)
    archetype: tuple = ()            # (mal_id, titul, kosinus) člena NEJBLÍŽ
                                     # těžišti = nejtypičtější představitel
                                     # nálady. `members` je řazený podle
                                     # ZNÁMKY, tedy ukazuje nejlíp hodnocené
                                     # členy, ne ty charakteristické -- tohle
                                     # je doplňuje, ne nahrazuje.


class TasteModel:
    def __init__(
        self,
        # Defaulty drž v synchronu s config.ModelCfg (hlídá to test
        # tests/test_config_defaults.py) -- cli.py sice předává hodnoty
        # explicitně, ale test harness a programové použití (README)
        # konstruují TasteModel jen s částí argumentů a dřív tak tiše
        # běžely s jinými prahy než produkce (HODNOCENI_PROJEKTU.md §5.3).
        shrinkage_k: float = 8.0,
        min_attr_count: float = 4.0,
        interaction_min_count: float = 8.0,
        interaction_min_lift: float = 0.30,
        interaction_triples: bool = False,
        intensity: dict[str, float] | None = None,
    ):
        self.K = shrinkage_k
        self.min_attr_count = min_attr_count
        self.int_min_count = interaction_min_count
        self.int_min_lift = interaction_min_lift
        self.use_triples = interaction_triples
        # Lexikon osy náročnosti {canon_klíč: −1..+1}; None = vestavěný
        # default (viz intensity.py). cli.py sem předává load_lexicon(...).
        self.intensity = intensity if intensity is not None else DEFAULT_LEXICON

        self.titles: list[Title] = []
        self.b0: float = 0.0
        self.effects: dict[str, AttrEffect] = {}
        self.all_attr_keys: set[str] = set()
        self.interactions: list[Interaction] = []
        self.triples: list[Triple] = []
        self.scale: float = 1.0           # globální faktor s z cross-validace
                                          # (singl efekty + párové interakce)
        self.scale_triples: float = 0.0   # VLASTNÍ faktor pro trojice, taky z CV.
                                          # Trojice jsou jiný řád důkazů než
                                          # singly/páry (řídší podpora, kandidáti
                                          # z klastrových signatur), takže sdílet
                                          # s nimi jedno `s` není o co opřít --
                                          # viz _calibrate_scale.
        self.cv_rmse: float = 0.0
        self.cv_mae: float = 0.0
        self.cv_rmse_no_triples: float = 0.0  # nejlepší CV RMSE při vypnutých
                                          # trojicích -- kolik reálně přinesly
        self.resid_std: float = 1.0       # pro intervaly predikce
        self.clusters: list[Cluster] = []
        self.cluster_feat_keys: set[str] = set()   # prostor "nálady" (genre/
                                          # theme/tag/demographic s dost vzorky).
                                          # Kandidátský vektor se na něj MUSÍ
                                          # omezit -- studio/formát/dekáda do
                                          # nálady nepatří a v kosinu by jen
                                          # nafukovaly jmenovatel.
        self._n_clusters: int | None = None   # předává se fold-modelům, ať
                                              # klastrují stejným postupem

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, titles: list[Title], n_clusters: int | None = None):
        self.titles = [t for t in titles if t.user_score > 0]
        if len(self.titles) < 20:
            raise ValueError(f"Příliš málo ohodnocených titulů ({len(self.titles)}).")
        self._n_clusters = n_clusters
        self._fit_baseline(self.titles)
        # cílová proměnná = odchylka od MÉ očekávané známky (po zohlednění komunity)
        self._resid = {t.mal_id: self._target(t) for t in self.titles}
        self._fit_effects()
        self._fit_interactions()
        # n_clusters se předává rovnou sem -- dřív se klastrovalo jednou tady
        # (vždy s k=None/auto, config se ignoroval) a pak ZNOVU explicitně
        # v cli.py s cfg.model.n_clusters, což zdvojovalo celý KMeans+silhouette
        # search zbytečně na každém běhu (ověřeno živě: druhé volání přepisovalo
        # výsledek prvního, ne ho doplňovalo).
        self._fit_clusters(n_clusters)
        # Trojice až PO klastrech -- kandidáty generují klastrové signatury.
        self.triples = []
        if self.use_triples:
            self._fit_triples()
        # Kalibrace až ÚPLNĚ NAKONEC, nad modelem v té podobě, v jaké bude
        # doopravdy predikovat. Dřív běžela PŘED klastrováním a trojicemi,
        # takže se `s` hledalo na modelu bez trojic a pak se aplikovalo na
        # predikci s nimi -- `s` bylo nadhodnocené a cv_rmse popisovala jiný
        # model, než který generoval doporučení (HODNOCENI_PROJEKTU.md §5.1).
        self._calibrate_scale()
        return self

    # ── Baseline: můj průměr + sklon vůči komunitě ───────────────────────────
    def _fit_baseline(self, titles):
        """
        Naučí ū (můj vážený průměr) a beta (jak moc kopíruju komunitní skóre).
        Predikce bez atributů = ū + beta·(komunita − c̄). Atributy pak vysvětlují
        zbytek. Tím komunitní skóre vstupuje jako JEDEN skalár, ne jako per-atribut
        confound (to byl problém čistého rezidua).
        """
        self.u_mean = self._weighted_mean([(t.user_score, t.weight) for t in titles])
        have_c = [t for t in titles if t.community is not None]
        if len(have_c) >= 10:
            self.c_mean = self._weighted_mean([(t.community, t.weight) for t in have_c])
            cov = sum(t.weight * (t.community - self.c_mean) * (t.user_score - self.u_mean)
                      for t in have_c)
            var = sum(t.weight * (t.community - self.c_mean) ** 2 for t in have_c)
            self.beta = (cov / var) if var > 1e-9 else 0.0
            self.beta = max(-0.5, min(1.5, self.beta))
        else:
            self.c_mean, self.beta = self.u_mean, 0.0
        # zachovaný offset vůči komunitě (jen pro report "distinctiveness")
        self.b0 = self._weighted_mean(
            [(t.user_score - t.community, t.weight) for t in have_c]) if have_c else 0.0

    def _baseline_pred(self, community):
        if community is None:
            return self.u_mean
        return self.u_mean + self.beta * (community - self.c_mean)

    def _target(self, t: Title):
        return t.user_score - self._baseline_pred(t.community)

    def _fit_effects(self):
        bucket: dict[str, list[tuple[float, float, Title]]] = defaultdict(list)
        meta: dict[str, AttrValue] = {}
        for t in self.titles:
            e = self._resid[t.mal_id]
            for key, av in t.attrs.items():
                bucket[key].append((e, av.weight * t.weight, t))
                meta[key] = av

        # Všechny pozorované klíče, i pod min_attr_count -- effects (níž) filtruje
        # na "dost důkazů pro fitnutý efekt", ale intensity_of() atributy čte
        # přímo z Title.attrs bez ohledu na to. Diagnostika lexikonu
        # (unrated_intensity_attrs) proto kontroluje proti TOMHLE, ne proti
        # self.effects.keys(). Metadata + efektivní počty se uchovávají kvůli
        # řazení diagnostiky podle důležitosti.
        self.all_attr_keys: set[str] = set(bucket.keys())
        self.all_attrs: dict[str, AttrValue] = dict(meta)
        self.attr_counts: dict[str, float] = {
            key: sum(w for _, w, _ in rows) for key, rows in bucket.items()
        }

        self.effects = {}
        for key, rows in bucket.items():
            n_eff = sum(w for _, w, _ in rows)
            if n_eff < self.min_attr_count:
                continue
            raw = self._weighted_mean([(e, w) for e, w, _ in rows])
            shrunk = (n_eff / (n_eff + self.K)) * raw
            # distinctiveness = o kolik to hodnotím výš než komunita (jen kde známe community)
            dpairs = [(r[2].user_score - r[2].community, r[1])
                      for r in rows if r[2].community is not None]
            distinct = self._weighted_mean(dpairs) if dpairs else 0.0
            ordered = sorted(rows, key=lambda r: -r[0])
            pos = [(r[2].mal_id, r[2].title, round(r[2].user_score, 1)) for r in ordered[:4]]
            neg = [(r[2].mal_id, r[2].title, round(r[2].user_score, 1)) for r in ordered[-4:][::-1]]
            self.effects[key] = AttrEffect(
                key=key, label=meta[key].label, category=meta[key].category,
                n_eff=n_eff, raw_mean=raw, effect=shrunk, distinct=distinct,
                titles_pos=pos, titles_neg=neg, spoiler=meta[key].spoiler,
            )

    def _fit_interactions(self):
        # Kandidáti = atributy s dost vzorky (jinak kombinatorika × šum)
        keyset = {k for k, e in self.effects.items() if e.n_eff >= self.int_min_count}
        # spočítej páry jen pro tituly (omezit kombinatoriku)
        pair_resid: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        for t in self.titles:
            present = sorted(k for k in t.attrs if k in keyset)
            e = self._resid[t.mal_id]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    # přítomnost páru vážená stejně jako u singl efektů:
                    # součin vah atributů (AniList rank) × franšízová váha
                    # titulu -- dřív se tag rank ignoroval (binární pár)
                    w = t.attrs[a].weight * t.attrs[b].weight * t.weight
                    pair_resid[(a, b)].append((e, w))

        self.interactions = []
        for (a, b), rows in pair_resid.items():
            n = sum(w for _, w in rows)
            if n < self.int_min_count:
                continue
            mean_pair = self._weighted_mean(rows)
            expected = self.effects[a].effect + self.effects[b].effect
            raw_lift = mean_pair - expected  # navíc oproti aditivnímu modelu
            # stejné empiricko-bayesovské smrštění jako u efektů -- pár na
            # hraně podpory dřív dostal plný lift (nekonzistence: singly
            # smrštěné byly, jejich kombinace ne). Práh int_min_lift se teď
            # vztahuje na SMRŠTĚNÝ lift, je tedy fakticky přísnější.
            lift = (n / (n + self.K)) * raw_lift
            if abs(lift) >= self.int_min_lift:
                self.interactions.append(Interaction(
                    a=a, b=b,
                    label=f"{self.effects[a].label} + {self.effects[b].label}",
                    n=n, lift=lift,
                    spoiler=self.effects[a].spoiler or self.effects[b].spoiler,
                ))
        self.interactions.sort(key=lambda x: -abs(x.lift))

    def _fit_triples(self):
        """
        Hierarchické synergie trojic (experiment, config
        model.interaction_triples). Kandidáti se negenerují slepou enumerací
        (na ~450 titulech by volné hledání v prostoru trojic dalo šum), ale
        z PODMNOŽIN KLASTROVÝCH SIGNATUR -- jádra nálad jsou přesně místa,
        kde má smysl se ptát "funguje tahle kombinace jen pohromadě?".

        Lift je definovaný nad nižšími řády (viz docstring Triple): odečítá
        se přesně to, co by predikce pro tu kombinaci dala ze singlů a
        prahem prošlých párů -- trojice tedy zachycuje JEN zbytek, žádné
        dvojité počítání.
        """
        from itertools import combinations

        self.triples = []
        candidates: set[tuple] = set()
        for c in self.clusters:
            sig_keys = sorted({s[0] for s in c.signature})
            candidates.update(combinations(sig_keys, 3))
        if not candidates:
            return

        pair_lift = {(it.a, it.b): it.lift for it in self.interactions}
        rows_by_triple: dict[tuple, list[tuple[float, float]]] = {c: [] for c in candidates}
        for t in self.titles:
            e = self._resid[t.mal_id]
            for combo in candidates:
                a, b, c = combo
                if a in t.attrs and b in t.attrs and c in t.attrs:
                    w = (t.attrs[a].weight * t.attrs[b].weight
                         * t.attrs[c].weight * t.weight)
                    rows_by_triple[combo].append((e, w))

        for combo, rows in rows_by_triple.items():
            n = sum(w for _, w in rows)
            if n < self.int_min_count:
                continue
            mean = self._weighted_mean(rows)
            a, b, c = combo
            expected = (self.effects[a].effect + self.effects[b].effect
                        + self.effects[c].effect
                        + pair_lift.get((a, b), 0.0)
                        + pair_lift.get((a, c), 0.0)
                        + pair_lift.get((b, c), 0.0))
            lift = (n / (n + self.K)) * (mean - expected)
            if abs(lift) >= self.int_min_lift:
                self.triples.append(Triple(
                    keys=combo,
                    label=" + ".join(self.effects[k].label for k in combo),
                    n=n, lift=lift,
                    spoiler=any(self.effects[k].spoiler for k in combo),
                ))
        self.triples.sort(key=lambda x: -abs(x.lift))

    def _resid_parts(self, attrs: dict[str, AttrValue]) -> tuple[float, float]:
        """
        Reziduální predikce rozložená na DVĚ nezávisle škálované složky:
        `(singly + páry, trojice)`, obojí PŘED globálními faktory.

        Rozdělení existuje proto, že každá složka má vlastní kalibrovaný
        faktor (`scale`, `scale_triples`) -- viz _calibrate_scale.
        """
        base = 0.0
        for key, av in attrs.items():
            e = self.effects.get(key)
            if e:
                base += e.effect * av.weight
        # interakce -- škálované vahami atributů KANDIDÁTA (analogie
        # `e.effect * av.weight` u singl efektů; slabě přítomný tag
        # neaktivuje synergii naplno)
        present = set(attrs)
        for it in self.interactions:
            if it.a in present and it.b in present:
                base += it.lift * attrs[it.a].weight * attrs[it.b].weight
        tri = 0.0
        for tr in self.triples:
            a, b, c = tr.keys
            if a in present and b in present and c in present:
                tri += (tr.lift * attrs[a].weight
                        * attrs[b].weight * attrs[c].weight)
        return base, tri

    def _raw_resid_pred(self, attrs: dict[str, AttrValue]) -> float:
        """Součet obou složek BEZ škálování (diagnostika). Pro predikci a
        řazení použij `affinity()` -- ta respektuje kalibrované faktory."""
        base, tri = self._resid_parts(attrs)
        return base + tri

    def affinity(self, attrs: dict[str, AttrValue]) -> float:
        """
        Kalibrovaná predikce afinity: `scale·(singly+páry) + scale_triples·trojice`.

        Veřejné API pro řazení (recommend.py, season.py) -- ty dřív sahaly
        na `_raw_resid_pred`, tedy na NEškálovaný součet. To bylo neškodné,
        dokud existoval jediný faktor (z-skóre společnou konstantu stejně
        vykrátí), ale se dvěma faktory na tom záleží: poměr scale_triples/scale
        je právě ta informace, kterou CV o trojicích zjistila.
        """
        base, tri = self._resid_parts(attrs)
        return self.scale * base + self.scale_triples * tri

    #: grid kandidátů pro `scale` i `scale_triples` (0.00, 0.05, …, 1.00)
    _SCALE_GRID = tuple(i / 20 for i in range(0, 21))

    def _calibrate_scale(self):
        """
        Najde faktory minimalizující CV RMSE; uloží i intervaly predikce.

        Bez trojic: 1D grid přes `s` (přesně dřívější chování).

        S trojicemi (`interaction_triples`): SPOLEČNÝ 2D grid přes
        `(s, s_triples)`. Trojice nedostávají tentýž faktor jako singly a
        páry, protože jsou to jiný řád důkazů -- řidší podpora a kandidáti
        vybíraní z klastrových signatur, ne z plné enumerace. Kolik jim
        věřit, je proto samostatná otázka a CV na ni umí odpovědět.
        Vyhodnocení jednoho bodu gridu je čistá aritmetika nad
        předpočítanými CV řádky (_eval_scale), takže 441 kombinací stojí
        zlomek sekundy -- fitovat se nic znovu nemusí.

        Při shodě RMSE vyhrává NIŽŠÍ faktor (grid je vzestupný a porovnává
        se ostrým `<`) -- konzervativnější varianta, konzistentní se
        smrštěním jinde v modelu.

        `cv_rmse_no_triples` = nejlepší RMSE dosažitelná s `s_triples = 0`;
        rozdíl proti `cv_rmse` je poctivá odpověď na „přinesly trojice
        vůbec něco?".
        """
        rows = self._cv_predictions()
        grid = self._SCALE_GRID

        best_rmse, best_mae, best_s, best_st = float("inf"), 0.0, 0.0, 0.0
        for s in grid:
            for st in (grid if self.use_triples else (0.0,)):
                rmse, mae = self._eval_scale(rows, s, st)
                if rmse < best_rmse:
                    best_rmse, best_mae, best_s, best_st = rmse, mae, s, st
        self.scale, self.scale_triples = best_s, best_st
        if not self.triples:
            # Fold-modely můžou mít trojice i když PLNÝ model žádnou neudrží
            # (jiná data → jiné lifty projdou prahem), takže grid umí vrátit
            # nenulové s_triples, které tady není co škálovat. Vynuluj, ať
            # atribut neříká něco, co na predikci nemá vliv.
            self.scale_triples = 0.0
        self.cv_rmse, self.cv_mae = best_rmse, best_mae

        self.cv_rmse_no_triples = min(
            self._eval_scale(rows, s, 0.0)[0] for s in grid)
        self.baseline_rmse, _ = self._eval_scale(rows, 0.0, 0.0)  # jen ū + beta·komunita
        self.resid_std = self.cv_rmse

    def _cv_predictions(self, folds: int = 5, seed: int = 42) -> list[tuple]:
        """
        Jednou přefituje fold-modely a pro každý out-of-fold titul vrátí
        čtveřici (baseline_predikce, singly+páry, trojice, skutečné_skóre).
        Vyhodnocení libovolného `(s, s_triples)` je pak čistá aritmetika nad
        těmito čtveřicemi (_eval_scale) -- žádné další fitování.

        S `interaction_triples` fold-model klastruje a fituje trojice SÁM,
        na svých 4/5 dat. Je to dražší (~0,3 s na fold; sklearn už je v tu
        chvíli naimportovaný z hlavního modelu, takže žádná studená cena),
        ale je to jediná varianta bez úniku informace: kandidáti trojic
        vznikají z klastrových signatur, takže převzít je hotové z plného
        modelu by do každého foldu protáhlo znalost jeho testovací pětiny --
        a přesně tomu se kalibrace vyhýbá.
        """
        rng = random.Random(seed)
        idx = list(range(len(self.titles)))
        rng.shuffle(idx)
        fold_of = {i: k % folds for k, i in enumerate(idx)}
        rows: list[tuple] = []
        for f in range(folds):
            train = [self.titles[i] for i in idx if fold_of[i] != f]
            test = [self.titles[i] for i in idx if fold_of[i] == f]
            sub = TasteModel(self.K, self.min_attr_count,
                             self.int_min_count, self.int_min_lift,
                             interaction_triples=self.use_triples,
                             intensity=self.intensity)
            sub.titles = train
            sub._fit_baseline(train)
            sub._resid = {t.mal_id: sub._target(t) for t in train}
            sub._fit_effects()
            sub._fit_interactions()
            if self.use_triples:
                sub._fit_clusters(self._n_clusters)
                sub._fit_triples()
            for t in test:
                base, tri = sub._resid_parts(t.attrs)
                rows.append((sub._baseline_pred(t.community), base, tri,
                             t.user_score))
        return rows

    @staticmethod
    def _eval_scale(rows: list[tuple], s: float,
                    s_triples: float = 0.0) -> tuple[float, float]:
        """(RMSE, MAE) pro dané `(s, s_triples)` nad předpočítanými CV
        predikcemi -- včetně ořezu predikce na 1–10, jako při skutečné
        predikci."""
        sq = ab = 0.0
        for base, raw, tri, y in rows:
            pred = max(1.0, min(10.0, base + s * raw + s_triples * tri))
            err = pred - y
            sq += err * err
            ab += abs(err)
        n = len(rows)
        rmse = math.sqrt(sq / n) if n else 0.0
        mae = ab / n if n else 0.0
        return rmse, mae

    # ── Predikce ─────────────────────────────────────────────────────────────

    def predict(self, attrs: dict[str, AttrValue], community: float | None):
        """
        Vrátí (predikce, dolní, horní, příspěvky) pro daný titul.
        příspěvky = seřazený list (label, category, value, spoiler) pro
        vysvětlení; spoiler=True znamená, že tag je na AniListu
        spoiler-flagged (obecně, nebo pro TENHLE konkrétní titul) a report
        ho umí přepínačem skrýt.
        """
        base = self._baseline_pred(community)
        pred = base + self.affinity(attrs)
        pred = max(1.0, min(10.0, pred))
        lo = max(1.0, pred - self.resid_std)
        hi = min(10.0, pred + self.resid_std)

        contribs = []
        for key, av in attrs.items():
            e = self.effects.get(key)
            if e and abs(e.effect) > 1e-6:
                contribs.append((e.label, e.category,
                                 self.scale * e.effect * av.weight,
                                 av.spoiler or e.spoiler))
        present = set(attrs)
        for it in self.interactions:
            if it.a in present and it.b in present:
                w_pair = attrs[it.a].weight * attrs[it.b].weight
                contribs.append((it.label, "interakce",
                                 self.scale * it.lift * w_pair, it.spoiler))
        for tr in self.triples:
            a, b, c = tr.keys
            if a in present and b in present and c in present:
                w_tri = attrs[a].weight * attrs[b].weight * attrs[c].weight
                # vlastní faktor -- příspěvek ve vysvětlení musí odpovídat
                # tomu, co skutečně vstoupilo do `pred` výš
                contribs.append((tr.label, "interakce",
                                 self.scale_triples * tr.lift * w_tri,
                                 tr.spoiler))
        contribs.sort(key=lambda x: -abs(x[2]))
        return pred, lo, hi, contribs

    def intensity_of(self, attrs: dict[str, AttrValue]) -> float:
        """
        Vážený průměr lexikonových skóre přítomných atributů ∈ [−1, +1].

        Klíče s hodnotou 0.0 (neutrální) nebo mimo lexikon se přeskakují --
        neředí výsledek, stejná sémantika jako dřívější nečlenství v
        HEAVY/LIGHT. Při lexikonu s hodnotami ±1 dává přesně starý vzorec
        (h − l)/(h + l); spojité hodnoty ho zjemňují.
        """
        num = 0.0
        den = 0.0
        for key, av in attrs.items():
            s = self.intensity.get(key, 0.0)
            if not s:
                continue
            num += av.weight * s
            den += av.weight
        return num / den if den else 0.0

    def _global_user_mean(self) -> float:
        return self._weighted_mean([(t.user_score, t.weight) for t in self.titles])

    # ── Klastrování nálad ────────────────────────────────────────────────────

    def _fit_clusters(self, k: int | None = None):
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
        except Exception:
            self.clusters = []
            return

        # Feature prostor: atributy s dost vzorky (stabilní osy)
        feat_keys = [k_ for k_, e in self.effects.items()
                     if e.n_eff >= max(4, self.min_attr_count)
                     and e.category in ("genre", "theme", "tag", "demographic")]
        if len(feat_keys) < 4:
            self.clusters = []
            return
        kidx = {k_: i for i, k_ in enumerate(feat_keys)}
        self.cluster_feat_keys = set(feat_keys)

        rows, meta = [], []
        for t in self.titles:
            v = [0.0] * len(feat_keys)
            for key, av in t.attrs.items():
                if key in kidx:
                    v[kidx[key]] = av.weight
            if sum(v) > 0:
                rows.append(v)
                meta.append(t)
        if len(rows) < 30:
            self.clusters = []
            return

        X = normalize(np.array(rows), norm="l2")
        # Franšízové váhy titulů (1/√k_eff apod., viz enrich.py) se propisují
        # i sem -- bez sample_weight by desetiřadá franšíza byla 10 plnohodnotných
        # bodů v prostoru nálad a táhla centroidy i velikosti klastrů, přestože
        # ve zbytku modelu (baseline/efekty/interakce) už je tlumená.
        w_all = np.array([t.weight for t in meta])
        if k is None:
            # vyber k podle siluety v rozsahu 4–7 (chceme interpretovatelné nálady)
            best_k, best_sil = 5, -1
            from sklearn.metrics import silhouette_score
            for kk in range(4, 8):
                if kk >= len(rows):
                    continue
                km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(
                    X, sample_weight=w_all)
                try:
                    # silhouette_score sample_weight nepodporuje -- výběr k
                    # zůstává nevážený (na tvar klastrů má váha vliv přes
                    # KMeans fit výš, tady jde jen o skóre dělení)
                    sil = silhouette_score(X, km.labels_)
                except Exception:
                    sil = -1
                if sil > best_sil:
                    best_sil, best_k = sil, kk
            k = best_k

        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=w_all)
        labels = km.labels_
        overall = np.average(X, axis=0, weights=w_all)

        clusters = []
        for c in range(k):
            members_i = [i for i in range(len(meta)) if labels[i] == c]
            if not members_i:
                continue
            sub = X[members_i]
            w_sub = w_all[members_i]
            centroid = np.average(sub, axis=0, weights=w_sub)
            distinct = centroid - overall
            sig_idx = distinct.argsort()[::-1][:6]
            signature = []
            for si in sig_idx:
                if distinct[si] <= 0:
                    continue
                key = feat_keys[si]
                # key se teď nese dál -> recommend.py._cluster_fit ho nemusí
                # zpětně dohledávat lineárním průchodem přes všechny effects
                # podle (label, category) -- bylo to zbytečně pomalé (dělalo
                # se to pro každého kandidáta × každý klastr × každou položku
                # signatury) a teoreticky křehké, kdyby dva různé atributy
                # měly stejný label i kategorii.
                signature.append((key, self.effects[key].label,
                                  self.effects[key].category,
                                  float(distinct[si]),
                                  self.effects[key].spoiler))
            mem = sorted(
                [(meta[i].mal_id, meta[i].title, meta[i].user_score) for i in members_i],
                key=lambda x: -x[2])
            # průměrné skóre, intenzita a afinita klastru vážené franšízovými
            # vahami -- `size` zůstává prostý počet titulů (zobrazovací údaj)
            w_tot = float(w_sub.sum())
            mean_score = float(sum(w * meta[i].user_score
                                   for w, i in zip(w_sub, members_i)) / w_tot)
            inten = float(sum(w * self.intensity_of(meta[i].attrs)
                              for w, i in zip(w_sub, members_i)) / w_tot)
            aff = float(sum(w * self._resid[meta[i].mal_id]
                            for w, i in zip(w_sub, members_i)) / w_tot)
            name = " / ".join(s[1] for s in signature[:3]) or f"Klastr {c+1}"
            # Těžiště se uchová CELÉ (nenulové složky), ne jen jako top-6
            # signatura: _cluster_fit na něm počítá vážený kosinus, takže
            # kandidát se poměřuje s plným profilem nálady. Dřív se ta
            # informace zahazovala a podobnost se počítala jen proti šesti
            # nejvýraznějším osám (HODNOCENI_PROJEKTU.md §5.4).
            cen = {feat_keys[i]: float(centroid[i])
                   for i in range(len(feat_keys)) if centroid[i] > 0}
            cen_norm = math.sqrt(sum(v * v for v in cen.values()))

            # Nejtypičtější člen ("archetyp"): nejvyšší kosinus k těžišti,
            # ale JEN mezi členy, které nesou aspoň mediánový počet atributů
            # nálady. Stejná metrika jako recommend.py::_cluster_fit, ať to,
            # co report ukazuje jako jádro nálady, odpovídá tomu, podle čeho
            # se kandidáti k náladám přiřazují.
            #
            # Proč ten filtr: kosinus je směrový, takže titul s JEDINÝM
            # atributem, který je zároveň hlavní osou těžiště, dostane vysokou
            # shodu, přestože z nálady nepokrývá skoro nic. Naměřeno: v jednom
            # klastru vyhrával "Petit Special" s 1 atributem (medián 10) a
            # korelace kosinu s počtem atributů tam byla −0,56. Medoid (nejvyšší
            # průměrná podobnost ke členům) to NEŘEŠÍ -- titul nesoucí jen
            # nejčastější atribut je podobný všem. Vážení pokrytím zas sklouzne
            # k "titulu s nejvíc tagy". Mediánový práh je odvozený z dat (ne
            # magická konstanta) a nikdy nevyprázdní výběr: aspoň polovina
            # členů ho vždy splní, takže není potřeba žádná záložní větev.
            arch = ()
            if cen_norm > 0:
                sims = (sub @ centroid) / cen_norm
                n_mood = [int(np.count_nonzero(sub[j])) for j in range(len(members_i))]
                thr = float(np.median(n_mood))
                # Shoda kosinu je běžná (tituly s identickou sadou atributů) --
                # rozhodne hlavní řada před vedlejším obsahem (vyšší franšízová
                # váha) a pak lepší známka, ať archetypem není náhodné OVA.
                best = max(
                    range(len(members_i)),
                    key=lambda j: (n_mood[j] >= thr,
                                   round(float(sims[j]), 12), float(w_sub[j]),
                                   meta[members_i[j]].user_score))
                bt = meta[members_i[best]]
                arch = (bt.mal_id, bt.title, float(sims[best]))

            clusters.append(Cluster(
                idx=c, name=name, size=len(mem),
                mean_user_score=mean_score, intensity=inten,
                signature=signature, members=mem, affinity=aff,
                centroid=cen, centroid_norm=cen_norm, archetype=arch,
            ))
        clusters.sort(key=lambda x: -x.size)
        self.clusters = clusters

    # ── Pomůcky ──────────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
        num = sum(v * w for v, w in pairs)
        den = sum(w for _, w in pairs)
        return num / den if den else 0.0

    def top_effects(self, n: int = 25, sign: int = 0):
        items = list(self.effects.values())
        if sign > 0:
            items = [e for e in items if e.effect > 0]
        elif sign < 0:
            items = [e for e in items if e.effect < 0]
        return sorted(items, key=lambda e: -abs(e.effect))[:n]

    def unrated_intensity_attrs(self, top: int | None = None) -> list[tuple[str, str, float]]:
        """
        Diagnostika intensity lexikonu -- OBRÁCENĚ než dřívější
        unmatched_intensity_keywords (ta hlídala odhadnuté klíče bez shody
        v datech; teď je množina klíčů exaktní z universa, takže se hlídá
        opačný směr): které POZOROVANÉ genre/theme/tag atributy nemají v
        lexikonu žádný záznam. Typicky tagy, které AniList přidal až po
        vygenerování intensity.yaml -- doplní se regenerací (--gen-intensity,
        existující hodnoty se zachovají).

        Klíč s explicitní hodnotou 0.0 se NEhlásí (je ohodnocený jako
        neutrální, ne zapomenutý). Volej AŽ PO fit().

        Vrací [(klíč, label, n_eff), ...] seřazené podle n_eff (nejčastější
        první = největší dopad na osu náročnosti). `top` výsledek ořízne.
        """
        out = [
            (key, av.label, self.attr_counts.get(key, 0.0))
            for key, av in self.all_attrs.items()
            if av.category in ("genre", "theme", "tag") and key not in self.intensity
        ]
        out.sort(key=lambda x: (-x[2], x[0]))
        return out[:top] if top is not None else out
