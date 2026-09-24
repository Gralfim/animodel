"""
selection.py — model VÝBĚRU: sáhnu po tom titulu vůbec?

═══════════════════════════════════════════════════════════════════════════
PROČ
═══════════════════════════════════════════════════════════════════════════
Model vkusu (taste.py) odhaduje, jakou známku dám, KDYŽ titul uvidím. Učí se
jen z toho, co jsem viděl -- a do seznamu se dostane jen to, co jsem si sám
vybral. Žánrům, kterým se vyhýbám (horor, thriller, krimi), v seznamu skoro
nic neodpovídá, takže je model vkusu bere jako NEUTRÁLNÍ a graf podobnosti
je pak klidně vytáhne nahoru (HODNOCENI_PROJEKTU.md §9d, nález #7).

Chybějící evidence se dá zčásti rekonstruovat z toho, co v seznamu CHYBÍ:
populární titul, který nemám ani na PTW, skoro jistě znám a vědomě ho
přeskakuju (podnět uživatele, §9d.4). Model vkusu a model výběru odpovídají
na dvě různé otázky, proto jsou to dvě oddělené složky kompozitu.

═══════════════════════════════════════════════════════════════════════════
JAK
═══════════════════════════════════════════════════════════════════════════
* Vesmír: top-N MAL titulů podle popularity (počtu členů), formát
  TV/Movie/ONA, premiéra aspoň MIN_AGE_DAYS zpět (o čerstvé sezóně ještě
  nemusím vědět), jen PRVNÍ díly franšíz (bez prequelu a parent story) --
  jinak by jedna přeskočená franšíza hlasovala tolikrát, kolik má řad.
* Popisek „zájem" = franšízu mám v seznamu v JAKÉMKOLI stavu, včetně PTW.
* Logistická regrese s L2 nad atributy (bez studií a staff) + log10(členů)
  jako kovariáta POVĚDOMÍ: méně známý titul chybí spíš proto, že o něm
  nevím (zájem klesá 45 % → 11 % mezi top-250 a #1001-2000), a tahle
  proměnná to z efektů atributů odfiltruje.
* Do kompozitu jde jen ATRIBUTOVÁ část logitu -- popularita je povědomí,
  ne preference.

Změřeno na reálném seznamu (§9d.4): 1 198 titulů, zájem 22 %, CV AUC 0,874
(samotná popularita 0,683). Nejsilněji se vyhýbám Primarily Male Cast
(zájem 3 %), Organized Crime, Police, Thriller, Horror, Crime, Suspense.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass, field

from .series import build_roots

log = logging.getLogger(__name__)

#: formáty vesmíru -- plnohodnotná díla. OVA/speciály se vybírají jinak
#: (jako doplněk rozjeté franšízy), do otázky „znám, a přesto nechci" nepatří.
MAIN_FORMATS = ("TV", "Movie", "ONA")
#: kolik dní po premiéře musí titul být, aby se jeho absence brala jako
#: vědomé přeskočení
MIN_AGE_DAYS = 180
#: kategorie mimo model: preference tvůrce se z popularity nevyčte a
#: studia/staff jen kódují identitu jedné franšízy
EXCLUDED_CATEGORIES = ("studio", "director", "writer")
#: min. vážený výskyt atributu ve vesmíru, aby byl featurou
MIN_FEATURE_COUNT = 5.0
#: L2 síla (sklearn C = 1/λ). Změřeno (§9d.4): C 0,1–0,3 → CV AUC 0,874,
#: 0,03 i 1,0 horší
C_L2 = 0.1
#: pod tolika tituly s/bez zájmu se model nefituje (není z čeho)
MIN_CLASS = 10
#: od jakého příspěvku k logitu se atribut ukáže jako „obvykle nevybíráš"
AVOID_THRESHOLD = 0.15
#: kategorie, které se v kartě ukazují jako „obvykle nevybíráš". Dekáda,
#: formát a zdroj v modelu zůstávají (sleduju hlavně novější tituly, to je
#: skutečný signál), ale „Obvykle nevybíráš: 2010s" nic nevysvětlí.
AVOID_DISPLAY_CATEGORIES = ("genre", "theme", "demographic", "tag")


@dataclass
class SelectionModel:
    """Naučený model výběru. `score` jde do kompozitu, `avoided` do karty."""
    coef: dict                 # {klíč atributu: koeficient logitu}
    labels: dict               # {klíč: label pro report}
    pop_coef: float            # koeficient log10(členů) -- jen diagnostika
    n: int                     # velikost vesmíru (první díly franšíz)
    n_engaged: int             # z toho se zájmem (v seznamu vč. PTW)
    cv_auc: float | None = None  # AUC out-of-fold, foldy po franšízách
    top_avoided: list = field(default_factory=list)  # [(label, koef)] nejzápornější
    categories: dict = field(default_factory=dict)   # {klíč: kategorie}

    def score(self, attrs: dict) -> float:
        """Atributová část logitu -- BEZ popularity."""
        return sum(self.coef.get(k, 0.0) * av.weight for k, av in attrs.items())

    def avoided(self, attrs: dict, n: int = 3) -> list[str]:
        """Labely atributů, které výběr nejvíc sráží (příspěvek ≤ −AVOID_THRESHOLD),
        jen z AVOID_DISPLAY_CATEGORIES."""
        items = sorted((self.coef.get(k, 0.0) * av.weight, k) for k, av in attrs.items()
                       if self.categories.get(k, av.category) in AVOID_DISPLAY_CATEGORIES)
        return [self.labels.get(k, k) for v, k in items if v <= -AVOID_THRESHOLD][:n]


def _has_prequel(rel: dict | None) -> bool:
    """Má titul PŘEDCHŮDCE (prequel nebo parent story) mezi anime?"""
    for r in (rel or {}).get("relations") or []:
        if (r.get("relation") or "").lower() in ("prequel", "parent story"):
            if any(e.get("type") == "anime" for e in r.get("entry") or []):
                return True
    return False


def _members(en) -> int:
    return int(((en.jikan or {}).get("members")) or 0)


def universe_entries(universe: dict, relations: dict,
                     today: _dt.date | None = None) -> list[int]:
    """MAL ID z vesmíru, které se počítají: hlavní formát, dost starý, první
    díl franšízy, se známým počtem členů."""
    cutoff = ((today or _dt.date.today())
              - _dt.timedelta(days=MIN_AGE_DAYS)).isoformat()
    out = []
    for mid, en in universe.items():
        j = en.jikan or {}
        if j.get("type") not in MAIN_FORMATS or _members(en) <= 0:
            continue
        aired = str(((j.get("aired") or {}).get("from")) or "")[:10]
        if not aired or aired > cutoff:
            continue
        if _has_prequel(relations.get(mid)):
            continue
        out.append(mid)
    return sorted(out)


def fit_selection(universe: dict, list_ids: set, relations: dict, *,
                  today: _dt.date | None = None,
                  with_cv: bool = True) -> SelectionModel | None:
    """
    universe   {mal_id: Enriched} -- populární tituly (s Jikan daty)
    list_ids   vše, co mám v seznamu (Completed/Watching/On-Hold/Dropped/PTW)
    relations  {mal_id: relace v Jikan tvaru} pro vesmír i seznam
               (Enricher.relations_data) -- kvůli franšízovým kořenům

    None = málo dat (pod MIN_CLASS titulů se zájmem nebo bez něj).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    ids = universe_entries(universe, relations, today)
    roots = build_roots(list(universe) + sorted(list_ids), relations)
    engaged_roots = {roots.get(m, m) for m in list_ids}
    y = np.array([roots.get(m, m) in engaged_roots for m in ids], dtype=int)
    if y.sum() < MIN_CLASS or (len(y) - y.sum()) < MIN_CLASS:
        log.warning(f"model výběru: málo dat ({len(y)} titulů, se zájmem "
                    f"{int(y.sum())}) -- složka do kompozitu nevstoupí")
        return None

    counts: dict = {}
    labels: dict = {}
    cats: dict = {}
    for m in ids:
        for k, av in universe[m].attrs.items():
            if av.category in EXCLUDED_CATEGORIES:
                continue
            counts[k] = counts.get(k, 0.0) + av.weight
            labels.setdefault(k, av.label)
            cats.setdefault(k, av.category)
    keys = sorted(k for k, c in counts.items() if c >= MIN_FEATURE_COUNT)
    kidx = {k: i for i, k in enumerate(keys)}
    X = np.zeros((len(ids), len(keys) + 1))
    for r, m in enumerate(ids):
        for k, av in universe[m].attrs.items():
            i = kidx.get(k)
            if i is not None:
                X[r, i] = av.weight
        X[r, -1] = math.log10(max(1, _members(universe[m])))

    cv_auc = _cv_auc(X, y, [roots.get(m, m) for m in ids]) if with_cv else None
    clf = LogisticRegression(C=C_L2, max_iter=2000).fit(X, y)
    coef = {k: float(c) for k, c in zip(keys, clf.coef_[0][:-1])}
    top = sorted(((k, v) for k, v in coef.items()
                  if cats[k] in AVOID_DISPLAY_CATEGORIES), key=lambda kv: kv[1])[:10]
    return SelectionModel(
        coef=coef, labels={k: labels[k] for k in keys},
        pop_coef=float(clf.coef_[0][-1]), n=len(ids), n_engaged=int(y.sum()),
        cv_auc=cv_auc, top_avoided=[(labels[k], v) for k, v in top],
        categories={k: cats[k] for k in keys})


def _cv_auc(X, y, groups) -> float | None:
    """AUC out-of-fold, 5 foldů po franšízách -- ať report poctivě ukáže,
    jak dobře jde výběr předpovědět (ne jestli zlepší doporučení; to
    rozhodne až historie, §9c.5)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    if len(set(groups)) < 5:
        return None
    ps = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if len(set(y[tr])) < 2:
            return None
        clf = LogisticRegression(C=C_L2, max_iter=2000).fit(X[tr], y[tr])
        ps[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, ps))
