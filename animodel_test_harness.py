"""
Ad-hoc funkční test TasteModel bez sítě — převádí ručně tagovaná data
z dřívějška v týhle konverzaci (franchise_tags.py, přiložen ve stejné složce)
na Title objekty a spouští model naživo, aby se ověřilo chování na
realistických datech (ne jen čtení kódu).

Spuštění: python3 animodel_test_harness.py
(očekává animodel/ balíček a franchise_tags.py ve stejné složce nebo na PYTHONPATH)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from franchise_tags import ALL
from animodel.taste import TasteModel, Title
from animodel.attributes import AttrValue

MOOD_KEYS = ["emo", "rom", "com", "act", "deep", "harem", "fluff"]

titles = []
import random
rng = random.Random(7)
for row in ALL:
    key, genres, emo, rom, com, act, deep, harem, fluff, score, tier = row
    attrs = {}
    for g in genres:
        attrs[g] = AttrValue(category="genre", weight=1.0, label=g.replace("_", " ").title())
    moodvals = dict(zip(MOOD_KEYS, [emo, rom, com, act, deep, harem, fluff]))
    for mk, v in moodvals.items():
        if v > 0:
            attrs[f"mood_{mk}"] = AttrValue(category="tag", weight=v / 5.0, label=mk)
    # Realističtější syntetická komunita: NEZÁVISLÁ na jeho skóre (jak by to
    # bylo doopravdy) -- typická MAL komunita sedí kolem 7.2-8.3 s vlastním
    # šumem, ne odvozená přímo z jeho skóre (to byl artefakt v první verzi
    # tohodle testu, který uměle vysvětloval skoro všechnu varianci baseline).
    community = max(5.5, min(9.0, rng.gauss(7.6, 0.7)))
    titles.append(Title(
        mal_id=hash(key) % 100000, title=key,
        user_score=float(score), community=community,
        attrs=attrs, weight=1.0,
    ))

print(f"Postaveno {len(titles)} syntetických Title objektů.\n")

t0 = time.time()
model = TasteModel(shrinkage_k=8.0)
model.fit(titles, n_clusters=None)
t_fit = time.time() - t0
print(f"fit() dokončen za {t_fit:.2f}s (klastrování teď uvnitř, jen jednou)")
print(f"beta={model.beta:.3f}  scale (s)={model.scale:.3f}  "
      f"CV RMSE={model.cv_rmse:.3f}  baseline RMSE={model.baseline_rmse:.3f}")
print(f"Počet klastrů po fit(): {len(model.clusters)}")

unrated = model.unrated_intensity_attrs(top=10)
print(f"\nPozorované atributy bez záznamu v intensity lexikonu (top 10): "
      f"{[(k, round(n, 1)) for k, _lab, n in unrated] or '—'}")

print("\n--- top 10 pozitivních efektů ---")
for e in model.top_effects(10, sign=1):
    print(f"  {e.label:30s} n={e.n_eff:5.1f}  effect={e.effect:+.3f}  distinct={e.distinct:+.2f}")

print("\n--- top 5 negativních efektů ---")
for e in model.top_effects(5, sign=-1):
    print(f"  {e.label:30s} n={e.n_eff:5.1f}  effect={e.effect:+.3f}")

print(f"\n--- klastry (k={len(model.clusters)}) ---")
for c in model.clusters:
    print(f"  {c.name:50s} n={c.size:3d}  avg_score={c.mean_user_score:.2f}  intensity={c.intensity:+.2f}")
