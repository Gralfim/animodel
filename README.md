# animodel

Nástroj, který z tvého MyAnimeList exportu postaví **model tvého anime vkusu**
a navrhne dosud neshlédnuté tituly. Náhrada za jednorázovou „AI analýzu" —
opakovatelná, laditelná, bez ručního mezikroku.

Jeden vstup (MAL XML export), jeden příkaz, dvě HTML stránky na výstupu:
`model.html` (profil vkusu) a `recommendations.html` (doporučení).

---

## Rychlý start

```bash
pip install -r requirements.txt
python -m animodel --export animelist.xml
```

Výstup najdeš v `output/`. MAL export stáhneš na
<https://myanimelist.net/panel.php?go=export> (vyber *Anime List*, rozbal `.gz`).

### Další volby

```bash
python -m animodel -e animelist.xml -c config.yaml   # vlastní ladění
python -m animodel -e animelist.xml --no-recommend   # jen model
python -m animodel -e animelist.xml --no-anilist     # jen MAL/Jikan
python -m animodel -e animelist.xml --shrinkage 12   # konzervativnější efekty
python -m animodel -e animelist.xml --user-cf        # + user-based CF (pomalé)
```

První běh je pomalejší (stahuje metadata přes Jikan a AniList); vše se cachuje
do `cache/`, takže další běhy jsou rychlé.

---

## Proč ne „obyčejná regrese"

Tvůj seznam má **silně omezený rozsah známek** — skoro nic pod 7, protože už při
výběru do PTW děláš náročný předvýběr. Lineární regrese na surových známkách tu
nemá co vysvětlovat: tvá známka ≈ komunita + skoro konstantní posun. (To je důvod,
proč předchozí pokus s Ridge regresí nedával přesvědčivé výsledky.)

animodel proto **necílí na známku, ale na odchylku**:

1. **Baseline.** Pro každý titul: `tvůj_průměr + β·(komunita − průměr_komunity)`.
   Komunita vstupuje jako *jeden* kalibrovaný sklon β, **ne** jako atribut — jinak
   by se „kvalita" počítala dvakrát a zkreslila by efekty (např. studia, která
   hodnotí vysoko i komunita).
2. **Cíl = afinita.** Co po odečtení baseline zbude. To je tvůj osobní podpis nad
   rámec toho, co by čekal kdokoli.
3. **Efekty atributů.** Pro každý atribut empiricko-bayesovsky **smrštěný** vážený
   průměr afinity: `efekt = (n/(n+K))·průměr`. Malé vzorky se táhnou k nule, takže
   tě neošálí žánr viděný 2×. Vedle efektu se počítá i `Δ komunita` = intuitivní
   „o kolik výš než dav to hodnotíš".
4. **Interakce.** Dvojice atributů, kde afinita převyšuje součet jednotlivých
   efektů — tvé „sladké tečky".
5. **Nálady (módy).** KMeans na normalizovaných atributových vektorech; počet
   klastrů se volí podle siluety. Každý klastr dostane **osu náročnosti**
   (těžké drama/psycho vs. lehká komedie/slice-of-life) — to je ta „emocionální
   únava", kvůli které mezi módy přepínáš.
6. **Kalibrace.** Globální škála efektů a interval predikce z 5-násobné
   cross-validace.

### Atributy se neudržují ručně

Žádný `config.yaml` plný seznamů žánrů. Atributy (žánry, témata, AniList tagy,
studia, zdroj, dekáda, formát, demografie) se **objevují samy z dat**.
`attributes.py` je kanonizuje a **dedupuje napříč zdroji** (MAL „Drama" a AniList
tag „Drama" = jeden atribut), aby se stejný koncept nezapočítal víckrát.
Franšízy (sequel/prequel) se přes union-find slučují a váží `1/√k`, aby oblíbená
desetidílná série nepřeválcovala model.

---

## Doporučení

Dvě nezávislé větve, sjednocené a deduplikované:

- **Atributová / obsahová** — z tvých oblíbených seedů se tahá MAL + AniList
  „recommendations" graf (item-based CF) a navíc discovery přes AniList tag-search
  na tvé nejcharakterističtější tagy.
- **Collaborative / uživatelská** (volitelná, `--user-cf`) — uživatelé s podobným
  vkusem a jejich vysoko hodnocené tituly.

Každý kandidát se skóruje kompozitem (sčítají se z-skóry):

```
composite = w_taste_fit · afinita+shoda_s_náladou
          + w_cf        · „doporučili to tvé oblíbené"
          + w_quality   · komunitní skóre
```

Řadí se podle kompozitu, **ne** podle predikované známky (ta se kvůli restrikci
rozsahu lepí na komunitní průměr a nerozlišuje). Predikovaná známka + interval se
počítá zvlášť jen pro zobrazení.

Vyhledává se **nezávisle na PTV**; tituly z tvého plan-to-watch se jen **označí**.
Už shlédnuté (Completed/Watching/On-Hold/Dropped) se vyřazují.

Pro každý titul: originální i anglický název, synopse, odůvodnění (které atributy
a které tvé oblíbené ho táhnou), MAL skóre, odhad tvého hodnocení jako interval,
a do jaké tvé nálady patří.

---

## Ladění (`config.yaml`)

Zkopíruj `config.example.yaml`. Nejčastější páčky:

| parametr | co dělá |
|---|---|
| `model.shrinkage_k` | vyšší = konzervativnější (malé vzorky víc tlumeny) |
| `model.n_clusters` | `null` = auto; nebo napevno počet nálad |
| `recommend.w_taste_fit / w_cf / w_quality` | váhy řazení doporučení |
| `recommend.min_community` | spodní hranice MAL skóre kandidátů |
| `recommend.high_score` | od jaké známky je titul „seed" |
| `enrich.use_anilist` | vypni pro rychlejší běh jen na MAL |

---

## Architektura

```
animodel/
  mal.py            parser MAL XML exportu
  sources/
    jikan.py        MAL data + recommendations + search (přes Jikan)
    anilist.py      AniList tagy + recommendations + tag-search
  attributes.py     kanonizace + deduplikace atributů napříč zdroji
  series.py         union-find slučování franšíz
  enrich.py         MAL ID → obohacené Title objekty (s cache)
  taste.py          jádro: baseline, afinitní efekty, interakce, nálady, predikce
  recommend.py      generování kandidátů + kompozitní skórování
  report.py         HTML prezentace (model + doporučení)
  config.py         laditelné parametry (žádné seznamy atributů)
  cli.py            orchestrace: python -m animodel
```

Programové použití:

```python
from animodel import Config, TasteModel, Enricher, Recommender
from animodel.mal import parse_export, split_by_status

cfg = Config()
entries, userinfo = parse_export("animelist.xml")
completed = [e for e in split_by_status(entries)["Completed"] if e.score]
titles = Enricher(cfg).build_titles(completed)
model = TasteModel(shrinkage_k=cfg.model.shrinkage_k).fit(titles)
model._fit_clusters(None)
for e in model.top_effects(15, sign=1):
    print(e.label, round(e.effect, 2))
```

---

## Pozn. k datům

Jikan i AniList jsou veřejná API s rate-limity; respektuj je (klient cachuje).
Komunitní skóre se bere primárně z MAL, fallback AniList — záměrně se **neprůměrují**
(jsou silně korelované, průměrování nepřináší informaci a riskuje zkreslení).
