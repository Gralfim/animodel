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
python -m animodel -e animelist.xml -o vystup         # jiná výstupní složka (default: output)
python -m animodel -e animelist.xml --cache muj_cache # jiná cache složka (default: cache)
python -m animodel -e animelist.xml --no-recommend   # jen model
python -m animodel -e animelist.xml --no-anilist     # jen MAL/Jikan
python -m animodel -e animelist.xml --no-jikan       # nouzový AniList-only režim (výpadek Jikanu)
python -m animodel -e animelist.xml --shrinkage 12   # konzervativnější efekty
python -m animodel -e animelist.xml --user-cf        # + user-based CF (pomalé)
python -m animodel -e animelist.xml --analyze        # jen přehled franšízových skupin, bez modelu
python -m animodel -e animelist.xml --gen-intensity  # (re)generace intensity.yaml (osa náročnosti)
python -m animodel -e animelist.xml --verbose        # + rutinní retry/rate-limit hlášky (INFO)
```

Kompletní přehled: `python -m animodel --help`.

První běh je pomalejší (stahuje metadata přes Jikan a AniList); vše se cachuje
do `cache/`, takže další běhy jsou rychlé. Default log level je WARNING (jen
skutečné problémy); `--verbose` přidá INFO úroveň s běžnými retry/rate-limit
zprávami, které jinak nejsou vidět.

### Nouzový režim bez Jikanu

Jikan (neoficiální MAL API) mívá výpadky, kdy většina requestů vrací 504 —
každý necachovaný titul pak stojí ~17 s marného čekání. `--no-jikan` (nebo
`enrich.use_jikan: false` v configu) přepne na čistě AniList data: žánry,
synopse, dekáda i franšízové vazby se vezmou z AniListu, komunitní skóre
z `averageScore`. Degradace je malá a ohraničená: chybí MAL recommendations
graf (CF signál stojí jen na AniList-rec + tag-search) a volitelný staff
signál. Stejné fallbacky fungují i per-titul v běžném režimu — když Jikan
selže jen pro některé tituly, doplní se z AniListu automaticky.

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
4. **Interakce.** Dvojice atributů, kde se afinita liší od součtu jednotlivých
   efektů — oběma směry: „sladké tečky" i kombinace, které si nesedí. Lift je
   smrštěný stejně jako efekty a vážený přítomností atributů; obojí vstupuje
   do predikce i řazení doporučení. Volitelně (`model.interaction_triples`)
   i **trojice nad jádry nálad**: kandidáti z klastrových signatur,
   hierarchický lift = zbytek nad singly a páry (žádné dvojité počítání).
5. **Nálady (módy).** KMeans na normalizovaných atributových vektorech; počet
   klastrů se volí podle siluety. Každý klastr dostane **osu náročnosti**
   (těžké drama/psycho vs. lehká komedie/slice-of-life) — to je ta „emocionální
   únava", kvůli které mezi módy přepínáš. Osu řídí **intensity lexikon**
   (`intensity.yaml`, viz níž): spojité hodnoty −1…+1 na atribut, generované
   z úplného universa tagů a revidovatelné v jednom souboru. Každá nálada má
   navíc **afinitu** — vážený průměr reziduí svých členů (o kolik ji hodnotíš
   nad baseline) — kterou doporučení používají místo surové známky.
6. **Kalibrace.** Globální škála efektů a interval predikce z 5-násobné
   cross-validace.

### Atributy se neudržují ručně

Žádný `config.yaml` plný seznamů žánrů. Atributy (žánry, témata, AniList tagy,
studia, zdroj, dekáda, formát, demografie) se **objevují samy z dat**.
`attributes.py` je kanonizuje a **dedupuje napříč zdroji** (MAL „Drama" a AniList
tag „Drama" = jeden atribut), aby se stejný koncept nezapočítal víckrát.
Franšízy (sequel/prequel/side story) se přes union-find slučují a členové
dostávají tlumené váhy: hlavní řady `1/√k_eff`, vedlejší obsah (OVA, speciály,
side story — poznané podle formátu nebo parent-story vazby) ještě míň
(`side_story_weight`, default poloviční příspěvek). Oblíbená desetidílná série
tak nepřeválcuje model, ale její opakované potvrzení vkusu se neztratí. Váhy se
propisují i do klastrování nálad; v doporučeních navíc platí limit
`seeds_per_franchise` (default 2), ať jedna franšíza nehlasuje pěti skoro
identickými rec grafy.

### Osa náročnosti: generovaný lexikon místo ručních seznamů

Jediné místo, kde je potřeba lidský úsudek, je „jak emočně těžký daný
žánr/tag je" — to z dat odvodit nejde. Řeší to `intensity.yaml`:

```bash
python -m animodel -e animelist.xml --gen-intensity
```

stáhne **úplné universum** atributů (AniList `MediaTagCollection` — všechny
tagy včetně popisu a kategorie; Jikan `/genres/anime` — všechny MAL
žánry/témata) a vygeneruje YAML s hodnotou −1.0 (nejlehčí) … +1.0 (nejtěžší)
pro každý klíč. Prefill hodnot: kurátorovaný seznam v `intensity.py` →
prior podle AniList kategorie (Theme-Comedy → lehké, Theme-Drama → těžké) →
0.0 (neutrální, do výpočtu nevstupuje). Řádky jsou seřazené podle četnosti
ve **tvém** seznamu, takže revizi začneš u atributů s největším dopadem.

Při regeneraci se tvé úpravy **vždy zachovají** — doplní se jen nové klíče
(např. tagy, které AniList přidal později; model je po fitu sám vypíše jako
„bez záznamu v lexikonu"). Bez souboru běží vestavěný default.

**Spoiler tagy** (AniList `isGeneralSpoiler`/`isMediaSpoiler` — Tragedy,
Tearjerker, …) vstupují do modelu normálně; jsou to nejsilnější signály osy
náročnosti. V HTML reportech nesou příznak a přepínač vpravo nahoře
(„spoiler tagy") je umí jedním klikem skrýt. Adult tagy zůstávají vyloučené
úplně.

---

## Doporučení

Dvě nezávislé větve, sjednocené a deduplikované:

- **Atributová / obsahová** — z tvých oblíbených seedů se tahá MAL + AniList
  „recommendations" graf (item-based CF), volitelně i Shikimori `/similar`
  (`enrich.use_shikimori` v configu, default vypnuto — naživo neověřený tvar
  odpovědi), a navíc discovery přes AniList tag-search na tvé
  nejcharakterističtější tagy.
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
| `model.intensity_lexicon` | cesta k intensity.yaml (osa náročnosti, viz `--gen-intensity`) |
| `model.side_story_weight` | vliv OVA/speciálů/side stories uvnitř franšízy (1.0 = bez rozlišení) |
| `model.interaction_triples` | experiment: synergie trojic nad jádry nálad |
| `recommend.seeds_per_franchise` | max. seedů z jedné franšízy (0 = bez limitu) |
| `recommend.w_taste_fit / w_cf / w_quality` | váhy řazení doporučení |
| `recommend.min_community` | spodní hranice MAL skóre kandidátů |
| `recommend.high_score` | od jaké známky je titul „seed" |
| `enrich.use_anilist` | vypni pro rychlejší běh jen na MAL |
| `enrich.use_jikan` | vypni pro nouzový AniList-only režim (viz `--no-jikan`) |
| `enrich.include_staff` | signál po režisérech/scenáristech (+1 Jikan volání/titul, default vypnuto) |
| `enrich.use_shikimori` | další zdroj „podobných anime" kandidátů (naživo neověřeno, default vypnuto) |
| `recommend.use_user_cf*` | user-based CF přes AniList a jeho ladění (viz `--user-cf`, dražší/pomalejší) |

Plný seznam parametrů (včetně výchozích hodnot) je v `config.example.yaml`.

---

## Architektura

```
animodel/
  mal.py            parser MAL XML exportu
  sources/
    __init__.py     sdílené utility (progress výpisy, Result typ pro úspěch/selhání)
    cache.py        sdílený cache primitiv (FileCache, cached_fetch) -- 1 klíč = 1 soubor
    http.py         sdílený retry/backoff driver (request_with_retry, rate limitery)
    jikan.py        MAL data + recommendations + search (přes Jikan)
    anilist.py      AniList tagy + recommendations + tag-search + user-based CF
    shikimori.py    volitelný zdroj "podobných anime" (/similar), default vypnuto
  attributes.py     kanonizace + deduplikace atributů napříč zdroji
  intensity.py      osa emocionální náročnosti: lexikon, prefill, --gen-intensity
  series.py         union-find slučování franšíz
  enrich.py         MAL ID → obohacené Title objekty (s cache)
  taste.py          jádro: baseline, afinitní efekty, interakce, nálady, predikce
  recommend.py      generování kandidátů + kompozitní skórování
  report.py         HTML prezentace (model + doporučení)
  config.py         laditelné parametry (žádné seznamy atributů)
  cli.py            orchestrace: python -m animodel
tests/              pytest sada nad sources/ (cache, retry/backoff, klienti) -- viz níž
```

Programové použití:

```python
from animodel import Config, TasteModel, Enricher, Recommender
from animodel.mal import parse_export, split_by_status

cfg = Config()
entries, userinfo = parse_export("animelist.xml")
completed = [e for e in split_by_status(entries)["Completed"] if e.score]
titles = Enricher(cfg).build_titles(completed)
model = TasteModel(
    shrinkage_k=cfg.model.shrinkage_k,
    min_attr_count=cfg.model.min_attr_count,
    interaction_min_count=cfg.model.interaction_min_count,
    interaction_min_lift=cfg.model.interaction_min_lift,
).fit(titles, n_clusters=cfg.model.n_clusters)
for e in model.top_effects(15, sign=1):
    print(e.label, round(e.effect, 2))
```

---

## Testování

Síťová/cache vrstva (`animodel/sources/`) má pytest sadu, která běží čistě
offline (žádné skutečné HTTP volání, žádné čekání na retry/backoff):

```bash
pip install -e ".[dev]"   # nebo jen: pip install pytest
pytest -q
```

Testy pokrývají cache sémantiku (úspěch/trvalé/dočasné selhání — kdy se smí a
nesmí zapsat cache záznam), sdílenou retry/backoff smyčku a per-klientské
chování (Jikan, AniList včetně stránkovaného user-based CF, Shikimori) nad
mockovaným `requests.Session`.

`animodel_test_harness.py` (v rootu, mimo `tests/`) je samostatný ad-hoc
skript, který ověřuje `taste.py` na ručně tagovaných datech
(`franchise_tags.py`) bez sítě — spouští se přímo (`python
animodel_test_harness.py`), ne přes pytest.

---

## Pozn. k datům

Jikan i AniList jsou veřejná API s rate-limity; respektuj je (klient cachuje).
Komunitní skóre se bere primárně z MAL, fallback AniList — záměrně se **neprůměrují**
(jsou silně korelované, průměrování nepřináší informaci a riskuje zkreslení).
