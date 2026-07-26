# Hodnocení projektu animodel

*Kolo 2 — stav ke 2026-07-25. Nezávislé review celého kódu (8 739 řádků Pythonu,
21 modulů + 17 testových souborů) po dokončení senpai redesignu, migrace na Tenrai
a sezónních doporučení. Předchozí verze tohohle dokumentu (kolo 1, 2026-07-14/16)
je uzavřená — skoro všechny její body jsou vyřešené; co z ní zůstává živé, je
níž výslovně převzaté. Nic v kódu jsem neměnil.*

*Nálezy označené **[ověřeno]** jsem změřil/spustil, ne odvodil ze čtení kódu.*

---

## 0. Shrnutí

Projekt je ve výrazně lepším stavu než v kole 1. Tři největší tehdejší rizika
padla: testy existují a jsou skutečné (**165 testů, 12,7 s, všechny zelené**
[ověřeno]), CF monolit se rozpadl na `usercf.py` (372 ř. čisté logiky nad
čtyřmetodovým klientským kontraktem) + síťová primitiva v klientovi, a
`anilist.py` spadl z 1 185 na 768 řádků. Metodika je konzistentní, zdůvodněná a
— což je u projektů tohohle typu vzácné — dokumentuje i **zamítnuté alternativy
a proč**.

> **Stav implementace (2026-07-26):** hotovo všech 8 krátkých oprav z §9,
> **§9.1** variantou (c) (trojice mají vlastní kalibrovanou škálu, fold-modely
> klastrují samy → do CV neprosakuje nic) a **§9.2** (vážený kosinus proti
> plnému těžišti nálady). Detaily a naměřené dopady u jednotlivých nálezů.
> Testů 165 → **187**, všechny zelené; defaultní cesta kalibrace ověřena jako
> bitově identická s HEAD. Otevřené zůstává §9.3 a rozvojové body §9.4–§9.8.

Zbývající nálezy nejsou architektonické, ale konkrétní a lokální. Čtyři, které
bych řešil první:

1. ~~**`interaction_triples: true` (tvůj aktuální config) rozbíjí kalibraci
   `scale`.**~~ **OPRAVENO** variantou (c): vlastní `scale_triples`, kalibrace
   až nad hotovým modelem, fold-modely klastrují samy. Na tvých datech CV dává
   trojicím poloviční váhu (`s = 0.30`, `s₃ = 0.15`) — a zároveň ukazuje, že
   při současném prahu nepřinášejí prakticky nic (§5.1, §9.1).
2. ~~**`Senpai.entries` drží plné seznamy pro celý vyhodnocený pool.**~~
   **OPRAVENO** — ~450 MB → ~5 MB (§5.2).
3. ~~**Retry smyčka spí i po posledním pokusu.**~~ **OPRAVENO** — 240 s → 150 s
   na request s vyčerpaným rate limitem (§5.3).
4. ~~**`_cluster_fit` systematicky penalizuje bohatě otagované tituly.**~~
   **OPRAVENO**: vážený kosinus proti plnému těžiště v prostoru nálady.
   Hlavní zisk nebyl v pořadí (top-10 beze změny), ale v **přiřazení nálady:
   85,4 % → 99,6 %** shody s tím, co KMeans rozhodl (§5.4, §9.2).

Dokumentace je nadprůměrná, ale má jedno konkrétní **zastaralé tvrzení**:
Shikimori „naživo neověřeno" už neplatí — tvar odpovědi je v cache ověřený a
`use_shikimori: true` běží (§8.2).

---

## 1. Návrh a analýza

### Co je na návrhu dobré

Východisko je správně identifikovaný **statistický problém dat, ne modelu**.
`ANALYZA.md` to má doložené číselně (411 ohodnocených, průměr 8,12, rozdělení
33/119/150/83/25/1/0 od desítky dolů, dvě dropnutá anime celkem) a z toho plyne
celý zbytek: při takhle omezeném rozsahu známek je regrese na skóre
bezpředmětná, protože `moje známka ≈ komunita + konstanta`.

Volba **rezidua vůči komunitě** jako cílové proměnné je z toho korektně
odvozený závěr, ne libovolné rozhodnutí. A hlavně je držená důsledně napříč
celým systémem: řazení doporučení jde podle kompozitu, ne podle predikované
známky (`recommend.py:368`); klastrová afinita se počítá z reziduí, ne ze
surových známek (`taste.py:608`); CF podobnost běží na komunitně-relativních
odchylkách, ne na surovém skóre (`usercf.py:180-182`). Když se v kole 1 našlo
místo, kde se surová známka do skórování vracela zadními vrátky
(`mean_user_score − u_mean` v `_cluster_fit`), opravilo se to i s vysvětlením
proč — komentář na `recommend.py:300-306` je učebnicový příklad, jak se má
dokumentovat oprava.

Druhý nosný princip — **žádné ruční seznamy atributů v configu** — je taky
skutečně dodržený, ne jen deklarovaný. `config.py` obsahuje výhradně laditelná
čísla. Jediná výjimka (osa emocionální náročnosti) se nedá z dat odvodit a je
řešená čistě: `intensity.yaml` s **exaktním** universem klíčů stahovaným z
AniList `MediaTagCollection` + `GenreCollection` + Jikan `/genres/anime`, lidský
úsudek jen v hodnotách, tvoje úpravy se při regeneraci nikdy nepřepíšou, a
`unrated_intensity_attrs()` sám hlásí, co v lexikonu chybí. To je lepší řešení
než původní HEAVY/LIGHT množiny o dvě úrovně, ne o jednu.

### Kde je návrh slabší

**Kompozitní skóre má čtyři konfigurovatelné váhy a jednu skrytou.**
`taste_fit = raw_resid + 0.5 * cfit` (`recommend.py:338`) — ta 0,5 rozhoduje,
jak moc má shoda s náladou vážit proti atributové afinitě, a je to jediné číslo
v celém řetězci skórování, které se nedá ladit z configu. Sedí v jednom řádku
mezi ostatními a nikde není zdůvodněné.

**Chybí zpětná vazba na kvalitu doporučení.** Model se validuje na predikci
známky (CV RMSE) — a dokument sám správně říká, že to je *špatná* metrika,
protože atributy nepomáhají hádat číslo, ale řadit. Jenže na to *řazení* pak
neexistuje žádná měřená zpětná vazba: nikde se nesleduje, jestli doporučení z
minulého běhu skončila v Completed a s jakou známkou. Přitom vstup pro to
existuje zdarma — každý nový MAL export je ground truth pro předchozí
doporučení. To je největší nevyužitá příležitost celého projektu (návrh v §9.1).

---

## 2. Architektura

```
mal.py ──┐
         ├─► enrich.py ──► taste.py (baseline, efekty, interakce, nálady)
sources/ ┘       │                     │
  cache.py       │                     ├─► recommend.py ──┬─► report.py (4× HTML)
  http.py        │                     │      │           │
  jikan.py       │                     └──────┤           │
  anilist.py ────┴──► usercf.py ───────────────┘           │
  shikimori.py                          season.py ─────────┘
```

### Silné stránky (a proč to nejsou jen dojmy)

**Vrstvy skutečně drží.** `sources/` neví nic o modelu, `attributes.py` neví nic
o síti, `taste.py` neví nic o HTML. Nejlepší důkaz je testovatelnost: `usercf.py`
(nejsložitější doménová logika v projektu) se testuje **bez sítě** proti fake
klientovi se čtyřmi metodami — `tests/test_usercf.py` má 341 řádků a nepotřebuje
ani mock `requests`. To se nedá zfalšovat dobrým komentářem; buď to jde, nebo ne.

**Tři sdílené primitivy odstranily tři paralelní implementace.** `cache.py`
(85 ř.) — jeden klíč = jeden soubor, `cached_fetch` s jedinou tříbodovou logikou
hit/success/permanent/transient. `http.py` (209 ř.) — jeden retry driver pro REST
i GraphQL, s injektovatelným `sleep` (proto testy běží 12 s a ne 20 minut).
`Result` typ — klasifikace selhání je součást návratové hodnoty, nedá se přečíst
pozdě ani zapomenout. Tohle je ta část, kde v kole 1 vznikaly bugy opakovaně, a
teď jich je strukturálně méně možných.

**Senpai redesign je největší architektonický zisk.** Bývalá 340řádková metoda
se šesti odpovědnostmi je nahrazená čtyřmi fázemi, z nichž **každá je samostatná
funkce s vlastními testy**: `discover_candidates` / `evaluate_candidate` /
`select_senpai` / `recommend_from_senpai`. Že `evaluate_candidate` bere
`userlist` jako obyčejný dict (ne klienta) je přesně ten druh rozhodnutí, který
dělá rozdíl mezi testovatelným a netestovatelným kódem.

### Zbývající architektonické nedostatky

| Co | Kde | Proč to vadí |
|---|---|---|
| CF výsledky se předávají **privátními atributy** | `recommend.py:251,270-271` → `cli.py:216,251` | `getattr(rec, "_cf_raw_results", [])` je implicitní kontrakt mezi dvěma moduly. Přejmenování `_cf_senpai` nic nerozbije — jen tiše přestane být CF report. |
| `model._raw_resid_pred()` volaný **z jiných modulů** | `recommend.py:336`, `season.py:98` | Privátní metoda je fakticky veřejné API. Buď `affinity()`, nebo součást `predict()`. |
| `rec._prequel_score` **dynamicky přišpendlený** na dataclass | `season.py:182,193` | `Recommendation` už má 4 sezónní volitelná pole — patří tam i tohle. Dnes funguje; s `slots=True` spadne. |
| `report.py` = 775 ř. s **CSS v f-stringu** | `report.py:63-150` | Zdvojené `{{}}` v celém stylopisu. Nejnepřátelštější kód projektu k editaci; ostatní moduly se čtou příjemně. |
| `cli.py::run()` = **215 ř. lineárně**, 4 režimy s early-return | `cli.py:42-257` | Číslování kroků se už rozjelo (`[4/4]` v sezónním režimu vs `[1/5]`…`[5/5]` jinde) — příznak, že to chce rozpad na `run_model()` / `run_season()` / `run_analyze()`. |
| **Jikan cache není v podsložce** | `jikan.py:44` | AniList má `cache/anilist`, CF `cache/cf_al`, Shikimori `cache/shikimori` — MAL data leží přímo v `cache/` (~10 tis. souborů vedle těch podsložek). Selektivní invalidace znamená glob, ne `rm -r`. |

---

## 3. Použité algoritmy

### 3.1 Baseline
`baseline = ū + β·(komunita − c̄)`, β z kovariance/rozptylu, ořez `[-0.5, 1.5]`.
Vážené průměry respektují franšízové váhy. Korektní; β je odhadnuté z ~400 bodů
se záměrně omezenou variancí, takže ořez je funkční pojistka, ne ozdoba.

### 3.2 Efekty atributů
`effect = n_eff/(n_eff+K) · vážený_průměr(reziduí)`. Standardní empirical-Bayes,
váhy = AniList tag rank × franšízová váha titulu. `min_attr_count` navíc atributy
s málo důkazy vynechá úplně (ne jen smrští k nule) — rozumné rozlišení.

### 3.3 Interakce a trojice
Páry: lift = `mean_pair − (effect_a + effect_b)`, **smrštěný stejným K** jako
efekty. Trojice: hierarchický lift nad singly *a* prahem prošlými páry
(`taste.py:348-353`) — odečítá se přesně to, co by predikce z nižších řádů dala,
takže žádné dvojité počítání. Kandidáti se negenerují slepou enumerací, ale z
podmnožin klastrových signatur; při n≈450 je to správné rozhodnutí.

**Konzistence smrštění je nejlepší vlastnost celé metodiky:** totéž `n/(n+K)` se
uplatní na singl efekty, párové lifty, trojicové lifty *i* na CF podobnost
(`usercf.py:186`). Čtyři nezávislá místa, jeden princip, žádné ad-hoc váhy.

### 3.4 Kalibrace `scale`
5-fold CV, grid 21 hodnot, fold-modely se fitují **jednou** (oprava z kola 1
drží: `_cv_predictions` + `_eval_scale`). Zbývá ale problém §5.1 (trojice).

### 3.5 Klastrování nálad
KMeans na L2-normalizovaných vektorech, jen genre/theme/tag/demographic (staff a
studia záměrně mimo — „nálada" nemá být určená tvůrcem), `sample_weight` =
franšízové váhy, `k` podle siluety 4–7. Že `silhouette_score` váhy nepodporuje a
výběr `k` je proto nevážený, je v kódu **přiznané**, ne zamlčené
(`taste.py:559-561`). Přesně takhle to má vypadat.

### 3.6 Skóre doporučení
Čtyři oddělené z-skóry (taste_fit / item-CF graf / user-CF / kvalita), item-CF
tlumený `log1p` proti šikmému rozdělení. Oddělení item-CF a user-CF do vlastních
kbelíků je změřené rozhodnutí (kolo 1: |příspěvek| 6,4 vs 1,6), ne odhad.

### 3.7 Senpai pipeline
Discovery podle **nejnižší popularity** (sdílení nišového titulu je silnější
signál shody než sdílení hitu) → plné seznamy → Pearson na plném překryvu se
smrštěním → penalizace za nepokryté oblíbené. Zásadní zlepšení proti kole 1:
podobnost se měří na **plných seznamech**, vzorek slouží jen k prioritizaci.
Dropnuté-se-známkou jako plnohodnotný datový bod je dobrý postřeh („zkusil a dal
3" nese víc informace než většina desítek).

Tvůj `user_cf_min_full_overlap: 200` proti ~411 ohodnoceným znamená, že senpai
musí mít ohodnoceno ~půl tvého seznamu. To je s cílem „pár ověřených senpai, ne
statistika anonymů" **konzistentní** — jen to stojí, co to stojí (§7.1).

---

## 4. Vlastní implementace

Co bych na implementaci nechal být, protože je to lepší než běžný standard:

- **Docstringy dokumentují zamítnuté alternativy.** `cache.py:8-15` popisuje
  konkrétní bug (`json.loads("null")` == cache miss), který ta struktura
  odstraňuje. `series.py:14-18` vysvětluje, proč `aggregate_entries` **zmizelo**.
  `recommend.py:151-162` popisuje, proč byl první circuit breaker mrtvá větev.
  Tohle není komentování kódu, tohle je zápis rozhodnutí — a jediná věc, která
  po roce zabrání to samé rozhodnutí udělat znovu špatně.
- **Cache verzování klíčů** (`mal_{id}_v2`, `userlist_{uid}_v2`) s vysvětlením
  proč: chybějící pole ve starém schématu nejde odlišit od „titul ho nemá".
  Správné a málokdo to udělá.
- **`ProgressAwareLogHandler`** — log zprávy rolují nad `\r` progress řádkou,
  místo aby ji rozsekaly. Detail, ale je vidět, že tenhle nástroj někdo
  **používá**, ne jen píše.
- **Injektovatelný `sleep`** napříč všemi klienty. Jediný důvod, proč 165 testů
  nad retry/backoff logikou běží 12 sekund.

Slabší místa implementace jsou v §5 a §6.

---

## 5. Potenciální chyby a křehký kód

### 5.1 `interaction_triples: true` rozbíjí kalibraci `scale` — VYSOKÁ

> **Stav (2026-07-26): OPRAVENO variantou (c)** — vlastní kalibrovaná škála
> pro trojice, bez jakéhokoli úniku informace.
>
> - `_calibrate_scale()` běží **až po** `_fit_clusters` + `_fit_triples`, tedy
>   nad modelem v té podobě, v jaké bude predikovat.
> - Grid je společný a 2D: `(s, s₃) ∈ {0, 0.05, …, 1}²`. Vyhodnocení bodu je
>   čistá aritmetika nad předpočítanými CV řádky, takže 441 kombinací stojí
>   zlomek sekundy.
> - Fold-modely **klastrují a fitují trojice samy** na svých 4/5 dat. Převzít
>   kandidáty z plného modelu (varianta b) by do každého foldu protáhlo znalost
>   jeho testovací pětiny — tomu se přesně chtělo vyhnout. Cena: **~0,31 s na
>   fold, ~1,6 s na běh** (sklearn už je naimportovaný z hlavního modelu; ta
>   dřív měřená „14 s" byla z 4/5 právě jen ten líný import).
> - `predict()` i příspěvky ve vysvětlení používají pro trojice `scale_triples`,
>   ne `scale` — číslo v reportu teď odpovídá tomu, co vstoupilo do predikce.
> - Nové veřejné `TasteModel.affinity()` = kalibrovaná afinita; `recommend.py`
>   a `season.py` ji používají místo neškálovaného `_raw_resid_pred`. Se dvěma
>   faktory na tom záleží: poměr `s₃/s` je právě to, co CV o trojicích zjistila
>   (z-skóre by vykrátilo jen jednu společnou konstantu, ne dvě různé).
> - `cv_rmse_no_triples` = nejlepší RMSE při `s₃ = 0`; rozdíl proti `cv_rmse`
>   je poctivá odpověď na „přinesly trojice vůbec něco?". Hlásí CLI i report.
>
> **Ověření na tvých datech** (458 titulů, `min_lift 0.25`, 4 trojice):
> `s = 0.30`, `s₃ = 0.15` — CV dává trojicím **polovinu** váhy singlů a párů;
> stará varianta jim dávala plnou. Křivka RMSE podle `s₃` má minimum v 0,15 a
> pak roste (0,89822 → 0,89872 při 0,30 → 0,91301 při 1,00), takže staré
> implicitní `s₃ = 0.30` už bylo za optimem. Zkreslení *reportovaného* RMSE
> bylo přitom malé (+0,0001) — jen proto, že prahem projdou 4 trojice;
> s jejich větším počtem by rostlo.
>
> **Regrese ověřena bitově:** v defaultní cestě (`interaction_triples: false`)
> dává nový kód proti HEAD identická čísla do 10 desetinných míst
> (cv_rmse/cv_mae/baseline/scale/beta/afinity klastrů) — změna je tam no-op.
> Testů 172 → **180**.
>
> **Vedlejší nález:** `scale_triples` umělo vyjít nenulové i když plný model
> žádnou trojici neudrží (fold-modely na jiných datech je mají). Pro predikci
> to nic neznamená, ale atribut lhal → vynuluje se.

**[ověřeno]** Pořadí v `TasteModel.fit()`:

```
_fit_baseline → _fit_effects → _fit_interactions → _calibrate_scale
              → _fit_clusters → self.triples = [] → _fit_triples
```

`_calibrate_scale()` běží, když `self.triples == []`, a `_cv_predictions()` staví
fold-modely, které trojice vůbec nefitují. `predict()` pak počítá
`base + scale · (singly + páry + trojice)` (`taste.py:377-381`).

Důsledky:
- **`scale` je systematicky nadhodnocená** pro model, který se skutečně používá.
  Zkalibrovala se na menší reziduální predikci, než jaká se pak sčítá.
- **`cv_rmse` / `baseline_rmse` v `model.html` popisují jiný model**, než který
  generoval doporučení. Interval predikce (`resid_std = cv_rmse`) taky.
- Platí to **jen** pro `interaction_triples: true` — což je právě tvůj
  `config.yaml` (`example` má `false`).

V kódu je poznámka, že fold-modely klastrování nedělají, protože je „drahé a
nestabilní na 4/5 dat" (`taste.py:188-191`) — to je legitimní důvod. Není ale
dotažený: buď se to má přiznat i ve výstupu, nebo se z toho má vyvodit důsledek.
Návrhy v §9.2.

### 5.2 `Senpai.entries` drží plné seznamy celého poolu — VYSOKÁ

> **Stav (2026-07-25): OPRAVENO.** `evaluate_candidates(keep_entries=False)`
> (default) po vyhodnocení `entries` zahodí — metriky pro `select_senpai()`
> jsou v tu chvíli spočítané. Vybraným pár senpai je doplní nová
> `hydrate_entries()` z cache (disk hit, zdarma). Paměť ~450 MB → ~5 MB.
> Testy `test_evaluate_candidates_drops_entries_but_keeps_metrics`,
> `test_hydrate_entries_refills_selected_from_cache` (+ opt-in a chybějící
> seznam). Cestou vylepšen `test_find_senpai_respects_exclude_users_from_config`:
> místo křehkého `list_calls == 1` teď kontroluje přímo to, co má
> (`100 not in client.fetched_uids`).

`evaluate_candidates()` vrací `Senpai` objekt pro **každého** vyhodnoceného
kandidáta a každý si nese `entries` (celý seznam uživatele, `usercf.py:75`).
`select_senpai()` z nich vybere 20, ale `evaluated` zůstává naživu až do konce
`find_senpai_recommendations()`.

**[ověřeno na tvé cache]** (280 vzorků z `cache/cf_al`):

| | entries/uživatel | v paměti |
|---|---|---|
| medián | 991 | ~228 KB |
| p90 | 2 454 | ~565 KB |
| max | 8 641 | ~2 MB |

→ `user_cf_candidate_pool: 200` (default) ≈ **45 MB**
→ `user_cf_candidate_pool: 2000` (tvůj config) ≈ **450 MB**, a to jen mediánem;
   s p90 ocasem reálně víc.

Oprava je malá a lokální: `entries` u nevybraných kandidátů zahodit (metriky pro
`select_senpai` už jsou spočítané), nebo je vůbec nedržet a pro 20 vybraných je
znovu přečíst — jsou na disku v cache, čtení je zdarma.

### 5.3 Retry smyčka spí i po posledním pokusu — STŘEDNÍ

> **Stav (2026-07-25): OPRAVENO.** Na posledním pokusu se místo `sleep()`
> loguje warning a vrací failure. Rozpočet pokusů ani `on_throttled()`
> se nemění — ubylo jen marné čekání (AniList 240 s → 150 s). Bonus:
> prázdný `retry_delays` už nespadne na `IndexError` (dřív sáhl na
> `retry_delays[-1]`). Testy `test_rate_limited_does_not_sleep_after_final_attempt`,
> `test_rate_limited_with_empty_retry_delays_does_not_crash`.

**[ověřeno]** `http.py:190-196`. Test s `retry_delays=[5,15,40,90]` a trvalým 429:

```
sleeps: [5, 15, 40, 90, 90]  → 240 s celkem, z toho 90 s po posledním pokusu
```

Na poslední iteraci (`i == attempts-1`) se `sleep(floor)` provede a hned na to
smyčka skončí a vrátí `Result.failure`. Není na co čekat.

- AniList (`RETRY_DELAYS = [5,15,40,90]`): **90 s nadarmo** na request.
- MAL API (`[2,5,10,30]`): 30 s nadarmo.
- Shikimori (`[2,5,10]`): 10 s nadarmo.

Při skutečném throttlingu v CF fázi (tisíce requestů) je to zásadní rozdíl. Fix
je jednořádkový: nespat, když `i == attempts - 1`.

### 5.4 `_cluster_fit` penalizuje bohatě otagované tituly — STŘEDNÍ

> **Stav (2026-07-26): OPRAVENO (§9.2).** `Cluster` si uchová celé těžiště
> (`centroid` + předpočítaná `centroid_norm`), model prostor nálady
> (`cluster_feat_keys`), a `_cluster_fit` počítá **vážený kosinus** proti
> plnému těžišti — jen v prostoru nálady a s vahami atributů. Odpadly obě
> zkreslení: cizí kategorie (studio/formát/dekáda/zdroj) už nenafukují
> jmenovatel a okrajový tag už neváží jako hlavní žánr. Magická 0,5
> vytažena do `recommend.cluster_fit_weight`.
>
> **Naměřeno na tvých datech** (458 titulů, 5 nálad, 172 featur):
>
> | | starý | nový |
> |---|---|---|
> | korelace `cluster_fit` s počtem atributů | −0,092 | **+0,034** |
> | shoda přiřazené nálady s tím, co rozhodl KMeans | 85,4 % | **99,6 %** |
> | medián `cluster_fit` | 0,351 | 0,669 |
> | sm. odchylka `0.5·cluster_fit` (vs. odchylka afinity) | 0,063 (19 %) | 0,094 (28 %) |
>
> Zkreslení tedy **existovalo a zmizelo**, ale byl slabší, než můj analytický
> příklad níž naznačoval: v reálných datech titul s víc atributy zároveň trefí
> víc klíčů signatury, takže se penalizace částečně sama vyruší — příklad
> „2× rozdíl" držel počet shod na třech, což se nestává. Skutečný přínos je
> jinde: **přiřazení nálady bylo dřív špatně u ~14 % titulů** a teď je
> prakticky přesné. Není to náhoda — KMeans na L2-normalizovaných vektorech
> je blízko sférickému k-means, takže kosinus k těžišti *je* (skoro) jeho
> vlastní rozhodovací pravidlo.
>
> **Zamítnutá alternativa:** místo těžiště použít vektor distinktivity
> (`centroid − globální průměr`), tedy to, čím se nálada *liší*. Vypadá
> lákavě — těžiště jsou hustá (39–166 nenulových složek ze 172), takže skoro
> každý titul má s každou náladou kladný přesah a relativní mezera mezi 1. a
> 2. náladou je jen 26,8 % (proti 85,9 % u distinktivity). Ale změřeno:
> distinktivita má shodu s KMeans jen **90,0 %**, tedy horší než těžiště.
> Větší mezera ≠ lepší přiřazení; rozhoduje, jestli reprodukujeme skutečné
> členství. Ponecháno těžiště.
>
> **Dopad na doporučení** (izolované A/B na reálném poolu 404 skórovaných
> kandidátů, vše ostatní identické): top-10 **100 %** shodných, top-20 85 %,
> top-40 92 %; medián posunu v pořadí 10 míst (p90 32, max 66). Řazení se
> tedy nahoře skoro nehýbe — **viditelně se změnil hlavně štítek nálady**,
> a to u 28 % kandidátů. Vedlejšek k rozvaze: člen nálady má při stejném
> `w = 0.5` teď **1,51× větší** vliv na `taste_fit` (jiný rozsah kosinu);
> původní poměr odpovídá `cluster_fit_weight ≈ 0.33`. Default zůstal 0,5 —
> tichou změnu vážení bych do opravy nemíchal, ale je to teď ladicí páčka.

`recommend.py:297`: `sim = inter / sqrt(len(sig_keys) · len(present))`

`present` = **všechny** atributy kandidáta (včetně studia, formátu, dekády,
source — kategorií, které v klastrovém prostoru vůbec nejsou), `sig_keys` = jen
top-6 signatura klastru. Kosinus na binárních množinách to formálně je, ale
jmenovatel míchá dva různé prostory.

**[ověřeno]** Skutečné rozdělení počtu atributů (599 titulů z tvé AniList cache,
po `anilist_min_rank: 30`): medián **14**, p90 **26**, max **56** (s Jikan žánry
a studii ještě víc). Při **stejných třech shodách**:

| atributů kandidáta | cluster_fit sim |
|---|---|
| 10 | 0,387 |
| 25 | 0,245 |
| 45 | 0,183 |

Rozdíl 2× čistě z bohatosti metadat. A protože počet tagů nad rank 30 koreluje s
popularitou (víc hlasujících = víc tagů nad prahem), je to systematický posun
proti dobře zdokumentovaným titulům — přes `0.5 * cfit` v `taste_fit` a `w_taste_fit`
až do finálního řazení.

Dvě cesty: normalizovat jen proti genre/theme/tag/demographic atributům
kandidáta, nebo (lépe) si v `_fit_clusters` **uložit centroidy** a počítat pravý
vážený kosinus — centroid se stejně počítá (`taste.py:580`) a zahazuje se z něj
všechno kromě top-6 labelů.

### 5.5 Tag-search discovery je strukturálně znevýhodněná — STŘEDNÍ

`recommend.py:248`: `bump(m["mal_id"], 0.0, None, "tag-search")` → `item_votes = 0`
→ `z_item(log1p(0))` = **dno** rozdělení → `w_cf · z_item` = −0,8 × (něco
kladného) jako fixní srážka.

Kandidát nalezený obsahovým discovery (tj. přesně ta větev, která má najít, co
graf podobnosti nezná) tedy startuje s penalizací, která nemá nic společného s
jeho shodou s vkusem. Pro kandidáty, které našel *i* graf, se to nesečte špatně
— ale ryze tag-search nálezy se v žebříčku systematicky neobjeví.

„Chybějící důkaz z grafu" má být `z = 0` (neutrální), ne `z = min`. Nejčistší
řešení: `item_votes = None` pro tag-search-only kandidáty a z-skóre počítat jen
z těch, kde graf nějaký hlas dal.

### 5.6 `except Exception: print(...)` polyká chyby po hodinách práce — STŘEDNÍ

> **Stav (2026-07-25): OPRAVENO.** `log.exception(...)` (traceback jde do
> stderr přes `ProgressAwareLogHandler`), a všechny tři `print()` v `_user_cf`
> nahrazeny `status()`, takže už nerozbíjejí `\r` progress řádku.

`recommend.py:275-276`:

```python
except Exception as exc:
    print(f"  user-CF: selhalo ({exc})")
```

Po několikahodinové CF fázi jakýkoli bug (KeyError, TypeError v novém kódu)
vypíše jeden řádek **bez tracebacku** a běh pokračuje, jako by CF prostě nic
nenašlo. To je přesně scénář, kdy chceš stacktrace nejvíc. `log.exception(...)`
řeší celé — zpráva zůstane, traceback se přidá do stderr.

Vedle toho tenhle blok používá `print()` místo `status()` (`recommend.py:263,274,276`),
takže rozbíjí `\r` progress řádku, kterou zbytek projektu pečlivě udržuje
(`sources/__init__.py`, `ProgressAwareLogHandler`).

### 5.7 Spoiler příznak efektu je „vyhrává poslední" — NÍZKÁ

`taste.py:235`: `meta[key] = av` v cyklu přes tituly → `AttrEffect.spoiler` a
`.label` pro atribut pochází z **posledního** titulu, který ho nesl. Když je tag
spoiler-flagged u titulu A a ne u titulu B, výsledek závisí na pořadí iterace.

Ironie: `attributes._add()` tenhle případ řeší správně (`new_s = prev.spoiler or spoiler`,
„opatrnější varianta vyhrává"), jen se to při agregaci přes tituly zapomene.
Dopad je jen na HTML přepínač, ne na model. Fix: OR přes tituly, stejná
konvence jako o vrstvu níž.

### 5.8 CF report bez stropu → 8 MB HTML — NÍZKÁ, ale nepříjemná

> **Stav (2026-07-25): OPRAVENO.** Nový `recommend.user_cf_report_top`
> (default 200, 0 = bez stropu) → `render_cf_recommendations_html(top=...)`.
> Vstup je seřazený dle `cf_score`, takže ořez bere skutečnou špičku, a
> hlavička hlásí obě čísla („Zobrazeno 200 z 1000"). Změřeno na syntetickém
> vstupu 1 000 karet: **769 kB → 160 kB**.

**[ověřeno]** `output/cf_recommendations.html` = **8,1 MB**.
`render_cf_recommendations_html()` iteruje **všechny** `cf_recs` bez ořezu, a
`recommend_from_senpai()` vrací každý titul s ≥2 hodnotiteli — při 20 senpai a
mediánu 991 položek jsou to tisíce karet. Ostatní reporty strop mají
(`top_n`, `top_per_cluster`, `season_top_new`); tenhle na něj zapomněl.

### 5.9 Drobnější křehkosti

| Co | Kde | Poznámka |
|---|---|---|
| Cache klíč watchers **neobsahuje `per_page`** | `anilist.py:673` | `watchers_{aid}_p{n}` — při změně `per_page` z 50 znamenají cachované stránky něco jiného. Dnes latentní (fixní default). |
| `_z()` deklaruje `-> dict`, vrací `lambda` | `recommend.py:78` | Kosmetika, ale mate. |
| `side_story_weight: 0.0` → `k_eff = 0` → váha 0 pro celou skupinu | `enrich.py:213-220` | Dokumentované jako „vyřadit z modelu", ale titul s `weight=0` pak jde do `KMeans(sample_weight=...)` a do `len(self.titles)`. Krajní hodnota, neošetřená. |
| `UnionFind.find()` je **rekurzivní** bez union-by-rank | `series.py:34-39` | Path compression řetězce krátí, ale sestavení může vyrobit dlouhý. U franšíz ≤20 členů neproblém; u `season.py`, kde se sjednocuje i `referenced` mimo seznam, je řetězec teoreticky delší. |
| `season.py` znovu obohacuje celý seznam | `season.py:149` | `enrich_ids(list(my_scores))` na ~450 ID, které `cli.py` obohatil o pár řádků dřív. Z cache, takže levné — ale je to ten samý vzorec zdvojené práce, který projekt už dvakrát opravoval. |
| `taste.py:175` `raise ValueError` nezachycené v CLI | `cli.py:154` | <20 titulů = traceback místo hlášky. |

---

## 6. Nedodělky

| Co | Kde | Stav |
|---|---|---|
| ~~**Mrtvý kód v klientech: 9 nepoužitých metod**~~ | `jikan.py`, `anilist.py`, `shikimori.py` | **SMAZÁNO 2026-07-25** (−224 ř.): `get_top_anime`, `search_anime`, `list_mal_features`, `list_all_staff`, `list_all_studios`, `list_all_tags`, `extract_tags`, `extract_animation_studios`, `batch_similar`. Nejhorší byl `extract_tags(exclude_spoiler=True)` — mrtvý kód nesoucí **opačnou** spoiler politiku, než jakou projekt od 2026-07 má. Doplněn i osiřelý odkaz na `list_all_staff` v `attributes.py`. |
| ~~**Osiřelá v1 CF cache**~~ | `cache/cf_al/` | **UKLIZENO 2026-07-25**: smazáno 2 309 souborů `userlist_*.json` = **108,5 MB**; aktivních 2 308 `_v2` (114 MB) + 3 432 watchers stránek zůstalo. Postup zapsán do README. **Pozor:** `rm cache/cf_al/userlist_*[0-9].json` je PAST — `v2` končí číslicí, takže by ten vzor smazal i aktivní cache. Správně je vylučovací `find … ! -name '*_v2.json'`. |
| `include_staff` | `config.py:57` | U tebe **zapnuto** (`config.yaml`), takže už není nedodělek — jen pozor, že to je +1 request/titul. |
| `use_shikimori` | `config.py:61` | U tebe **zapnuto** a fakticky ověřené (§8.2) — dokumentace o tom neví. |
| Chybějící ladicí cesta pro atributy | — | `--analyze` řeší franšíze, `unrated_intensity_attrs()` lexikon. Neexistuje nic pro „co se nesloučilo v `attributes.canon`/`ALIAS`" — tedy pro jediné místo, kde tichá chyba systematicky nafoukne efekt (dva klíče pro jeden koncept). Návrh v §9.4. |
| Validace configu | `config.py:138-152` | **Uzavřeno v kole 1 vědomě** — neznámé klíče se hlásí warningem, rozsahy se nekontrolují. Nevracím se k tomu: failure path je bezpečná, jen pomalá. |
| Cache TTL / `--refresh-cache` | — | **Uzavřeno rozhodnutím v kole 1** (ruční mazání). Bod výš o osiřelé v1 cache to nemění, jen připomíná, že „ruční" znamená občas ručně. |
| LICENSE | — | Není žádná. |

---

## 7. Provozní náklady současné konfigurace

Není to chyba (v kole 1 potvrzeno jako vědomý experiment za hranicí původního
návrhu), ale pro plánování stojí za vyčíslení, protože čísla v `config.yaml` se
proti `config.example.yaml` liší 6–20×.

### 7.1 CF fáze

| parametr | tvůj `config.yaml` | `example` |
|---|---|---|
| `user_cf_seed_count` | 300 | 50 |
| `user_cf_users_per_seed` | 1000 | 100 |
| `user_cf_candidate_pool` | 2000 | 200 |
| `user_cf_min_full_overlap` | 200 | 40 |

- **Discovery:** 300 seedů × (1000 ÷ 50 na stránku) = **~6 000 stránkových requestů**.
- **Plné seznamy:** scan budget `max(2000, 2000×3)` = až **6 000 pokusů** pro 2 000
  použitelných.
- Při adaptivním AniList delay 0,7–4 s → **řádově hodiny** na studené cache.

Per-page i per-user cache to dělá **přerušitelné a obnovitelné**, což je správná
odpověď na tuhle cenu. Tvá cache to potvrzuje: 3 432 watchers stránek + 2 308
userlistů už na disku [ověřeno]. Zbývající riziko je §5.2 (paměť) — to je jediná
věc na téhle konfiguraci, kterou bych opravdu řešil.

### 7.2 Ostatní odchylky od example

`interaction_min_lift: 0.25` (vs 0.30), `interaction_triples: true` (vs false, viz
**§5.1**), `include_staff: true`, `use_shikimori: true`, `min_community: 6.0` (vs 6.5).
V `config.yaml` **nejsou** `seeds_per_franchise`, `user_cf_fav_score`,
`user_cf_fav_miss_penalty`, `user_cf_exclude_users` → berou se defaulty
(2 / 9.0 / 0.3 / []), což je v pořádku, ale znamená to, že penalizace za
nepokryté oblíbené běží, i když o ní v configu není řádek.

---

## 8. Stav dokumentace

### 8.1 Co je výborné

`README.md` (400+ ř.) je nadstandardní: vysvětluje **proč ne obyčejná regrese**,
ne jen jak se to spustí; má tabulku všech ladicích páček; popisuje nouzový režim
bez Jikanu i s odhadem, co se degraduje. Módní docstringy jsou ještě lepší (§4).
`ANALYZA.md` má trvalou hodnotu jako záznam premisy (rozdělení známek), na které
celá metodika stojí.

`config.example.yaml` je plnohodnotná dokumentace s komentáři u každého klíče —
lepší reference než většina „full parameter list" sekcí.

### 8.2 Zastaralá tvrzení o Shikimori — jediný faktický rozpor

> **Stav (2026-07-25): OPRAVENO** na všech čtyřech místech (docstring modulu,
> docstring `get_similar`, README 2×, `config.example.yaml`). Text teď říká,
> co platí: tvar ověřen, prostý seznam bez skóre podobnosti, pozicové
> váhování je proto jediná varianta; default vypnuto kvůli ceně, ne kvůli
> nejistotě. Doplněn i postřeh o síle signálu (viz níž).

**[ověřeno]** V `cache/shikimori/` je **41 odpovědí**, všechny `found=true`.
Tvar je potvrzený: **prostý seznam** objektů s klíči
`{id, name, russian, score, kind, episodes, episodes_aired, aired_on, released_on, status, image, url}`
— tj. **žádné explicitní skóre/pořadí podobnosti**.

Z toho plyne dvojí:
1. **Pozicová heuristika `rank_hint = 1/(i+1)` je potvrzeně správná volba**, ne
   provizorium — API sílu podobnosti nevrací, takže lepší signál neexistuje.
2. **Tři místa tvrdí, že tvar odpovědi je neověřený, a to už neplatí:**
   `shikimori.py:98-105` (docstring), `README.md` (2×: „naživo neověřený tvar
   odpovědi", „naživo neověřeno, default vypnuto") a `config.example.yaml`
   („zkontroluj tvar odpovědi před ostrým zapnutím"). Přitom u tebe
   `use_shikimori: true` **běží**.

Vedlejší postřeh z těch dat: jeden seed vrátil **133** podobných titulů. Při
`rank_hint = 1/(i+1)` a ořezu na `candidates_per_seed: 25` klesne příspěvek z
1,0 (i=0) na 0,04 (i=24), zatímco MAL-rec dává typicky `(1+log1p(votes))·w ≈ 6,5`.
Shikimori tedy prakticky přispívá **novými kandidáty**, ne přeskládáním pořadí —
což je fajn vědět a nikde to není.

### 8.3 Další mezery v dokumentaci

- ~~**Chyby v uživatelsky viditelném textu.**~~ **OPRAVENO 2026-07-25.**
  `report.py:727-728`: „spř**í**něných duší" / „spř**í**něné duše" (chybělo
  *z*) a „hodnoto**wí**" (→ „hodnotí"); `report.py:625` mělo „tvém" složené
  z kombinujícího akutu místo předkomponovaného „é".
- ~~**Terminologická dvojkolejnost**~~ — **SJEDNOCENO** na „senpai" (kód,
  README i celý CF report; dřív měl tentýž dokument v tabulce „senpai" a
  v kartách jiný termín).
- **NOVÝ NÁLEZ (opraven při té příležitosti): `report.py` měl od řádku 617
  dál zdrojový kód psaný literálními `\uXXXX`/`\xNN` escape sekvencemi**
  místo UTF-8 — 39 řádků, zbytek souboru přitom normální. Ve *stringech* je
  Python interpretuje, takže rendrovaný výstup byl správný, ale **čtyři
  komentáře byly fakticky nečitelné** (v komentáři se escape neinterpretuje).
  Celá oblast převedena na čisté UTF-8 (`\xa0` → `&nbsp;`, ať ve zdrojáku
  nejsou neviditelné znaky); výstup se tím nemění. Pozor při další editaci
  toho souboru — snadno se to zavede znovu.
- **Chybí zápis o ceně CF konfigurace** (§7.1). Za rok bude „proč to běželo pět
  hodin" otázka bez odpovědi; tři řádky komentáře v `config.yaml` to řeší.
- **`HODNOCENI_PROJEKTU.md` (tento dokument) a `CHANGELOG_review.md` (23 KB)**
  jsou historie, ale nikde to není napsané. Návrh: `docs/history/` a jednořádkový
  odkaz z README.
- **`animodel_test_harness.py` + `franchise_tags.py`** leží v rootu vedle
  produkčního kódu. README je zmiňuje, ale kořen projektu tím vypadá
  neuklizeně — `tools/` nebo `dev/` by pomohlo.

---

## 9. Návrhy na rozvoj

Řazeno podle poměru přínos/náklad. Body 1–5 jsou opravy nálezů z §5, bod 6+ je
nový potenciál.

### Krátké opravy (hodiny) — ✅ HOTOVO 2026-07-25

| # | Co | Výsledek |
|---|---|---|
| **1** | ✅ Nespat po posledním pokusu v retry smyčce (§5.3) | 240 s → 150 s na request s vyčerpaným rate limitem; navíc odolné vůči prázdnému `retry_delays`. 2 nové testy |
| **2** | ✅ Zahodit `Senpai.entries` u nevybraných kandidátů (§5.2) | ~450 MB → ~5 MB; nová `hydrate_entries()`. 4 nové testy + zpevněný existující |
| **3** | ✅ Strop na CF report (§5.8) | `user_cf_report_top` (default 200); 769 kB → 160 kB na 1 000 karet, hlavička hlásí ořez |
| **4** | ✅ `log.exception` + `status()` v CF handleru (§5.6) | Traceback se nezahazuje; progress řádka se nerozbíjí |
| **5** | ✅ Překlepy + terminologie (§8.3) | „spřízněné"/„hodnotí"/předkomponované „é"; sjednoceno na „senpai". **Navíc:** celý CF blok `report.py` převeden z literálních escape sekvencí na UTF-8 |
| **6** | ✅ Shikimori dokumentace (§8.2) | 4 místa: docstring modulu i `get_similar`, README 2×, `config.example.yaml` |
| **7** | ✅ Úklid osiřelé v1 CF cache (§6) | 2 309 souborů / 108,5 MB smazáno; postup + varování před `*[0-9].json` pastí v README |
| **8** | ✅ Smazat 9 mrtvých metod (§6) | −224 řádků z klientů; odstraněn i osiřelý import a odkaz v `attributes.py` |

Souhrn: **15 souborů**, testy 165 → **171** (všechny zelené), pipeline
ověřená ostrým během (`--no-recommend` nad 458 tituly, β=+0,49,
CV RMSE 0,899).

### Metodické opravy (den)

**9.1 Vyřešit trojice vs. kalibrace `scale` (§5.1)** — ✅ **HOTOVO 2026-07-26
variantou (c)**, detaily v §5.1. Zvolena nejpřesnější varianta, protože se
ukázalo, že je i dost levná (~1,6 s na běh); kompromis (b) s drobným únikem
informace nebyl potřeba. Původní rozvaha:
- **(a) Přiznat to.** Do `model.html` doplnit, že při `interaction_triples: true`
  je `cv_rmse` spočtená bez trojic. Nejmenší práce, žádná změna chování.
- **(b) Zahrnout trojice do CV.** `_cv_predictions` by muselo fitovat klastry na
  foldech — přesně to, co je označené jako „drahé a nestabilní na 4/5 dat". Dá se
  obejít: **kandidátské trojice předat z plného modelu** a na foldu jen
  přepočítat lifty. Klastrování se nefituje, kandidáti jsou fixní, únik informace
  je omezený na výběr kandidátů (obhajitelný — je to strukturální volba, ne
  parametr).
- **(c) Vlastní `scale_triples`** kalibrovaná zvlášť. Nejpřesnější, nejvíc práce.

~~Doporučuji **(b)**, s **(a)** jako okamžitou záplatou.~~ → Zvoleno **(c)**;
měření ukázalo, že per-fold klastrování stojí jen ~0,31 s, takže se za
přesnost neplatilo skoro nic a kompromis (b) byl zbytečný.

**Co z toho vyšlo o samotném experimentu s trojicemi.** Když už kalibrace umí
říct, kolik trojice přinesly, tady je sweep přes `interaction_min_lift` na
tvých datech (458 titulů):

| `min_lift` | párů | trojic | `s` | `s₃` | CV RMSE | bez trojic | přínos |
|---|---|---|---|---|---|---|---|
| 0.30 | 7 | 0 | 0.30 | — | 0.9064 | 0.9071 | +0.0007 |
| **0.25** (tvůj) | 21 | 4 | 0.30 | 0.15 | 0.8982 | 0.8986 | +0.0004 |
| 0.20 | 46 | 11 | 0.30 | 0.15 | 0.8895 | 0.8906 | +0.0010 |
| **0.15** | 126 | 19 | 0.30 | 0.25 | **0.8818** | 0.8859 | **+0.0041** |
| 0.10 | 330 | 21 | 0.25 | **0.00** | 0.8857 | 0.8857 | +0.0000 |
| 0.05 | 635 | 11 | 0.15 | **0.00** | 0.9078 | 0.9078 | +0.0000 |

Dvě čitelné věci: na tvém aktuálním prahu trojice **prakticky nepřinášejí nic**
(+0,0004), a při volnějších prazích (0,10 a níž) jim CV dá `s₃ = 0`, tedy
„nevěř jim vůbec" — páry už tu strukturu vysvětlí. Nejlepší bod je `min_lift
0.15` (19 trojic, `s₃ = 0.25`, +0,0041).

**Ber to jako hint, ne doporučení k přepnutí**, a to ze dvou důvodů: (1) vybírat
`min_lift` podle CV RMSE je samo selekce na CV, takže část toho zlepšení je
optimismus; (2) CV RMSE je podle vlastní metodiky projektu ta *špatná* metrika —
model má řadit, ne hádat číslo. Poctivě to rozhodne až §9.4 (zpětná vazba
z historie).

**9.2 Klastrová podobnost přes centroidy (§5.4).** ✅ **HOTOVO 2026-07-26**,
naměřené dopady v §5.4. Implementováno podle plánu (těžiště + prostor nálady
v modelu, vážený kosinus, `cluster_fit_weight` místo zadrátované 0,5), navíc
ověřena a zamítnuta varianta s distinktivitou. Testů 180 → **187**.

Shrnutí přínosu: přiřazení nálady 85,4 % → **99,6 %** shody s KMeans,
zkreslení podle počtu atributů odstraněno (−0,092 → +0,034), řazení nahoře
stabilní (top-10 beze změny). Původní rozvaha:

> V `_fit_clusters` uložit `centroid` vektor + `feat_keys` do `Cluster`;
> `_cluster_fit` pak počítá vážený kosinus kandidátova vektoru (jen
> genre/theme/tag/demographic, stejné váhy jako při fitu) proti centroidu.
> Odstraní to zkreslení podle bohatosti metadat, využije informaci, která se
> dnes zahazuje, a zároveň **zjemní** signál: dnes je to shoda s 6 slovy, pak
> by to byla shoda s celým profilem nálady. Zároveň vytáhnout magickou 0,5
> z `taste_fit` do configu (§1).

**9.3 Neutrální z-skóre pro chybějící graf (§5.5).** `item_votes = None` u
tag-search-only kandidátů, z-skóre počítat jen z nenulových. Obsahové discovery
tím dostane šanci, kterou dnes strukturálně nemá.

### Nový potenciál (návrhy k rozhodnutí)

**9.4 Zpětná vazba z historie — největší nevyužitá příležitost.**
Každý nový MAL export je **ground truth pro předchozí doporučení** a dnes se
zahazuje. Konkrétně: uložit každý běh do `output/history/{datum}.json` (mal_id,
composite, taste_fit, pred, pořadí). Při dalším běhu spárovat s aktuálním
exportem a spočítat, co se z doporučených skutečně dostalo do Completed a s jakou
známkou.

Z toho vypadne přesně ta metrika, která dnes chybí (§1): **hit rate a průměrná
známka doporučených titulů** — tedy validace *řazení*, ne predikce čísla. A jako
druhý krok: `w_taste_fit / w_cf / w_user_cf / w_quality` se pak dají ladit proti
měřenému výsledku místo odhadem. To je posun od „model, který vypadá rozumně" k
„model, o kterém vím, že funguje", a vstupní data pro to už existují.

**9.2b Archetyp nálady** (doplněno 2026-07-26 na návrh uživatele) — ✅ HOTOVO.
Každá nálada dostane **nejtypičtějšího představitele** (`Cluster.archetype`),
který ji v `model.html` pojmenuje konkrétním titulem. Doplňuje to seznam členů,
který je řazený podle **známky**, a ukazuje tedy nejlíp hodnocené členy, ne ty
charakteristické; v kartě jsou proto teď oba, popsané („nejtypičtější titul" /
„nejlépe hodnocené"). Metrika je tentýž kosinus k těžišti, jaký používá
`_cluster_fit`, takže co report ukazuje jako jádro nálady, odpovídá tomu, podle
čeho se přiřazují kandidáti.

Výsledek na tvých datech:

| nálada | archetyp | shoda |
|---|---|---|
| Slice of Life / Romance / Coming of Age | Kimi ni Todoke 3rd Season | 85 % |
| Harem / Ecchi / Female Harem | Bokutachi wa Benkyou ga Dekinai! | 86 % |
| Fantasy / Magic / Adventure | Dungeon ni Deai wo Motomeru… (DanMachi) | 70 % |
| Comedy / Shounen / Chibi | Seitokai Yakuindomo Movie | 70 % |
| Military / Cute Girls Doing Cute Things / School | Girls & Panzer: Saishuushou Part 1 | 88 % |

**Nutná korekce naivní verze.** Čistý kosinus k těžišti vybíral u komediální
nálady *„Kanojo, Okarishimasu Petit Special"* — titul s **jediným** atributem
v prostoru nálady proti mediánu 10. Kosinus je směrový, takže titul nesoucí jen
dominantní osu těžiště dosáhne vysoké shody, i když z nálady nepokrývá skoro
nic; korelace kosinu s počtem atributů byla v tom klastru **−0,56**. Vyzkoušeno
a zamítnuto:

- **medoid** (nejvyšší průměrná podobnost ke všem členům) — *nepomohl*: titul
  nesoucí jen nejčastější atribut je podobný všem, takže vyhrál znovu;
- **kosinus × pokrytí těžiště** — sklouzlo k „titulu s nejvíc tagy": u komedie
  vybralo Spy x Family (38 atributů, ale kosinus jen 0,54 a vůbec ne chibi).

Použit **mediánový práh**: archetyp se hledá jen mezi členy s aspoň mediánovým
počtem atributů nálady. Práh je odvozený z dat (ne magická konstanta), drží
vysoký kosinus (0,70–0,88 napříč náladami) a nikdy nevyprázdní výběr — aspoň
polovina členů ho splní vždy, takže není potřeba žádná záložní větev.

Testy: past je reprodukovaná syntetickou fixture s rozptýleným těžištěm a
`test_archetype_skips_degenerate_sparse_members` **nejdřív doloží, že čistý
kosinus by degenerovaný titul opravdu vybral** (aby test nemohl zvacuovat) a
teprve pak, že implementace na něj nesedne. První verze toho testu vacuous byla
— čistý kosinus na ní vybíral správně, takže filtr netestovala vůbec.

**9.5 Diagnostika kanonizace atributů (§6).** `--analyze-attrs`: vypsat klíče,
které se liší jen málo (Levenshtein ≤2, nebo shodné po odstranění stop-slov) a
**nejsou** v `ALIAS`. Jediný tichý selhací mód `attributes.py` je „dva klíče pro
jeden koncept" a ten systematicky nafukuje efekty — přesně to, čemu má modul
zabránit. Levné, jednorázově užitečné.

**9.6 Vyčistit privátní rozhraní mezi vrstvami (§2).** `Recommender.recommend()`
by vracel `RecommendResult(recs, senpai, cf_raw)` místo tří privátních atributů;
`_raw_resid_pred` → veřejné `affinity()`; `_prequel_score` → skutečné pole
`Recommendation`. Nic to nerozbije a odstraní tři místa, kde tichá regrese
neshodí testy.

**9.7 Rozpad `cli.py::run()`** na `run_model / run_season / run_analyze /
run_gen_intensity` nad společným `prepare(cfg)`. Sjednotit číslování kroků
(`[4/4]` vs `[5/5]`).

**9.8 Volitelně: Jikan cache do `cache/mal/`** (§2) — sjednotí to s ostatními
třemi klienty a udělá selektivní invalidaci `rm -r`-schopnou. Vyžaduje jednorázový
přesun ~10 tis. souborů, jinak se cache znovu stáhne. Čistě kosmetika, spíš
„až se to bude hodit".

---

## 10. Co bych neměnil

- **Reziduální cíl + zdůvodnění restrikce rozsahu.** Nosná myšlenka, správně
  odvozená z dat a důsledně držená (§1).
- **Jedno `n/(n+K)` na čtyřech místech** (efekty, páry, trojice, CF podobnost).
  Nejlepší vlastnost celé metodiky — jeden princip, žádné ad-hoc váhy (§3.3).
- **`Result` + `cached_fetch` + `request_with_retry`.** Tři primitivy, které
  celou třídu bugů z kola 1 dělají strukturálně nemožnou (§2).
- **Docstringy se zamítnutými alternativami.** Ať se refaktoruje cokoli, tyhle
  komentáře přenést s sebou. Jsou to nejcennější řádky v repozitáři (§4).
- **Senpai pipeline jako čtyři testovatelné funkce** nad úzkým klientským
  kontraktem (§2).
- **Komunitní skóre bez průměrování MAL/AniList** — zdůvodněné korelací, správně.
- **`--gen-intensity` s exaktním universem a zachováním úprav.** Řeší jedinou
  nutně lidskou část úsudku tak, že se nedá tiše rozjet s daty (§1).

---

*Analytická část dokumentu (§1–§8) popisuje stav při review. Osmička krátkých
oprav (§9, tabulka) je od 2026-07-25 **implementovaná** — jednotlivé nálezy
nesou stav v citovaném bloku. Otevřené zůstávají metodické opravy §9.1–§9.3
(nejdřív §9.1(a) jako okamžitá záplata kalibrace `scale` při
`interaction_triples: true`) a rozvojové body §9.4–§9.8, z nichž §9.4
(zpětná vazba z historie běhů) je jediný, který přidává skutečně novou
schopnost, ne úklid.*
