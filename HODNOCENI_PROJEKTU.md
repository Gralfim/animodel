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
> Testů 165 → **256**, všechny zelené; defaultní cesta kalibrace ověřena jako
> bitově identická s HEAD. §9.3 **zamítnuta** (její premisa byla chybná —
> viz §5.5), ale při jejím ověřování se našel a opravil vážnější bug: obsahová
> discovery větev byla kvůli AND sémantice `tag_in` **mrtvá**. Otevřené
> zůstává už jen druhý krok §9.4 — ladit váhy kompozitu proti naměřenému výsledku, což jde až s několika snapshoty za sebou.

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

### 5.5 Discovery větev byla MRTVÁ (původní diagnóza byla chybná) — VYSOKÁ

> **Stav (2026-07-26): OPRAVENO — ale úplně jinak, než §9.3 navrhovala.**
> Při implementaci se ukázalo, že **původní nález byl ve dvou bodech
> špatně**, a skutečný problém byl vážnější. Nechávám tu obojí, ať je vidět,
> co neplatilo.

**Co jsem tvrdil:** `bump(m["mal_id"], 0.0, None, "tag-search")` dá
`item_votes = 0` → `z_item(log1p(0))` je dno rozdělení → `w_cf · z_item` je
„fixní srážka", kvůli které se ryze tag-search nálezy do žebříčku nedostanou.
Navrhované řešení: `item_votes = None` a z-skóre jen z kandidátů, kde graf
nějaký hlas dal.

**Chyba č. 1 — centrování na pořadí nemá vliv.** Kompozit je součet z-skóre,
tedy lineárních transformací. V rozdílu dvou kandidátů se průměr vykrátí:

```
comp_i − comp_j = Σ w_k · (x_ki − x_kj) / sd_k        (mean_k se vyruší)
```

Ověřeno numericky na reálném poolu: kompozit s centrováním a bez něj dává
**identické pořadí** a liší se o konstantu (6,566218 pro každého kandidáta).
„Je na dně rozdělení" tedy není srážka v řazení; jediné, co z-skóre reálně
řídí, je měřítko (`sd`) a váha `w_cf`. Návrh §9.3 by navíc byl **škodlivý**:
posunout „bez důkazu" na `z = 0` znamená ohodnotit ho jako *průměrný* důkaz,
což by discovery nálezy vystřelilo nad kandidáty s podprůměrnou, ale skutečnou
podporou grafu.

**Chyba č. 2 — tag-search kandidáti neexistovali.** V reálném poolu bylo
tag-search-only kandidátů **0 ze 414**. Ne proto, že by se propadli, ale proto
že discovery větev nevracela nic.

**Skutečný nález: `search_by_tags` byla mrtvá.** AniList `tag_in` má **AND**
sémantiku — vrací jen tituly nesoucí *všechny* vyjmenované tagy. Posílalo se
tam pět nejcharakterističtějších tagů najednou, což reálně nesplní nic.
Ověřeno živě: `["Iyashikei"]` → 50 titulů, `["Iyashikei","Reincarnation"]` →
18, tři niche tagy → **0**. Celá obsahová discovery větev tedy tiše vracela
prázdno a **nešlo si toho všimnout** — prázdný výsledek je legitimní návratová
hodnota, ne chyba, takže nepadl ani warning.

Opraveno dotazem **na každý tag zvlášť** se sjednocením výsledků. Dopad:

| | před | po |
|---|---|---|
| kandidátů v poolu | 404 | **515** |
| z toho čistě obsahových (bez grafu) | 0 | **111** |
| nejlepší obsahový nález | — | **#5** (Mushoku Tensei III, taste_fit +1,41, MAL 8,88) |

Ten titul rec graf nezná (je čerstvý) a model ho vytáhl čistě podle obsahu —
přesně k tomu ta větev je. Zbytek se drží při zemi (medián #517 z 675 při
`pages=2`), což je v pořádku: chybějící podpora grafu *je* legitimní mínus.

Volba `pages=1` na tag: druhá stránka přidala 160 kandidátů k obohacení, ale
do top-40 z nich neprošel ani jeden — stejný přínos za dvojnásobnou cenu.
Discovery teď stojí 5 requestů (dřív 2, které nevracely nic).

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

**9.3 Neutrální z-skóre pro chybějící graf (§5.5).** ❌ **ZAMÍTNUTO
2026-07-26 — premisa byla chybná**, viz §5.5. Centrování z-skóre pořadím
nehýbe (ověřeno numericky: identické pořadí, konstantní posun 6,566218), takže
žádná „fixní srážka" neexistovala; a navržená změna by byla škodlivá — „bez
důkazu" by se ohodnotilo jako *průměrný* důkaz.

✅ Místo toho opraven **skutečný** problém, který se pod tím schovával:
`search_by_tags` posílala pět tagů do `tag_in`, který má **AND** sémantiku, a
celá obsahová discovery větev tak tiše vracela **0 kandidátů**. Po opravě
(dotaz per tag) pool 404 → 515, z toho 111 čistě obsahových nálezů a jeden
na **#5**. Testy `test_search_by_tags_*` v `tests/test_anilist.py`.

### Nový potenciál (návrhy k rozhodnutí)

**9.4 Zpětná vazba z historie — největší nevyužitá příležitost.**
✅ **ZÁKLAD HOTOV 2026-07-26** (`animodel/history.py`, 20 testů).

> **Změna proti původnímu návrhu (podnět uživatele):** snapshoty se
> **neklíčují datem, ale otiskem stavu seznamu**. Ladění parametrů znamená
> desítky běhů nad týmž exportem a datové klíčování by z nich udělalo
> desítky skoro identických záznamů, které by evaluaci jen ředily (tentýž
> stav světa započítaný mnohokrát).
>
> Otisk = sha256 nad seřazenými trojicemi `(mal_id, status, score)`, tedy nad
> tím, co model konzumuje a co zároveň tvoří ground truth. **Záměrně ne hash
> souboru:** `my_watched_episodes` a `my_finish_date` se v exportu mění
> průběžně, takže hash bajtů by se lišil po každém odsledovaném dílu, aniž by
> se změnilo cokoli podstatného. Naopak přesun do Completed nebo změna známky
> otisk změní — a to je nový stav světa, který si zaslouží vlastní záznam.
> Stejný otisk = přepis (poslední běh vyhrává). Ověřeno: **tři ladicí běhy
> s různým `--shrinkage` daly jeden soubor.**
>
> Název `{počet_hodnocených}_{otisk}.json` (druhá varianta z podnětu) dává
> složce chronologickou čitelnost — seznam v čase roste, takže abecední
> řazení odpovídá časovému.
>
> **Metriky.** Klíčová není hit rate, ale **Δ proti kontrole**: průměrná
> známka doporučených titulů dokoukaných od snapshotu proti průměru
> *ostatních* titulů dokoukaných za tutéž dobu. Teprve to odpovídá na
> „jsou doporučení lepší než to, co bych si vybral sám?". Hit rate je vždy
> jen spodní odhad (jmenovatel je celý snapshot, čitatel to, co se stihlo
> dokoukat). Vedle toho průměrná známka **po desítkách pořadí** — přímá
> validace *řazení*, tedy přesně to, co CV RMSE změřit neumí.
>
> Ověřeno end-to-end na simulovaném budoucím exportu (tři doporučené tituly
> přesunuté do Completed + kontrolní skupina): `doporučené 8.67 (n=3) vs.
> ostatní nové 6.50 (n=4), Δ +2.17`, pořadí 1–10 průměr 9.00, 11–20 průměr
> 8.00. Simulovaný snapshot i testovací `history/` po ověření smazány, ať
> nekazí budoucí vyhodnocení skutečných běhů.
>
> **Co zbývá:** (a) HTML sekce místo jen CLI výpisu, (b) druhý krok z návrhu
> níž — ladit `w_taste_fit / w_cf / w_user_cf / w_quality` proti naměřenému
> výsledku, což jde ale až s několika snapshoty za sebou. Původní záměr:
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

**9.2c Doplnění osy náročnosti + strukturální mezera v universu**
(2026-07-26) — ✅ HOTOVO. Diagnostika `unrated_intensity_attrs()` po ostrém
běhu nahlásila 10 pozorovaných atributů bez hodnoty; při jejich revizi vyšel
najevo i **skutečný bug**:

> `build_universe()` stahovalo z MAL jen `filter=genres` (18 položek) a
> `filter=themes` (52). MAL má ale **třetí skupinu `explicit_genres`**
> (Ecchi, Erotica, Hentai), která se nestahovala nikdy. `build_attributes()`
> přitom MAL žánry nefiltruje — vylučují se jen AniList `isAdult` **tagy**,
> což je jiná, mnohem granulárnější množina. **Erotica** se proto v datech
> pozorovala, ale do `intensity.yaml` se nemohla dostat ani regenerací a
> hlásila se jako neohodnocená pořád dokola. `Ecchi` tu mezeru náhodou
> obcházelo přes AniList `GenreCollection`; Erotica tam ekvivalent nemá.
> Opraveno + test `test_build_universe_includes_mal_explicit_genres`.

Přiřazené hodnoty (kalibrované proti sousedním klíčům v lexikonu, ne od stolu;
zapsané do `CURATED` i do uživatelova `intensity.yaml`):

| atribut | hodnota | proč |
|---|---|---|
| `adult_cast` (20×) | **+0.1** | jako seinen/josei — dospělejší rámování, ale i spousta lehkých komedií |
| `video_game` (5×) | **−0.1** | hra jako prostředí/námět; ne `high_stakes_game` (+0.5) |
| `love_status_quo` (4×) | **−0.2** | vztah se záměrně neposouvá = pohodlí (opak `love_triangle` +0.1) |
| `performing_arts` (2×) | **+0.1** | divadlo/rakugo — dramatičtější než `music` (−0.2) |
| `visual_arts` (2×) | **−0.1** | tvůrčí slice-of-life |
| `magical_sex_shift` (2×) | **−0.3** | komediální premisa, blízko `parody` |
| `erotica` (2×) | **−0.2** | méně komediální než `ecchi` (−0.3) |

**Vědomě ponecháno na 0.0** (revidováno, ne zapomenuto — explicitní nula se
v diagnostice už nehlásí):

- `award_winning` (14×) — je to marker **uznání/kvality**, ne emocionální
  náročnosti. Kladná hodnota by na osu náročnosti propašovala komunitní
  kvalitu, kterou si model všude jinde pečlivě odděluje.
- `workplace` (4×) — genuinely bimodální (Shirobako vs. vyhoření), nenulová
  hodnota by přidala šum.
- `girls_love` / `boys_love` (1× / 1×) — spektrum od lehkého romcomu po
  těžké drama; spolehlivé znaménko neexistuje.

Dopad je malý a je to tak správně: intenzity nálad se posunuly nejvýš
o **0,012** (nízkofrekvenční atributy s mírnými hodnotami). Jde o úplnost
a o to, že diagnostika je teď čistá (1 → 0 neohodnocených).

**9.5 Diagnostika kanonizace atributů (§6).** ✅ **HOTOVO 2026-07-26** —
`--analyze-attrs` + `attributes.find_near_duplicate_keys()`.

> **Naivní verze byla k ničemu a musel jsem ji přepsat.** Původní návrh
> (Levenshtein ≤2 nad celými klíči) dal na živých datech **19 dvojic, z toho
> 2 skutečné** — znaková vzdálenost totiž nerozliší „jiný token" od „jiná
> koncovka": `female_protagonist` ↔ `male_protagonist` má vzdálenost 2 úplně
> stejně jako `video_game` ↔ `video_games`, přestože první dvojice jsou dva
> různé koncepty a druhá jeden. S takovým poměrem šumu by nástroj nikdo
> nepoužil.
>
> Přepsáno na porovnání **po slovech**: (1) shodná slova po stemmingu
> množného čísla, nezávisle na pořadí; (2) jednoslovné klíče s editační
> vzdáleností ≤2 **a** společným prefixem ≥ 70 % délky kratšího (tedy
> „liší se jen koncovkou"). Výsledek na týchž datech: **2 dvojice, obě
> skutečné**, žádný falešný nález.
>
> | nález | zdroje | výskyty |
> |---|---|---|
> | `video_game` ↔ `video_games` | MAL téma vs. AniList tag | 12× a 33× |
> | `anthropomorphic` ↔ `anthropomorphism` | MAL téma vs. AniList tag | 4× a 9× |
>
> Oba páry doplněny do `ALIAS`. Evidence se tím slévá místo aby se dělila —
> u `video_game` šlo o 12 + 33 titulů rozpadlých na dvě poloviny.
>
> **Vedlejší nález při té příležitosti:** přidání aliasu odhalilo latentní
> chybu v `load_lexicon()`. Když dva řádky `intensity.yaml` spadnou po
> kanonizaci na tentýž klíč, tiše vyhrál ten pozdější v souboru — takže
> `video_games: 0` („neohodnoceno") by přepsalo `video_game: -0.1`. Teď se
> kolize hlásí a nenulová hodnota má přednost před nulou. Týkalo by se to
> každého budoucího aliasu, ne jen tohohle. `--analyze-attrs`: vypsat klíče,
které se liší jen málo (Levenshtein ≤2, nebo shodné po odstranění stop-slov) a
**nejsou** v `ALIAS`. Jediný tichý selhací mód `attributes.py` je „dva klíče pro
jeden koncept" a ten systematicky nafukuje efekty — přesně to, čemu má modul
zabránit. Levné, jednorázově užitečné.

**9.6 Vyčistit privátní rozhraní mezi vrstvami (§2).** ✅ **HOTOVO 2026-07-26.**

> - `Recommender.recommend()` vrací `RecommendResult(recs, senpai, cf_raw)`
>   místo aby CF část předával privátními atributy čtenými přes
>   `getattr(rec, "_cf_raw_results", [])`. Ten kontrakt byl implicitní:
>   přejmenování atributu by nic nerozbilo, jen by tiše zmizel CF report.
>   Výsledek je iterovatelný a má `len()`, takže volající, kterého CF
>   nezajímá, se chová jako dřív.
> - `Recommendation.prequel_score` je skutečné pole místo dynamicky
>   přišpendleného `rec._prequel_score` čteného přes `getattr` — fungovalo,
>   ale rozbilo by se tiše při `slots=True`.
> - `model._raw_resid_pred()` volaný z jiných modulů nahradilo veřejné
>   `affinity()` (už v §9.1) — `_raw_resid_pred` zůstává jako interní
>   diagnostika neškálovaného součtu. `Recommender.recommend()`
by vracel `RecommendResult(recs, senpai, cf_raw)` místo tří privátních atributů;
`_raw_resid_pred` → veřejné `affinity()`; `_prequel_score` → skutečné pole
`Recommendation`. Nic to nerozbije a odstraní tři místa, kde tichá regrese
neshodí testy.

**9.7 Rozpad `cli.py::run()`** ✅ **HOTOVO 2026-07-26.**

> Jedna 215řádková funkce se čtyřmi early-return větvemi je teď `run()`,
> které jen vybere režim, plus `run_analyze / run_analyze_attrs /
> run_gen_intensity / run_season / run_recommend` nad sdíleným
> `RunContext` (načtený seznam + enricher + odvozené `watched_ids`/`ptw_ids`).
> Vytažené i `_apply_overrides`, `_check`, `_exclude_own_account`,
> `_render_cf_report`, `_record_history`.
>
> **Číslování kroků drží třída `Steps`** — každý režim řekne, kolik kroků má,
> a pořadí se dopočítá. Ruční `[4/4]` vs. `[1/5]` se tím rozejít nemůže.
>
> Ironická poznámka k tomu: **při prvním ostrém běhu se to rozešlo hned zas**
> (`--analyze-attrs` tiskl `[3/2]`, protože jsem do součtu zapomněl vlastní
> krok navíc). Opraveno dvakrát: `_mode_steps()` teď vrací jen kroky *po*
> `prepare()` a celek se skládá jako `PREPARE_STEPS + _mode_steps()`, takže
> vztah je vidět; a `Steps` na přetečení zaloguje warning, aby to příště
> nemohlo projít potichu. Ověřeno na všech pěti režimech. na `run_model / run_season / run_analyze /
run_gen_intensity` nad společným `prepare(cfg)`. Sjednotit číslování kroků
(`[4/4]` vs `[5/5]`).

**9.8 Jikan cache do `cache/mal/`** (§2) ✅ **HOTOVO 2026-07-26.**

> Všichni tři klienti teď berou **kořen** cache a podsložku si doplní sami
> (`cache/mal`, `cache/anilist` + `cache/cf_al`, `cache/shikimori`). Kořen
> tak obsahuje jen podsložky a invalidace jednoho zdroje je `rm -r cache/mal`
> místo globu. Cestou zmizela i asymetrie, kdy Shikimori jako jediný
> očekával už složenou cestu a volající mu ji skládal
> (`f"{cache_dir}/shikimori"`).
>
> Migrace uživatelovy cache provedena jednorázově: **10 427 souborů / 112 MB**
> přesunuto do `cache/mal/`. Ověřeno průkazně — po zakázání
> `requests.Session.get/post` se celý seznam (458 titulů) obohatil **bez
> jediného síťového requestu**, takže se nic neztratilo.
>
> **Automatickou migraci jsem záměrně nepřidal.** Byl by to kus kódu, který
> po jednom použití zůstane v repozitáři navždy a zhorší čitelnost přesně té
> vrstvy, kterou tenhle bod čistí. Místo toho je v README jednořádkový příkaz
> pro případ, že se někde objeví stará cache (jiný stroj, záloha). Testy
> `test_each_client_uses_its_own_subdirectory`,
> `test_cache_root_holds_no_loose_files`,
> `test_enricher_gives_all_clients_the_same_root`.

Původní poznámka: — sjednotí to s ostatními
třemi klienty a udělá selektivní invalidaci `rm -r`-schopnou. Vyžaduje jednorázový
přesun ~10 tis. souborů, jinak se cache znovu stáhne. Čistě kosmetika, spíš
„až se to bude hodit".

---

## 9b. Revize zdrojů dat (2026-08-05)

Kontrola po ~týdnu provozu: co se změnilo u používaných zdrojů a je-li venku
něco nového, co by projektu pomohlo. Stav endpointů **změřen živě**, ne
odvozen z dokumentace.

| zdroj | stav | pozn. |
|---|---|---|
| **Tenrai** (default) | ✅ 200, 0,04–0,61 s | deklaruje 99,9 % uptime, pokrývá 95 % Jikan endpointů |
| **Jikan** (fallback v configu) | ❌ **HTTP 504** | výpadek trvá |
| **AniList** | ✅ 200, ~0,5 s | 90 req/min beze změn |
| **Shikimori** | ✅ 200, 1,6 s | v1 REST funguje; jejich docs už doporučují GraphQL |

Jikanův výpadek není náhoda: v issue trackeru běží od dubna do července 2026
řada hlášení (#591, #594, #606 „Anime API is down", #608, #609, #610) **bez
reakce údržby**. Zakomentovaný fallback v `config.example.yaml` je tedy
dekorace; migrace na Tenrai byla správná.

Další zjištění bez dopadu na kód: **`anime-offline-database` byla 4. 7. 2026
archivována** (nejčastěji doporučovaný offline dataset, teď zmražený na ~40
tis. záznamech). Z nových zdrojů nic nepřesvědčilo — **ids.moe/AnimeAPI**
(mapování ID) by jen zpřesnilo dnešní zkratku „Shikimori ID ≈ MAL ID",
**AniDB** by duplikoval AniList tagy za cenu registrace a přísných limitů,
**Kitsu/Simkl** nabízí třetí komunitní skóre, které projekt vědomě neprůměruje.

### Co z toho vzniklo v kódu

**AniList umí dodat staff v téže dávce po 50** (role „Director", „Series
Composition", „Original Creator" sedí doslova na `DIRECTOR_POSITIONS`/
`WRITER_POSITIONS`). Přidáno do `_MEDIA_FIELDS` spolu s `countryOfOrigin`;
cache klíč bumpnut na `_v3`.

> **Původní doporučení („přesunout staff na AniList, ušetří ~460 requestů")
> se při měření ukázalo jako polovičně chybné** a skončilo jako
> konfigurovatelná volba, ne jako výměna defaultu:
>
> 1. **Míchat zdroje per titul by byla chyba.** Každý píše jména jinak —
>    „Mizushima, Tsutomu" vs. „Tsutomu Mizushima". Ze ~159 jmen se doslovně
>    shodovalo **9**. Tentýž člověk by dostal dva klíče podle toho, který
>    zdroj titul pokryl, a evidence by se rozdělila na poloviny — přesně ten
>    selhací mód, na který je `--analyze-attrs`. Řešeno dvakrát: nová
>    `attributes.person_key()` (klíč ze seřazených slov jména, průnik po
>    normalizaci **100** místo 9) **a** pravidlo „jeden zdroj na běh".
> 2. **Úspora requestů je jednorázová a už je zaplacená.** Kdo `include_staff`
>    používá, má staff dávno v cache (tady 5 130 souborů), takže by přechod
>    nepřinesl nic a jen ubral data.
> 3. **AniList je mělčí.** Změřeno na 458 titulech: 571 osob / 121 atributů
>    nad prahem (Jikan) vs. 464 / 94 (AniList). V živém modelu je rozdíl
>    menší, protože práh se aplikuje na *vážený* počet (franšízové váhy):
>    **24 vs. 20 staff efektů**, CV RMSE 0,8996 vs. 0,9012. Zvětšení
>    `perPage` z 10 na 25 pokrytí nezlepšilo (+1 titul za dvojnásobná data),
>    takže nejde o useknutí, ale o chybějící data.
>
> Výsledek: `enrich.staff_source` = `jikan` (default, beze změny chování)
> nebo `anilist`. AniList se vyplatí na **studené cache** (94 atributů zdarma
> proti 121 za ~460 requestů) a v `--no-jikan` režimu, kde je jediný možný.

**`countryOfOrigin`** přidán jako atribut kategorie `origin`, a to **jen když
není japonský** — JP tvoří ~95 % seznamu, takže jako atribut by to byla
konstanta s nulovým efektem. Na tomhle seznamu je dopad zatím **nulový**
(446 JP, 1 KR, 11 neznámo), takže žádný efekt neprojde prahem; smysl to má do
budoucna, kdyby přibyla donghua nebo korejská tvorba.

**Nepřidáno:** `duration` (délka dílu). Zpřesnila by detekci vedlejšího obsahu,
ale to je změna franšízových vah — vlastní téma s vlastním měřením, ne přílepek
k revizi zdrojů. Až se pro to někdo rozhodne, bude to stát další bump cache.

---

## 9c. Revize predikcí, doporučení a historie (2026-09-15)

*Kolo 3. Zadání: lepší predikce a doporučení + jak využít historii hodnocení
ke zpřesnění současných a k novým statistikám. Postup: čtyři analytické
průchody kódem a daty (31 návrhů), pak **nezávislé přeměření a adversariální
revize** v šesti skupinách (každá měřila znovu vlastním kódem) a kontrola
úplnosti. Vše offline nad cache, bez sítě, bez zásahu do repozitáře; nad
skutečným seznamem (487 ohodnocených, 3 snapshoty, 339 titulů s datem
dokončení). Čísla jsou změřená, ne odvozená ze čtení kódu.*

### 9c.1 Nálezy v kódu

| # | nález | stav |
|---|---|---|
| **1** | **Duplicitní položky v AniList userlistech — VYSOKÁ.** `_fetch_user_animelist` (`anilist.py:700-718`) prochází všechny listy `MediaListCollection` **včetně vlastních (custom)** a nededuplikuje podle `mal_id`. Titul zařazený ve status listu i ve vlastním se počítá dvakrát: nafouknutý `overlap` (a tím `min_full_overlap`), zdvojená váha v Pearsonovi, zkreslený `personal_avg` a v `rec_count` jeden senpai jako dva hodnotitelé. Změřeno: duplicity v **451 z 2000** vyhodnocených seznamů (79 269 položek), **2 z 20** dnešních senpai vybraní jen díky nim, **40 titulů** projde `min_raters=2` jen díky duplicitě, Bühlmannovo K 3,21 → **21,3** po deduplikaci. | ✅ **OPRAVENO** — dedup při ČTENÍ, takže platí i pro existující v2 cache bez nového stahování |
| **2** | **Podobnost senpai odměňuje konstantní hodnotitele — VYSOKÁ.** `evaluate_candidate` (`usercf.py:182-184`) koreluje `(moje − komunita)` s `(jeho − komunita)`. Kdo dává všemu maximum, je čistý obraz komunity a při β≈0,49 vyjde vysoce podobný: teoretická korelace **0,392**, u vybraných 0,41–0,48. Změřeno: **16 z 20** vybraných senpai má sd normalizovaných známek < 0,05 (v eligible poolu 5,9 %), korelace skóre se sd **−0,571**, první hodnotitel se sd ≥ 0,1 je až 22. v pořadí. Normalizace škál je přitom správná (ověřeno na surových datech 2000 seznamů, žádná známka nepřesahuje dělitel formátu) — chyba je v tom, co podobnost odměňuje. | otevřené (krok 6) |
| **3** | **User-CF složka nese podruhé komunitu — VYSOKÁ.** `recommend_from_senpai` vrací `(komunita + diff + bonus)·10` (`usercf.py:337-344`), což jde přes `bump` jako `user_votes` do vlastního z-skóre s vahou 0,6 — vedle `w_quality` 0,3. Změřeno na poolu: korelace `user_cf_signal` s komunitou **0,86–0,93**, rozptyl komunitní části 0,78 proti 0,11 u rozdílové. S dnešními konstantními senpai je to fakticky druhá kopie komunitního skóre. Porušuje princip „komunita vstupuje jen jedním sklonem". | otevřené (krok 6, jen spolu s #2) |
| **4** | **Afinita nemá intercept — STŘEDNÍ.** Součet smrštěných efektů (`taste.py:405-423`) není centrovaný (in-sample průměr **+0,19**) a `_calibrate_scale` (`taste.py:470-491`) hledá `s` podle RMSE bez interceptu. Predikovaná známka je proto systematicky nadsazená: CV bias **−0,17** proti +0,01 u samotné baseline, na disjunktních časových oknech −0,25. Po centrování počítaném ve foldu: CV RMSE −0,028, bias na oknech **−0,07**, žebříček skoro beze změny (top-40 37–38/40, top-100 98–99/100). Vysvětluje i nález „plný model má horší RMSE než baseline" a asymetrii chyb. | otevřené (krok 5) |
| **5** | **Náhodná CV prosakuje franšízami — STŘEDNÍ.** `_cv_predictions` (`taste.py:508-511`) přiděluje foldy po titulech, ale **95 ze 103** franšíz se rozpadne do víc foldů a **355 z 371** franšízových titulů má sourozence v tréninku. Grouped CV podle `series_root` (18 seedů): cv_rmse 0,892 → **0,925** (znaménko rozdílu se neotočí ani u jednoho seedu), OOF Spearman 0,39 → 0,26, zisk nad baseline 0,053 → 0,024. | otevřené (krok 5) |
| **6** | **Mid-franchise sequely v doporučeních — STŘEDNÍ.** Kandidáti se skórují po jednotlivých MAL ID (`recommend.py:400-446`), franšíze se používají jen pro váhy a limit seedů. V dnešním top-100 je **6** pokračování s neviděným prequelem hlavního formátu (v top-40 **3**), plus 4 duplicitní franšízy; ve snapshotech bylo #1 „3-gatsu no Lion 2nd Season", zatímco 1. řada byla #3. Všech 8 nálezů má mezi zdroji user-CF — odpovídá to otevřenému nápadu „filtrovat mid-franchise sequely v senpai doporučeních". Pozor: detekce **musí** kontrolovat formát prequelu, jinak prologová OVA dá falešný nález (2 z 8). | otevřené (krok 7) |
| **7** | **`saved_at` není datum exportu — STŘEDNÍ.** Je to čas POSLEDNÍHO běhu nad otiskem (ladicí běhy ho posouvají). Doloženo: Toaru Index II/III mají `my_finish_date` 07-19 a 07-26, ale ve snapshotu uloženém 07-28 ve `watched_ids` nejsou. Jakákoli sazba „za 30 dní" nad `saved_at` je vychýlená. | ✅ **OPRAVENO** — `export_date` = mtime souboru exportu |
| **8** | **`load_snapshots` řadilo podle `n_rated`** (`history.py:186`) — počet hodnocených klesne, když titul odhodnotíš nebo smažeš, a pořadí přestane být chronologické; okna ledgeru potřebují čas. | ✅ **OPRAVENO** — řadí podle `saved_at` |
| **9** | **Mrtvá větev** `cli.py:381-383` („starší snapshoty zatím bez měřitelného výsledku") — `evaluate` vracelo None jen při shodném otisku, který je odfiltrovaný o dva řádky výš. | ✅ odpadlo s evaluate v2 |
| **10** | Drobnosti: `en.community or self.model.c_mean` (`recommend.py:426,433`) by tiše nahradilo `community == 0.0`; `_is_side_content` (`enrich.py:91-116`) aplikuje formátové pravidlo i na standalone tituly, ačkoli komentář u `SIDE_FORMATS` slibuje jen skupiny k>1 (podmínku dnes drží až volající); `season.py` neaplikuje `min_community` a jeho `taste_fit` nemá `cluster_fit`, takže „shoda s vkusem" znamená v sezónním a globálním reportu jiné číslo. | otevřené (krok 7) |

### 9c.2 Co ukázaly časové řezy (`my_finish_date`)

Nový měřicí nástroj revize: fit na titulech dokončených před T, test na
pozdějších. Dává **10–20× víc** prospektivních bodů než samotné zásahy
doporučení a je to jediný způsob, jak rozhodnout strukturální volby.

- **Afinita mimo čas řadí slabší, než slibuje CV:** test Spearman
  0,36/0,26/0,16/0,20 proti CV 0,34/0,40/0,39/0,44. Rozdíl ale táhne hlavně
  **poslední okno** (od 2026-05, harémové maratony): stratifikovaně po
  disjunktních oknech 0,39, bez posledního okna **0,56**. Období 2026-07..09
  je konfundované a jako důkaz se nedá brát samo o sobě.
- **Vnořené řezy nejsou nezávislá potvrzení** (161 ⊃ 128 ⊃ 115 ⊃ 76).
  Kritéria typu „CI nad nulou na 3 ze 4 řezů" počítají fakticky tentýž test
  čtyřikrát. Disjunktní okna mají n = 33/13/39/76 a bootstrap musí být
  klastrovaný po franšízách (CI o 30–50 % širší než iid).
- **OOF metriky řadí strukturální volby obráceně než čas:** Kendall tau mezi
  pořadím konfigurací podle CV a podle času je **−0,47 až −0,87**, a to pro
  RMSE i pro Spearman. O `min_lift`, K a párech se tedy podle CV rozhodovat
  nemá — a rovnou z toho plyne i zamítnutí `cv_spearman` jako „lepší" metriky.
- **Únik přes nedatované tituly je zanedbatelný:** ze 148 hodnocených bez
  data má premiéru po řezu 2026-01-01 **jediný** (a po 2026-03-01 žádný).
  Filtr podle data premiéry je levná pojistka, ne nutnost.
- **Otázka párů zůstává otevřená.** Čistý efekt vypnutí párů (proti téže
  konfiguraci bez trojic) je +0,040 [−0,03; +0,12] pooled; celý sedí
  v **pokračováních** (+0,21 na 37 titulech), na nových franšízách je −0,006.
  Default se proto nemění a rozhodne se to až na nekonfundovaných oknech.

### 9c.3 Změřená zamítnutí (neopakovat)

| co | čím to padlo |
|---|---|
| **Trojice** (`interaction_triples`) | `s₃` skáče 0–0,45 podle zamíchání CV a na plných datech je 0 (žebříček tedy beze změny), fit je s nimi ~5,5 s pomalejší; mimo čas pooled +0,035 [−0,01; +0,09]. **Vypnuto v `config.yaml`**, kód zůstává |
| **K = 4** | CV RMSE ho preferuje (0,908 vs 0,914), časový split ho zamítá: Spearman **−0,042** [−0,08; −0,00] na 4 ze 4 řezů. K a počet párů jsou navíc svázané přes práh na **smrštěném** liftu (`taste.py:332-333`): K4 pustí 33–45 párů, K8 10–19, K16 jen 1–2 — každé srovnání K je zároveň srovnáním párů |
| **K podle kategorie, kvadratická/kubická baseline, backfitting** | grouped CV 0,9116 (proti 0,9142) / 0,9456→0,9570→0,9625 / 0,9059 při **horším** Spearmanu 0,280 |
| **Recency vážení podle stáří hodnocení** | zisk je artefakt: nenormalizované vážení sníží sumu vah, tím zesílí smrštění `n/(n+K)` a ořízne efekty pod `min_attr_count`. Kontrola „uniform" (stejná suma vah, žádná informace o čase) reprodukuje **celý** zisk (0,372 vs 0,363); normalizovaná recency dává +0,02 s CI přes nulu |
| **Únava po náročných titulech** | korelace rezidua s intenzitou 3 předchozích titulů 0,001 [−0,11; 0,10]; autokorelace intenzity po vyloučení dvojic téže franšízy 0,016 |
| **Report posunu vkusu s permutační nulou** | síla **3 %** pro posun −0,5 u atributu s n_eff ≥ 10 (spolehlivě detekovatelný je až posun o celý bod, kterého by sis všiml sám); gate by skoro vždy mlčel |
| **Asymetrický interval predikce z OOF kvantilů** | mimo čas ±`cv_rmse` pokrývá 0,68–0,78 při nominálních 0,68, kvantilový interval by nadpokrýval ještě víc. Původní „pokrytí 0,63" míchalo chyby ze seskupené OOF s `cv_rmse` z náhodné CV. Bias patří řešit u zdroje (nález #4) |
| **Výběr počtu nálad podle stability (ARI)** | žádný měřitelný dopad (OOF Spearman ±0,01, top-40 39/40) za +3,3 s na fit; silueta navíc **není** na úrovni šumu (permutovaný null 0,022–0,030 proti 0,071–0,089). Kdo chce stabilní jména nálad, má `n_clusters: 4` |
| **`cv_spearman` jako rozhodovací metrika** | pořadí konfigurací mimo čas předpovídá obráceně (tau −0,47 až −0,87), stejně jako RMSE |
| **Ladění vah kompozitu na retro-poolu** | rozlišení nestačí: v retro-poolu je jen 44–45 % pozdějších shlédnutí, top-100 je 27 % poolu (náhodné pořadí dá 16–18 zásahů z 51–60) a marginály mění mezi řezy znaménko. Silná původní čísla navíc pocházela z poolu s **dnešním** user-CF |
| **Franšízový efekt v predikované známce pokračování** | na vlastním kritériu (−0,03 RMSE na 3 ze 4 řezů) neuspěl: −0,02/−0,02/0,00/+0,01. Zůstává nanejvýš jako vysvětlující údaj v kartě |

### 9c.4 Chyby v metodice měření (poučení pro příští experimenty)

Revize našla víc chyb v *měření* než v samotném kódu. Stojí za zapsání,
protože se budou opakovat:

1. **Selection leak ve franšízovém backtestu.** Detekce pokračování běžela
   nad `watched ∪ cíle ∪ top-400`, takže cíl (později shlédnutý titul) byl
   rozpoznán kdekoli v poolu, zatímco nezkonvertované pokračování jen
   v top-400. Konverze vyšla 78–81 %; po opravě **38–43 %**, medián pořadí
   #291 → **#1008**, AUC kompozitu uvnitř sekce 0,29 → **0,69**. Celá teze
   „pokračování mají nejvyšší úspěšnost a kompozit je řadí špatně" na tom stála.
2. **Srovnání proti produkci míchá dva zásahy.** „Páry OFF" se měřilo proti
   konfiguraci *s trojicemi*; čistý efekt párů je pak podstatně menší.
3. **Retro-pool s dnešním user-CF.** Pooly 4060/3963 obsahovaly 4046 položek
   ze současné senpai cache a jen ~370 z rec grafů, takže „citlivost `w_cf`"
   měřila hlavně poměr zdrojů kandidátů v poolu, ne váhu.
4. **Tautologické kritérium.** Rozdíl reziduální a surové delty je identicky
   `β·(rozdíl průměrné komunity)`, takže test „liší se aspoň o 0,3" projde vždy.
5. **Interval jen z kontrolní skupiny.** Bootstrap přes kontrolu při n_rec = 1
   dává pokrytí 0,36–0,51 — strana doporučení s jedním bodem rozptyl nenese.
6. **Offline pooly nemají discovery větev.** `search_by_tags` se záměrně
   necachuje, takže každá offline rekonstrukce poolu tag-search postrádá.
   Závěry o top-100 to nemění, absolutní velikosti poolů ano.

### 9c.5 Historie hodnocení: co z ní jde vytěžit

Data dnes: 3 snapshoty (7 týdnů), 23 nových shlédnutí, z toho **2 z doporučení
a obě z jedné franšízy**. Většina nových shlédnutí jsou pokračování a vedlejší
obsah rozjetých franšíz, které v poolu doporučení vůbec nejsou.

Simulace síly (franšízové dvojice, pozorovaná konverze) říká, na co čekat:

| za | zásahů doporučení | síla pro skutečnou Δ = +0,5 | pro Δ = +1,0 |
|---|---|---|---|
| 6 měsíců | ~4–7 | 0,31–0,33 | 0,58–0,71 |
| 12 měsíců | ~7–15 | 0,35–0,50 | 0,72–0,93 |

Z toho plyne pořadí priorit: **hustota dat je důležitější než chytřejší
metrika**. Nejhustší zpětná vazba nejsou zásahy doporučení (~1 měsíčně), ale
**porovnání uložené predikce se skutečnou známkou u všech nově shlédnutých
titulů** (~13 měsíčně). Proto snapshot v2 ukládá log predikcí i mimo pool.

Implementováno v tomto kole:

- **Snapshot v2** (`history.py`): stav celého seznamu (`status`, známka,
  `finish_date`), `export_date` z mtime exportu, `u_mean`/`c_mean` a
  z-parametry složek kompozitu, celý pool se složkami (`affinity` a
  `cluster_fit` zvlášť, aby šlo přeladit i `cluster_fit_weight`) a log
  predikcí mimo pool. Pool a log jdou do vedlejšího `{n}_{otisk}.pool.json`
  bez odsazení. **6 desetinných míst** je nutnost: se 4 se přepočtené pořadí
  celého poolu liší u 374 kandidátů.
- **Log predikcí** (`cli._prediction_log`): PTW + neviděné díly franšíz do
  2 kroků od viděných a PTW titulů. Změřené pokrytí 23 nových shlédnutí:
  pool + PTW 5/23, +1 krok 14/23, **+2 kroky 17/23**.
- **evaluate v2** (`history.evaluate_history`): ledger událostí — každé nové
  shlédnutí jednou, přiřazené oknu mezi snapshoty, s první expozicí. Nahrazuje
  blok za každý snapshot (tentýž titul se dřív počítal vícekrát; dnes 4 proti
  2 skutečným událostem). Metriky: počty po oknech a kohortách místo sazeb,
  známky **po franšízách** surově i vůči baseline z komunity s 95 % intervalem
  ze sd celého seznamu, kontrola i bez pokračování rozjetých franšíz, kbelíky
  pořadí až od 3 titulů, a prospektivní porovnání predikce se známkou zvlášť
  pro nové franšízy a pokračování.

Co se vědomě **nezavedlo**: smrštěná delta (smrštění k nule by „málo dat"
ukázalo jako „žádný efekt"), stavový automat trychtýře R→P→W→C (za 7 týdnů
jediný přechod R→P; užitečná část je jedno číslo v řádku okna), sazby
konverze za 30 dní (potřebují `export_date` a ≥ 15 událostí v kohortě),
percentil v uloženém poolu (dává smysl až s poolem v historii a ~20
událostmi) a penalizace za opakované ignorování doporučení.

### 9c.6 Verdikty 31 návrhů

Po nezávislém přeměření: **4 keep, 21 revise, 6 drop**. Vybrané (plný seznam
s čísly je v příloze revize):

| id | návrh | verdikt |
|---|---|---|
| P01 | snapshot v2 (stav seznamu, pool, z-parametry) | keep (upraveno: celý pool, 6 míst, mtime místo max finish_date) |
| P02 | časová validace na `my_finish_date` | revise — jako vývojářský skript, disjunktní okna, cluster bootstrap |
| P03 | ukončit experiment s trojicemi | keep |
| P04 | grouped CV podle `series_root` | keep (spolu s centrováním, krok 5) |
| P05 | retro-pool a ladění vah | revise — jen jako nástroj, váhy neměnit |
| P06–P09, P15–P19, P26 | metriky historie | revise — konsolidováno do jednoho evaluate v2 |
| P10 | `cv_spearman` jako metrika | **drop** |
| P11 | sekce pokračování s vlastním řazením | revise — jen vyřadit z „nových objevů" |
| P12 | jedna karta na franšízu | revise — bez přepočtu skóre, se štítkem „začni od" |
| P13 + P14 | senpai na reziduích + user-CF jako reziduum | keep — jen společně a po deduplikaci |
| P20–P23, P27 | páry, franšízový efekt, K, opakovaná CV | revise/odloženo |
| P24 | sezónní view | revise — jen `min_community` a sdílený `taste_fit` |
| P25 | HTML historie | revise — odloženo, minimální rozsah |
| P28, P29, P30, P31 | stabilita k, interval, drift, recency | **drop** (P31 = zamítnutí zapsat) |

### 9c.7 Plán a stav

| krok | co | stav |
|---|---|---|
| **1** | `interaction_triples: false`; změřená zamítnutí do dokumentace | ✅ hotovo |
| **2** | deduplikace AniList userlistů | ✅ hotovo |
| **3** | snapshot v2 + log predikcí | ✅ hotovo |
| **4** | evaluate v2 (ledger, franšízové intervaly, kohorty, kontrola predikcí) | ✅ hotovo |
| **5** | časový harness (`backtest.py`, `--backtest`) → centrování afinity + grouped CV + R=3 zamíchání, jednou změnou kalibrace | ✅ hotovo |
| **6** | senpai na reziduích + user-CF jako smrštěné reziduum | ✅ hotovo |
| **7** | jeden průchod seskupením franšíz: karta na franšízu, pokračování mimo „nové objevy", sekce „Z tvého PTW", `cluster_fit` se smrštěním, sezónní view | ✅ hotovo |
| **8** | až s daty (≥ 6 měsíců): páry × franšízový efekt, percentil v poolu, sazby kohort, HTML historie, případné ladění vah | otevřené |

### 9c.8 Co ukázal první běh po implementaci (2026-09-16)

Ověřeno offline nad reálnou cache; testů 258 → **292**, všechny zelené.

**Kalibrace (krok 5).** Nekorigovaná afinita měla na tréninku posun
**+0,625** — přesně ta chybějící konstanta z nálezu #4. Po vycentrování:

| veličina | před | po |
|---|---|---|
| `scale` | 0,30 | **0,35** |
| CV RMSE | 0,8917 (náhodné foldy) | **0,9104** (po franšízách, poctivější) |
| průměr predikce in-sample | nadsazený | **8,04** proti skutečnému průměru 8,07 |
| bias na časových oknech | −0,25 | **−0,03** |

Časová validace (4 disjunktní okna, bootstrap po franšízách) navíc obrátila
dřívější závěr „model je horší než baseline": pooled RMSE **0,808 → 0,757**,
Spearman **+0,348 [+0,17; +0,50]**. Tři ze čtyř oken mají Spearman 0,43–0,55;
poslední (od 2026-06-27, harémové maratony) −0,01 — potvrzuje, že to období
je konfundované a jako důkaz se brát nedá.

**User-CF (krok 6).** Konstantních hodnotitelů mezi vybranými senpai
**16/20 → 0/20**, podobnost vybraných je teď 0,17–0,28 (reziduální, tedy
nižší čísla než dřívější 0,43 vůči komunitě). Korelace user-CF složky
s komunitním skóre **0,86–0,93 → −0,001**: komunita vstupuje do kompozitu
jen jednou.

**Franšízové pohledy (krok 7).** Přehled má 40 objevů, 15 titulů z PTW a
26 pokračování rozjetých sérií ve vlastních sekcích. Mid-franchise sequelů
zbyl v objevech **1 ze 40** (se štítkem „začni od: Little Busters!"), zbytek
se sbalil pod lépe hodnocený díl téže franšízy.

Řazení se změnilo znatelně (shoda s předchozím modelem: top-40 27/40,
top-100 79/100), což odpovídá tomu, že se současně vyměnili senpai, změnil
tvar `cluster_fit` a povyrostla `scale`. Jestli je to zlepšení, rozhodne až
historie — proto ledger a log predikcí (kroky 3 a 4) běží první.

---

## 9d. Proč doporučení odporují vkusu (2026-09-24)

*Kolo 4. Podnět: top-3 žebříčku z 2026-09-16 — **Tokyo Revengers,
Higurashi no Naku Koro ni, Death Note** — jde proti tomu, co o svém vkusu
víš (silná romantická linka ≈ +1 ke známce, hodně akce odpuzuje, temný a
ošklivý obsah taky) i proti tomu, co navrhují LLM. Rozbor offline nad cache
a nad uloženým poolem snapshotu `00487_10d03f8ccdeb` (6 615 kandidátů), bez
sítě a bez zásahu do kódu. Offline přestavěný model sedí s během (`scale`
0,35, CV RMSE 0,9104). Nic z návrhů není implementované.*

**Závěr v jedné větě:** nejde o jednu chybu, ale o čtyři naskládané
mechanismy — graf podobnosti má kvůli normalizaci několikanásobnou váhu,
jeho seedy se mezi devítkami vybírají abecedně, model vkusu strukturálně
ředí nejsilnější preferenci (romantiku) a o temném obsahu nejsou v seznamu
data. Testy (292/292) nic z toho nezachytí — nic nepadá, jen se špatně váží.

### 9d.1 Rozpad top-3

Vážená z-skóre složek kompozitu (přepočteno z uložených `z_params`):

| titul | vkus | graf | user-CF | kvalita | kompozit |
|---|---|---|---|---|---|
| Tokyo Revengers | +3,08 | **+5,54** | +0,67 | +0,36 | 9,65 |
| Higurashi no Naku Koro ni | +2,28 | **+4,95** | +1,91 | +0,39 | 9,52 |
| Death Note | +2,15 | **+5,27** | +1,06 | +0,75 | 9,23 |

Hlasy z grafu přicházejí u všech tří jen ze tří až čtyř seedů:

| titul | hlasy | z toho | podle zdroje |
|---|---|---|---|
| Tokyo Revengers | 71,4 | Steins;Gate 30,9 · Boku dake ga Inai Machi 26,0 · S;G 0 14,6 | AniList 49,3 · MAL 20,9 · Shikimori 1,2 |
| Higurashi | 45,4 | S;G 25,4 · S;G 0 10,7 · Boku dake 9,3 | AniList 34,4 · MAL 10,7 · Shikimori 0,4 |
| Death Note | 58,4 | S;G 26,0 · Boku dake 18,3 · Bakuman 14,2 | AniList 31,9 · MAL 25,5 · Shikimori 1,1 |

### 9d.2 Nálezy

| # | nález | návrh | stav |
|---|---|---|---|
| **1** | **Graf podobnosti má kvůli normalizaci několikanásobnou váhu — VYSOKÁ.** Z-skóre složek se počítají přes celý pool (`recommend.py:491-494`). S user-CF má pool 6 615 titulů, hlas z grafu ale jen **399** z nich (6 108 přinesli jen senpai). `log1p(hlasy)` má proto průměr 0,134 a sd 0,599 a **jakýkoli** titul z grafu dostane +3,5 až +6 bodů, zatímco celá složka vkusu má sd 1. Graf tak funguje jako brána: **40/40** nových objevů má hlasy z grafu a tituly s nejlepší shodou s vkusem mimo graf se nahoru nedostanou (Kanon 2006 #110, Air #61, Summer Pockets #174). Efektivní síla `w_cf: 0.8` tak závisí na tom, kolik user-CF-only titulů je v poolu — se zapnutým user-CF se znásobí, aniž se váha změnila. | Parametry z-skóre počítat z **obsahového poolu** (graf + tag-search), ne z celého. Zvážit ořez \|z\| ≤ 2 nebo percentilovou normalizaci, aby žádná složka nemohla fungovat jako brána. Efektivní váhy kompozitu pak nezávisí na tom, jestli je user-CF zapnuté. | ✅ **OPRAVENO** 2026-09-24 — parametry vkusu, grafu a kvality z obsahového poolu, user-CF přes celý pool; ořez \|z\| nebyl potřeba (graf −1,3…+2,1, vkus ±3,3, kvalita ±2,4) |
| **2** | **Seedy se mezi devítkami vybírají abecedně — VYSOKÁ.** `_seeds` (`recommend.py:167-168`) řadí jen podle známky, remízy rozhoduje pořadí exportu, a to je abecední. 30 desítek (po limitu 2 na franšízu) + prvních **10 ze 134 devítek**: 2.5-jigen no Ririsa … Bakuman., Boku dake ga Inai Machi, Boku no Kokoro no Yabai Yatsu. Clannad, Kimi ni Todoke, Fruits Basket, White Album 2, Plastic Memories ani Oregairu seedem nejsou nikdy. Dva ze čtyř seedů, které přivedly top-3, jsou tam **jen díky písmenu B**. Váha seedu `score − ū + 1` navíc nerozlišuje, *proč* titul máš v oblibě: Steins;Gate (komunita 9,07) má stejnou váhu jako Domestic na Kanojo, ač reziduum má +1,19 proti +2,40. | Řadit seedy podle **rezidua** (`model.residuals()`, o kolik víc se ti titul líbil, než čeká baseline), sekundárně podle známky. Top-40 podle rezidua: Domestic na Kanojo, Yuragi-sou, Hige wo Soru, Koi wa Ameagari, Russia-go Alya-san, 5-toubun … Boku dake (reziduum +0,57) a Bakuman vypadnou. **Nesimulováno** — nové seedy nemají rec graf v cache. K ověření navíc: normalizovat příspěvek seedu jeho celkovým objemem hlasů, ať populární seed (Steins;Gate: hrany se stovkami hlasů) nepřehlasuje romantické seedy, jejichž doporučení se rozptylují do méně známých titulů. | ✅ **OPRAVENO** 2026-09-24 — řazení (reziduum, známka, mal_id); normalizace objemem hlasů seedu otevřená |
| **3** | **Součet marginálních průměrů ředí časté atributy a netrestá absenci — VYSOKÁ.** Efekt = smrštěný průměr rezidua titulů, které atribut **mají** (`taste.py:302-303`); titul bez atributu dostane 0. Efekt tak vychází ≈ (1 − p)·Δ, kde p je podíl atributu v seznamu. Romantika (p = 0,61): skutečný kontrast +0,40 bodu (romantika bez akce proti zbytku +0,50), efekt **+0,146**, po `scale` 0,35 **≈ +0,05 bodu**. Akce: skutečně −0,34, v modelu ≈ −0,08. Death Note za absenci romantiky neztratí nic. Součet 27–40 korelovaných tagů navíc nutí CV stáhnout `scale` na 0,35, čímž utlumí i ty skutečné signály — nejhlasitější jsou pak vzácné tagy s extrémním průměrem (Rehabilitation +0,44, Age Gap +0,40, Alternate Universe +0,32). U Death Note se sčítají tropy romantických dramat, které jsou tam okrajové (Unrequited Love, Yandere, Kuudere, Suicide, Amnesia). Pár **Psychological + Supernatural (+0,34)** stojí na Bakemonogatari, Nekomonogatari, Bunny Girl Senpai, Yofukashi no Uta a Fruits Basket — romantických příbězích s nadpřirozeným twistem. Zásluhu romantiky, kterou model neumí přiznat jí samotné, si připíše tahle dvojice a přenese ji na Death Note i Higurashi. | **Ridge regrese nad centrovanými atributy** (cíl zůstává reziduum, alpha ze CV, ~60). V grouped CV je stejně přesná (RMSE **0,9106** proti 0,9104, Spearman 0,416 proti 0,418), ale strukturu má rozumnou: nahoře široké rysy (TV, Male Protagonist, Light novel, Heterosexual, Romance), dole Episodic a **Action**; Rehabilitation +0,10 místo +0,44, Decouzon +0,06 místo +0,30. Na kandidátech spadne afinita Higurashi ze **110. na 652.** místo z 907, Death Note ze **133. na 607.** CV to nerozliší, protože měří jen na tvém seznamu, kde temné tituly nejsou. Podle §9c.2 se o struktuře rozhoduje **časovými okny** (`--backtest`), ne CV — tam ji ověřit před zavedením. Páry po zavedení přeměřit (část z nich byla jen zástupce utlumených singlů). | ✅ **OPRAVENO** 2026-09-24 — `model.effect_model: ridge` (default), α = 60; rozhodnuto backtestem, §9d.7 |
| **4** | **AniList žánry se slučují binárně — STŘEDNÍ.** `build_attributes` přidává AniList žánry bezpodmínečně s vahou 1,0 (`attributes.py:208-209`). MAL u Tokyo Revengers romantiku nemá (Action, Drama; Delinquents, Time Travel), AniList ano — a model ji pak počítá stejně jako u Toradory. Pro zdroj, formát a dekádu už AniList slouží jen jako fallback, u žánrů ne. | Žánry **primárně z MAL**, AniList jen když MAL žádné nemá. Změřeno spolu s #5: CV RMSE 0,9104 → **0,9063** (v šumu, ale ne hůř). | ✅ **OPRAVENO** 2026-09-24 |
| **5** | **„Script" zahrnuje lokalizaci — STŘEDNÍ.** `WRITER_POSITIONS` obsahuje `script` (`attributes.py:83-84`) a MAL pod ním vede i překladatele titulků/dabingu. „Writer: Decouzon, Mélanie" (+0,30, druhý největší kladný příspěvek u Tokyo Revengers) je francouzská lokalizace u Tokyo Revengers i Sakamichi no Apollon; podobně „Mattos, Sidney". Všech 30 staff efektů má n_eff 4–8, fakticky tedy kódují identitu jedné franšízy. Staff je zapnutý jen v `config.yaml` (default vypnuto). | Vyřadit `script` z `WRITER_POSITIONS` (hlavního scenáristu kryje `series composition`), nebo v `config.yaml` vrátit `include_staff: false`. | ✅ **OPRAVENO** 2026-09-24 — `script` vyřazen; varianta „Script jen bez Series Composition" zamítnuta (lokalizace prosakovala dál: Salva, Stocker 4×, Decouzon 2×) |
| **6** | **`min_attr_count` zahazuje vzácnou negativní evidenci — STŘEDNÍ.** Atribut pod prahem 4 (`taste.py:300`) se vyřadí, místo aby ho jen smrštilo `n/(n+K)`. Horor máš u 2 titulů (Highschool of the Dead, Shinsekai yori; rezidua −0,82 a −1,40), efekt Horror je proto 0, tedy „neutrální" — a Higurashi nese žánr Horror. Totéž Delinquents u Tokyo Revengers (AniList rank 95). | Snížit práh na ~1,5–2 a nechat malé vzorky na smrštění. Pozor na vazbu z §9c.3: práh interaguje se sumou vah, takže změnu ověřit časovými okny. | ✅ **OPRAVENO** 2026-09-24 — práh 1,5 (i v `config.yaml`), rozhodnuto backtestem, §9d.7 |
| **7** | **Chybí negativní evidence (selection bias) — VYSOKÁ, strukturální.** Tituly, kterým se vyhýbáš, v seznamu nejsou; dropnuté máš 2, obě bez známky. Model se tak nemá odkud naučit, že temný obsah vadí: co nezná, bere jako neutrální, a graf s kvalitou to pak vytáhnou nahoru. Tohle je rozdíl proti LLM, které tvé výslovné „ne temné, ne moc akce" použije jako pevné pravidlo. | Viz §9d.4 (populární tituly mimo seznam jako slabý negativní signál); jako doplněk explicitní averze v configu (horor, gore, delikventi) — jde proti principu „žádné ruční seznamy atributů", je to vědomé rozhodnutí. | 🟡 **ČÁSTEČNĚ** 2026-09-24 — model výběru + sekce „znáš, ale nemáš v plánu" (§9d.7); explicitní averze v configu k rozhodnutí |
| **8** | **Osa náročnosti do řazení nevstupuje a míchá smutné s ošklivým — STŘEDNÍ.** `intensity_of` se používá jen pro popis nálad. Lexikon dává Tragedy +1,0 stejně jako Body Horror a Torture, přitom tragédie ti sedí (reziduum +0,18). | Rozdělit na dvě osy — emoční tíha (tragédie, tearjerker) a pochmurnost/brutalita (horor, gore, body horror) — a teprve tu druhou zvážit jako penalizaci. Vyžaduje druhý sloupec v `intensity.yaml`. | otevřené |
| **9** | **User-CF je dnes převážně šum — STŘEDNÍ.** Senpai mají reziduální podobnost r = 0,17–0,28 a jejich špička jsou mainstreamové akční tituly (Gintama, Solo Leveling, Shingeki no Kyojin, Dr. Stone). Do kompozitu to přidává ±2 body (Higurashi +1,9) a hlavně nafukuje pool, čímž spouští #1. | Dokud historie neukáže přínos: `w_user_cf: 0`, nebo user-CF-only kandidáty nedávat do poolu kompozitu (CF report ponechat). | otevřené |
| **10** | **„Proč:" míchá důvody se srážkami — NÍZKÁ, ale matoucí.** Karta ukazuje top-6 příspěvků podle absolutní hodnoty, záporné jen červeně (`report.py:385-390`). „Action + Drama" (−0,31) a „Action" u Tokyo Revengers jsou srážky, ne důvody. | Dva řádky: „Pro:" a „Proti:". | ✅ **OPRAVENO** 2026-09-24 — `report._why_html`, karta nese 12 příspěvků (`WHY_KEEP`), ukazuje max. 5 „Pro" a 3 „Proti" |

Menší odchylky `config.yaml` od example, které se na výsledku podílejí:
`include_staff: true` (#5), `use_user_cf: true` s agresivním nastavením
(300 seedů × 1000 sledujících, pool 2000, překryv ≥ 200; #1, #9),
`use_shikimori: true` (data odvozená z MAL, takže nejde o nezávislý hlas; na
top-3 ale jen 0,4–1,2 z 45–71 hlasů) a `min_community: 6.0` místo 6,5.

### 9d.3 Co by opravy změnily (what-if nad uloženým poolem)

Počítáno nad 507 kandidáty z obsahových větví (+400 user-CF-only s nejlepší
shodou s vkusem, obohaceno z cache bez sítě). Seedy podle rezidua (#2)
simulovat nešlo. Pořadí ve variantách je bez sbalení franšíz do jedné
karty (druhé díly jsou z top-10 vyškrtané ručně), u hlubších pozic se tedy
může o pár míst lišit.

| varianta | top-10 nových objevů (bez PTW) | Tokyo Rev. | Higurashi | Death Note |
|---|---|---|---|---|
| **dnes** (report) | Tokyo Revengers, Higurashi, Death Note, Charlotte, Koi to Yobu ni wa Kimochi Warui, Ano Natsu de Matteru, Kimi ga Nozomu Eien, Sakamichi no Apollon, Koi to Uso, Osananajimi ga Zettai ni Makenai Love Comedy | 1 | 2 | 3 |
| z-skóre jen přes obsahový pool (#1) | Higurashi, Tokyo Revengers, Kimi ga Nozomu Eien, Death Note, Sakamichi no Apollon, Koi to Yobu…, Charlotte, Tantei wa Mou Shindeiru, Gosick, Ano Natsu de Matteru | 2 | 1 | 4 |
| + ridge afinita, bez user-CF (#1, #3, #9) | Tokyo Revengers, Sakamichi no Apollon, Gosick, Charlotte, Kiznaiver, Ano Natsu de Matteru, Summertime Render, Taishou Otome Otogibanashi, Kimi ga Nozomu Eien, Given | 1 | >20 | >20 |
| + žánry z MAL, bez staff (#4, #5), `w_cf` 0,8 | Sakamichi no Apollon, Ano Natsu de Matteru, Gosick, Tokyo Revengers, Charlotte, Summertime Render, Kimi ga Nozomu Eien, Koi to Uso, Cross Game, Kiznaiver | 4 | 73 | 39 |
| totéž, `w_cf` 0,4 | Gosick, Sakamichi no Apollon, Ano Natsu de Matteru, Kimi ga Nozomu Eien, Cross Game, Hachimitsu to Clover II, ef: A Tale of Memories., Tokyo Revengers, Kiznaiver, Taishou Otome Otogibanashi | 8 | 112 | 59 |

Poučení z tabulky:

- **Samotná oprava normalizace (#1) nestačí** — mění poměr sil, ne obsah
  grafu. Higurashi a Death Note padají až s ridge (#3), definitivně se
  žánry z MAL (#4).
- **Tokyo Revengers zůstává kolem 4.–8. místa** i po všech opravách: má
  AniList tag Love Triangle (rank 73), dramatickou linku, cestování časem a
  silný graf ze Steins;Gate. Pomůže oprava seedů (odpadne 26 hlasů z Boku
  dake), úplně dolů ho ale dostane jen signál o averzi k akci/delikventům,
  a ten v ohodnoceném seznamu není (#6, #7). Změřeně ho dostane dolů signál
  vyhýbání z §9d.4 (#4 → #25).
- Tagy neříkají, **kolik** titulu tvoří romantika — Tokyo Revengers a
  Toradora mají obě „Romance" i „Love Triangle". To je strop obsahového
  modelu nad tagy a důvod, proč LLM, která ví, o čem příběh *je*, radí jinak.

### 9d.4 Nápad: populární tituly mimo seznam jako slabý negativní signál

*Podnět uživatele (2026-09-24).* Titul s vysokým MAL skóre, který nemáš
shlédnutý ani na PTW, skoro jistě znáš — a vyhýbáš se mu vědomě kvůli popisu
nebo tagům. Je to přesně ta chybějící negativní evidence z nálezu #7:
předvýběr, který export neobsahuje, se dá zčásti rekonstruovat z toho, **co
v něm chybí**. V literatuře je to model výběru / expozice (implicitní
zpětná vazba, „missing not at random") — druhá, na modelu vkusu nezávislá
otázka: model vkusu odhaduje *jakou známku dáš, když to uvidíš*, tohle
*jestli po tom vůbec sáhneš*.

**Měření (offline, cache, bez sítě).** Cache pokrývá **100 %** top-1000 a
**99 %** top-2000 MAL titulů podle popularity (počtu členů), takže vesmír je
úplný.

- **Vesmír:** top-2000 podle popularity, formát TV/Movie/ONA, premiéra
  ≤ 2025, **jen první díly franšíz** (bez prequelu a parent story) — jinak
  by jedna přeskočená franšíza hlasovala tolikrát, kolik má řad. Celkem
  **1 198** titulů.
- **Popisek:** „zájem" = franšízu máš v seznamu v jakémkoli stavu včetně PTW
  (kořen z union-findu). Zájem má **263** (22 %).

Zájem silně klesá s popularitou, takže u méně známých titulů je absence
slabší důkaz:

| popularita | 1–250 | 251–500 | 501–1000 | 1001–2000 |
|---|---|---|---|---|
| podíl se zájmem | 45 % | 34 % | 24 % | 11 % |

Atributy s nejsilnějším vyhýbáním (smrštěné log-odds proti průměru):
**Primarily Male Cast** (zájem 3 % z n=120), Organized Crime 0 %, Police 2 %,
**Thriller 5 %**, **Horror 6 %**, Crime 6 %, Post-Apocalyptic 6 %, Cyberpunk,
**Suspense 8 %**, Super Robot, Idol, Space, Historical, **Mystery 10 %**,
Sports; z formátů ONA (0 z 30) a zdroj Game (0 z 28). Nejvyšší zájem: Harem
52 %, Cohabitation, Ecchi, Female Harem, Nudity, Love Triangle, **Romance
39 %** (n=500), Heterosexual, Unrequited Love. Vyhýbání tedy zachytí přesně
to, co model vkusu kvůli chybějícím datům vidět nemůže (#6, #7).

Naučitelnost — logistická regrese s L2 nad týmiž atributy + log(členů)
jako kovariáta povědomí, 5 foldů po franšízách:

| prediktor | CV AUC |
|---|---|
| jen popularita | 0,683 |
| jen atributy | 0,852 |
| **atributy + popularita** (C = 0,1) | **0,874** |

**Časová kontrola:** model natrénovaný na stavu ze snapshotu 2026-07-28
řadí 8 franšíz, které od té doby přibyly do seznamu nebo PTW, s **AUC 0,946**
proti 0,781 u samotné popularity (medián pořadí ~30 z 943). Pozor: 8
událostí je málo a skoro všechny spadají do konfundovaného období
harémových maratonů (§9c.2) — Kore wa Zombie, Nazo no Kanojo X, Asterisk,
Kaifuku Jutsushi… — takže číslo je spíš horní odhad.

Pravděpodobnost zájmu (out-of-fold) u dnešní top-3 a pro srovnání:
Higurashi **0,05**, Tokyo Revengers **0,20**, Death Note **0,22** —
proti Horimiya 0,79, Kimi ni Todoke 0,77, Kaguya-sama 0,64, Kanon (2006)
0,41, Air 0,40.

**Dopad na žebříček** (nad variantou ridge + z-skóre z obsahového poolu +
`w_cf` 0,4 z §9d.3; složka = z-skóre atributové části logitu, **bez**
popularity):

| varianta | Tokyo Rev. | Death Note | Higurashi | top-5 |
|---|---|---|---|---|
| bez signálu | 4 | 59 | 107 | Gosick, Sakamichi no Apollon, Kiznaiver, Tokyo Revengers, Taishou Otome |
| + atributová složka, váha 0,5 | **25** | 106 | 203 | Gosick, Taishou Otome, Kiznaiver, Kimi ga Nozomu Eien, Sakamichi no Apollon |
| + atributová 0,5 + srážka titulu 0,5·povědomí | 37 | 134 | 238 | Taishou Otome, Kimi ga Nozomu Eien, ef, Watari-kun no xx ga Houkai, HachiKuro II |

Korelace atributové složky s ridge afinitou na obsahovém poolu je **0,68** —
částečný překryv, ne duplikát.

**Návrh.**

1. **Atributová složka „výběr"** do kompozitu (`w_select` ~0,5): logistická
   regrese nad populárními prvními díly franšíz, do kompozitu jen atributová
   část logitu (popularita je povědomí, ne preference). V kartě jako důvod
   „Proti: Primarily Male Cast — takovým titulům se vyhýbáš".
2. **Srážku za konkrétní titul zatím ne.** Každý kandidát v „nových
   objevech" je z definice mimo seznam i PTW, takže srážka podle povědomí
   se rovná srážce za popularitu. Postihne i tituly, na které jen
   ještě nedošlo — z top-5 vypadly Gosick a Sakamichi no Apollon. Lepší
   varianta k rozhodnutí: populární přeskočené tituly nesrážet, ale ukázat
   ve vlastní malé sekci „Znáš, ale nemáš v plánu — opravdu ne?".
3. **Sjednotit s historií:** titul, který se ukázal mezi doporučeními a do
   PTW nepřibyl, má povědomí ≈ 1 bez ohledu na popularitu. §9c.5 penalizaci
   za ignorovaná doporučení vědomě odložila; tady na ni jde navázat stejným
   mechanismem.

**Rizika a podmínky.**

- **Bublina:** výběr silně táhne k harému/ecchi (zájem 45–52 %), přičemž
  model vkusu u nich vidí jen mírně kladné reziduum. Proto jen střední váha
  a vysvětlení na kartě.
- **Povědomí:** vesmír omezit na tituly s premiérou aspoň ~6 měsíců zpět
  a popularitu nechat v modelu jako kovariátu, jinak se „neznám" splete
  s „nechci".
- **Náklady:** dnes je vesmír v cache hlavně díky user-CF poolu. Bez
  user-CF (krok 9 v §9d.5) je potřeba jednorázově stáhnout
  `top/anime?filter=bypopularity` (80 stránek po 25) + `/full` pro chybějící
  tituly; pak jen přírůstky.
- **Validace:** AUC měří, jestli jde výběr předpovědět, ne jestli doporučení
  zlepší. O váze rozhodne až ledger v historii (§9c.5).

### 9d.5 Plán

Pořadí podle poměru dopad/cena; strukturální změny (#3, #6) rozhodnout
časovými okny (`--backtest`), ne CV (§9c.2).

| krok | co | nálezy | cena | stav |
|---|---|---|---|---|
| **1** | parametry z-skóre z obsahového poolu (ořez \|z\| nebyl potřeba) | #1 | řádky | ✅ hotovo |
| **2** | seedy podle rezidua | #2 | řádek | ✅ hotovo |
| **3** | žánry primárně z MAL, `script` pryč z `WRITER_POSITIONS` | #4, #5 | řádky | ✅ hotovo |
| **4** | „Pro:" / „Proti:" v kartě | #10 | řádky | ✅ hotovo |
| **5** | ridge místo součtu marginálních průměrů, přeměřit páry | #3 | den | ✅ hotovo (α = 60) |
| **6** | nižší `min_attr_count` | #6 | řádek | ✅ hotovo (1,5) |
| **7** | atributová složka „výběr" (vyhýbání se populárním titulům) do kompozitu; srážku za titul nahradit sekcí „znáš, ale nemáš v plánu" | #7, §9d.4 | den | ✅ hotovo |
| **8** | dvě osy náročnosti, pochmurnost jako penalizace | #8 | den + revize lexikonu | otevřené |
| **9** | user-CF z kompozitu, dokud historie neukáže přínos | #9 | config | k rozhodnutí |
| **10** | explicitní averze v configu | #7 | řádky | k rozhodnutí (proti principu „žádné ruční seznamy") |

### 9d.6 Co ukázal běh po krocích 1–4 (2026-09-24)

Plný běh s `config.yaml` (user-CF zapnuté), výstup mimo `output/`, bez
zápisu do historie. Testů 292 → **298**, všechny zelené.

**Model.** CV RMSE **0,910 → 0,905** (baseline 0,951), `scale` beze změny
0,35. Žánr Drama má po přechodu na MAL žánry efekt **+0,29** (n = 74) místo
+0,16 (n = 110) — volnější AniList „Drama" ho ředil. Počet nálad podle
siluety klesl z 5 na **4**. Nejsilnější nálada je teď „Drama / Coming of
Age / Romance" (134 titulů, afinita **+0,29**) místo dřívějšího „Slice of
Life / Romance / Coming of Age" (+0,18).

**Seedy.** Mezi seedy už není Steins;Gate (reziduum +1,19), Boku dake ga
Inai Machi ani Bakuman. Hlasy z grafu u Tokyo Revengers klesly **71 → 15**,
u Higurashi **45 → 11** (zbyl jen Steins;Gate 0).

**Žebříček nových objevů** (top-40, shoda se starým 31/40):

| | dřív | teď |
|---|---|---|
| Tokyo Revengers | 1 | 12 |
| Higurashi no Naku Koro ni | 2 | 22 |
| Death Note | 3 | mimo top-40 |
| Kanon (2006) | ~110 v poolu | **18** |
| top-5 | Tokyo Revengers, Higurashi, Death Note, Charlotte, Koi to Yobu… | **Kimi ga Nozomu Eien, Charlotte, Koi to Uso, Koi to Yobu…, ef: A Tale of Memories.** |

Z top-40 vypadly mimo jiné Another, Death Note, Summertime Render a Mawaru
Penguindrum. Přibyly Kanon (2006), Hachimitsu to Clover II, Myself;
Yourself, Onegai☆Teacher, Shakugan no Shana, ale i **Madoka Magica (#40)**
a Dr. Stone. Madoka přichází grafem z Bakemonogatari a Steins;Gate 0
(39 hlasů) — přesně případ pro složku „výběr" z §9d.4 (odhad zájmu
0,18–0,25). Graf má hlasy u 39 ze 40 titulů, bránou ale už není: nad
uloženým poolem z 2026-09-16 dává složka grafu titulu mimo graf −1,0 a
nejsilnějšímu titulu z grafu +1,7 bodu (dřív −0,2 a +6,0).

Karta: „Pro" a „Proti" jsou oddělené, například Charlotte „Proti: P.A.
Works, Original, Super Power", Tokyo Revengers už nemá „Action + Drama"
mezi důvody.

**Co zbývá:** vzácné tagy dál vedou tabulku efektů (Rehabilitation +0,44,
Age Gap +0,40) a scenáristé s n ≈ 5 zůstávají (Akao Deko +0,31, Akasaka
Aka −0,48) — to řeší až ridge (krok 5). Temné tituly přicházející grafem
(Madoka, Higurashi) zachytí až složka „výběr" (krok 7).

### 9d.7 Ridge a model výběru (kroky 5–7, 2026-09-24)

**Backtest** (`backtest.run`, 4 disjunktní okna podle `my_finish_date`, řezy
2025-09-09 / 2026-02-20 / 2026-05-17 / 2026-06-27; Spearman afinity vůči
reziduu, 95% interval klastrovaný po franšízách; atributy už po krocích 1–4).
`scale` naráželo u ridge na strop gridu 1,0, grid je proto rozšířený do 2,0
(marginálního modelu se to netýká, optimum má 0,35):

| varianta | `scale` | CV RMSE | Spearman vše | nové franšízy | pokračování | RMSE (debiased) | okna |
|---|---|---|---|---|---|---|---|
| marginal, min 4 (dosavadní) | 0,35 | 0,9046 | +0,424 [+0,28; +0,56] | +0,420 | +0,523 | 0,741 (0,739) | .57 .51 .47 .16 |
| marginal, min 2 | 0,35 | 0,9065 | +0,457 | +0,454 | +0,547 | 0,736 | .67 .54 .46 .16 |
| ridge α40, min 2 | 0,95 | 0,9061 | +0,459 | +0,441 | +0,494 | 0,731 | .57 .56 .48 .27 |
| ridge α60, min 4 | 1,10 | 0,9068 | +0,447 | +0,444 | +0,473 | 0,734 | .55 .50 .53 .25 |
| ridge α60, min 2 | 1,10 | 0,9056 | +0,465 | +0,459 | +0,476 | 0,730 | .56 .54 .52 .28 |
| **ridge α60, min 1,5** | 1,10 | **0,9053** | **+0,465** [+0,34; +0,57] | +0,459 | +0,469 | **0,729** (0,726) | .57 .54 .51 .27 |
| ridge α100, min 2 | 1,40 | 0,9058 | +0,467 | +0,470 | +0,465 | 0,730 | .53 .52 .54 .31 |
| ridge α150, min 2 | 1,70 | 0,9064 | +0,466 | +0,477 | +0,454 | 0,731 | .53 .48 .56 .29 |

Ridge s prahem 1,5–2 tvoří plošinu (α 40–150), rozdíly proti dosavadnímu
modelu jsou menší než šířka intervalu, ale ve stejném směru ve všech
souhrnných metrikách a ve 3 ze 4 oken; nejvíc se zlepšilo poslední,
konfundované okno (.16 → .27). **Vybráno α = 60, `min_attr_count` 1,5**:
nejnižší RMSE i CV RMSE a jako jediná varianta pustí do modelu horor (n_eff
1,82). Marginální model zůstává jako `effect_model: marginal` (s páry a
trojicemi) pro srovnání.

**Model výběru** (`selection.py`, krok 7) podle §9d.4: vesmír z
`get_top_popular` (Tenrai `top/anime?filter=bypopularity`, 80 stránek,
cachuje se), obohacení bez staff, složka `w_select` 0,5 v kompozitu
(z-skóre přes obsahový pool jako vkus), řádek „Obvykle nevybíráš" v kartě
(jen žánry/témata/demografie/tagy — „2010s" v modelu zůstává, ale nic
nevysvětlí). **Srážka za titul se nezavedla**; populární franšízy mimo
seznam i PTW (`known_popularity` 500, rozhoduje nejpopulárnější díl) jdou
místo toho do sekce „Znáš, ale nemáš v plánu". V produkčním běhu: 1 204
titulů, zájem 22 %, CV AUC **0,865**, nejzápornější koeficienty Primarily
Male Cast, Supernatural, Ojou-sama, Idol, Crime.

**Plný běh** (`config.yaml`, user-CF zapnuté, zápis historie do kopie
`history/` — snapshot se sloupcem `select` se uložil a ledger nad staršími
snapshoty proběhl). Testů 298 → **313**.

| | 2026-09-16 | po krocích 1–4 | po krocích 5–7 |
|---|---|---|---|
| Tokyo Revengers | #1 | #12 | pool #44, sekce „znáš" |
| Higurashi no Naku Koro ni | #2 | #22 | pool #96, sekce „znáš" |
| Death Note | #3 | mimo top-40 | pool #446 |
| Kanon (2006) | pool #80 | #18 | #28 (pool #41) |

(„pool #" = pořadí v celém poolu bez PTW, před sbalením franšíz a
rozdělením do sekcí.)

Nové objevy, top-10: Ano Natsu de Matteru, Koi to Uso, Kimi ga Nozomu Eien,
Osananajimi ga Zettai ni Makenai Love Comedy, Sakamichi no Apollon, ef: A
Tale of Memories., Koi to Yobu ni wa Kimochi Warui, Just Because!, Yumemiru
Danshi wa Genjitsushugisha, Araburu Kisetsu no Otome-domo yo. Sekce „Znáš,
ale nemáš v plánu": OreImo, Gosick, Charlotte, Isekai wa Smartphone, Shakugan
no Shana, Ore Monogatari!!, Sankarea, Toki wo Kakeru Shoujo, Accel World,
Madoka Magica. Karta teď ukazuje i absence — u Koi to Uso „Proti: bez Female
Harem", u OreImo „Proti: bez Romance, bez Drama".

**Co zbývá otevřené:** kroky 8–10 z §9d.5 (dvě osy náročnosti, user-CF
z kompozitu, explicitní averze), normalizace objemem hlasů seedu (#2),
propojení modelu výběru s historií („doporučeno a do PTW nepřidáno" ⇒
povědomí ≈ 1, §9d.4 bod 3) a ověření váhy `w_select` až na ledgeru historie.

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
