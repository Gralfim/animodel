# Hodnocení projektu animodel

*Komplexní review kódu (~4400 řádků Pythonu, 18 souborů) provedené nezávisle na
`CHANGELOG_review.md` — ten dokumentuje, co už bylo opraveno v předchozím kole;
tady jde o čerstvý pohled na aktuální stav včetně věcí, které předchozí review
nezachytilo. Nic v kódu jsem neměnil, jde čistě o analýzu a doporučení.*

---

## 0. Shrnutí

Projekt je v podstatně lepším stavu, než by se čekalo u nástroje tohoto rozsahu
vyvíjeného iterativně přes AI asistenty — modelovací jádro (`taste.py`) má
promyšlenou, dobře zdůvodněnou metodiku a síťová vrstva (`sources/`) prošla
viditelně důkladným kolem oprav (cache-poisoning, retry klasifikace, circuit
breaker). Hlavní zbývající rizika nejsou "bugy" v klasickém smyslu, ale:

1. **`config.yaml` má u `user_cf_*` parametrů hodnoty, které se řádově liší od
   doporučených defaultů** a mohou vést k mnohahodinovému běhu s malou šancí
   na výsledek (detail v §5.1) — **doporučuji zkontrolovat před ostrým spuštěním.**
2. **Kalibrace `scale` v `TasteModel._calibrate_scale()` přefituje model
   21× místo 1× na fold** — reálná neefektivita, ne jen kosmetika (§5.2).
3. **Nulové pokrytí testy** mimo jeden ad-hoc skript bez síťové vrstvy — nejsložitější
   a historicky nejchybovější kód (`sources/anilist.py`) nemá jediný test (§7.1).
4. Několik menších architektonických nedodělků (nepoužitá `aggregate_entries`,
   neověřený Shikimori, žádná cache invalidace) — vědomě zapsané, ne skryté.

Dál je to rozepsané po tématech: zdroje dat, architektura, algoritmy, pak nálezy
řazené podle závažnosti.

---

## 1. Zdroje dat a parametry popisu anime

| Zdroj | Co dodává | Status |
|---|---|---|
| **MAL XML export** (`mal.py`) | jediný vstup: `mal_id`, skóre, status, epizody, data | povinný |
| **Jikan** (`sources/jikan.py`) | žánry, témata, demografie, zdroj (manga/LN/…), formát, studia, rok, synopse, staff (režie/scénář), MAL recommendations graf | vždy zapnuto |
| **AniList** (`sources/anilist.py`) | 500+ granulárních tagů s rankem 0–100, studia, AniList recommendations graf, tag-search discovery, user-based CF | výchozí zapnuto, `--no-anilist` vypne |
| **Shikimori** (`sources/shikimori.py`) | jen `/similar` endpoint jako další zdroj kandidátů | **výchozí vypnuto**, naživo neověřeno |

**Kanonizace a deduplikace (`attributes.py`)** je koncepčně nejsilnější část
datové vrstvy: atribut se převede na kanonický klíč (`canon()`), synonyma napříč
zdroji se sloučí přes `ALIAS` mapu (např. MAL "Sci-Fi" ↔ AniList "Sci-fi"), a při
kolizi vyhrává vyšší váha + kategorie dle priority. To brání dvojímu počítání
téhož konceptu, což by jinak systematicky nafukovalo efekty populárních žánrů.

**Franšízy (`series.py`)** se slučují union-findem přes Jikan `relations`
(sequel/prequel/alternative version/side story) a aktivně používaná cesta
(`enrich.py::build_titles`) váhuje každého člena `1/√k` — ne kolaps na jeden
záznam. To je rozumný kompromis (desetidílná oblíbená série nepřeváží model,
ale signál se úplně neztratí).

**Komunitní baseline** (`community_baseline()`) bere MAL skóre, fallback AniList,
**záměrně bez průměrování** (zdůvodněno korelací zdrojů — rozumné rozhodnutí,
zdokumentované i v README).

### Poznámka k datovému rozsahu
Uživatelův seznam (viz `ANALYZA.md`) má silně omezený rozsah známek (skoro nic
pod 7) — to je premisa, na které stojí celá volba metodiky (reziduum místo
regrese na skóre). Je to důsledně promyšlené a en konzistentně zdůvodněné napříč
`taste.py`, `README.md` i `ANALYZA.md`.

---

## 2. Architektura

```
mal.py ──┐
         ├─► enrich.py (Jikan+AniList+Shikimori → Title) ─► taste.py (fit model)
sources/ ┘                                                        │
                                                                   ▼
                                              recommend.py (kandidáti + skóre)
                                                                   │
                                                                   ▼
                                                    report.py (2× HTML výstup)
```

Silné stránky:
- **`config.py` neobsahuje žádné seznamy atributů** — jen laditelné parametry
  (shrinkage, prahy, váhy). To je explicitně pojmenovaný designový princip a
  kód se ho drží důsledně (jediná výjimka je HEAVY/LIGHT lexikon, viz §7.2).
- **Jasné oddělení vrstev**: síťová vrstva (`sources/`) neví nic o modelu;
  `attributes.py` neví nic o síti; `taste.py` neví nic o HTML. Změna v jedné
  vrstvě nevyžaduje zásah v ostatních — ověřeno tím, že `report.py` (621 řádků
  čistě prezentační logiky) se dá číst bez pochopení zbytku systému.
- **`Result` typ** (`sources/__init__.py`) nahrazující dřívější mutable
  `_last_failure_kind` side-channel je solidní refaktor — klasifikace selhání
  (permanent/transient) je teď nedílnou součástí návratové hodnoty, nedá se
  přečíst ve špatný moment.
- **`animodel_test_harness.py`** dokazuje, že model byl ověřen na realistických
  datech (54 ručně tagovaných titulů), ne jen podle toho, že kód proběhne.

Slabší místa architektury:
- **`sources/anilist.py` má 1185 řádků v jedné třídě** — mísí HTTP/retry vrstvu,
  cache správu (3 různé cache formáty: `mal_*.json`, `cf_al/watchers_*.json`,
  `cf_al/userlist_*.json`), doménovou logiku (tag extrakce, popularita) i
  user-based CF (samo o sobě ~340řádková metoda). Modul dělá práci, kterou by
  jinde bylo přirozené rozdělit na klienta + repository + CF engine. Není to
  akutní problém (kód je funkčně korektní a dobře komentovaný), ale je to místo,
  kde se příští bug nejsnáz schová.
- **Batch (50 ID) a single-item cesty v AniList klientovi mají oddělenou,
  ne úplně symetrickou cache-sentinel logiku** (`{}` vs. chybějící soubor vs.
  `None`) — funguje, ale vyžaduje čtení kódu k pochopení, ne jen dokumentace.

---

## 3. Algoritmy — jak model skutečně počítá

### 3.1 Baseline (vkusový posun vůči komunitě)
```
baseline = tvůj_průměr + β·(komunita − průměr_komunity)
```
β se odhaduje jako kovariance/rozptyl (jednoduchá lineární regrese na jeden
prediktor), ořezané do `[-0.5, 1.5]`. Rozumná pojistka proti extrapolaci mimo
pozorovaný rozsah, ale stojí za zmínku, že β je odhadnuto jen z ~400 bodů se
záměrně omezenou variancí komunitního skóre u titulů, které uživatel vůbec
sleduje — odhad tedy nemusí být extrémně stabilní (ořez to jen limituje, neřeší).

### 3.2 Efekty atributů (empirical-Bayes shrinkage)
```
effect(attr) = n_eff/(n_eff + K) · vážený_průměr(reziduí)
```
Standardní James–Stein přístup, korektně implementovaný (vážený průměr respektuje
franšízové váhy i AniList tag-rank jako váhu). `min_attr_count` navíc atributy s
příliš málo důkazy úplně vynechá z reportu (ne jen smrští k nule).

### 3.3 Interakce
Páry atributů, kde společný výskyt vysvětluje víc, než součet jejich
samostatných efektů (`lift`). Kombinatorika je omezená jen na atributy, které
už prošly `interaction_min_count` filtrem efektů — rozumné, brání kombinatorické
explozi.

### 3.4 Kalibrace `scale` + cross-validace
5-fold CV přes grid 21 hodnot `s ∈ {0, 0.05, …, 1.0}`, vybírá `s` s nejnižším
RMSE. Report pak srovnává `cv_rmse` s `baseline_rmse` (`s=0`) — malý rozdíl je
interpretován správně jako "atributy nepomáhají hádat číslo, ale pomáhají řadit"
(důsledek restrikce rozsahu, ne chyba modelu). **Implementační neefektivita
zde je popsaná v §5.2.**

### 3.5 Klastrování nálad
KMeans (sklearn) na L2-normalizovaných atributových vektorech (jen genre/theme/
tag/demographic — staff a studia záměrně vynechány, aby "nálada" nebyla ovlivněná
tvůrcem). Počet klastrů `k` se hledá siluetou v rozsahu 4–7, pokud není napevno
v configu. Osa "náročnosti" (HEAVY/LIGHT) je jediná ručně kurátorovaná datová
struktura v celém projektu — viz §7.2.

### 3.6 Skóre doporučení
```
composite = w_taste_fit·z(taste_fit) + w_cf·z(cf_signal) + w_quality·z(community)
```
kde `taste_fit` kombinuje predikované reziduum s kosinovou podobností k
nejbližšímu náladovému klastru. Řazení podle kompozitu místo podle predikované
známky je metodicky konzistentní s bodem 3.4 — správné rozhodnutí, ne jen
kosmetika.

### 3.7 User-based CF (`similar_users_recommendations`)
Nejsofistikovanější (a nejkřehčí, viz §7.1) část kódu: seedy se vybírají podle
**nejnižší popularity** (ne nejvyššího skóre) — sdílení nišového titulu je
silnější signál shody vkusu. Podobnost uživatelů je vážená Pearsonova korelace
na komunitně-relativních diferenciálech (ne surové skóre), takže odstraňuje
efekt "hodnotím obecně přísně/velkoryse". Zohledňuje i rozdílné škály hodnocení
napříč uživateli (`POINT_100` vs. `POINT_10` atd.) a soukromé/smazané účty
řeší korektně (skenuje pool dál, místo aby ztratila místo v `top_users`).
Metodicky je to v pořádku; provozně je to nejnákladnější a nejkonfiguračně
nejcitlivější část systému (§5.1).

---

## 4. Co je dobře udělané a nemělo by se měnit

Aby review nebyl jen seznam problémů — tohle jsou rozhodnutí, která bych
neotáčel:

- Reziduální cíl místo regrese na skóre + zdůvodnění restrikcí rozsahu —
  metodicky správně a je to i ověřené na fixture (CV RMSE ≈ baseline RMSE je
  očekávaný, ne alarmující výsledek).
- `Result` typ a permanent/transient klasifikace chyb na jednom místě
  (`is_permanent_status`) — brání přesně té třídě bugů, která podle
  `CHANGELOG_review.md` dřív existovala (cache poisoning dočasným výpadkem).
- Automatická kanonizace atributů bez ručního configu — skutečně dodržený
  princip, ne jen deklarovaný.
- `unmatched_intensity_keywords()` jako sebe-diagnostika HEAVY/LIGHT lexikonu —
  chytrý způsob, jak dát budoucímu ladění zpětnou vazbu bez nutnosti to ručně
  auditovat.

---

## 5. Nálezy — chyby a neefektivity

### 5.1 `config.yaml`: extrémní hodnoty pro `user_cf_*` (vysoká priorita)

Aktuální `config.yaml` (na disku, se `use_user_cf: true`) má:

```yaml
user_cf_min_overlap: 80
user_cf_top_users: 200
user_cf_seed_count: 350
user_cf_users_per_seed: 4950
```

Pro srovnání, `config.example.yaml` (a defaulty v `config.py`) mají `4 / 120 / 25
/ 100`. Rozdíl je 15–50×, ne drobné doladění. Konkrétní dopady:

- **Náklady:** `user_cf_seed_count=350` × až desítky stránek na seed (strop
  `MAX_WATCHER_PAGE=100`, typicky méně díky `NO_PROGRESS_PAGE_LIMIT`) × adaptivní
  AniList delay 0.7–4s → realisticky hodiny běhu jen na fázi sběru sledujících,
  než se vůbec dostane k `top_users=200` stahování celých seznamů.
- **Šance na výsledek:** `min_overlap=80` znamená, že kandidát musí sdílet
  **80 z 350** nejméně populárních titulů s uživatelem, aby byl vůbec zvažován.
  To je výrazně přísnější práh, než jaký typicky projde i u aktivních fanoušků
  podobného vkusu — reálné riziko je, že po hodinách stahování vyjde `candidates
  = []` (kód to sice korektně zaloguje a vrátí prázdný list, nespadne — ale
  čas se neušetří).

Toto není bug v kódu — kód dělá přesně to, co configu řekne. Je to ale
konfigurace, která vypadá jako omylem přenesená čísla z jiného experimentu
(např. záměna "kolik titulů mám celkem" za "kolik sdílených seedů žádám"), a
stojí za ověření/vysvětlení, než se `--user-cf` pustí naostro.

### 5.2 `TasteModel._calibrate_scale()`: 21× zbytečné přefitování (střední priorita)

> **Stav (2026-07-14): OPRAVENO.** Fold-modely se fitují jen jednou
> (`_cv_predictions`), grid přes `s` je čistá aritmetika (`_eval_scale`).
> Ověřeno numericky identickými výsledky (scale/CV RMSE/MAE/baseline na
> harness datech beze změny), 116 → 6 fitů, kalibrace ~12× rychlejší.
> Regresní testy v `tests/test_taste_calibration.py`.

`_calibrate_scale()` prochází 21 kandidátních hodnot `s` a pro každou volá
`_cross_val(s)`. Uvnitř `_cross_val()` se ale pro každý z 5 foldů **znovu**
volá `sub._fit_baseline()` + `sub._fit_effects()` + `sub._fit_interactions()`
— a tenhle fit **vůbec nezávisí na `s`** (`s` vstupuje až v predikčním kroku
`pred = sub._baseline_pred(...) + s * sub._raw_resid_pred(...)`).

Důsledek: samotné (drahé) přefitování modelu na foldu se dělá 21× 5 = 105×,
místo 5× (jednou na fold, s výsledkem znovupoužitým pro všech 21 hodnot `s`).
U ~400 titulů to prakticky ještě běží v řádu sekund, ale je to přesně ten typ
neefektivity, který projekt už jednou řešil jinde (dvojité klastrování,
lineární průchod v `_cluster_fit` — viz `CHANGELOG_review.md` bod 2) — tady
zůstal nepovšimnutý. Oprava je lokální: v `_cross_val` přesunout fit foldu
mimo smyčku přes `s`, nebo v `_calibrate_scale` fitovat foldy jednou a předávat
hotové `sub` modely dovnitř.

### 5.3 Rozjeté defaulty `TasteModel.__init__` vs. `config.py` (nízká priorita)

`TasteModel.__init__` má vlastní defaulty (`min_attr_count=3.0,
interaction_min_count=6.0, interaction_min_lift=0.25`), které se liší od
`ModelCfg` v `config.py` (`4.0, 8.0, 0.30`). `cli.py` vždy předává explicitní
hodnoty z configu, takže se to v produkčním běhu neprojeví — ale
`animodel_test_harness.py` i příklad v `README.md` konstruují
`TasteModel(shrinkage_k=...)` bez zbytku parametrů, takže **oba běží s jinými
prahy, než jaké produkční `cli.py` používá**. Drobné, ale je to past pro
někoho, kdo bude ladit podle výstupu test harness a čekat shodu s ostrým
během.

---

## 6. Nedodělky (vědomě otevřené, zapsané v kódu)

Většina z nich je už poctivě okomentovaná přímo v kódu — jde spíš o
konsolidovaný přehled, co čeká na rozhodnutí:

| Co | Kde | Stav |
|---|---|---|
| `aggregate_entries` (kolaps franšíze na 1 záznam) | `series.py:130` | funkční, ale **nenapojené** — konkuruje aktivní `1/√k` váze v `enrich.py`. Rozhodnutí (zachovat jako `--aggregate-mode` alternativu, nebo smazat) čeká na tebe. |
| Shikimori `/similar` | `sources/shikimori.py` | výchozí vypnuto — tvar odpovědi (rank vs. holý seznam) není ověřený naživo, `REQUEST_DELAY=1.0` je odhad. |
| `include_staff` (režie/scénář jako signál) | `config.py` | funkční, výchozí vypnuto (cena: +1 Jikan volání/titul). |
| Cache bez expirace | `sources/*.py` | **žádný TTL ani `--refresh-cache` flag** — jednou stažené `averageScore`/`popularity`/tagy zůstávají v cache navždy, i když se použije nástroj opakovaně za měsíce. U aktuálně vysílaných sérií (kde se komunitní skóre rychle mění) to může tiše zkreslovat `w_quality` a `min_community` filtr. Není zdokumentované jako záměr, spíš chybějící funkce. |
| `--analyze` | `cli.py` | doplněné (podle changelogu), ale je to jediná diagnostická cesta — žádný ekvivalent pro "co se nematchlo v attributes" mimo `unmatched_intensity_keywords()`. |

---

## 7. Křehké konstrukce

### 7.1 Nulové pokrytí automatickými testy nad síťovou vrstvou (vysoká priorita)

`animodel_test_harness.py` je jediný test v repozitáři a testuje výhradně
`taste.py` na syntetických datech bez sítě. **Nic netestuje:**
- `mal.py::parse_export` (parsování XML, `or 0` fallbacky) — jednoduché, ale
  přesně typ kódu, kde se tiše rozbije formát exportu po změně na straně MAL.
- `attributes.py::canon`/`resolve_alias`/`_add` (kanonizace, kolize vah/kategorií).
- `series.py` union-find (spojování franšíz) — netriviální logika s edge-case
  (cyklické reference, chybějící `relations`).
- **`sources/anilist.py`** — 1185 řádků, historicky nejchybovější soubor podle
  `CHANGELOG_review.md` (cache poisoning, mrtvý circuit breaker, zdvojená
  klasifikace chyb). Právě retry/cache invarianty (permanent vs. transient,
  kdy se smí/nesmí zapsat cache) jsou přesně ten typ logiky, kterou je snadné
  při budoucí úpravě tiše znovu rozbít — a nic by to nezachytilo dřív než další
  ruční audit.

Není potřeba plný pytest apparát — i pár testů nad `Result`/`is_permanent_status`
klasifikací a nad `_add()`/`canon()` kolizemi by pokrylo největší riziko za
zlomek úsilí.

### 7.2 HEAVY/LIGHT lexikon (`taste.py`) — přiznaná výjimka z vlastního principu

Projekt jinde důsledně odmítá ruční seznamy atributů (viz `attributes.py`
docstring), ale osa "emocionální náročnosti" je jediná ručně kurátorovaná
množina klíčů (`HEAVY`/`LIGHT`, ~30 položek). Kód to sám otevřeně komentuje a
dává nástroj na diagnostiku (`unmatched_intensity_keywords()`) — takže to není
skrytá nekonzistence, ale je to jediné místo v systému, kde se "co je co"
rozhoduje před-datově, ne z dat. Riziko: pokud uživatel sleduje anime s
konceptem náročnosti, který lexikon nezná (a `canon()` na něj netrefí), titul
dostane `intensity=0` (neutrální) tiše, bez chyby — ne proto, že by titul byl
skutečně neutrální.

### 7.3 `similar_users_recommendations` — jedna metoda, šest odpovědností

Viz §2 — ~340 řádků: výběr seedů podle vzácnosti, IDF váhování, stránkované
stahování s cache, Pearsonova korelace, řešení soukromých účtů, finální
agregace a řazení, to vše v jedné metodě jedné třídy. Funkčně to (podle
changelogu) prošlo reálným testováním, ale každá budoucí úprava jedné
odpovědnosti riskuje nechtěně zasáhnout ostatní — typicky přesně tam, kde v
`CHANGELOG_review.md` už vznikly tři různé bugy (body 5, 6, 8, 9).

### 7.4 `side story` v defaultní množině slučovaných relací

`series.py::SERIES_RELATION_TYPES` obsahuje `"side story"` s komentářem
"volitelné — některé side story jsou samostatné", ale je zapnuté ve výchozí
množině. Spin-off, který je tonálně jiný než hlavní série (např. komediální
chibi vedlejšák k dramatu), se tak sloučí do stejné franšíze a jeho hodnocení
dostane sníženou váhu `1/√k`, i když uživatelův pocit z něj mohl být nezávislý
signál, ne opakování stejné preference. Malý dopad (týká se jen titulů se
side-story vazbou), ale je to tichý předpoklad, ne ověřené rozhodnutí.

### 7.5 Config bez validace rozsahů

`Config.load()` (`config.py`) přebírá jakoukoli hodnotu z YAML bez kontroly
typu nebo rozsahu (`setattr(sub, kk, vv)`) — proto mohla vzniknout situace z
§5.1 beze slova varování. Přidání i jen orientačních sanity-checků (např. varovat,
když `user_cf_seed_count` výrazně převyšuje doporučenou hodnotu, nebo když
`min_overlap` přesáhne `seed_count`) by podobné case zachytilo dřív, než
uživatel čeká hodiny na prázdný výsledek.

---

## 8. Doporučení — pořadí podle poměru přínos/náklad

1. **Ověř `config.yaml` `user_cf_*` hodnoty** (§5.1) — buď je to záměr (pak
   stojí za komentář v souboru proč), nebo je to omyl a stačí vrátit na
   defaulty z `config.example.yaml`. Nulová cena opravy, vysoký dopad.
2. **Oprav zbytečné 21× přefitování v `_calibrate_scale`** (§5.2) — lokální
   změna ve dvou metodách, měřitelné zrychlení `fit()`.
3. **Přidej hrstku testů nad `sources/__init__.py` a `attributes.py`** (§7.1) —
   nejlevnější pojistka proti regresi v nejkřehčí a nejvíc opravované části
   kódu.
4. **Rozhodni osud `aggregate_entries`** (§6) — buď smazat, nebo zapojit jako
   `--aggregate-mode` přepínač; dead code s aktivní alternativou matoucí pro
   budoucí čtenáře.
5. **Cache TTL / `--refresh-cache` flag** (§6) — nutné až při delším provozu
   napříč měsíci, ne akutní teď.
6. Sjednoť defaulty `TasteModel.__init__` s `ModelCfg` (§5.3) — kosmetika,
   ale snadná.

---

*Poznámka: tento dokument je analytický, žádné změny v kódu jsem neprováděl.
Pokud chceš, můžu kterýkoli bod z §8 rovnou implementovat — nejlíp začít
bodem 1 (jen kontrola configu, žádný kód) a bodem 2 (malá, izolovaná oprava).*
