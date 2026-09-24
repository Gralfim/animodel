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
python -m animodel -e animelist.xml --analyze-attrs  # diagnostika kanonizace atributů
python -m animodel -e animelist.xml --gen-intensity  # (re)generace intensity.yaml (osa náročnosti)
python -m animodel -e animelist.xml --season         # doporučení pro aktuální vysílanou sezónu
python -m animodel -e animelist.xml --season 2026 summer  # konkrétní sezóna
python -m animodel -e animelist.xml --backtest       # časová validace modelu (out-of-time)
python -m animodel -e animelist.xml --no-history     # nezapisuj běh do historie
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
proč předchozí pokus s Ridge regresí na surových známkách nedával přesvědčivé
výsledky. Na *odchylce* od komunity — viz níž — naopak ridge funguje nejlépe.)

animodel proto **necílí na známku, ale na odchylku**:

1. **Baseline.** Pro každý titul: `tvůj_průměr + β·(komunita − průměr_komunity)`.
   Komunita vstupuje jako *jeden* kalibrovaný sklon β, **ne** jako atribut — jinak
   by se „kvalita" počítala dvakrát a zkreslila by efekty (např. studia, která
   hodnotí vysoko i komunita).
2. **Cíl = afinita.** Co po odečtení baseline zbude. To je tvůj osobní podpis nad
   rámec toho, co by čekal kdokoli.
3. **Efekty atributů.** **Ridge regrese** afinity na **centrované** atributy
   (`model.effect_model: ridge`, penalizace `ridge_alpha`): každý efekt platí
   „při ostatním stejném". Dvě věci, které dřívější průměry neuměly:
   - **chybějící oblíbený atribut sráží** — titul bez romantiky dostane
     `−β·μ` (μ = jak častá je romantika v seznamu). Průměr afinity titulů
     *s* atributem vycházel jen ≈ (1 − p)·rozdíl, takže romantika (61 %
     seznamu) měla po kalibraci vliv ~0,05 bodu a Death Note za to, že ji
     nemá, neztratil nic;
   - **zastupitelné tagy se dělí o jeden efekt** místo aby se sčítaly
     (Romance + Heterosexual + Love Triangle…), a řídké atributy se
     smršťují k nule samy.
   Vedle efektu se počítá i `Δ komunita` = intuitivní „o kolik výš než dav to
   hodnotíš". Rozpad predikce v kartě ukazuje i absence („bez Romance").
   Backtest na 4 časových oknech: Spearman +0,424 → **+0,465**, RMSE 0,741 →
   **0,729** (HODNOCENI §9d.7).
4. **Interakce** (jen `effect_model: marginal`, dřívější model). Tam se efekt
   počítá jako smrštěný průměr `(n/(n+K))·průměr` a navíc dvojice atributů,
   kde se afinita liší od součtu jednotlivých efektů; volitelně
   (`model.interaction_triples`) i trojice nad jádry nálad. V ridge režimu
   páry nejsou — část z nich byla jen zástupce utlumených singlů (pár
   „Psychological + Supernatural" nesl zásluhu romantiky z Monogatari a
   Bunny Girl Senpai a přenášel ji na Death Note).
5. **Nálady (módy).** KMeans na normalizovaných atributových vektorech; počet
   klastrů se volí podle siluety. Kandidát se k náladě přiřazuje **váženým
   kosinem proti celému těžišti** klastru, a jen v prostoru nálady
   (žánry/témata/tagy/demografie — studio, formát ani dekáda do nálady
   nepatří). Tím přiřazení odpovídá tomu, jak klastry vznikly: shoda
   s tím, co KMeans skutečně rozhodl, je 99,6 % (proti 85 %, když se
   podobnost počítala jen proti šesti zobrazovaným osám). Každá nálada navíc
   dostane **archetyp** — svého nejtypičtějšího představitele (člen nejblíž
   těžišti), který ji v reportu pojmenuje konkrétním titulem; bere se jen mezi
   členy s aspoň mediánovým počtem atributů nálady, aby archetypem nebyl
   krátký special nesoucí jedinou dominantní osu.
   Každý klastr dostane **osu náročnosti**
   (těžké drama/psycho vs. lehká komedie/slice-of-life) — to je ta „emocionální
   únava", kvůli které mezi módy přepínáš. Osu řídí **intensity lexikon**
   (`intensity.yaml`, viz níž): spojité hodnoty −1…+1 na atribut, generované
   z úplného universa tagů a revidovatelné v jednom souboru. Každá nálada má
   navíc **afinitu** — vážený průměr reziduí svých členů (o kolik ji hodnotíš
   nad baseline) — kterou doporučení používají místo surové známky.
6. **Kalibrace.** Globální škála efektů a interval predikce z 5-násobné
   cross-validace, jejíž **foldy se dělí po franšízách** (díl série v tréninku
   prozradí testu její identitu přes studio, staff a řídké tagy — 95 ze 103
   franšíz se dřív rozpadlo do víc foldů) a sčítají se přes **tři zamíchání**,
   ať čísla nekolísají podle seedu. Afinita se navíc **centruje** tréninkovým
   průměrem (u ridge je centrovaná z konstrukce). Škála může být i nad 1 —
   ridge koeficienty jsou smrštěné penalizací a CV jim amplitudu vrací.
   Trojice (jsou-li zapnuté) mají **vlastní škálu** — jsou jiný
   řád důkazů než singly a páry (řidší podpora, kandidáti z klastrových
   signatur), tak se i kalibrují zvlášť: společný grid přes `(s, s₃)`.
   Fold-modely si přitom klastrují a hledají trojice **samy** na svých 4/5 dat,
   aby do cross-validace neprosákla znalost testovací části. Report i CLI
   hlásí, o kolik trojice CV RMSE zlepšily (`bez trojic by CV RMSE bylo …`) —
   ať je vidět, jestli se ten experiment vyplácí.

### Atributy se neudržují ručně

Žádný `config.yaml` plný seznamů žánrů. Atributy (žánry, témata, AniList tagy,
studia, zdroj, dekáda, formát, demografie) se **objevují samy z dat**.
`attributes.py` je kanonizuje a **dedupuje napříč zdroji** (MAL „Drama" a AniList
tag „Drama" = jeden atribut), aby se stejný koncept nezapočítal víckrát. Totéž
platí pro osoby: klíč režiséra/scenáristy se skládá ze seřazených slov jména,
takže „Mizushima, Tsutomu" (Jikan) a „Tsutomu Mizushima" (AniList) jsou jeden
atribut. Za scenáristu se nepočítá holé „Script" — MAL pod ním vede i
lokalizaci (překladatele titulků). Země původu vstupuje jen když **není**
japonská (JP tvoří ~95 % každého seznamu, jako atribut by to byla konstanta).

**Žánry se berou z MAL**, AniList je doplní jen tam, kde MAL žádné nemá (jako
zdroj, formát a dekádu). AniList žánry jsou volnější a při binárním sloučení
MAL přebíjely: Tokyo Revengers má na MAL Action + Drama, AniList přidal
Romance a model ji počítal stejně jako u Toradory. AniList **tagy** (s vahou
podle ranku) zůstávají.

Tichý selhací mód téhle vrstvy je „dva klíče pro jeden koncept": nikde to
nespadne, jen se evidence rozdělí na dvě poloviny, obě se silněji smrští
k nule a efekt zmizí. `--analyze-attrs` proto hledá dvojice atributů, které
vypadají jako týž koncept a `ALIAS` je nespojuje — porovnáním **po slovech**
(shodná slova v jiném tvaru/pořadí, nebo jednoslovné klíče lišící se jen
koncovkou). Na živých datech to našlo `Video Game` ↔ `Video Games` a
`Anthropomorphic` ↔ `Anthropomorphism` — v obou případech MAL a AniList
pojmenovaly totéž jinak; oba páry jsou teď v `ALIAS`.
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
  „recommendations" graf (item-based CF). Seedy (známka ≥ `high_score`, max.
  `max_seeds`) se vybírají podle **rezidua** — o kolik víc se ti titul líbil,
  než čeká baseline z komunity — ne jen podle známky: remízy mezi desítkami
  a devítkami dřív rozhodovalo abecední pořadí exportu. Volitelně i Shikimori `/similar`
  (`enrich.use_shikimori` v configu, default vypnuto — stojí +1 request na seed
  a přináší hlavně *nové kandidáty*, pořadí ovlivní jen okrajově: endpoint
  vrací prostý seznam bez skóre podobnosti, takže se váží pozicí), a navíc
  discovery přes AniList tag-search na tvé nejcharakterističtější tagy.
- **Collaborative / uživatelská** (volitelná, `--user-cf`) — hledání „senpai":
  pár (default 20) uživatelů s **ověřeně** podobným vkusem, kteří viděli víc než
  ty. Discovery jde přes tvé nejméně populární tituly (sdílení nišového titulu
  je silný signál), ale překryv a podobnost se počítá až na **plných seznamech**
  kandidátů (Pearson na komunitně-relativních odchylkách, smrštěný velikostí
  překryvu) — ne na vzorku. Skóre navíc **sráží nepokryté oblíbené** tituly:
  kdo tvé nejlepší známky nemá ohodnocené ani na PTW, je slabší průvodce
  (`user_cf_fav_miss_penalty`). **Dropnuté tituly se známkou** se počítají jako
  plnohodnotné hodnocení — „zkusil a dal 3" o shodě vkusu řekne víc než většina
  desítek. Podobnost se měří na **reziduích**: na mé straně známka minus to, co
  u mě podle komunity čeká model, na jeho straně odchylka od **jeho vlastní**
  baseline (`norm ≈ a + b·komunita`). Bez toho vycházel jako nejpodobnější ten,
  kdo jen kopíruje dav — kdo dává všemu desítku, je čistý obraz komunity a na
  reálných datech tvořili takoví hodnotitelé 16 z 20 vybraných senpai.
  Senpai jsou vidět jmenovitě v CF reportu i s metrikami. Jejich signál
  (**smrštěná odchylka od baseline** bez komunitního skóre) má v kompozitu
  vlastní složku, ale od 2026-09 s **výchozí vahou 0**: časový test ukázal,
  že tvé pozdější známky sám předpovídá jen slabě (Spearman +0,19) a
  k modelu vkusu nic nepřidá — s váhou 0,6 se pořadí dokonce zhoršilo
  (0,46 → 0,43; HODNOCENI §9d.8). Senpai dál dodávají **kandidáty** do poolu
  (odtud přišly tituly jako Summer Pockets) a CF report.
  Tvůj vlastní účet se vylučuje (podle jména z MAL exportu; máš-li na AniListu
  jinou přezdívku, přidej ji do `recommend.user_cf_exclude_users`) — import
  vlastního seznamu má podobnost 1.00 a doporučil by ti jen to, co už máš.

Každý kandidát se skóruje kompozitem (sčítají se z-skóry pěti oddělených
složek):

```
composite = w_taste_fit · afinita+shoda_s_náladou
          + w_cf        · „doporučili to tvé oblíbené" (graf, log-tlumené hlasy)
          + w_user_cf   · user-based CF (podobní uživatelé; default 0, viz výš)
          + w_quality   · komunitní skóre
          + w_select    · „sáhneš po tom vůbec?" (model výběru)
```

**Model výběru** (`selection.py`) doplňuje, co model vkusu z principu vidět
nemůže: čemu se vyhýbáš. Model vkusu se učí jen z toho, co máš shlédnuté, a
žánrům, kterým se vyhýbáš, v seznamu skoro nic neodpovídá — bere je proto
jako neutrální. Model výběru se učí z **top-2000 MAL titulů podle
popularity** (jen první díly franšíz, aspoň půl roku staré): populární
titul, který nemáš ani na PTW, skoro jistě znáš a vědomě ho přeskakuješ.
Logistická regrese nad atributy s popularitou jako kovariátou povědomí (méně
známý titul chybí spíš proto, že o něm nevíš); do kompozitu jde jen
atributová část. Na reálném seznamu CV AUC 0,874 a nejvíc se vyhýbá
Primarily Male Cast, Organized Crime, Police, Thriller, Horror a Crime.
Karta titulu, který model výběru sráží, ukazuje řádek **„Obvykle
nevybíráš: …"**.

Graf podobnosti a user-CF mají **každý vlastní z-skóre a váhu** — ve sdíleném
kbelíku by šikmé rozdělení hlasů grafu user-CF utopilo a přebilo i model vkusu.
Průměr a rozptyl pro z-skóre vkusu, grafu a kvality se počítají z
**obsahového poolu** (kandidáti z grafu a tag-search), ne z celého: user-CF
do poolu přidá tisíce titulů s nulou v grafu, rozptyl grafu se tím zhroutí a
každý titul z grafu dostane pár bodů navíc jen za to, že v grafu je. Váhy
`w_*` tak znamenají totéž se zapnutým i vypnutým user-CF.
Slabé hrany grafu (`min_mal_rec_votes`, `min_anilist_rec_rating`) se zahazují:
jednotky hlasů jsou šum, skutečně podobné série mívají hlasů desítky.

Řadí se podle kompozitu, **ne** podle predikované známky (ta se kvůli restrikci
rozsahu lepí na komunitní průměr a nerozlišuje). Predikovaná známka + interval se
počítá zvlášť jen pro zobrazení.

Vyhledává se **nezávisle na PTW**; už shlédnuté (Completed/Watching/On-Hold/
Dropped) se vyřazují. Výstup je rozdělený na **tři sekce nad týmž poolem**:

- **Nové objevy** — jádro přehledu (`top_n`), bez PTW, bez pokračování sérií,
  které už máš rozjeté, a bez populárních titulů mimo plán (další bod).
- **Znáš, ale nemáš v plánu** (`known_top`) — franšízy s dílem mezi
  `known_popularity` (default 500) nejpopulárnějšími na MAL, které nemáš
  v seznamu ani na PTW. Nejspíš o nich víš a vědomě je přeskakuješ, takže
  mezi objevy jen zabíraly místo; nesrážejí se, jen mají vlastní sekci.
- **Z tvého plan-to-watch** (`ptw_top`) — tvůj vlastní výběr seřazený týmž
  kompozitem. Dřív PTW tituly zabíraly 15–16 ze 40 míst přehledu.
- **Pokračování tvých sérií** — franšízy, ze kterých už něco máš. Hlídáš si je
  sám (všechna 4 v dnešní stovce jsi měl v PTW), takže v objevech jen zabírala
  místo.

**Jedna karta = jedna franšíza:** díly téže série se sbalí pod ten s nejvyšším
kompozitem a ostatní se jen vyjmenují. Když je kartou pokračování, jehož
předchozí díl jsi neviděl, dostane štítek **„začni od: X"** (jen hlavní formáty
— bez té kontroly by se za začátek série označila i prologová OVA).

Pro každý titul: originální i anglický název, synopse, odůvodnění — zvlášť
**„Pro"** (atributy, které ho táhnou nahoru) a **„Proti"** (co ho sráží) — a
které tvé oblíbené ho doporučily, MAL skóre, odhad tvého hodnocení jako
interval a do jaké tvé nálady patří.

### Zpětná vazba z historie

Každý nový MAL export je **ground truth pro předchozí doporučení** — a zároveň
pro predikci u všech titulů, které jsi mezitím dokoukal. Každý běh se proto
zapíše do `history/` a při dalším **změněném** exportu se vyhodnotí:

```
── Zpětná vazba z historie (2 snapshoty od 2026-07-28) ─────────────
  okna mezi snapshoty:
    2026-07-28 → 2026-08-23: nově shlédnuto 14 · z doporučení 0 · do PTW přidáno 12 (z doporučení: #3 3-gatsu no Lion)
    2026-08-23 → dnes: nově shlédnuto 9 · z doporučení 2: #1 3-gatsu no Lion 2nd Season (rozkoukáno, 7), #3 3-gatsu no Lion (7) · do PTW přidáno 1
  vyzkoušeno: z doporučení mimo PTW 2/77 · z doporučení na PTW 0/29 · z PTW mimo doporučení 5/160
  známky dokoukaných (průměr po franšízách; vůči očekávání = známka − baseline z komunity):
    doporučené               7.00 · vůči očekávání -1.46  (1 franšíza, 1 titul)
    ostatní, nové franšízy   7.20 · vůči očekávání -0.37  (5 franšíz, 6 titulů)
    ostatní vč. pokračování  7.30 · vůči očekávání -0.37  (9 franšíz, 18 titulů)
    Δ proti novým franšízám: -0.20 (95% ±2.18) · vůči očekávání -1.09 (±2.02) — zatím neprůkazné
  predikce vs. skutečná známka (uložená predikce u 1/19 dokoukaných; záporný bias = predikce nadsazené):
    nové franšízy  n=1 · bias -2.07 · RMSE 2.07
```

Co je na tom podstatné:

- **Každé nové shlédnutí se počítá jednou** (ledger událostí), přiřazené oknu
  mezi snapshoty a *první* expozici v doporučeních. Dřív se vyhodnocoval každý
  snapshot zvlášť, takže titul doporučený třikrát se započítal třikrát.
- **Průměry jsou po franšízách, ne po titulech.** Osm dílů jedné série není
  osm nezávislých důkazů — a přesně tak vypadá typické období sledování.
- **Kontrola se hlásí dvakrát:** proti novým franšízám (o ty doporučovač
  soutěží) a proti všem ostatním včetně pokračování rozjetých sérií.
- **„Vůči očekávání"** = známka minus baseline z komunity. Doporučené tituly
  mají vyšší komunitní skóre z konstrukce, takže surová Δ míchá vkus s výběrem
  kvality; rozdíl obou čísel je přesně `β·(rozdíl průměrné komunity)`.
- **Interval, ne jen číslo.** Při jednotkách událostí je Δ neprůkazná a je to
  ve výpisu vidět; na rozlišení Δ = +0,5 je potřeba řádově rok sběru.
- **Predikce se ověřuje i u titulů, které nikdo nedoporučil.** Nových hodnocení
  přibývá ~13 měsíčně, zásahů doporučení ~1 — tohle je nejhustší zpětná vazba,
  jakou ze seznamu dostaneš.

**Co snapshot ukládá** (schéma 2): stav celého seznamu (status, známka, datum
dokončení), datum exportu (mtime souboru), parametry baseline i z-skóre,
top-100 doporučení a **co report skutečně ukázal** (karty po sekcích — nové
objevy, „znáš, ale nemáš v plánu", PTW, pokračování). Ve vedlejším souboru `{n}_{otisk}.pool.json` pak **celý
pool kandidátů** se složkami kompozitu (~400 kB) a **log predikcí** pro tituly
mimo pool: celé PTW a neviděné díly franšíz do dvou kroků od toho, co máš
shlédnuté nebo v plánu. Díky poolu jde později přepočítat pořadí s jinými
vahami bez nového běhu, díky logu má většina nově shlédnutých titulů uloženou
predikci (samotný pool + PTW by pokryl 5 z 23 nových shlédnutí, s franšízovými
sousedy je to 17 z 23).

> **První běh po upgradu** stáhne metadata pro ~100 titulů, které dosud nebyly
> potřeba (PTW mimo pool a franšízoví sousedé). Další běhy je berou z cache.

**Snapshoty se klíčují otiskem stavu seznamu, ne datem.** Ladění parametrů
znamená desítky běhů nad týmž exportem; datové klíčování by z nich udělalo
desítky skoro identických záznamů a evaluaci by to ředilo. Otisk se počítá
z trojic `(mal_id, status, score)` — tedy z toho, co model konzumuje a co
zároveň tvoří ground truth. Odsledovaný díl ani datum dokončení otisk
nezmění (hash souboru by se změnil), přesun titulu do Completed nebo změna
známky ano. Stejný otisk = přepis, takže **ladicí běhy hromadí jeden záznam,
ne sto**. Soubory se jmenují `{počet_hodnocených}_{otisk}.json`, aby šla
složka číst chronologicky.

Vypnout jde `--no-history` nebo `recommend.save_history: false`.

### Jak z historie vytěžit nejvíc

Úzké hrdlo není frekvence exportů, ale tempo sledování (~13 nových hodnocení
měsíčně proti ~1 titulu, který vzejde z doporučení). Z toho plyne pár
praktických věcí:

- **Exportuj a spusť běh zhruba jednou měsíčně.** Častěji se vyplatí jen tehdy,
  když chceš zachytit cestu *doporučeno → PTW*: přidání do PTW export nedatuje,
  je vidět jen jako rozdíl mezi dvěma snapshoty. Běh nad teplou cache je levný.
- **Nech si staré exporty.** `animelist.xml` archivovaný s datem v názvu je
  jediný způsob, jak historii zrekonstruovat, kdyby se `history/` ztratila —
  a nese data dokončení, ze kterých žije `--backtest`.
- **Zálohuj `history/`** (včetně `*.pool.json`). Je to jediná validační data,
  která nejdou dopočítat zpětně; každý běh bez nich je nevratně ztracený.
- **Hodnoť i to, co dropneš.** „Zkusil a dal 3" nese víc informace než většina
  desítek — dropnutý titul bez známky je signál, který se zahodí.
- **Váhy kompozitu nelaď podle cross-validace.** Pořadí konfigurací podle CV
  vychází mimo čas obráceně; na strukturální volby je `--backtest`, na váhy je
  potřeba počkat na dost událostí v historii.

### Sezónní doporučení (`--season`)

`python -m animodel -e animelist.xml --season` vygeneruje
`recommendations_season.html` pro aktuální vysílanou sezónu (auto-detekce z data;
nebo napevno `--season 2026 summer`). Dvě sekce:

- **Pokračování tvých sérií** — nové řady sérií, jejichž předchozí díly hodnotíš
  `≥ season_min_prequel_score` (default 7). Řazeno podle tvé známky předchozí řady.
- **Nové tituly pro tebe** — zbytek sezóny řazený podle **taste_fit** (obsahová
  shoda s modelem vkusu). Collaborative signály se tu nepoužívají — čerstvě
  vysílané série nemají graf podobnosti ani hodnocení od senpai. Pokračování
  série, kterou nemáš shlédnutou, se sem zařadí s poznámkou.

U běžících sérií se zobrazí **datum posledního dílu**, dopočítané z AniList
`nextAiringEpisode` + počtu epizod. Airing data se (na rozdíl od statických
metadat) necachují — mění se týdně.

---

## Ladění (`config.yaml`)

Zkopíruj `config.example.yaml`. Nejčastější páčky:

| parametr | co dělá |
|---|---|
| `model.shrinkage_k` | jen `effect_model: marginal` — vyšší = konzervativnější (malé vzorky víc tlumeny) |
| `model.n_clusters` | `null` = auto; nebo napevno počet nálad |
| `model.intensity_lexicon` | cesta k intensity.yaml (osa náročnosti, viz `--gen-intensity`) |
| `model.side_story_weight` | vliv OVA/speciálů/side stories uvnitř franšízy (1.0 = bez rozlišení) |
| `model.interaction_triples` | experiment: synergie trojic nad jádry nálad (vlastní kalibrovaná škála; CLI hlásí, kolik reálně přinesly) |
| `recommend.seeds_per_franchise` | max. seedů z jedné franšízy (0 = bez limitu) |
| `recommend.cluster_fit_weight` | váha shody s náladou uvnitř `taste_fit` (0 = nálady neřadí) |
| `model.effect_model` / `model.ridge_alpha` | `ridge` (default) nebo `marginal` (dřívější průměry + páry); síla penalizace ridge |
| `recommend.w_taste_fit / w_cf / w_user_cf / w_quality / w_select` | váhy 5 složek řazení doporučení |
| `recommend.known_popularity` / `known_top` | hranice popularity pro sekci „znáš, ale nemáš v plánu" (0 = vypnuto) |
| `recommend.min_mal_rec_votes / min_anilist_rec_rating` | prahy síly hrany v grafu podobnosti |
| `recommend.min_community` | spodní hranice MAL skóre kandidátů (nově i v sezónním pohledu) |
| `recommend.ptw_top` | kolik titulů v sekci „z tvého PTW" |
| `recommend.high_score` | od jaké známky je titul „seed" |
| `enrich.use_anilist` | vypni pro rychlejší běh jen na MAL |
| `enrich.use_jikan` | vypni pro nouzový AniList-only režim (viz `--no-jikan`) |
| `enrich.include_staff` | signál po režisérech/scenáristech (default vypnuto) |
| `enrich.staff_source` | `jikan` (+1 request/titul, hlubší) nebo `anilist` (zdarma v dávce, mělčí) |
| `enrich.use_shikimori` | další zdroj „podobných anime" kandidátů (default vypnuto, +1 request/seed) |
| `recommend.user_cf_report_top` | strop karet v `cf_recommendations.html` (0 = bez stropu) |
| `recommend.use_user_cf` + `user_cf_*` | senpai pipeline (viz `--user-cf`): počet senpai, velikost poolu, min. plný překryv… |

Plný seznam parametrů (včetně výchozích hodnot) je v `config.example.yaml`.

---

## Architektura

```
animodel/
  mal.py            parser MAL XML exportu
  sources/
    __init__.py     sdílené utility (progress výpisy, Result typ pro úspěch/selhání)
    cache.py        sdílený cache primitiv (FileCache, cached_fetch) -- 1 klíč = 1 soubor
                    (každý klient dostane kořen cache a doplní si podsložku)
    http.py         sdílený retry/backoff driver (request_with_retry, rate limitery)
    jikan.py        MAL data + recommendations + search (Jikan-kompat. API: Tenrai/Jikan)
    anilist.py      AniList tagy + recommendations + tag-search + user-based CF
    shikimori.py    volitelný zdroj "podobných anime" (/similar), default vypnuto
                    (tvar odpovědi ověřen: prostý seznam, váží se pozicí)
  attributes.py     kanonizace + deduplikace atributů napříč zdroji
  intensity.py      osa emocionální náročnosti: lexikon, prefill, --gen-intensity
  usercf.py         user-based CF: senpai pipeline (discovery -> plné seznamy -> výběr)
  selection.py      model výběru: čemu se mezi populárními tituly vyhýbáš
  season.py         sezónní doporučení (--season): pokračování + nové tituly + finále
  history.py        záznam běhů (klíč = otisk seznamu) + zpětná vazba z nového exportu
  backtest.py       časová validace nad my_finish_date (--backtest): disjunktní
                    okna, bootstrap po franšízách, nové franšízy vs. pokračování
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

MAL data se tahají z **Jikan-kompatibilního API**. Default je
[Tenrai](https://tenrai.org/) (`https://api.tenrai.org/v1`) — spolehlivá náhrada
Jikanu s identickým v4 schématem; původní Jikan (`https://api.jikan.moe/v4`) měl
od července 2026 trvalé 504 výpadky. Přepnutí je jen změna
`enrich.anime_api_base_url` v configu (cache je společná — klíč dle endpointu,
ne hostu). AniList je druhý, nezávislý zdroj (tagy, rec graf, user-CF).

Všechna API mají rate-limity; respektuj je (klient cachuje). Komunitní skóre se
bere primárně z MAL, fallback AniList — záměrně se **neprůměrují** (jsou silně
korelované, průměrování nepřináší informaci a riskuje zkreslení).

### Úklid cache

Každý zdroj má vlastní podsložku, takže jde invalidovat jednotlivě:

```
cache/
  mal/        MAL data přes Jikan-kompat. API (/full, /staff, /recommendations, sezóny)
  anilist/    AniList Media, doporučení, universum tagů
  cf_al/      user-CF: stránky sledujících + seznamy uživatelů
  shikimori/  /similar (jen když enrich.use_shikimori)
```

```bash
rm -r cache/mal        # jen MAL data, AniList i CF zůstanou
```

> Verzované klíče: AniList Media je od 2026-08-05 na `_v3` (přibyl `staff`
> a `countryOfOrigin`). Starší `mal_*_v2.json` se už nečtou — smaž je:
> ```bash
> find cache/anilist -name 'mal_*_v2.json' -delete
> ```
>
> Pokud máš cache z verze před 2026-07-26, leží MAL data přímo v `cache/`.
> Přesuň je jednou do `cache/mal/`, jinak se stáhnou znovu (u ~10 tis.
> titulů jde o hodiny):
> ```bash
> mkdir -p cache/mal && find cache -maxdepth 1 -type f -name '*.json' -exec mv {} cache/mal/ \;
> ```

Cache nemá expiraci (záměr — maž ručně, když chceš čerstvá data; 1 request =
1 soubor, takže jde smazat i jen část). **Pozor na verzované klíče:** když se
změní schéma odpovědi, klient přejde na nový suffix (`mal_{id}_v2`,
`userlist_{uid}_v2`) a **staré soubory zůstanou ležet** — nikdy se už nečtou,
ale zabírají místo. Po přechodu na `_v2` u user-CF seznamů to bylo přes 100 MB
mrtvých dat. Osiřelé verze poznáš podle chybějícího suffixu — maž je ale
**vylučovací** podmínkou, ne shell wildcardem:

```bash
# správně: vše krom _v2
find cache/cf_al -name 'userlist_*.json' ! -name '*_v2.json' -print   # kontrola
find cache/cf_al -name 'userlist_*.json' ! -name '*_v2.json' -delete
```

> `rm cache/cf_al/userlist_*[0-9].json` **nepoužívej** — `v2` končí číslicí,
> takže by ten vzor smazal i aktivní `_v2` soubory.
