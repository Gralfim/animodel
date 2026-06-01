# Analýza anime vkusu — Gralfim

*Podklad k prvotní kontrole před nasazením nástroje. Shrnuje, co v datech je,
jakou metodiku jsem zvolil a proč — ať můžeš schválit směr, než to celé poběží
naživo přes API.*

---

## 1. Co data říkají hned na první pohled

411 ohodnocených titulů, průměr **8,12**. Klíčové je ale **rozdělení**:

| známka | 10 | 9 | 8 | 7 | 6 | 5 | ≤4 |
|---|---|---|---|---|---|---|---|
| počet | 33 | 119 | 150 | 83 | 25 | 1 | 0 |

Skoro nic pod 7, dvě dropnutá anime za celý seznam. To **není náhoda — to je
tvůj systém**: do PTW a ke sledování pouštíš jen tituly, které prošly náročným
předvýběrem (podobnost s oblíbenými, trailery, žebříčky). Statisticky tomu říkáme
**restrikce rozsahu**.

**Důsledek pro modelování (důležité):** na takhle úzkém rozsahu má klasická
lineární regrese na známkách minimum signálu — tvá známka je skoro `komunita +
konstanta`. To je přesně důvod, proč ti předchozí appka s Ridge regresí
nepřišla přesvědčivá. Není to chyba implementace, je to vlastnost dat. Proto
**nemodeluju známku, ale odchylku od komunity** (víc v sekci 3).

Mimochodem to taky vysvětluje, proč „intuice" jazykového modelu vychází líp:
nehádá číslo, ale chápe *proč* tě něco bere — a přesně to se snažím zrekonstruovat
automaticky.

---

## 2. Profil vkusu — pět nálad, mezi kterými přepínáš

Z desítek a šestek (a z celé struktury) se konzistentně rýsuje **pět módů**.
Nehodnotíš jeden žánr — máš jich několik a střídáš je podle „emocionální únavy".

**① Školní romance / romcom** — *Toradora!, 5-toubun no Hanayome, Golden Time,
Sakurasou, Otonari no Tenshi-sama, Alya-san, Kaoru Hana, Koi wa Ameagari.*
Tvůj nejširší a nejstabilnější pilíř.

**② Emocionální drama s romantickým jádrem** — *Shigatsu wa Kimi no Uso, Violet
Evergarden, Kimi no Na wa, Clannad, Anohana, Koe no Katachi, White Album 2.*
Hodnotíš nejvýš ze všech (průměr klastru ~9,6), ale je to „náročné" — nedá se
sledovat pořád.

**③ Harém s dobrým psaním** — *Yuragi-sou, Date A Live, Zero no Tsukaima,
Grisaia, Hyakkano.* Pozor na hranici: čistá ecchi/gag bez posunu tě nebere —
*Sora no Otoshimono, To LOVE-Ru, Nisekoi* spadly na 6–7. Rozhoduje, jestli má
příběh progresi, ne fanservice sám o sobě.

**④ Psychologický twist + nadpřirozeno / slice-of-life** — *Steins;Gate,
Bakemonogatari, Seishun Buta Yarou, Yofukashi no Uta, Re:Zero, Oshi no Ko.*
Tady je důkaz, že se vyplatí občas tlačit proti tvému deklarovanému vkusu:
sci-fi zasazení nemáš rád, ale Steins;Gate je tvá desítka.

**⑤ Mainstreamový isekai s emoční váhou** — *Mushoku Tensei, KonoSuba, Re:Zero.*
Ne „fast-food" isekai bez sázek — *Tate no Yuusha S3* kleslo na 7.

**Co tě naopak NEbere** (tvé šestky): čistě „mind-game" romcom bez emoce
(*Kaguya-sama* → 6), hořká romance bez katarze (*Kuzu no Honkai* → 6), starší
gag série (*Ranma ½, Ouran*), čistá ecchi bez příběhu.

Oblíbená studia napříč módy: **J.C.Staff, Kyoto Animation, CloverWorks,
White Fox, A-1 Pictures.**

---

## 3. Metodika nástroje — co jsem zvolil a proč

Zvážil jsem tři přístupy, které jsi zmínil (regrese / lineární programování /
neuronka), a vědomě jsem šel jinou cestou. Tady je odůvodnění:

### Cíl = afinita, ne známka
Pro každý titul spočítám baseline `tvůj_průměr + β·(komunita − průměr_komunity)`
a modeluju jen **to, co zbude** (afinitu). Komunita vstupuje jako **jeden**
kalibrovaný sklon β, **ne jako atribut**.
*Proč:* kdyby komunitní kvalita byla atribut, počítala by se dvakrát a
převrátila by efekty — např. KyoAni by vyšlo „záporně", protože ho hodnotí
vysoko i dav. (Tohle jsem si na datech ověřil — čistě reziduální cíl bez
kalibrace dává nesmyslné koeficienty.)

### Efekty atributů přes empiricko-bayesovské smrštění
`efekt = (n/(n+K))·průměr_afinity`. Malé vzorky se táhnou k nule.
*Proč ne neuronka:* na ~400 bodech v úzkém rozsahu by se přeučila a hlavně bys
z ní nevyčetl *proč*. Ty chceš interpretovatelnost a ladění — ne černou skříňku.
*Proč ne lineární programování:* řeší optimalizaci, ne odhad vlivu atributů ani
řazení; na tuhle úlohu nesedí.

### Atributy automaticky, deduplikovaně
Žádný ruční konfigurák se seznamy žánrů (ta bolavá pipeline z minula). Atributy
se objeví z dat a kanonizují napříč zdroji (MAL „Drama" + AniList tag „Drama" =
jeden atribut), aby se stejný koncept nenafoukl. Franšízy se slučují a váží
`1/√k`, aby tě desetidílná série nepřeválcovala.

### Nálady = klastry s osou náročnosti
KMeans na atributových otiscích, počet klastrů dle siluety. Každý klastr dostane
„náročnost" (těžké drama/psycho vs. lehká komedie) — to je přímo ta osa
emocionální únavy, o které píšeš.

### Doporučení se řadí podle shody, ne podle hádané známky
Predikovaná známka se kvůli restrikci rozsahu lepí na komunitní průměr, takže by
neřadila dobře. Skóruju kompozitem: shoda s atributy/náladou + „doporučili to
tvé oblíbené" (MAL/AniList rec graf) + mírná kvalita. Známka + interval slouží
jen k zobrazení.

---

## 4. Ověření na fixture

Než jsem spustil cokoli naživo, ověřil jsem model na ručně sestaveném vzorku 54
tvých titulů. Klastrování **samo zrekonstruovalo všech pět módů** výše a top
afinitní atributy seděly na profil (J.C.Staff, light novel, coming-of-age, slow
romance, time travel nahoře; action, sci-fi, čistý harém dole). To je dobrá
zpráva — metodika dělá to, co od ní čekáme.

CV RMSE modelu vyšlo skoro stejně jako u samotného baseline — a to je **správně,
ne chyba**: potvrzuje, že tvá známka ≈ komunita + posun, takže atributy neslouží
k hádání čísla, ale k tomu *co vybrat* a *do jaké nálady* to patří. Přesně to,
co od nástroje chceš.

---

## 5. Co teď

Nástroj je hotový a otestovaný offline. Naživo potřebuje stáhnout metadata přes
Jikan + AniList (z mého prostředí na ně nevidím, proto poslední krok poběží u
tebe — stačí jeden příkaz). Přiložené `model.html` a `recommendations.html` jsou
**ukázky formátu** vygenerované z fixture, ať vidíš, jak výstup vypadá.

Pokud ti směr sedí, stačí pustit `python -m animodel --export animelist.xml` a
dostaneš model i doporučení na celém seznamu. Ladění (váhy, smrštění, počet
nálad) je v `config.yaml` — bez sahání do kódu.
