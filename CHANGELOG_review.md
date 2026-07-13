# Changelog — opravy a doplnění

*Řazeno přesně podle zadané priority: nejdřív co zkresluje výsledky, pak výkon,
pak chybějící funkcionalita. Každá položka odkazuje na finding z `animodel_code_review.md`.*

Ověřeno po každém kroku pomocí `animodel_test_harness.py` (bez sítě, na syntetických
datech) + syntax check přes `ast.parse` na všech 11 upravených souborech.

---

## 0. Formát logování a progress výpisů (dodatečná oprava)

Šlo přesně o to, cos popsal. Ověřil jsem to empiricky (test v `sources/__init__.py`
docstringu) — dvě věci se skládaly dohromady:

**Bufferování.** Všech 7 `print(..., end="\r")` progress výpisů (2× `jikan.py`,
5× `anilist.py`) nikdy explicitně neflushovalo. Když stdout není TTY
(přesměrování do souboru, `tee`, spousta IDE konzolí, CI), Python ho plně
bufferuje — text se neobjeví vůbec, dokud se buffer nezaplní nebo proces
neskončí. `log.warning`/`log.error` naproti tomu jde přes stderr
(nebufferované), takže se objeví okamžitě. Výsledek: v logu vidíš prakticky
jen "chyby", zatímco progress buď zmizí, nebo se ukáže najednou zpackaný na
konci běhu.

**Chybějící `logging.basicConfig()`** — nikde v kódu nebyla, takže Python
spadl na `logging.lastResort`: ukazuje jen WARNING+, ale BEZ formátu (žádné
"WARNING:", žádná časová značka) — hlášky tak vypadaly jako obyčejný text,
nerozeznatelné od progress výpisů.

Oprava:
- nový `animodel/sources/__init__.py` s `progress()`/`progress_done()`/`status()`
  — flushují vždy, `progress()` navíc dorovnává mezerami na šířku předchozího
  volání (ať kratší text nenechá viditelný zbytek delšího předchozího řádku)
- `cli.py::main()` teď volá `logging.basicConfig(format="%(levelname)s [%(name)s] %(message)s", ...)`
- nový `--verbose`/`-v` flag: default level je WARNING (jen skutečné problémy),
  `--verbose` zapne i INFO (routinní retries)
- v `jikan.py` i `anilist.py` jsem **snížil úroveň** rutinních "429, zkouším
  znovu" hlášek z WARNING na INFO — teď se defaultně vůbec nezobrazí (nejsou
  problém, API je prostě rate-limitované), zatímco skutečné vyčerpání pokusů
  zůstává ERROR a vidíš ho vždycky

Ověřeno reprodukcí přesně tvýho symptomu (skript v `/tmp`, viz commit) — před
opravou stdout obsahoval jeden slepenec `\r`-ů zobrazený až na konci,
zatímco stderr měl 4 čisté "Rate limit" řádky. Po opravě: stdout ukazuje
progress průběžně (díky flush), stderr defaultně jen skutečnou chybu.

---

## 9. "Private User" 404 — odfiltrovat co nejdřív, bez ztráty místa

Nejdřív k té první možnosti z otázky (jiný způsob, jak získat seznam
soukromého uživatele): není. Ověřil jsem si to přímo proti AniListově
dokumentaci -- OAuth2 je potřeba přesně pro čtení privátních seznamů,
a bez autorizace jako TEN konkrétní uživatel se k datům nedá dostat.
Nejde o mezeru, kterou by šlo obejít, je to schválně takhle -- takže jsem
se držel jen druhé možnosti (časnější odfiltrování).

**Ověřil jsem i to, jestli AniList nabízí nějaké pole jako `isPrivate`,
podle kterého by šlo poznat soukromý profil PŘED pokusem o stažení seznamu**
-- stáhl jsem si celou referenci `User` typu (`docs.anilist.co/reference/object/user`)
a nic takového tam není. Zjistí se to jedině pokusem.

Daná fakta (žádná predikce, jen pokus) znamenají, že nejlepší dostupná
oprava je: (1) **nevyřazovat kandidáty na `top_users` HNED při výběru** --
`similar_users_recommendations` teď drží celý seřazený pool a prochází ho
POSTUPNĚ, dokud nenajde `top_users` POUŽITELNÝCH (ne jen zkusených) --
soukromý/smazaný účet se přeskočí a jde se na dalšího kandidáta, místo aby
natrvalo sebral slot; (2) **využít cache mezi běhy** -- už existující
`_cf_save(ck, ["UNKNOWN", []])` pro permanentní selhání teď funguje jako
plnohodnotný filtr: na PŘÍŠTÍM běhu se tenhle uid ani nezkusí, cache-check
na začátku smyčky ho přeskočí bez jediného requestu.

Nový parametr `scan_budget_factor` (default 3.0) je pojistka proti
neomezenému skenování, kdyby byl podíl soukromých účtů extrémně vysoký --
`max_attempts = top_users * scan_budget_factor`. Když se dosáhne stropu a
nenasbíralo se dost použitelných uživatelů, zaloguje se o tom jasná
zpráva s návrhem zvýšit `scan_budget_factor`.

Cestou jsem opravil i zavádějící log zprávu (`_ul_failed` počítalo
permanentní i dočasná selhání dohromady, ale text hlásil jen "dočasně") --
teď jsou to dva oddělené počitadla (`_ul_failed_transient` vs.
`_ul_skipped_empty`).

Ověřeno na skutečné `similar_users_recommendations()` se dvěma scénáři:
(a) 8 soukromých před 4 reálnými (schválně nejhorší pořadí) -- ukázalo
mimo jiné, že `scan_budget_factor=3.0` může být v extrémních případech
málo, proto je teď nastavitelný; (b) 6 soukromých a 6 reálných prokládaně
(realističtější) -- `top_users=5` správně našlo přesně 5 použitelných po
proskenování 10 z 12 kandidátů, bez ztráty jediného slotu.

---

## 8. Refaktor na `Result` typ — nahrazuje mutable side-channel

Popsal jsi to přesně: `_last_failure_kind` byl fragilní vzor (mutable stav
klienta, co se čte samostatnou metodou HNED po volání a je snadné ho
přečíst ve špatný moment) a měl jsem `_mark_from_status` definovanou, ale
nikde nepoužitou -- oba postřehy sedí a `_get`/`_post` skutečně měly
klasifikaci řešenou dvěma nezávislými, mírně odlišnými cestami.

**Poctivě:** i po opakovaném pročtení starého kódu jsem nenašel konkrétní
řádek, kde by `_last_failure_kind` čtený hned po `result is None` dal
špatnou odpověď na 521 -- klasifikace (5xx mimo `PERMANENT_HTTP_CODES` →
transient) vypadala logicky správně i tam. Nedokážu tedy s jistotou říct
PROČ se to v praxi chovalo jinak. Co ale udělat šlo: přepsat to na typ, kde
se tahle třída chyby nedá vůbec vyrobit, a přesně tohle jsem udělal.

**`Result` (`sources/__init__.py`)** — `@dataclass` s `ok`, `data`,
`permanent`. `_post`/`_get` přejmenovány na `_request`, vrací `Result`
místo `dict | None`. Klasifikace se navíc dělá JEDNOU, přímo na
`resp.status_code`, PŘED `raise_for_status()` -- ne dodatečně v except
větvi jako dřív. Sdílená čistá funkce `is_permanent_status(code)` nahrazuje
`FailureTrackingMixin` úplně (ta třída už v kódu není).

**`jikan.py`** — `_get()` rozdělen na `_request()` (čistě síťová vrstva,
vrací `Result`) + `_get()` (JEDINÉ místo, které rozhoduje o cache -- tři
jasné větve: úspěch / permanent / transient). Volající (`get_anime`,
`get_anime_staff`, `get_recommendations`, `search_anime`) se nemuseli měnit
vůbec, protože `_get()`'s vnější rozhraní zůstalo stejné.

**`anilist.py`** — `_request()` nemůže centralizovat cache stejně jako
Jikan (8 volajících míst má každé jiný cache formát/klíč), takže těch 8
míst (`get_anime`, `get_anime_batch` fallback, `get_recommendations`,
`search_by_tags`, `_media_popularity`, oba `_user_cf` cache body,
`QUERY_USER_NAMES`) teď čte `result.ok`/`result.data`/`result.permanent`
přímo z návratové hodnoty -- ne přes samostatné volání metody na klientovi.

**Ověřeno, tentokrát na skutečné `similar_users_recommendations()`, ne jen
izolovaně na `_request`:** připravil jsem cache přesně tak, jak by ji
nechal reálný `get_anime` běh (MAL→AniList ID mapping), namockoval
`_request` na `Result.failure(permanent=False)` (přesně to, co dá 521 po
vyčerpání retry) a zavolal celou metodu se 2 seedy —
`find cache -iname 'watchers_*' | wc -l` vrátilo **0**. Se
`Result.failure(permanent=True)` (simuluje GraphQL/400 chybu) stejný test
vytvořil `watchers_101.json` i `watchers_102.json` správně. Obě větve teď
ověřené přímo na produkční metodě.

---

## 7. Stejné ošetření rozšířeno na zbytek `sources/` a refaktorováno sdíleně

*(Pozn.: `FailureTrackingMixin` popsaná v týhle sekci byla o kolo později
nahrazená `Result` typem -- viz sekce 8 výš. Nechávám tuhle sekci jako
záznam postupu, ne jako popis současného stavu kódu.)*

Na žádost jsem stejnou permanent/transient distinkci (viz bod 6) natáhl i na
zbylá místa, kde se cachuje výsledek externího volání -- a přesunul
klasifikaci na jedno sdílené místo, ať ji `jikan.py`/`anilist.py`
nemají každý svoji kopii.

**`FailureTrackingMixin` (`sources/__init__.py`)** — nová sdílená třída:
`PERMANENT_HTTP_CODES`, `_reset_failure()`, `_mark_permanent()`,
`_mark_transient()`, `_mark_from_status()`, `last_failure_was_permanent()`.
Jak `JikanClient`, tak `AniListClient` z ní teď dědí -- `PERMANENT_HTTP_CODES`
je ověřeně (`is`) tatáž instance u obou, ne dvě kopie stejného čísla.

**`anilist.py`** — `_post()` přepsán na `self._mark_permanent()`/
`_mark_transient()` místo přímého nastavování `_last_failure_kind`; lokální
duplicitní `last_failure_was_permanent()` smazána (dědí se). `get_anime()`
a `get_recommendations()` teď při `result is None` navíc kontrolují
`last_failure_was_permanent()` -- při trvalém selhání se cachuje `{}`/`[]`
jako konečné, při dočasném se cache nedotkne (stejný vzor jako u
`_user_cf` z minula).

**`jikan.py`** — jednodušší případ: všechny veřejné metody
(`get_anime`, `get_anime_staff`, `get_recommendations`, `search_anime`)
cachují výhradně přes centrální `_get()`, takže stačila JEDNA úprava tam,
ne úprava v každé metodě zvlášť. `_get()` teď: (a) 404 klasifikuje jako
permanent (chování při cachování beze změny, jen teď je to i navenek
viditelné přes `last_failure_was_permanent()`), (b) nově i 400/422 selže
HNED bez plýtvání celým retry schedulem (ověřeno mockovaně: 0.0s místo
plánovaných ~17-47s), (c) síťové chyby a vyčerpaný 429 zůstávají transient,
beze změny v cachovacím chování.

Ověřeno mockovanými testy (bez potřeby živé sítě) pro oba klienty zvlášť:
400→permanent+rychlé selhání, 404→permanent+cachuje se, 500→transient+plný
retry beze změny, čistá síťová chyba→transient, GraphQL chyba→permanent.

---

## 6. `_user_cf` — cache se nesmí kazit dočasným selháním, ale trvalé smí

Tvůj popis (prázdné `[]` cache soubory po prvním běhu, méně jich po smazání
a rerunu) přesně sedí na to, co jsem po prostudování kódu potvrdil:
`_post()` vracel `None` uniformně pro VŠECHNO selhání (400, 404, 429
vyčerpané, 500, timeout, GraphQL chyba) a volající metody v
`similar_users_recommendations` to nedokázaly rozlišit — `fetch_failed`/
"cache nezměněna" logika tam už BYLA (a fungovala správně pro "nekešuj při
selhání"), ale neuměla poznat rozdíl mezi "zkus příště znovu" a "tohle se
nikdy nezmění".

**Nová klasifikace v `AniListClient`:**
- `PERMANENT_HTTP_CODES = {400, 404, 422}` — natrvalo špatný request
  (špatné ID, malformed query). Retry se stejnými parametry nikdy neuspěje.
- GraphQL chyba (`"errors"` pole i při HTTP 200) — taky klasifikováno jako
  permanent (typicky problém s konkrétní proměnnou, ne dočasný stav serveru).
- Vše ostatní (5xx, timeout, síťová chyba, vyčerpaný 429) — transient.
- `client.last_failure_was_permanent()` — nová veřejná metoda, volající si
  po `result = self._post(...)` (když `result is None`) může zeptat, co se
  stalo.

**Bonus oprava našlá při testování:** retry loop předtím čekal celý
backoff schedule (5+15+40+90s) i na 400, přestože je zjevné, že to nikdy
nepovede jinak. Teď se u permanentních kódů vzdá HNED po prvním pokusu —
ověřeno mockovaným testem (0.2s místo 60s), zatímco 500/timeout/GraphQL
pořád projde celým plánovaným retry (ověřeno, taky mockovaně, 60.0-60.1s).

**Použití ve `similar_users_recommendations` (obě místa, watchers i
userlist fetch):** při selhání se teď kešuje, JEN KDYŽ `last_failure_was_permanent()`
vrátí True (částečné/prázdné výsledky jsou v tom případě konečné, retry by
stejně nepomohl). Při dočasném selhání se cache nedotkne vůbec — přesně jak
jsi chtěl.

Neřešeno vědomě: stejná distinkce by se dala použít i v `get_anime`/
`get_recommendations` (`sources/anilist.py`) a analogicky v `jikan.py` --
zeptal ses konkrétně na `_user_cf`, takže jsem se držel tam. Mechanismus
(`last_failure_was_permanent()`) je ale teď hotový a dal by se použít i tam
jako levný follow-up.

---

## 5. Circuit breaker pro nedostupnou službu (nalezeno při testování Shikimori)

Při ověřování Shikimori integrace (viz výš) jsem narazil na reálný problém:
v sandboxu bez síťového přístupu `_gather_candidates` visel na desítky minut,
protože žádný ze tří klientů (jikan/anilist/shikimori) neměl žádnou ochranu
proti "celá služba je nedostupná" (na rozdíl od "jen rate-limited, retry
pomůže"). Každý neúspěšný request na každý seed zvlášť si odsedí celý svůj
retry+backoff cyklus -- u AniListu s adaptivním zpomalením klidně 60s na
JEDNO volání.

Řešení má dvě části a přiznávám k tomu i vlastní chybu cestou:

- **První verze breakeru byla postavená na `except Exception` a byla to
  fakticky mrtvá větev.** `JikanClient`/`AniListClient`/`ShikimoriClient`
  svoje selhání interně pohlcují a vždy vrací `[]`/`None`, nikdy nevyhodí
  výjimku ven -- takže `try/except` kolem volání nikdy nic nechytilo.
  Vypadalo to opraveně (kód se tvářil, že něco dělá), ale instrumentovaný
  test (počítání skutečných volání) ukázal, že se to chovalo úplně stejně
  jako bez breakeru. Opraveno na měření ELAPSED TIME bez ohledu na to, co
  se vrátí -- pomalá odpověď (5s+) je jediný spolehlivý signál, že klient
  interně vyčerpal retry.
- **A2 tag-search sekce** (`self.enr.anilist.search_by_tags(...)`, volaná
  jednou mimo per-seed smyčku) měla stejnou díru, jen mimo hlavní cyklus --
  dodatečně objeveno, když breaker v A1 fungoval, ale celkový čas pořád
  neseděl. Sdílí teď stav s "AniList-rec" breakerem (stejná služba).

Práh: 20s promarněných na jeden zdroj, než se přeskočí zbytek dávky. Ověřeno
živě (v tomhle sandboxu, kde jsou všechny tři služby blokované egress
proxy) — `_gather_candidates` teď doběhne za ~2 min i v absolutním
nejhorším případě (všechny 3 zdroje nedostupné), místo dřívějšího
neomezeného visení (desítky minut až nekonečno).

**Neošetřeno zatím:** `_user_cf` (sekce B, `--user-cf`) dělá vlastní
vícekrokové AniList volání napříč více uživateli a nesdílí tenhle breaker.
Je to už dnes opt-in a v komentářích označené jako pomalé, ale při skutečné
nedostupnosti AniListu by čelilo stejnému riziku. Nechávám jako známý,
zapsaný gap pro příště, ne jako součást týhle opravy.

---

## 4. Nový zdroj kandidátů: Shikimori

Na základě průzkumu (Annict/Bangumi/Shikimori, viz diskuze) vyšlo, že jediná
jasně a opakovaně potvrzená přidaná hodnota napříč třemi prověřenými
regionálními alternativami je Shikimoriho funkční `/animes/{id}/similar`
endpoint — ani bohatší tagový prostor, ani lepší přístup k uživatelským
seznamům pro CF se nikde nepotvrdily (Annict nemá tagy vůbec a jeho
per-uživatelské skóre je hrubší než AniListovo; Bangumi má tagy, ale
volné/nekurátorované, a chybí mu oficiální cesta k "kdo tohle ohodnotil";
Shikimori samo o tagách/user-rates nemá o nic víc než co už dává Jikan).

Přidáno tedy úzce, jen na tohle:

- **`animodel/sources/shikimori.py`** (nový) — minimální klient jen pro
  `get_similar(mal_id)`. Využívá zkratku potvrzenou nezávisle přes projekt
  animeApi (dva konkrétní shodné příklady MAL/Shikimori ID): Shikimori ID
  je až na výjimky STEJNÉ číslo jako MAL ID, takže žádná zvláštní resoluční
  služba není potřeba — při 404 (mimo pravidlo) se to prostě přeskočí,
  logováno na INFO (očekávaný jev, ne chyba).
- **`recommend.py::_gather_candidates`** — nová větev vedle MAL-rec/AniList-rec,
  stejný try/except+log vzor, váhuje podle pozice v seznamu (`rank_hint`),
  protože jsem nemohl ověřit, jestli endpoint vrací i explicitní skóre
  podobnosti — zkontroluj skutečnou odpověď při prvním běhu a uprav váhování,
  pokud ano.
- **`enrich.py::Enricher`** — `self.shikimori`, stejný podmíněný vzor jako
  `self.anilist` (`cfg.enrich.use_shikimori`).
- **`config.py` / `config.example.yaml`** — `use_shikimori: bool = False`
  (default vypnuto, protože endpoint tvar nebyl ověřený naživo).

Rate limit pro Shikimori (`REQUEST_DELAY = 1.0` v `shikimori.py`) je
konzervativní odhad, ne ověřené číslo z jejich aktuální dokumentace — bez
síťového přístupu v dev sandboxu jsem si ho nemohl ověřit. Doladit podle
skutečného chování.

---

## 1. Zkreslují výsledky

**HEAVY/LIGHT lexikon (`taste.py`)** — Přes web search jsem ověřil, co z 35+14
položek odpovídá reálné AniList tagové/žánrové taxonomii. Potvrzeno: `tragedy`,
`survival`, `war`, `politics`, `philosophy`, `suicide`, `gore`, `bullying`,
`dystopian`, `psychological`, `horror`, `thriller`, `drama`, `military`,
`suspense`. Odstraněno bez důkazu a bez náhrady: `loss`, `grief`, `existential`,
`trauma`, `mature_themes`, `organized_crime` — tyhle by v `canon()` výstupu
pravděpodobně nikdy nic netrefily. `death`, `depression`, `crime` jsem nechal
jako best-guess (běžné koncepty, ale bez přímého důkazu).

Přidal jsem `TasteModel.unmatched_intensity_keywords()` — po `fit()` řekne
přesně, které klíče z HEAVY/LIGHT se v datech NIKDY neobjevily (kontroluje
proti všem pozorovaným atributům, ne jen těm, co prošly prahem pro fitnutý
efekt — to by dávalo falešně pozitivní "nedošlo" u vzácných, ale reálných
shod). `cli.py` to teď po fitu vypíše samo. Na tvém reálném (bohatém) tagovém
prostoru čekám podstatně méně nematchů, než ukázal test na mém zjednodušeném
fixture — to je očekávané, ne chyba.

**`AniListClient.get_recommendations` (`sources/anilist.py`)** — Dřív se při
selhání requestu i tak zapsalo `[]` natrvalo do cache (na rozdíl od
pečlivého `get_anime`, který mezi "selhalo" a "potvrzeně prázdné" rozlišuje).
Teď se při `result is None` cache nedotkne a jen zaloguje warning — zkusí se
to znovu příště.

**`mal.py::parse_export`** — `series_animedb_id` byl jediné pole bez `or 0`
fallbacku, co mají všechna ostatní. Sjednoceno, ať prázdný (ale přítomný) tag
nespadne na `ValueError`.

---

## 2. Zhoršují výkon

**Dvojité klastrování** — `TasteModel.fit()` teď bere `n_clusters` jako
parametr místo aby vždycky klastrovalo interně s `k=None` a čekalo na
přepsání zvenku. `cli.py` (řádek, co dřív volal `model._fit_clusters(cfg...)`
hned po `fit()`) teď jen `model.fit(titles, n_clusters=cfg.model.n_clusters)`.
Stejná oprava v `README.md`, jehož příklad měl tentýž zdvojený vzor.

**`recommend.py::_cluster_fit`** — `Cluster.signature` teď nese kanonický
klíč rovnou (`taste.py::_fit_clusters` ho stejně měl po ruce), takže
`_cluster_fit` nemusí pro každého kandidáta × každý klastr × každou položku
signatury dělat lineární průchod přes všechny `model.effects` podle shody
labelu+kategorie. Opraveno na obou místech v `taste.py` (dataclass + stavění
signatury) a v `recommend.py`. Musel jsem doladit i `report.py` (2 místa),
který signaturu rozbaloval jako 3-tuple — jinak by tahle oprava potichu
rozbila HTML report.

---

## 3. Doplněná funkcionalita

**Signál po režisérech/scenáristech** — `jikan.py` měl hotové
`get_anime_staff`/`get_staff_batch`, ale nikde nenapojené. Teď:
- `attributes.py::build_attributes` bere nový `staff` parametr, rozlišuje
  `director`/`writer` jako samostatné kategorie na osobu+roli (ne jen na
  osobu — dobrý režisér nemusí být dobrý scenárista)
- nové kategorie jsou vyloučené z mood-klastrování (to filtruje jen
  genre/theme/tag/demographic), takže preference tvůrce neovlivní "náladu"
- `EnrichCfg.include_staff: bool = False` (config.py + config.example.yaml) —
  default vypnuto, protože stojí 1 Jikan volání navíc na titul
- `enrich.py::Enricher.enrich_ids` staff stáhne a předá dál, jen když je flag zapnutý
- `jikan.py::list_all_staff` teď importuje `DIRECTOR_POSITIONS`/`WRITER_POSITIONS`
  z `attributes.py` místo duplicitní definice na dvou místech

**`--analyze` flag** — `print_series_groups` byla napsaná "pro `--analyze`",
který ale v `cli.py` neexistoval. Přidán, včetně early-return (nepočítá celý
model, jen stáhne/použije cachovaná Jikan data a vypíše franšízové skupiny).

**`candidates_per_seed`** — definovaný v configu, nikde čtený. Teď omezuje,
kolik MAL-rec/AniList-rec doporučení se per seed skutečně použije
(`recs[:self.rc.candidates_per_seed]`).

**Rozbitý import v `series.py::aggregate_entries`** — opraveno
(`from .mal import MalEntry` místo neexistujícího `mal_parser`), ale
**záměrně nenapojeno** do aktivní pipeline. Tahle funkce dělá kolaps franšízy
na jeden záznam s max skóre — konkuruje aktivně používanému vážení `1/√k`
v `enrich.py`. Rozhodnutí, jestli chceš obojí (např. jako přepínatelný
`--aggregate-mode`), nebo tohle smazat, nechávám na tobě — je to
architektonická volba, ne bug fix.

**Drobnosti**: stale docstring hlavičky (`mal_parser.py` → `mal.py`,
`jikan_client.py` → `jikan.py`, `anilist_client.py` → `anilist.py`); holé
`except Exception: pass` v `recommend.py::_gather_candidates` teď aspoň
loguje warning s výjimkou; opakované 429 v `jikan.py::_get`, co vyčerpaly
všechny pokusy, se teď taky zaloguje (dřív se to tiše vracelo jako `None`
bez stopy, na rozdíl od síťových chyb).

---

## Co jsem NEudělal

- Nesmazal jsem `aggregate_entries`/`_series_title`/`_common_prefix` — jen
  opravil import. Řekl jsi "raději doplnit než rušit", a dá se poznat, jak
  to bylo myšlené, takže jsem to nechal jako funkční, ale nenapojenou
  alternativu k rozhodnutí.
- `jikan.py::search_anime` jsem nechal beze změny — nenašel jsem jasné místo,
  kam by měla být napojená bez další diskuze o designu discovery fáze.
- Neřešil jsem `sfw=false` v tý samé metodě, ze stejného důvodu (nepoužívaná
  metoda, netriviální rozhodnutí).
