"""
shikimori.py — Shikimori (rusky mluvící obdoba MAL/AniList) klient, jen pro
"podobná anime" doporučení -- viz code review diskuze o obohacení
recommend.py::_gather_candidates o další, nezávislý zdroj kandidátů.

Dokumentace: https://shikimori.one/api/doc
Rate limit: nemám k dispozici přesné aktuální číslo z jejich docs (žádný
síťový přístup v dev sandboxu) -- REQUEST_DELAY níž je konzervativní odhad,
DOLAĎ podle skutečného chování při prvním ostrém běhu.

Proč jen "similar", ne plnohodnotný klient jako Jikan/AniList: code review
prověřil i tagy (Shikimori nemá nic bohatšího než základní žánry, spíš
srovnatelné s tím, co už dává Jikan) a user-list přístup pro CF (nejistý,
nepotvrzeno, jestli jde filtrovat podle anime napříč uživateli) -- jediná
jasně a opakovaně potvrzená přidaná hodnota je funkční
`/animes/{id}/similar` endpoint, tak se na něj klient omezuje. Když by se
časem ukázalo, že tagy/user-rates přece jen k něčemu jsou, tenhle soubor je
přirozené místo, kam je přidat.

KLÍČOVÝ ZKRATKOVÝ TRIK (ověřeno přes nezávislý cross-reference projekt
animeApi, ne přímo přes Shikimori samotné): Shikimori ID je až na výjimky
STEJNÉ číslo jako MAL ID (potvrzeno na dvou konkrétních příkladech --
Cowboy Bebop myanimelist=1/shikimori=1, "Iruma-kun 3rd Season"
myanimelist=49784/shikimori=49784 -- a jejich dokumentace to i explicitně
říká: "shikimori IDs are basically the same as myanimelist IDs"). Díky
tomu se dá MAL ID použít rovnou jako Shikimori ID bez zvláštní resoluční
služby -- při 404 se to prostě přeskočí (viz get_similar níž), místo aby
se stavěla složitější (a další závislost přidávající) cesta přes title
search nebo přes animeApi jako prostředníka.
"""

import json
import time
import logging
from pathlib import Path
import requests

from . import progress_done

log = logging.getLogger(__name__)

BASE_URL = "https://shikimori.one/api"
REQUEST_DELAY = 1.0             # konzervativní odhad, viz pozn. výš -- doladit
RETRY_DELAYS = [2, 5, 10]
MAX_RETRIES = 3


class ShikimoriClient:
    def __init__(self, cache_dir: str = "cache/shikimori", user_agent: str = "animodel"):
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # Shikimori vyžaduje identifikovatelné User-Agent (potvrzeno napříč
        # více nezávislými wrapper knihovnami, co si to samy nastavují) --
        # bez něj riskuješ, že tě budou blokovat/omezovat přísněji.
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request = 0.0

    def _wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)

    def _cache_file(self, key: str) -> Path:
        return self.cache_path / f"{key.replace('/', '_')}.json"

    def _get(self, endpoint: str):
        ck = self._cache_file(endpoint)
        if ck.exists():
            cached = json.loads(ck.read_text(encoding="utf-8"))
            return cached["data"] if cached.get("_found", True) else None

        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(len(RETRY_DELAYS) + 1):
            self._wait()
            try:
                resp = self.session.get(url, timeout=15)
                self._last_request = time.time()

                if resp.status_code == 404:
                    # Titul na Shikimori pod tímhle ID není -- typicky proto,
                    # že jde o jednu z výjimek ze zkratkového ID triku (viz
                    # docstring modulu). INFO ne WARNING: očekávaný, běžný jev.
                    ck.write_text(json.dumps({"_found": False}), encoding="utf-8")
                    log.info(f"Shikimori 404 (mimo zkratkové ID pravidlo?): {endpoint}")
                    return None

                if resp.status_code == 429:
                    wait = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 30
                    log.info(f"Shikimori rate limit 429, čekám {wait}s…")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                ck.write_text(json.dumps({"_found": True, "data": data}, ensure_ascii=False),
                             encoding="utf-8")
                return data

            except requests.RequestException as e:
                if attempt < len(RETRY_DELAYS) - 1:
                    log.warning(f"Shikimori chyba {e}, retry za {RETRY_DELAYS[attempt]}s…")
                    time.sleep(RETRY_DELAYS[attempt])
                else:
                    log.error(f"Shikimori selhalo po {MAX_RETRIES} pokusech: {url}")
                    return None
        log.error(f"Shikimori selhalo po {MAX_RETRIES} pokusech (rate limit): {url}")
        return None

    def get_similar(self, mal_id: int) -> list[dict]:
        """
        Podobná anime k `mal_id` přes Shikimori (MAL ID použité rovnou jako
        Shikimori ID, viz docstring modulu). Vrací [{mal_id, title}, ...],
        prázdný list když 404/nic nenajde -- NENÍ to signál "nemá rád",
        jen "nedostupné/nedohledatelné", stejná opatrnost jako u ostatních
        klientů v sources/ (viz get_anime v jikan.py).

        POZN.: přesný tvar odpovědi (jestli obsahuje sílu/pořadí podobnosti,
        nebo jen holý seznam) jsem si nemohl ověřit naživo (žádný síťový
        přístup v dev sandboxu) -- kód počítá s tím, že je to prostý seznam
        a váhuje podle POZICE v seznamu (dřívější = podobnější, běžná
        konvence pro tenhle typ endpointu), ne podle nějakého skóre.
        Zkontroluj skutečnou odpověď při prvním ostrém běhu a uprav, pokud
        API vrací i explicitní sílu podobnosti -- to by bylo přesnější
        než pozice.
        """
        data = self._get(f"animes/{mal_id}/similar")
        if not data:
            return []
        out = []
        for i, item in enumerate(data):
            if item.get("id"):
                out.append({
                    "mal_id": item["id"],  # zkratka: Shikimori ID == MAL ID (viz modul docstring)
                    "title": item.get("name", ""),
                    "rank_hint": 1.0 / (i + 1),  # pozice v seznamu jako proxy síly, ne potvrzené skóre
                })
        return out

    def batch_similar(self, mal_ids: list[int], show_progress=True) -> dict[int, list[dict]]:
        out = {}
        total = len(mal_ids)
        for i, mid in enumerate(mal_ids):
            out[mid] = self.get_similar(mid)
            if show_progress and i % 5 == 0:
                from . import progress
                progress(f"  Shikimori podobná anime: {i}/{total}…")
        if show_progress:
            progress_done(f"  Shikimori podobná anime: hotovo ({total} seedů).")
        return out
