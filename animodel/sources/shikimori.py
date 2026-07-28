"""
shikimori.py — Shikimori (rusky mluvící obdoba MAL/AniList) klient, jen pro
"podobná anime" doporučení -- viz code review diskuze o obohacení
recommend.py::_gather_candidates o další, nezávislý zdroj kandidátů.

Dokumentace: https://shikimori.one/api/doc
Rate limit: nemám k dispozici přesné aktuální číslo z jejich docs --
REQUEST_DELAY níž je konzervativní odhad. Při ostrém běhu (2026-07)
nezpůsobil žádné 429, takže je spíš opatrný než těsný.

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

import time
import logging
from pathlib import Path
from typing import Callable

import requests

from .cache import FileCache, cached_fetch
from .http import (
    FixedRateLimiter, Attempt, attempt_success, attempt_permanent,
    attempt_rate_limited, attempt_retryable, request_with_retry,
)

log = logging.getLogger(__name__)

BASE_URL = "https://shikimori.one/api"
REQUEST_DELAY = 1.0             # konzervativní odhad, viz pozn. výš -- doladit
RETRY_DELAYS = [2, 5, 10]


class ShikimoriClient:
    def __init__(self, cache_dir: str = "cache", user_agent: str = "animodel",
                 sleep: Callable[[float], None] = time.sleep):
        # `cache_dir` je KOŘEN cache (stejná konvence jako Jikan/AniList) --
        # podsložku si klient doplní sám. Dřív tenhle jediný očekával už
        # hotovou cestu a volající mu ji skládal (`f"{cache_dir}/shikimori"`),
        # což byla zbytečná asymetrie mezi třemi jinak stejnými klienty.
        self._cache = FileCache(Path(cache_dir) / "shikimori")
        self._rate_limiter = FixedRateLimiter(REQUEST_DELAY)
        self._sleep = sleep
        self.session = requests.Session()
        # Shikimori vyžaduje identifikovatelné User-Agent (potvrzeno napříč
        # více nezávislými wrapper knihovnami, co si to samy nastavují) --
        # bez něj riskuješ, že tě budou blokovat/omezovat přísněji.
        self.session.headers.update({"User-Agent": user_agent})

    def _classify(self, resp: requests.Response, url: str) -> Attempt:
        if resp.status_code == 404:
            # Titul na Shikimori pod tímhle ID není -- typicky proto, že jde
            # o jednu z výjimek ze zkratkového ID triku (viz docstring modulu).
            return attempt_permanent(f"HTTP 404 {url}")
        if resp.status_code == 429:
            return attempt_rate_limited()
        if not resp.ok:
            return attempt_retryable(f"HTTP {resp.status_code} {url}")
        return attempt_success(resp.json())

    def _request(self, endpoint: str):
        url = f"{BASE_URL}/{endpoint}"
        return request_with_retry(
            perform=lambda: self.session.get(url, timeout=15),
            classify=lambda resp: self._classify(resp, url),
            rate_limiter=self._rate_limiter,
            retry_delays=RETRY_DELAYS,
            label="Shikimori",
            sleep=self._sleep,
        )

    def _get(self, endpoint: str):
        return cached_fetch(self._cache, endpoint, lambda: self._request(endpoint))

    def get_similar(self, mal_id: int) -> list[dict]:
        """
        Podobná anime k `mal_id` přes Shikimori (MAL ID použité rovnou jako
        Shikimori ID, viz docstring modulu). Vrací [{mal_id, title}, ...],
        prázdný list když 404/nic nenajde -- NENÍ to signál "nemá rád",
        jen "nedostupné/nedohledatelné", stejná opatrnost jako u ostatních
        klientů v sources/ (viz get_anime v jikan.py).

        TVAR ODPOVĚDI: OVĚŘENO naživo (2026-07-25, 41 odpovědí v cache).
        Je to **prostý seznam** objektů s poli
        `{id, name, russian, score, kind, episodes, episodes_aired,
        aired_on, released_on, status, image, url}` -- tedy **žádné
        explicitní skóre ani pořadí podobnosti**. Pozicové váhování
        (`rank_hint = 1/(i+1)`, dřívější = podobnější) je proto potvrzeně
        jediná dostupná varianta, ne provizorium.

        Pozn. k síle signálu: seznamy bývají dlouhé (pozorováno až 133
        položek), takže při `candidates_per_seed` ořezu klesne rank_hint
        z 1,0 na ~0,04. Shikimori tedy přispívá hlavně NOVÝMI kandidáty,
        pořadím už jen okrajově -- MAL/AniList rec graf dává na týchž
        pozicích řádově vyšší váhy (log1p z hlasů).
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
