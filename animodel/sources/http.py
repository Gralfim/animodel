"""
http.py — sdílený retry/backoff driver pro sources/ (Jikan, AniList, Shikimori).

Rozklad na tři nezávisle testovatelné kusy:
  1. RateLimiter    — řídí čekání PŘED requestem (fixní, nebo adaptivní).
     `wait()`/`record_call()` se volají 1× na `request_with_retry()` call
     (ne 1× na pokus uvnitř retry smyčky) -- odpovídá dřívějšímu chování
     obou klientů (mezera mezi POKUSY v rámci jednoho requestu se řeší
     explicitním retry delay, ne rate limiterem).
  2. classify(resp) — dodává si každý klient sám (status-code větvení +
     doménová sémantika jako GraphQL "errors" pole). Smí sama volat
     `resp.raise_for_status()`/`resp.json()` -- výjimky odtud driver
     zachytí stejně jako výjimky z `perform()`.
  3. request_with_retry() — obecná řídící smyčka nad 1) a 2).

`sleep` je všude injektovatelný parametr (default `time.sleep`) -- testy
dosadí no-op, což je dělá rychlými, a je to zárodek budoucí async/paralelní
varianty (driver sám o sobě je bez sdíleného mutable stavu mimo to, co mu
dá `rate_limiter`, takže je bezpečné mít víc instancí běžet nezávisle na
různých klíčích).

Rozpočet pokusů je ZÁMĚRNĚ ASYMETRICKÝ (odpovídá původnímu, v praxi
ověřenému chování obou klientů):

  - RATE_LIMITED (429): `len(retry_delays) + 1` pokusů, poslední delay
    slouží i jako floor pro pokusy za koncem seznamu. Rate limit se
    vyřeší sám -- čekat déle se vyplatí.
  - RETRYABLE / síťová výjimka (5xx, timeout): `len(retry_delays)` pokusů,
    spí se jen `retry_delays[:-1]`. Při výpadku služby se cena násobí
    KAŽDÝM postiženým titulem v dávce (a enrich fáze nemá circuit
    breaker), takže tady se vzdáváme o pokus dřív -- dočasné selhání se
    stejně necachuje a příští běh to zkusí znovu zadarmo.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import requests

from . import Result

log = logging.getLogger(__name__)


# ── Rate limiting ────────────────────────────────────────────────────────

class RateLimiter:
    """Rozhraní (duck-typed) -- konkrétní implementace níž."""

    def wait(self, sleep: Callable[[float], None]) -> None:
        raise NotImplementedError

    def record_call(self) -> None:
        """Zavolat hned po obdržení HTTP odpovědi (i chybové) -- NE po
        výjimce bez odpovědi. Aktualizuje časovou základnu pro příští `wait()`."""
        raise NotImplementedError

    def on_throttled(self) -> None:
        """Zavolat při KAŽDÉM 429 uvnitř retry smyčky (může být víckrát)."""
        pass

    def on_success(self) -> None:
        """Zavolat jednou, jen když request nakonec uspěje."""
        pass


class FixedRateLimiter(RateLimiter):
    """Konstantní minimální mezera mezi requesty (Jikan, Shikimori)."""

    def __init__(self, delay: float):
        self.delay = delay
        self._last = 0.0

    def wait(self, sleep: Callable[[float], None]) -> None:
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            sleep(self.delay - elapsed)

    def record_call(self) -> None:
        self._last = time.time()


class AdaptiveRateLimiter(RateLimiter):
    """Mezera, která roste po sérii 429 a postupně klesá zpět k base rate
    po úspěších (AniList). Beze změny oproti dřívější `_current_delay`/
    `_consecutive_429` logice v AniListClient."""

    def __init__(self, base: float, max_delay: float, growth: float = 1.5):
        self.base = base
        self.max_delay = max_delay
        self.growth = growth
        self.current = base
        self._consecutive_429 = 0
        self._last = 0.0

    def wait(self, sleep: Callable[[float], None]) -> None:
        elapsed = time.time() - self._last
        if elapsed < self.current:
            sleep(self.current - elapsed)

    def record_call(self) -> None:
        self._last = time.time()

    def on_throttled(self) -> None:
        self._consecutive_429 += 1
        self.current = min(self.max_delay, self.base * (self.growth ** self._consecutive_429))

    def on_success(self) -> None:
        if self._consecutive_429 > 0:
            self._consecutive_429 = max(0, self._consecutive_429 - 1)
            self.current = max(self.base, self.base * (self.growth ** self._consecutive_429))


# ── Klasifikace jednoho pokusu ──────────────────────────────────────────

@dataclass(frozen=True)
class Attempt:
    """Výsledek `classify(resp)` -- co se má s TOUHLE odpovědí udělat."""
    kind: str                 # "success" | "permanent" | "rate_limited" | "retryable"
    data: object = None       # payload pro "success"
    wait: float = 0.0         # navržené čekání pro "rate_limited" (např. Retry-After)
    detail: str = ""          # popis pro log ("permanent"/"retryable")


def attempt_success(data) -> Attempt:
    return Attempt(kind="success", data=data)


def attempt_permanent(detail: str) -> Attempt:
    return Attempt(kind="permanent", detail=detail)


def attempt_rate_limited(wait: float = 0.0) -> Attempt:
    return Attempt(kind="rate_limited", wait=wait)


def attempt_retryable(detail: str) -> Attempt:
    return Attempt(kind="retryable", detail=detail)


# ── Obecná retry smyčka ──────────────────────────────────────────────────

def request_with_retry(
    *,
    perform: Callable[[], requests.Response],
    classify: Callable[[requests.Response], Attempt],
    rate_limiter: RateLimiter,
    retry_delays: list[float],
    label: str,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Sdílený retry/backoff driver pro REST (Jikan/Shikimori) i GraphQL (AniList)."""
    # Asymetrické rozpočty (viz docstring modulu): 429 dostane o pokus víc
    # než ostatní dočasné chyby -- rate limit se čekáním vyřeší, výpadek
    # serveru spíš ne, a jeho cena se násobí každým titulem v dávce.
    attempts = len(retry_delays) + 1        # strop smyčky (dosažitelný jen přes 429)
    retryable_attempts = len(retry_delays)  # rozpočet pro 5xx/síťové chyby

    # Rate limit (mezera OD PŘEDCHOZÍHO requestu) se čeká jen JEDNOU, před
    # celým pokusem o tenhle `_request()` call -- ne před KAŽDÝM pokusem
    # uvnitř retry smyčky níž. Mezery MEZI pokusy už řeší mnohem delší
    # explicitní backoff (`retry_delays`), další čekání by tam bylo
    # redundantní. Přesně odpovídá dřívějšímu chování obou klientů.
    rate_limiter.wait(sleep)

    for i in range(attempts):
        try:
            resp = perform()
            rate_limiter.record_call()
            outcome = classify(resp)
        except requests.RequestException as exc:
            if i < retryable_attempts - 1:
                log.warning(f"{label}: chyba ({exc}), retry za {retry_delays[i]}s…")
                sleep(retry_delays[i])
                continue
            log.error(f"{label}: selhalo po {i + 1} pokusech: {exc}")
            return Result.failure(permanent=False)

        if outcome.kind == "success":
            rate_limiter.on_success()
            return Result.success(outcome.data)

        if outcome.kind == "permanent":
            log.warning(f"{label}: natrvalo selhal (nebude se opakovat): {outcome.detail}")
            return Result.failure(permanent=True)

        if outcome.kind == "rate_limited":
            rate_limiter.on_throttled()
            floor = retry_delays[i] if i < len(retry_delays) else retry_delays[-1]
            wait = max(floor, outcome.wait)
            log.info(f"{label}: rate limit (pokus {i+1}/{attempts}), čekám {wait}s…")
            sleep(wait)
            continue

        # "retryable"
        if i < retryable_attempts - 1:
            log.warning(f"{label}: {outcome.detail}, retry za {retry_delays[i]}s…")
            sleep(retry_delays[i])
            continue
        log.error(f"{label}: selhalo po {i + 1} pokusech: {outcome.detail}")
        return Result.failure(permanent=False)

    # Sem se dá dojít jen když KAŽDÝ pokus skončil na rate_limited (viz
    # `continue` výš) -- vyčerpaný rate limit je transient, příští běh
    # (nebo i jen o pár minut později) může uspět.
    return Result.failure(permanent=False)
