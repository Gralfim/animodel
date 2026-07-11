"""
sources/__init__.py — sdílené utility pro Jikan/AniList klienty.

Progress výpisy (přes \r) i běžné stavové řádky bez explicitního
flush() se při nezanoření na TTY (přesměrování do souboru, mnoho IDE
konzolí, `tee`, CI…) chovají jako plně bufferované -- text se neobjeví
vůbec, dokud proces neskončí nebo se buffer nezaplní (typicky 4-8 kB).
`log.warning`/`log.error` naproti tomu jde přes stderr, které je
default nebufferované/řádkově bufferované, takže se objeví okamžitě.

Výsledek beze zásahu: v logu vidíš prakticky jen "chyby" (rate limit
warningy), zatímco skutečný progress buď zmizí úplně, nebo se ukáže
najednou, zpackaně, až na konci běhu -- přesně nahlášený problém.
"""
from dataclasses import dataclass

_LAST_WIDTH = 0


def progress(msg: str) -> None:
    """
    Vypíše stavový řádek přes \\r (přepisuje se na místě) a rovnou flushne.

    Doplní mezerami na šířku předchozího volání, ať kratší text nenechá
    viditelný zbytek delšího předchozího řádku za sebou.
    """
    global _LAST_WIDTH
    padded = f"{msg:<{_LAST_WIDTH}}"
    _LAST_WIDTH = len(msg)
    print(f"\r{padded}", end="", flush=True)


def progress_done(msg: str) -> None:
    """Uzavře sekvenci progress() volání skutečným novým řádkem a flushne."""
    global _LAST_WIDTH
    padded = f"{msg:<{_LAST_WIDTH}}"
    _LAST_WIDTH = 0
    print(f"\r{padded}", flush=True)


def status(msg: str) -> None:
    """Běžný informační řádek (bez \\r), ale explicitně flushnutý -- ať se
    neobjeví se zpožděním za pozdějšími (nebufferovanými) log.warning voláními."""
    print(msg, flush=True)


PERMANENT_HTTP_CODES = {400, 404, 422}


def is_permanent_status(status_code) -> bool:
    """
    HTTP kód, kde je request natrvalo špatný (neplatné ID, malformed query
    apod.) -- opakování se STEJNÝMI parametry nikdy nepovede jinak.
    `status_code=None` (čistě síťová chyba bez HTTP odpovědi vůbec --
    spojení, timeout) je vždy transient, ne permanent.

    Sdíleno mezi JikanClient._request a AniListClient._request, ať obě
    mají identickou definici "co je natrvalo špatně" na jednom místě,
    ne dvě nezávislé (a časem snadno rozjíždějící se) kopie.
    """
    return status_code in PERMANENT_HTTP_CODES


@dataclass
class Result:
    """
    Výsledek jednoho síťového volání (JikanClient._request / AniListClient._request).

    Nahrazuje dřívější přístup přes mutable `self._last_failure_kind`
    side-channel (klient si "pamatoval", jak dopadlo poslední volání, a
    volající se musel zeptat samostatnou metodou HNED po volání, jinak
    riskoval, že čte stav z něčeho úplně jiného). Tenhle typ nese
    klasifikaci PŘÍMO v návratové hodnotě -- nedá se přečíst pozdě,
    přepsat mezitímním voláním, ani zapomenout zkontrolovat ve špatný čas.

    ok=True:  `data` obsahuje úspěšnou odpověď.
    ok=False: `data` je None; `permanent` říká, jestli má smysl retry:
        permanent=True  -- 400/404/422/GraphQL sémantická chyba. Retry se
                           stejnými parametry nikdy neuspěje, takže je
                           bezpečné zacachovat i prázdný/částečný výsledek
                           jako KONEČNÝ.
        permanent=False -- 5xx/timeout/síťová chyba/vyčerpaný rate limit.
                           Příští pokus (i jen o pár minut/běhů později)
                           může uspět -- NESMÍ se cachovat jako "selhalo"
                           natrvalo, jinak se dočasný výpadek nerozezná od
                           titulu/uživatele, co fakt nic nemá.
    """
    ok: bool
    data: dict | list | None = None
    permanent: bool = False

    @classmethod
    def success(cls, data) -> "Result":
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, permanent: bool) -> "Result":
        return cls(ok=False, data=None, permanent=permanent)
