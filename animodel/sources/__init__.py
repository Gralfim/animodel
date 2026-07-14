"""
sources/__init__.py — sdílené utility pro Jikan/AniList klienty.

Progress výpisy (přes \r) i běžné stavové řádky bez explicitního
flush() se při nezanoření na TTY (přesměrování do souboru, mnoho IDE
konzolí, `tee`, CI…) chovají jako plně bufferované -- text se neobjeví
vůbec, dokud proces neskončí nebo se buffer nezaplní (typicky 4-8 kB).
`log.warning`/`log.error` naproti tomu jde přes stderr, které je
default nebufferované/řádkově bufferované, takže se objeví okamžitě.

Druhá polovina téhož problému (řešená v 2026-07): i s flushi se log
zprávy (stderr) v terminálu MÍCHAJÍ do rozepsané \r progress řádky
(stdout) -- warning se přilepí za progress text a další progress ho
zpřepisuje napůl, takže při sérii chyb (např. výpadek Jikanu) není
aktuální progress vidět vůbec. Řeší `ProgressAwareLogHandler` níž:
před vypsáním log záznamu progress řádku smaže, po něm ji překreslí.
Log zprávy tak "rolují nad" trvale viditelným progress řádkem.
"""
import logging
import sys
from dataclasses import dataclass

_LAST_WIDTH = 0          # šířka pro padding příštího progress() volání
_LAST_MSG: str | None = None    # text aktivní (rozepsané) progress řádky
_DISPLAYED_WIDTH = 0     # skutečná šířka právě vykreslené řádky (po paddingu)


def _stdout_isatty() -> bool:
    """Vytaženo do funkce kvůli testovatelnosti (monkeypatch)."""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def progress(msg: str) -> None:
    """
    Vypíše stavový řádek přes \\r (přepisuje se na místě) a rovnou flushne.

    Doplní mezerami na šířku předchozího volání, ať kratší text nenechá
    viditelný zbytek delšího předchozího řádku za sebou.
    """
    global _LAST_WIDTH, _LAST_MSG, _DISPLAYED_WIDTH
    padded = f"{msg:<{_LAST_WIDTH}}"
    _LAST_WIDTH = len(msg)
    _LAST_MSG = msg
    _DISPLAYED_WIDTH = len(padded)
    print(f"\r{padded}", end="", flush=True)


def progress_done(msg: str) -> None:
    """Uzavře sekvenci progress() volání skutečným novým řádkem a flushne."""
    global _LAST_WIDTH, _LAST_MSG, _DISPLAYED_WIDTH
    padded = f"{msg:<{_LAST_WIDTH}}"
    _LAST_WIDTH = 0
    _LAST_MSG = None
    _DISPLAYED_WIDTH = 0
    print(f"\r{padded}", flush=True)


def _clear_progress_line() -> None:
    """Smaže rozepsanou progress řádku z terminálu (bez ztráty stavu --
    `_redraw_progress_line()` ji umí vrátit). Mimo TTY nedělá nic: do
    přesměrovaného souboru by mazací sekvence jen přidávala balast."""
    if _LAST_MSG is None or not _stdout_isatty():
        return
    print("\r" + " " * _DISPLAYED_WIDTH + "\r", end="", flush=True)


def _redraw_progress_line() -> None:
    global _DISPLAYED_WIDTH
    if _LAST_MSG is None or not _stdout_isatty():
        return
    _DISPLAYED_WIDTH = len(_LAST_MSG)
    print(f"\r{_LAST_MSG}", end="", flush=True)


def status(msg: str) -> None:
    """Běžný informační řádek (bez \\r), ale explicitně flushnutý -- ať se
    neobjeví se zpožděním za pozdějšími (nebufferovanými) log.warning voláními.
    S aktivní progress řádkou spolupracuje stejně jako log handler: uklidí
    ji, vypíše se, a nechá ji překreslit."""
    _clear_progress_line()
    print(msg, flush=True)
    _redraw_progress_line()


class ProgressAwareLogHandler(logging.StreamHandler):
    """
    StreamHandler (typicky nad stderr), který se nemíchá do \\r progress
    řádky na stdout: před emitem záznamu ji smaže, po emitu překreslí.
    V terminálu tak log zprávy rolují NAD progress řádkem, který zůstává
    trvale viditelný dole -- místo dřívějšího slepence "progress text +
    přilepený warning + napůl přepsaný další progress".

    Clear/redraw jsou no-op mimo TTY a obalené try/except -- kosmetika
    výstupu nikdy nesmí shodit samotné logování.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _clear_progress_line()
        except Exception:
            pass
        super().emit(record)
        try:
            _redraw_progress_line()
        except Exception:
            pass


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
