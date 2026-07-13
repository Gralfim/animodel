"""request_with_retry(): obecná retry/backoff smyčka sdílená Jikan/AniList/
Shikimori klienty. Testováno čistě přes fake perform()/classify() -- žádné
skutečné HTTP mockování, jen kontrolní logika driveru samotného.
"""
import requests

from animodel.sources.http import (
    request_with_retry, attempt_success, attempt_permanent,
    attempt_rate_limited, attempt_retryable, RateLimiter,
    FixedRateLimiter, AdaptiveRateLimiter,
)


class RecordingRateLimiter(RateLimiter):
    """Nulová prodleva (nezavolá `sleep`), ale zaznamenává volání -- ať jde
    ověřit, že driver volá `on_throttled`/`on_success`/`record_call` ve
    správný okamžik."""

    def __init__(self):
        self.waits = 0
        self.records = 0
        self.throttled = 0
        self.successes = 0

    def wait(self, sleep):
        self.waits += 1

    def record_call(self):
        self.records += 1

    def on_throttled(self):
        self.throttled += 1

    def on_success(self):
        self.successes += 1


def make_driver(steps, retry_delays=(1, 2, 3)):
    """`steps`: list, kde každá položka je buď `"exc"` (perform() vyhodí
    RequestException) nebo `Attempt` (perform() "uspěje" a classify() vrátí
    přesně tenhle Attempt). Vrací (Result, perform_call_count, rate_limiter,
    zaznamenané `sleep()` argumenty)."""
    perform_calls = []
    sleeps = []

    def perform():
        perform_calls.append(1)
        step = steps[len(perform_calls) - 1]
        if step == "exc":
            raise requests.ConnectionError("boom")
        return step

    def classify(resp):
        return resp

    rl = RecordingRateLimiter()
    result = request_with_retry(
        perform=perform,
        classify=classify,
        rate_limiter=rl,
        retry_delays=list(retry_delays),
        label="test",
        sleep=lambda s: sleeps.append(s),
    )
    return result, len(perform_calls), rl, sleeps


def test_success_on_first_attempt():
    result, n, rl, sleeps = make_driver([attempt_success({"x": 1})])
    assert result.ok is True
    assert result.data == {"x": 1}
    assert n == 1
    assert rl.successes == 1
    assert sleeps == []


def test_retryable_then_success_uses_retry_delays():
    steps = [attempt_retryable("500"), attempt_retryable("500"), attempt_success("ok")]
    result, n, rl, sleeps = make_driver(steps, retry_delays=[1, 2, 3])
    assert result.ok is True
    assert result.data == "ok"
    assert n == 3
    assert sleeps == [1, 2]


def test_permanent_stops_immediately_without_sleeping():
    result, n, rl, sleeps = make_driver([attempt_permanent("400 bad")], retry_delays=[1, 2, 3])
    assert result.ok is False
    assert result.permanent is True
    assert n == 1
    assert sleeps == []


def test_retryable_exhausts_budget_then_transient_failure():
    """Rozpočet pro 5xx/retryable je ZÁMĚRNĚ o pokus kratší než pro 429
    (`len(retry_delays)` pokusů, spí se jen `retry_delays[:-1]`) -- při
    výpadku služby se cena násobí každým titulem v dávce, zatímco rate
    limit se čekáním vyřeší sám. Odpovídá původnímu chování obou klientů."""
    retry_delays = [1, 2, 3]
    steps = [attempt_retryable("500")] * len(retry_delays)
    result, n, rl, sleeps = make_driver(steps, retry_delays=retry_delays)
    assert result.ok is False
    assert result.permanent is False
    assert n == len(retry_delays)
    assert sleeps == retry_delays[:-1]


def test_request_exception_uses_same_budget_as_retryable():
    """Síťová výjimka (bez HTTP odpovědi) sdílí kratší rozpočet s
    retryable větví -- obě znamenají 'služba teď nefunguje'."""
    retry_delays = [1, 2, 3]
    steps = ["exc"] * len(retry_delays)
    result, n, rl, sleeps = make_driver(steps, retry_delays=retry_delays)
    assert result.ok is False
    assert result.permanent is False
    assert n == len(retry_delays)
    assert sleeps == retry_delays[:-1]


def test_request_exception_then_success():
    steps = ["exc", attempt_success("ok")]
    result, n, rl, sleeps = make_driver(steps, retry_delays=[1, 2, 3])
    assert result.ok is True
    assert n == 2


def test_rate_limited_wait_is_max_of_floor_and_hint():
    steps = [attempt_rate_limited(wait=99), attempt_success("ok")]
    result, n, rl, sleeps = make_driver(steps, retry_delays=[1, 2, 3])
    assert result.ok is True
    assert rl.throttled == 1
    assert sleeps == [99]   # header hint (99) > floor (retry_delays[0]=1)


def test_rate_limited_floor_wins_when_hint_smaller():
    steps = [attempt_rate_limited(wait=0), attempt_success("ok")]
    result, n, rl, sleeps = make_driver(steps, retry_delays=[5, 10])
    assert sleeps == [5]


def test_rate_limited_gets_full_budget_of_len_plus_one():
    """429 dostane plný rozpočet `len(retry_delays)+1` pokusů (na rozdíl
    od retryable/exception, viz test výš) -- rate limit se čekáním vyřeší."""
    retry_delays = [1, 2]
    steps = [attempt_rate_limited(wait=0)] * (len(retry_delays) + 1)
    result, n, rl, sleeps = make_driver(steps, retry_delays=retry_delays)
    assert result.ok is False
    assert result.permanent is False
    assert n == len(retry_delays) + 1
    assert rl.throttled == len(retry_delays) + 1


def test_rate_limited_then_retryable_respects_shorter_retryable_cutoff():
    """Po 429 na prvním pokusu se retryable chyba na druhém pokusu už
    vzdává (index pokusu se počítá globálně, retryable rozpočet je
    len(retry_delays)=2 → druhý pokus je jeho poslední)."""
    steps = [attempt_rate_limited(wait=0), attempt_retryable("500")]
    result, n, rl, sleeps = make_driver(steps, retry_delays=[1, 2])
    assert result.ok is False
    assert result.permanent is False
    assert n == 2
    assert sleeps == [1]   # jen 429 čekání; retryable na posledním pokusu už nespí


def test_on_success_never_called_when_all_attempts_fail():
    retry_delays = [1]
    steps = [attempt_retryable("x")] * (len(retry_delays) + 1)
    result, n, rl, sleeps = make_driver(steps, retry_delays=retry_delays)
    assert rl.successes == 0


def test_record_call_happens_on_every_response_but_not_on_exception():
    steps = ["exc", attempt_retryable("500"), attempt_success("ok")]
    result, n, rl, sleeps = make_driver(steps, retry_delays=[1, 2, 3])
    assert result.ok is True
    # "exc" nikdy nedostane odpověď (žádný record_call), zbylé 2 pokusy ano
    assert rl.records == 2


# ── RateLimiter implementace (Jikan: fixní, AniList: adaptivní) ─────────────

def test_fixed_rate_limiter_waits_out_remaining_delay():
    rl = FixedRateLimiter(delay=0.05)
    sleeps = []
    rl.wait(sleeps.append)          # nikdy žádný request -> nic k čekání
    assert sleeps == []
    rl.record_call()
    rl.wait(sleeps.append)          # hned po sobě -> musí počkat ~celý delay
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 0.05


def test_adaptive_rate_limiter_grows_on_throttle_and_decays_on_success():
    rl = AdaptiveRateLimiter(base=1.0, max_delay=10.0, growth=2.0)
    assert rl.current == 1.0

    rl.on_throttled()
    assert rl.current == 2.0
    rl.on_throttled()
    assert rl.current == 4.0

    rl.on_success()
    assert rl.current == 2.0
    rl.on_success()
    assert rl.current == 1.0   # zpátky na base


def test_adaptive_rate_limiter_caps_at_max_delay():
    rl = AdaptiveRateLimiter(base=1.0, max_delay=3.0, growth=2.0)
    for _ in range(10):
        rl.on_throttled()
    assert rl.current == 3.0


def test_adaptive_rate_limiter_on_success_is_noop_without_prior_throttle():
    rl = AdaptiveRateLimiter(base=1.0, max_delay=10.0, growth=2.0)
    rl.on_success()
    assert rl.current == 1.0
