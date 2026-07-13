"""JikanClient: retry/cache chování nad mockovaným requests.Session.get.

Test `test_404_is_cached_and_never_refetched` je přímá regrese na dřívější
bug: JikanClient._get() ukládal trvalé selhání jako `None` (JSON `null`),
což při čtení kolidovalo s "cache miss" -- 404 tituly se tak necachovaly a
každý běh je zkoušel stáhnout znovu.
"""
import requests

from animodel.sources.jikan import JikanClient, RETRY_DELAYS


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text or ""
        self.headers = {}

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        return self._json


class ScriptedGet:
    """Nahrazuje `session.get` -- vrací naskriptované odpovědi (nebo
    vyhodí ConnectionError pro `"exc"`), jednu na volání."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, url, timeout=15):
        self.calls += 1
        if self.calls > len(self.script):
            raise AssertionError(
                f"session.get zavolán {self.calls}x, ale naskriptováno jen "
                f"{len(self.script)} odpovědí -- test očekával míň síťových volání"
            )
        step = self.script[self.calls - 1]
        if step == "exc":
            raise requests.ConnectionError("boom")
        return step


def make_client(tmp_path, script, no_sleep):
    client = JikanClient(cache_dir=str(tmp_path), sleep=no_sleep)
    get = ScriptedGet(script)
    client.session.get = get
    return client, get


def test_get_anime_success_is_cached(tmp_path, no_sleep):
    resp = FakeResponse(200, {"data": {"mal_id": 1, "title": "Steins;Gate"}})
    client, get = make_client(tmp_path, [resp], no_sleep)

    result = client.get_anime(1)
    assert result == {"mal_id": 1, "title": "Steins;Gate"}
    assert get.calls == 1

    # druhé volání je z cache, žádný další network call
    result2 = client.get_anime(1)
    assert result2 == {"mal_id": 1, "title": "Steins;Gate"}
    assert get.calls == 1


def test_404_is_cached_and_never_refetched(tmp_path, no_sleep):
    client, get = make_client(tmp_path, [FakeResponse(404)], no_sleep)

    assert client.get_anime(1) is None
    assert get.calls == 1

    # REGRESNÍ TEST: druhé volání nesmí jít na síť -- 404 je trvalé a musí
    # zůstat cachované jako "nenalezeno", ne kolidovat s "cache miss".
    assert client.get_anime(1) is None
    assert get.calls == 1


def test_400_gives_up_immediately_without_wasting_retry_schedule(tmp_path, no_sleep):
    client, get = make_client(tmp_path, [FakeResponse(400)], no_sleep)

    assert client.get_anime(1) is None
    assert get.calls == 1          # ne 5 -- permanent selže na první pokus
    assert no_sleep.calls == []    # žádné čekání se neplýtvalo


def test_429_retries_then_succeeds(tmp_path, no_sleep):
    resp_ok = FakeResponse(200, {"data": {"mal_id": 1, "title": "X"}})
    client, get = make_client(
        tmp_path, [FakeResponse(429), FakeResponse(429), resp_ok], no_sleep,
    )

    result = client.get_anime(1)
    assert result == {"mal_id": 1, "title": "X"}
    assert get.calls == 3


def test_500_exhausts_retries_and_is_not_cached(tmp_path, no_sleep):
    # rozpočet pro 5xx je len(RETRY_DELAYS) pokusů -- o jeden míň než 429
    # (výpadek serveru se čekáním nevyřeší, jeho cena se násobí dávkou)
    attempts = len(RETRY_DELAYS)
    client, get = make_client(tmp_path, [FakeResponse(500)] * attempts, no_sleep)

    assert client.get_anime(1) is None
    assert get.calls == attempts

    # transientní selhání se NEcachuje -- druhé volání zkusí síť znovu
    client.session.get = ScriptedGet([FakeResponse(500)] * attempts)
    assert client.get_anime(1) is None
    assert client.session.get.calls == attempts


def test_connection_error_exhausts_retries_and_is_not_cached(tmp_path, no_sleep):
    attempts = len(RETRY_DELAYS)
    client, get = make_client(tmp_path, ["exc"] * attempts, no_sleep)

    assert client.get_anime(1) is None
    assert get.calls == attempts


def test_get_recommendations_unwraps_entries(tmp_path, no_sleep):
    resp = FakeResponse(200, {"data": [
        {"entry": {"mal_id": 5, "title": "Y"}, "votes": 12},
        {"entry": {}, "votes": 3},   # bez mal_id -- musí se vynechat
    ]})
    client, get = make_client(tmp_path, [resp], no_sleep)

    recs = client.get_recommendations(1)
    assert recs == [{"mal_id": 5, "title": "Y", "votes": 12}]
