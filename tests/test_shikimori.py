"""ShikimoriClient: stejná retry/cache matice jako Jikan, přes sdílený
`request_with_retry`/`cached_fetch`, jen kratší (méně veřejných metod)."""
import requests

from animodel.sources.shikimori import ShikimoriClient, RETRY_DELAYS


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
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, url, timeout=15):
        self.calls += 1
        if self.calls > len(self.script):
            raise AssertionError("víc síťových volání, než bylo naskriptováno")
        step = self.script[self.calls - 1]
        if step == "exc":
            raise requests.ConnectionError("boom")
        return step


def make_client(tmp_path, script, no_sleep):
    client = ShikimoriClient(cache_dir=str(tmp_path), sleep=no_sleep)
    get = ScriptedGet(script)
    client.session.get = get
    return client, get


def test_get_similar_success_is_cached(tmp_path, no_sleep):
    resp = FakeResponse(200, [{"id": 5, "name": "Y"}, {"id": 6, "name": "Z"}])
    client, get = make_client(tmp_path, [resp], no_sleep)

    result = client.get_similar(1)
    assert result == [
        {"mal_id": 5, "title": "Y", "rank_hint": 1.0},
        {"mal_id": 6, "title": "Z", "rank_hint": 0.5},
    ]
    assert get.calls == 1
    assert client.get_similar(1) == result
    assert get.calls == 1


def test_404_is_permanent_and_cached_as_empty(tmp_path, no_sleep):
    client, get = make_client(tmp_path, [FakeResponse(404)], no_sleep)

    assert client.get_similar(1) == []
    assert get.calls == 1
    assert client.get_similar(1) == []
    assert get.calls == 1


def test_429_retries_then_succeeds(tmp_path, no_sleep):
    ok = FakeResponse(200, [{"id": 5, "name": "Y"}])
    client, get = make_client(tmp_path, [FakeResponse(429), ok], no_sleep)

    result = client.get_similar(1)
    assert result == [{"mal_id": 5, "title": "Y", "rank_hint": 1.0}]
    assert get.calls == 2


def test_500_exhausts_retries_and_is_not_cached(tmp_path, no_sleep):
    # rozpočet pro 5xx je len(RETRY_DELAYS) pokusů -- o jeden míň než 429
    attempts = len(RETRY_DELAYS)
    client, get = make_client(tmp_path, [FakeResponse(500)] * attempts, no_sleep)

    assert client.get_similar(1) == []
    assert get.calls == attempts

    client.session.get = ScriptedGet([FakeResponse(500)] * attempts)
    assert client.get_similar(1) == []
    assert client.session.get.calls == attempts
