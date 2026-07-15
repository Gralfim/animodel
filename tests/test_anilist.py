"""AniListClient: retry/cache chování nad mockovaným requests.Session.post,
včetně per-stránkové cache pro user-based CF watchers (§ hlavní oprava
agregace -- viz plán: dřív byl celý stránkovaný seed jeden cache soubor
zapsaný až na konci, takže dočasné selhání na pozdní stránce zahodilo i
už úspěšně stažené dřívější stránky)."""
import requests

from animodel.sources.anilist import AniListClient, RETRY_DELAYS


class FakeResponse:
    def __init__(self, status_code, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json = json_data
        self.text = text or ""
        self.headers = headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        return self._json


class ScriptedPost:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, url, json=None, timeout=20):
        self.calls += 1
        if self.calls > len(self.script):
            raise AssertionError(
                f"session.post zavolán {self.calls}x, ale naskriptováno jen "
                f"{len(self.script)} odpovědí"
            )
        step = self.script[self.calls - 1]
        if step == "exc":
            raise requests.ConnectionError("boom")
        return step


def make_client(tmp_path, script, no_sleep):
    client = AniListClient(cache_dir=str(tmp_path), sleep=no_sleep)
    post = ScriptedPost(script)
    client.session.post = post
    return client, post


def test_get_anime_success_is_cached(tmp_path, no_sleep):
    media = {"id": 10, "idMal": 1, "title": {"romaji": "X"}}
    resp = FakeResponse(200, {"data": {"Media": media}})
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_anime(1) == media
    assert post.calls == 1
    assert client.get_anime(1) == media   # z cache
    assert post.calls == 1
    # cache klíč nese verzi schématu dotazu (v2 = genres/description/
    # seasonYear/relations) -- starý mal_{id} bez verze se už nepoužívá,
    # jinak by staré soubory bez nových polí vypadaly jako platná data
    assert client._cache.has("mal_1_v2")
    assert not client._cache.has("mal_1")


def test_confirmed_missing_media_is_permanent_and_cached(tmp_path, no_sleep):
    resp = FakeResponse(200, {"data": {"Media": None}})
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_anime(1) is None
    assert post.calls == 1
    assert client.get_anime(1) is None    # z cache, žádný další request
    assert post.calls == 1


def test_graphql_errors_field_is_permanent(tmp_path, no_sleep):
    resp = FakeResponse(200, {"errors": [{"message": "Invalid idMal"}]})
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_anime(1) is None
    assert post.calls == 1
    assert client.get_anime(1) is None
    assert post.calls == 1


def test_http_404_is_permanent(tmp_path, no_sleep):
    client, post = make_client(tmp_path, [FakeResponse(404, text="not found")], no_sleep)

    assert client.get_anime(1) is None
    assert post.calls == 1
    assert client.get_anime(1) is None
    assert post.calls == 1


def test_429_uses_retry_after_header_as_wait_floor(tmp_path, no_sleep):
    media = {"id": 10, "idMal": 1, "title": {"romaji": "X"}}
    throttled = FakeResponse(429, headers={"Retry-After": "20"})
    ok = FakeResponse(200, {"data": {"Media": media}})
    client, post = make_client(tmp_path, [throttled, ok], no_sleep)

    assert client.get_anime(1) == media
    assert post.calls == 2
    # RETRY_DELAYS[0]=5 je floor, Retry-After=20 je vyšší -> vyhrává 20
    assert no_sleep.calls == [20]


def test_500_exhausts_retries_and_is_not_cached(tmp_path, no_sleep):
    # rozpočet pro 5xx je len(RETRY_DELAYS) pokusů -- o jeden míň než 429
    attempts = len(RETRY_DELAYS)
    client, post = make_client(tmp_path, [FakeResponse(500)] * attempts, no_sleep)

    assert client.get_anime(1) is None
    assert post.calls == attempts

    client.session.post = ScriptedPost([FakeResponse(500)] * attempts)
    assert client.get_anime(1) is None
    assert client.session.post.calls == attempts


def test_get_tag_collection_success_is_cached(tmp_path, no_sleep):
    tags = [
        {"name": "Tearjerker", "description": "Sad.", "category": "Theme-Drama",
         "isAdult": False, "isGeneralSpoiler": False},
    ]
    resp = FakeResponse(200, {"data": {"MediaTagCollection": tags}})
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_tag_collection() == tags
    assert post.calls == 1
    assert client.get_tag_collection() == tags   # z cache
    assert post.calls == 1


def test_get_tag_collection_transient_failure_returns_empty_uncached(tmp_path, no_sleep):
    client, post = make_client(tmp_path, [FakeResponse(500)] * len(RETRY_DELAYS), no_sleep)
    assert client.get_tag_collection() == []
    # transient -> necachováno, příště se zkusí znovu
    assert client._cache.get("tag_collection") is None


def test_get_genre_collection_success_is_cached(tmp_path, no_sleep):
    resp = FakeResponse(200, {"data": {"GenreCollection": ["Comedy", "Drama"]}})
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_genre_collection() == ["Comedy", "Drama"]
    assert post.calls == 1
    assert client.get_genre_collection() == ["Comedy", "Drama"]   # z cache
    assert post.calls == 1


def test_get_recommendations_empty_success_is_cached_permanently(tmp_path, no_sleep):
    resp = FakeResponse(200, {"data": {"Media": {"recommendations": {"nodes": []}}}})
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_recommendations(1) == []
    assert post.calls == 1
    assert client.get_recommendations(1) == []
    assert post.calls == 1   # prázdný, ale ÚSPĚŠNÝ výsledek se necachuje jako miss


# ── CF watchers: per-stránková cache ────────────────────────────────────────

def _watchers_page_response(entries, has_next):
    return FakeResponse(200, {"data": {"Page": {
        "pageInfo": {"hasNextPage": has_next},
        "mediaList": [
            {"score": raw, "user": {"id": uid, "name": name}}
            for uid, name, raw in entries
        ],
    }}})


def test_watchers_page_success_is_cached_independently(tmp_path, no_sleep):
    page1 = _watchers_page_response([(1, "u1", 80), (2, "u2", 90)], has_next=False)
    client, post = make_client(tmp_path, [page1], no_sleep)

    result = client._fetch_watchers_page(anilist_id=42, page=1, per_page=50)
    assert result.ok is True
    assert result.data == {
        "entries": [[1, "u1", 80], [2, "u2", 90]],
        "has_next": False,
    }


def test_watchers_page_skips_unscored_entries(tmp_path, no_sleep):
    page1 = _watchers_page_response([(1, "u1", 80), (2, "u2", 0)], has_next=False)
    client, post = make_client(tmp_path, [page1], no_sleep)

    result = client._fetch_watchers_page(anilist_id=42, page=1, per_page=50)
    assert result.data["entries"] == [[1, "u1", 80]]


def test_get_watcher_entries_resumes_from_failed_page_without_refetching_earlier_ones(
    tmp_path, no_sleep,
):
    page1 = _watchers_page_response([(1, "u1", 80), (2, "u2", 90)], has_next=True)
    page2_failures = [FakeResponse(500)] * len(RETRY_DELAYS)
    client, post = make_client(tmp_path, [page1, *page2_failures], no_sleep)

    # users_per_seed=10 nutí pokračovat na stránku 2 (page1 má jen 2 uživatele)
    watchers = client.get_watcher_entries(anilist_id=42, users_per_seed=10, per_page=50)
    assert watchers == [[1, "u1", 80], [2, "u2", 90]]
    assert post.calls == 1 + len(page2_failures)

    # Stránka 1 je cachovaná zvlášť a zůstala na disku i po selhání stránky 2.
    assert client._cf_cache.get("watchers_42_p1") is not None
    assert client._cf_cache.get("watchers_42_p1")["found"] is True
    # Stránka 2 selhala DOČASNĚ -> nic se pro ni nezapsalo.
    assert client._cf_cache.get("watchers_42_p2") is None

    # Druhé volání: stránka 1 se vezme z cache (žádný network call), zkusí
    # se jen stránka 2 -- tentokrát úspěšně.
    page2_ok = _watchers_page_response([(3, "u3", 70)], has_next=False)
    client.session.post = ScriptedPost([page2_ok])
    watchers2 = client.get_watcher_entries(anilist_id=42, users_per_seed=10, per_page=50)
    assert watchers2 == [[1, "u1", 80], [2, "u2", 90], [3, "u3", 70]]
    assert client.session.post.calls == 1   # jen stránka 2, stránka 1 z cache


def test_get_watcher_entries_stops_at_users_per_seed(tmp_path, no_sleep):
    page1 = _watchers_page_response([(1, "u1", 80), (2, "u2", 90)], has_next=True)
    client, post = make_client(tmp_path, [page1], no_sleep)

    watchers = client.get_watcher_entries(anilist_id=42, users_per_seed=2, per_page=50)
    assert watchers == [[1, "u1", 80], [2, "u2", 90]]
    assert post.calls == 1   # nezkoušel stránku 2, i když has_next=True byl


def test_get_watcher_entries_stops_when_permanently_failed_page(tmp_path, no_sleep):
    page1 = _watchers_page_response([(1, "u1", 80)], has_next=True)
    page2_permanent = FakeResponse(400, text="bad request")
    client, post = make_client(tmp_path, [page1, page2_permanent], no_sleep)

    watchers = client.get_watcher_entries(anilist_id=42, users_per_seed=10, per_page=50)
    assert watchers == [[1, "u1", 80]]
    assert post.calls == 2

    # Trvalé selhání JE cachované (retry by stejně nikdy nepomohl) -- třetí
    # volání se o stránku 2 vůbec nepokusí přes síť.
    client.session.post = ScriptedPost([])
    watchers2 = client.get_watcher_entries(anilist_id=42, users_per_seed=10, per_page=50)
    assert watchers2 == [[1, "u1", 80]]
    assert client.session.post.calls == 0


# ── CF: plné seznamy uživatelů ─────────────────────────────────────────────

def _userlist_response(entries, fmt="POINT_10"):
    """entries: [(status, score, idMal, avg), ...]"""
    return FakeResponse(200, {"data": {"MediaListCollection": {
        "lists": [{"entries": [
            {"status": st, "score": sc,
             "media": {"idMal": mid, "averageScore": avg,
                       "title": {"romaji": f"A{mid}"}}}
            for st, sc, mid, avg in entries
        ]}],
        "user": {"mediaListOptions": {"scoreFormat": fmt}},
    }}})


def test_user_animelist_keeps_scored_dropped_and_collects_planning(tmp_path, no_sleep):
    resp = _userlist_response([
        ("COMPLETED", 9, 1, 80),
        ("DROPPED", 3, 2, 75),      # dropnutý SE známkou -> platný datový bod
        ("DROPPED", 0, 3, 70),      # dropnutý BEZ známky -> nenese signál
        ("PLANNING", 0, 4, 85),     # fronta -> zvlášť, ne mezi entries
        ("COMPLETED", 0, 5, 60),    # neohodnocený -> pryč
        ("CURRENT", 8, 6, 0),       # bez komunitního skóre -> pryč
    ])
    client, post = make_client(tmp_path, [resp], no_sleep)

    data = client.get_user_animelist(42)
    assert data["fmt"] == "POINT_10"
    assert data["entries"] == [[1, 9, 80, "A1"], [2, 3, 75, "A2"]]
    assert data["planning"] == [4]

    # cache klíč nese verzi schématu (v2 = pole `planning`) -- starý
    # `userlist_{uid}` by se tvářil jako platná data bez PTW informace
    assert client._cf_cache.has("userlist_42_v2")
    assert not client._cf_cache.has("userlist_42")
    assert client.get_user_animelist(42) == data   # z cache
    assert post.calls == 1


def test_user_animelist_private_is_permanent_none(tmp_path, no_sleep):
    resp = FakeResponse(404, text='{"errors":[{"message":"Private User"}]}')
    client, post = make_client(tmp_path, [resp], no_sleep)

    assert client.get_user_animelist(42) is None
    assert post.calls == 1
    assert client.get_user_animelist(42) is None   # z cache, žádný další request
    assert post.calls == 1
