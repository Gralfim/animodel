"""Model výběru (selection.py, HODNOCENI_PROJEKTU.md §9d.4): populární tituly,
které nemám ani v seznamu, ani na PTW, jsou slabý doklad toho, čemu se vyhýbám."""
import datetime as dt
import random

from animodel.attributes import AttrValue
from animodel.enrich import Enriched
from animodel.selection import SelectionModel, fit_selection, universe_entries

TODAY = dt.date(2026, 9, 24)


def _en(mid, tags, *, fmt="TV", members=100_000, aired="2015-04-01", rel=None):
    attrs = {t: AttrValue("tag", 1.0, t.title()) for t in tags}
    jikan = {"type": fmt, "members": members,
             "aired": {"from": f"{aired}T00:00:00+00:00"},
             "relations": rel or []}
    return Enriched(mal_id=mid, title=f"T{mid}", title_en="", community=7.5,
                    attrs=attrs, jikan=jikan)


def _prequel(of):
    return [{"relation": "Prequel", "entry": [{"mal_id": of, "type": "anime"}]}]


def test_universe_keeps_only_old_enough_main_format_franchise_entries():
    uni = {
        1: _en(1, ["x"]),                                   # ok
        2: _en(2, ["x"], fmt="OVA"),                        # vedlejší formát
        3: _en(3, ["x"], aired="2026-08-01"),               # moc čerstvé
        4: _en(4, ["x"], rel=_prequel(1)),                  # druhá řada
        5: _en(5, ["x"], members=0),                        # bez povědomí
    }
    rel = {m: {"relations": e.jikan["relations"]} for m, e in uni.items()}
    assert universe_entries(uni, rel, TODAY) == [1]


def _synthetic(n=80, seed=3):
    """Polovina romance (většinou v seznamu), polovina horor (nikdy)."""
    rng = random.Random(seed)
    uni, listed = {}, set()
    for i in range(1, n + 1):
        horror = i % 2 == 0
        uni[i] = _en(i, ["horror" if horror else "romance", "school"],
                     members=rng.randint(50_000, 2_000_000))
        if not horror and rng.random() < 0.7:
            listed.add(i)
    return uni, listed


def test_fit_learns_what_is_avoided():
    uni, listed = _synthetic()
    rel = {m: {"relations": []} for m in uni}
    sel = fit_selection(uni, listed, rel, today=TODAY)
    assert isinstance(sel, SelectionModel)
    assert sel.coef["horror"] < 0 < sel.coef["romance"]
    assert sel.n == 80 and sel.n_engaged == len(listed)
    assert sel.score(uni[2].attrs) < sel.score(uni[1].attrs)
    assert sel.avoided(uni[2].attrs) == ["Horror"]
    assert sel.avoided(uni[1].attrs) == []


def test_sequel_in_list_marks_franchise_entry_as_engaged():
    """Mám v seznamu druhou řadu -- první díl populární franšízy tedy není
    „přeskočený", i když ho v seznamu nemám."""
    uni, listed = _synthetic()
    target = next(m for m in uni if m % 2 == 1 and m not in listed)
    rel = {m: {"relations": []} for m in uni}
    rel[9999] = {"relations": _prequel(target)}
    before = fit_selection(uni, listed, rel, today=TODAY, with_cv=False)
    after = fit_selection(uni, listed | {9999}, rel, today=TODAY, with_cv=False)
    assert after.n_engaged == before.n_engaged + 1


def test_avoided_shows_only_content_categories():
    """Dekáda v modelu zůstává (skutečný signál), ale v kartě „Obvykle
    nevybíráš: 2010s" nic nevysvětlí."""
    sel = SelectionModel(coef={"horror": -1.0, "2010s": -2.0},
                         labels={"horror": "Horror", "2010s": "2010s"},
                         pop_coef=1.0, n=100, n_engaged=20,
                         categories={"horror": "genre", "2010s": "decade"})
    attrs = {"horror": AttrValue("genre", 1.0, "Horror"),
             "2010s": AttrValue("decade", 1.0, "2010s")}
    assert sel.avoided(attrs) == ["Horror"]
    assert sel.score(attrs) == -3.0


def test_too_little_data_gives_no_model():
    uni = {i: _en(i, ["x"]) for i in range(1, 30)}
    rel = {m: {"relations": []} for m in uni}
    assert fit_selection(uni, {1, 2}, rel, today=TODAY) is None
