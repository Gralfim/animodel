"""Struktura CLI (§9.7), rozhraní mezi vrstvami (§9.6) a diagnostika
kanonizace atributů (§9.5).

Tyhle tři body spolu souvisí čitelností: `run()` už jen vybírá režim,
doporučovač vrací hodnotu místo privátních atributů a nová diagnostika má
vlastní režim v téže struktuře.
"""
import pytest

from animodel import cli
from animodel.attributes import ALIAS, find_near_duplicate_keys, resolve_alias
from animodel.config import Config
from animodel.recommend import RecommendResult


# ── §9.7: číslování kroků se nedá rozjet ─────────────────────────────────

def test_steps_numbering_is_derived_not_hardcoded(capsys):
    steps = cli.Steps(3)
    steps("první")
    steps("druhý")
    steps("třetí")
    out = capsys.readouterr().out
    assert out.splitlines() == ["[1/3] první", "[2/3] druhý", "[3/3] třetí"]


def test_steps_detail_is_indented_under_step(capsys):
    cli.Steps.detail("výsledek")
    assert capsys.readouterr().out == "      výsledek\n"


class _Args:
    """Minimální náhrada argparse.Namespace pro _mode_steps."""
    analyze = analyze_attrs = gen_intensity = no_recommend = False
    season = None


@pytest.mark.parametrize("attr,expected_total", [
    ("analyze", 2),           # parse + franšízy
    ("analyze_attrs", 3),     # parse + sběr + hledání duplicit
    ("gen_intensity", 3),     # parse + frekvence + universum
    ("no_recommend", 3),      # parse + metadata + model
])
def test_mode_step_totals(attr, expected_total):
    """`_mode_steps` vrací kroky PO prepare; celkem = PREPARE_STEPS + to."""
    args = _Args()
    setattr(args, attr, True)
    assert cli.PREPARE_STEPS + cli._mode_steps(args) == expected_total


def test_full_and_season_step_totals():
    assert cli.PREPARE_STEPS + cli._mode_steps(_Args()) == 5     # plný běh
    args = _Args()
    args.season = []                                            # `--season`
    assert cli.PREPARE_STEPS + cli._mode_steps(args) == 4


def test_steps_warns_when_mode_declares_too_few(caplog):
    """Regrese: `--analyze-attrs` hlásil dva kroky, ale dělal tři, takže se
    tisklo `[3/2]`. Přetečení se teď ozve místo aby tiše prošlo."""
    steps = cli.Steps(2)
    with caplog.at_level("WARNING"):
        steps("a")
        steps("b")
        assert "kroků víc" not in caplog.text
        steps("c")
    assert "kroků víc než ohlášeno" in caplog.text


def test_check_rejects_running_without_any_metadata_source(tmp_path):
    cfg = Config()
    cfg.mal_export = str(tmp_path / "x.xml")
    cfg.enrich.use_jikan = cfg.enrich.use_anilist = False
    assert "žádný zdroj metadat" in cli._check(cfg)


def test_check_reports_missing_export(tmp_path):
    cfg = Config()
    cfg.mal_export = str(tmp_path / "chybi.xml")
    assert "nenalezen" in cli._check(cfg)


def test_check_passes_for_valid_setup(tmp_path):
    export = tmp_path / "list.xml"
    export.write_text("<myanimelist/>", encoding="utf-8")
    cfg = Config()
    cfg.mal_export = str(export)
    assert cli._check(cfg) is None


def test_own_account_is_excluded_from_senpai_search():
    cfg = Config()
    cli._exclude_own_account(cfg, {"user_name": "Gralfim"})
    assert "Gralfim" in cfg.recommend.user_cf_exclude_users
    # opakované volání ho nepřidá dvakrát (ani v jiné velikosti písmen)
    cli._exclude_own_account(cfg, {"user_name": "gralfim"})
    assert len(cfg.recommend.user_cf_exclude_users) == 1


# ── §9.6: výsledek místo privátních atributů ─────────────────────────────

def test_recommend_result_carries_cf_payload():
    r = RecommendResult(recs=[1, 2, 3], senpai=["s"], cf_raw=[{"mal_id": 9}])
    assert r.recs == [1, 2, 3]
    assert r.senpai == ["s"] and r.cf_raw == [{"mal_id": 9}]


def test_recommend_result_is_iterable_and_sized():
    """Volající, kterého CF nezajímá, ať se chová jako dřív (iterace/len)."""
    r = RecommendResult(recs=[1, 2, 3])
    assert list(r) == [1, 2, 3] and len(r) == 3
    assert r.senpai == [] and r.cf_raw == []


def test_recommendation_has_prequel_score_field():
    """Dřív se to na dataclass přišpendlovalo dynamicky a četlo přes getattr
    -- tiše by se to rozbilo při slots=True."""
    from animodel.recommend import Recommendation
    r = Recommendation(mal_id=1, title="T", title_en="", community=7.0,
                       pred=8.0, pred_lo=7.0, pred_hi=9.0, taste_fit=0.0,
                       cf_signal=0.0, composite=0.0, ptw=False,
                       cluster_name="", why=[], cf_seeds=[])
    assert r.prequel_score == 0.0
    assert "prequel_score" in Recommendation.__dataclass_fields__


# ── §9.5: diagnostika kanonizace ─────────────────────────────────────────

def test_finds_singular_plural_near_duplicates():
    pairs = find_near_duplicate_keys(["assassin", "assassins", "romance"])
    assert [(a, b) for a, b, _w, _d in pairs] == [("assassin", "assassins")]


def test_finds_same_words_in_different_order():
    pairs = find_near_duplicate_keys(["police_female", "female_police"])
    assert len(pairs) == 1
    a, b, why, dist = pairs[0]
    assert {a, b} == {"police_female", "female_police"}
    assert "shodná slova" in why and dist == 0


def test_finds_singular_plural_across_word_boundary():
    """Tvar téhož konceptu (jako živý nález MAL „Video Game" vs. AniList
    „Video Games"). Syntetické klíče schválně -- ty skutečné už ALIAS
    sjednocuje, takže by se (správně) nehlásily."""
    pairs = find_near_duplicate_keys(["space_ship", "space_ships"])
    assert [(a, b) for a, b, _w, _d in pairs] == [("space_ship", "space_ships")]


def test_finds_suffix_variants_of_single_word():
    pairs = find_near_duplicate_keys(["symbolic", "symbolism"])
    assert len(pairs) == 1
    assert "kmen" in pairs[0][2]


def test_different_concepts_sharing_a_word_are_not_reported():
    """Znakový Levenshtein tyhle hlásil (vzdálenost 2 stejně jako u
    video_game/video_games) a dělal tím 17 falešných nálezů z 19."""
    keys = ["female_protagonist", "male_protagonist",
            "primarily_female_cast", "primarily_male_cast",
            "female_harem", "male_harem"]
    assert find_near_duplicate_keys(keys) == []


def test_coincidental_single_word_similarity_is_not_reported():
    """`acting`/`action` mají vzdálenost 2, ale společný prefix jen 67 % --
    liší se uprostřed, ne v koncovce, takže to nejsou tvary téhož slova."""
    assert find_near_duplicate_keys(
        ["acting", "action", "medical", "medieval", "femboy", "tomboy",
         "asexual", "bisexual", "alchemy", "archery"]) == []


def test_short_keys_are_ignored_as_noise():
    """U tříznakových klíčů je vzdálenost 2 skoro náhoda -- hlásit `war`
    vs. `wax` by utopilo skutečné nálezy v šumu."""
    assert find_near_duplicate_keys(["war", "wax", "cat", "car"]) == []


def test_pairs_already_unified_by_alias_are_not_reported():
    """ALIAS je řešení, ne nález: co už spojuje, se hlásit nesmí."""
    assert resolve_alias("scifi") == resolve_alias("sci_fi")   # premisa testu
    assert find_near_duplicate_keys(["scifi", "sci_fi"], min_len=4) == []


def test_live_findings_are_resolved_and_no_longer_reported():
    """Dvojice nalezené `--analyze-attrs` na živých datech (MAL vs. AniList
    názvosloví téhož konceptu) jsou vyřešené v ALIAS -- nástroj je proto
    hlásit přestal. Kdyby alias někdo odebral, tenhle test spadne."""
    assert find_near_duplicate_keys(
        ["video_game", "video_games",
         "anthropomorphic", "anthropomorphism"]) == []


def test_distant_keys_are_not_reported():
    assert find_near_duplicate_keys(["comedy", "psychological", "military"]) == []


def test_result_is_deterministic_regardless_of_input_order():
    keys = ["assassins", "assassin", "female_police", "police_female"]
    assert find_near_duplicate_keys(keys) == find_near_duplicate_keys(keys[::-1])


def test_alias_targets_do_not_flag_themselves():
    """Identita v ALIAS (`harem: harem`) nesmí vyrobit dvojici sama se sebou."""
    pairs = find_near_duplicate_keys(list(ALIAS) + list(ALIAS.values()))
    assert all(a != b for a, b, _w, _d in pairs)
