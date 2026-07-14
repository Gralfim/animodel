"""Defaulty TasteModel.__init__ musí zůstat v synchronu s config.ModelCfg --
cli.py předává hodnoty explicitně, ale test harness a programové použití
(README příklad) konstruují TasteModel jen s částí argumentů; při rozjetých
defaultech by tiše běžely s jinými prahy než produkce (HODNOCENI §5.3)."""
import inspect

from animodel.config import ModelCfg
from animodel.taste import TasteModel


def test_tastemodel_defaults_match_model_cfg():
    sig = inspect.signature(TasteModel.__init__).parameters
    cfg = ModelCfg()
    assert sig["shrinkage_k"].default == cfg.shrinkage_k
    assert sig["min_attr_count"].default == cfg.min_attr_count
    assert sig["interaction_min_count"].default == cfg.interaction_min_count
    assert sig["interaction_min_lift"].default == cfg.interaction_min_lift
    assert sig["interaction_triples"].default == cfg.interaction_triples
