from pathlib import Path
import sys

import pytest

from swccg_policy.efficient import EfficientConfig
from swccg_policy.table_bundle import TableBundle
from swccg_policy.v4_checkpoint import new_models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v31_stage import VARIANTS, check_variant

TABLES = Path(__file__).resolve().parents[1] / "runs/knowledge-v3.1/tables-decipher"


def test_default_variant_is_the_current_config_default():
    defaults = EfficientConfig.__dataclass_fields__
    assert {key: defaults[key].default for key in VARIANTS["a10c"]} == VARIANTS["a10c"]


def test_a10_normalizes_every_component_at_full_label_gain():
    assert VARIANTS["a10"] == {"component_norm": True, "component_norm_scope": "all", "component_label_gain": 1.0}


@pytest.mark.skipif(not TABLES.is_dir(), reason="Decipher V3.1 tables not present")
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fresh_models_match_their_variant(variant):
    tables = TableBundle.load(TABLES)
    actor, critic = new_models(tables, seed=906101, **VARIANTS[variant])
    check_variant(actor.cfg, variant)
    check_variant(critic.cfg, variant)
    other = next(name for name in VARIANTS if name != variant)
    with pytest.raises(ValueError):
        check_variant(actor.cfg, other)
