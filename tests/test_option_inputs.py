"""Plan option-inputs-and-balance-probe: v2 option inputs (integer values, the pointed-at card and location)."""
import copy

import torch

from swccg_policy.efficient import EfficientConfig, EfficientPolicy
from swccg_policy.v4_data import parse, stack
from test_efficient_model import fixture


def records(record):
    integer = copy.deepcopy(record)
    integer[4] = "INTEGER"
    integer[9] = [["<int>", [], -1, "", str(value), value] for value in range(5)]
    card = copy.deepcopy(record)
    card[4] = "CARD_SELECTION"
    card[9] = [["act", [], source, None, str(source), None] for source in (0, 2, 7)]
    return [record, integer, card]


def models():
    tables, cfg, record = fixture()
    legacy = EfficientPolicy(cfg, tables).eval()
    extended = EfficientPolicy(EfficientConfig(**{**cfg.to_dict(), "option_inputs": "v2"}), tables).eval()
    missing, unexpected = extended.load_state_dict(legacy.state_dict(), strict=False)
    assert not unexpected and missing and all(key.startswith(("option_value", "option_scalars", "option_source",
                                                              "option_reference_log")) for key in missing)
    return tables, legacy, extended, records(record)


def test_extension_preserves_the_function():
    tables, legacy, extended, rows = models()
    for row in rows:
        batch = stack([parse(row, tables)])
        torch.testing.assert_close(extended(batch), legacy(batch), atol=0, rtol=0)


def test_new_inputs_separate_integer_and_card_options():
    tables, legacy, extended, rows = models()
    with torch.no_grad():
        torch.nn.init.normal_(extended.option_value.weight, std=0.5)
        torch.nn.init.normal_(extended.option_scalars.weight, std=0.5)
        torch.nn.init.normal_(extended.option_source_card.weight, std=0.5)
    integer = stack([parse(rows[1], tables)])
    # Legacy: options 2..4 differ only through value / 60, a ~1e-6 logit spread; v2 separates them clearly.
    spread = lambda logits: float(logits[0, 2:5].max() - logits[0, 2:5].min())
    assert spread(extended(integer)) > 100 * spread(legacy(integer))
    card = stack([parse(rows[2], tables)])
    assert not torch.allclose(legacy(card), extended(card))


def test_gradients_reach_every_new_parameter():
    tables, legacy, extended, rows = models()
    extended.train()
    batch = stack([parse(row, tables) for row in rows])
    extended(batch)[batch["omask"]].sum().backward()
    for name in ("option_value.weight", "option_scalars.weight", "option_source_card.weight",
                 "option_reference_log.weight"):
        gradient = dict(extended.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().sum() > 0, name


def test_torchscript_export_matches_eager(tmp_path):
    from swccg_policy.v4_native import export_ts
    tables, legacy, extended, rows = models()
    with torch.no_grad():
        torch.nn.init.normal_(extended.option_value.weight, std=0.5)
        torch.nn.init.normal_(extended.option_source_card.weight, std=0.5)
    traced = export_ts(extended, tmp_path / "actor.ts")
    for row in rows:
        batch = stack([parse(row, tables)])
        head = torch.cat([batch[key][:, None] for key in ("dtype", "qtmpl", "qbp", "side")] + [batch["glob"]], 1)
        arguments = [batch[key] for key in ("tok", "tokf", "tmask", "opt", "omask")] + [head, batch["kgrid"]]
        torch.testing.assert_close(extended(batch), traced(*arguments), atol=1e-5, rtol=1e-5)


def test_location_state_of_the_pointed_at_location():
    tables, cfg, record = fixture()
    cfg5 = EfficientConfig(**{**cfg.to_dict(), "tensor_abi": 5, "location_icons": True, "location_state": True})
    legacy = EfficientPolicy(cfg5, tables).eval()
    extended = EfficientPolicy(EfficientConfig(**{**cfg5.to_dict(), "option_inputs": "v2"}), tables).eval()
    extended.load_state_dict(legacy.state_dict(), strict=False)
    row = copy.deepcopy(record)
    row[8] = [token + [0, 0, 0, 0] for token in row[8]]
    # A location token: power 4 / 6 and ability 2 / 0 packed as h(us) + 512 h(them) with h = 2x, drains 1 / 2.
    row[8].append([1, "1_3", 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 8 + 512 * 12, 4, 1, 2])
    location = len(row[8]) - 1
    row[4] = "CARD_SELECTION"
    row[9] = [["act", [], 0, None, "a", None], ["act", [], location, None, "b", None]]
    batch = stack([parse(row, tables, width=8)])
    torch.testing.assert_close(extended(batch), legacy(batch), atol=0, rtol=0)
    extended.train()
    extended(batch)[batch["omask"]].sum().backward()
    gradient = extended.option_source_location[-1].weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
