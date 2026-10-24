"""Unlimited capacity (Integer.MAX_VALUE in the card database) becomes 16; other huge values are rejected."""
import pytest

from swccg_policy.cards import UNLIMITED, build_features, numeric_value


def test_unlimited_capacity_is_16():
    card = {"id": "4_167", "title": "Executor", "pilotCapacity": UNLIMITED, "passengerCapacity": UNLIMITED, "armor": 7}
    assert numeric_value(card, "pilotCapacity") == 16.0
    assert numeric_value(card, "armor") == 7.0
    assert numeric_value(card, "forfeit") is None


def test_other_huge_values_rejected():
    with pytest.raises(ValueError):
        numeric_value({"id": "x", "power": 2147483647}, "power")


def test_real_database_has_no_huge_features():
    import json
    from swccg_policy.cards import DEFAULT_DB
    if not DEFAULT_DB.exists():
        pytest.skip("card database not available")
    d = json.load(open(DEFAULT_DB))
    cards = d if isinstance(d, list) else list(d.get("cards", d).values()) if isinstance(d.get("cards", d), dict) else d["cards"]
    x, cols = build_features(cards)
    assert float(abs(x).max()) < 100
