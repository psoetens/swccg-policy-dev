"""Amendment 7 probe budget and blueprint-disjoint printed-power task."""

import hashlib

import numpy as np


GATE_BUDGET = {"steps": 6000, "lr": 3e-4, "batch": 40, "eval": 2000}
TASK_VERSION = "amendment7-powered-blueprint-halves"


def gate_budget(configuration):
    return all(configuration[key] == value for key, value in GATE_BUDGET.items())


def printed_split(tables):
    cards = tables.cards.cards
    names = list(cards["feature_names"])
    power = cards["features"][:, names.index("num:power")]
    has_power = cards["features"][:, names.index("has:power")] > 0
    real = np.arange(4, len(tables.cards.ids))
    # Hash order is stable by blueprint, independent of power, title and table row order.
    powered = sorted(real[has_power[real]].tolist(),
                     key=lambda index: hashlib.sha256(tables.cards.ids[index].encode()).digest())
    midpoint = len(powered) // 2
    pools = {False: np.asarray(powered[:midpoint]), True: np.asarray(powered[midpoint:])}
    missing = real[~has_power[real]]
    report = {"version": TASK_VERSION, "split": "SHA256(blueprint UTF-8), sorted, first half train; rest heldout",
              "missing_power_cards": len(missing), "powered_cards": len(powered), "ties": "excluded"}
    for heldout, pool in pools.items():
        values, counts = np.unique(np.round(power[pool] * 5, 5), return_counts=True)
        report["heldout" if heldout else "train"] = {
            "cards": len(pool), "blueprints": [tables.cards.ids[index] for index in pool],
            "power_counts": {str(float(value)): int(count) for value, count in zip(values, counts, strict=True)}}
    return power, pools, missing, report


def printed_example(rng, power, choices, missing=None):
    if missing is None:
        while True:
            pair = rng.choice(choices, 2, replace=False)
            if power[pair[0]] != power[pair[1]]:
                break
    else:
        # Informational only: preserve the former zero fallback for one missing-power card.
        powered = choices[power[choices] != 0]
        pair = rng.permutation([rng.choice(missing), rng.choice(powered)])
    options = np.array([[1, 0, 0, pair[0], 0, 0], [1, 0, 0, pair[1], 0, 0]], np.int32)
    item = {"tok": np.array([[0, 0, 0, 1, 0, 0, 1, 0]], np.int32),
            "tokf": np.zeros((1, 4), np.float32), "opt": options,
            "head": np.array([4, 5, 0, 0] + [0] * 18, np.int32)}  # converted to schema 4 by probe_efficient.sample
    return item, int(power[pair[1]] > power[pair[0]])
