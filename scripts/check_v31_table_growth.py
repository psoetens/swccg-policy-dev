"""Real legacy-to-V card extension and text-prior initialization/growth checks."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from swccg_policy.text_priors import GrowingPrintedCards
from swccg_policy.v4_tables import CardTable


def main():
    torch.manual_seed(3103)
    torch.set_num_threads(1)
    old = CardTable.load("runs/knowledge-v3.1/legacy/cards-v1.npz")
    grown = CardTable.load("runs/knowledge-v3.1/legacy/cards-v2-virtual.npz")
    old.assert_prefix_of(grown)
    new_blueprint = grown.ids[len(old.ids)]
    try:
        old.lookup(new_blueprint)
        raise AssertionError("Legacy table admitted virtual input")
    except ValueError as exception:
        if "requires explicit table growth" not in str(exception):
            raise
    assert grown.lookup(new_blueprint) == len(old.ids)
    columns = list(map(str, old.cards["feature_names"]))
    cfg = SimpleNamespace(d=256, numeric_columns=[index for index, name in enumerate(columns)
                                                if name.startswith(("num:", "has:"))])
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = {}
    for device in devices:
        module = GrowingPrintedCards(cfg, old.cards["features"], old.cards["text_emb"]).to(device).eval()
        head = nn.Linear(256, 9).to(device).eval()
        value = nn.Linear(256, 1).to(device).eval()
        with torch.no_grad():
            old_table = module.table().clone()
            logits = head(old_table).clone()
            values = value(old_table).clone()
            module.grow(grown.cards["features"], grown.cards["text_emb"])
            prefix = module.table()[:len(old.ids)]
            results[device] = {"prior_equal": torch.equal(old_table, prefix),
                               "logits_equal": torch.equal(logits, head(prefix)),
                               "values_equal": torch.equal(values, value(prefix))}
            assert all(results[device].values()), results
    report = {"passed": True,
              "scope": "Real card rows and projection/readout modules; full-policy fixture test pending C",
              "legacy_rows": len(old.ids), "extension_rows": len(grown.ids) - len(old.ids),
              "legacy_sha": old.sha, "extension_sha": grown.sha, "devices": results,
              "virtual_refused_before_and_accepted_after": True,
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                  (Path(__file__), Path("src/swccg_policy/text_priors.py"), Path("src/swccg_policy/v4_tables.py"))}}
    Path("reports/data/knowledge-v3.1/card-growth-components.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
