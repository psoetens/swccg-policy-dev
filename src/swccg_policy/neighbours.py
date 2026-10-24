"""Card-embedding sanity check: nearest neighbours of well-known cards, before and after training.

"Before" = the frozen initialisation inputs (text embedding and structured features, each normalised and concatenated);
"after"  = the trained card table E[c] = MLP(features, text) + delta[c]. Neighbours are restricted to cards that occur in
the training decks, so the lists reflect cards the policy actually saw.

    uv run python -m swccg_policy.neighbours --checkpoint checkpoints/bc-v1/model.pt --out reports/data/neighbours.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .decks import load_pool, REINSTATED
from .train import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROBES = ["Darth Vader, Dark Lord Of The Sith", "Luke Skywalker, Jedi Knight", "Han Solo", "Boba Fett", "Emperor Palpatine",
          "Sense", "Alter", "Houjix", "Millennium Falcon", "Tatooine: Mos Eisley", "Executor", "Nabrun Leids",
          "Obi-Wan Kenobi", "Chewbacca"]


def normed(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cards", default=str(ROOT / "data/cards/card_table.npz"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "reports/data/neighbours.md"))
    a = ap.parse_args()
    z = np.load(a.cards)
    ids, titles = [str(x) for x in z["ids"]], [str(x) for x in z["titles"]]
    pool = load_pool(REINSTATED)
    heldout = set(json.load(open(ROOT / "data/schedules/pool.json"))["heldout"])
    train_bps = {b for k, d in pool.items() if k not in heldout for b in d["cards"]}
    cand = np.array([i for i, b in enumerate(ids) if b in train_bps])
    freq = {}
    for k, d in pool.items():
        if k in heldout:
            continue
        for b in d["cards"]:
            freq[b] = freq.get(b, 0) + 1
    before = np.concatenate([normed(z["text_emb"]), normed(z["features"])], axis=1)
    model, _ = load_checkpoint(a.checkpoint)
    with torch.no_grad():
        after = model.cards.table().float().cpu().numpy()
    lines = ["# Card-embedding neighbours (cosine), restricted to cards in the training decks", ""]
    chosen = []
    for t in PROBES:
        opts = [i for i in cand if titles[i] == t]
        if opts:
            chosen.append(max(opts, key=lambda i: freq.get(ids[i], 0)))
        if len(chosen) >= a.probes:
            break
    for name, M in (("before (text ⊕ structured features)", before), ("after training (E = MLP + δ)", after)):
        lines += [f"## {name}", ""]
        Mn = normed(M[cand])
        pos = {int(c): j for j, c in enumerate(cand)}
        for i in chosen:
            sims = Mn @ Mn[pos[int(i)]]
            order = [int(cand[j]) for j in np.argsort(-sims) if int(cand[j]) != int(i)][:a.k]
            lines.append(f"- **{titles[i]}** ({ids[i]}, in {freq.get(ids[i], 0)} decks): "
                         + ", ".join(f"{titles[j]} ({ids[j]})" for j in order))
        lines.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
