"""Gate C's approved FP32 top-two near-tie rule."""

import numpy as np


def greedy_match(reference, candidate):
    reference = np.asarray(reference, np.float32)
    candidate = np.asarray(candidate, np.float32)
    if reference.ndim != 1 or candidate.shape != reference.shape or not len(reference):
        raise ValueError("Greedy parity requires matching nonempty logit vectors")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Greedy parity requires finite logits")
    # Stable order gives an unambiguous pair even if three or more choices tie.
    top = np.argsort(-reference, kind="stable")[:2]
    expected = int(top[0])
    actual = int(candidate.argmax())
    gap = float(reference[top[0]] - reference[top[1]]) if len(top) == 2 else None
    strict = actual == expected
    near_tie = not strict and gap is not None and gap < 1e-6 and actual in top
    return {"matched": bool(strict or near_tie), "strict": strict, "used_near_tie": bool(near_tie),
            "python_choice": expected, "backend_choice": actual, "python_top_two": top.tolist(),
            "python_gap": gap}
