#!/usr/bin/env python3
"""Java PolicySampler vs Python rl.sample_sequence (plan rollout-throughput §1.5, sampling gate).

1. 10,000 random cases (options 2-40, none flags, k 1-4, excluded options, temperature 1 or 0.7, logits with spreads from 0.1
   to ~300 like real ones): Python samples a sequence + log-probability; Java recomputes the log-probability of that sequence.
   Gate: max |difference| <= 1e-5 (float32 numpy vs double Java).
2. 20 fixed cases, 100,000 Java samples each: chi-square of the first-pick counts against the exact first-pick
   probabilities. Gate: every p-value > 0.01 (with 20 tests, expect ~0.2 below 0.01 by chance; report all).

    uv run python scripts/sampler_check.py
"""
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swccg_policy.rl import sample_sequence  # noqa: E402


def case(rng: random.Random, nrng: np.random.Generator):
    n = rng.randint(2, 40)
    spread = 10 ** rng.uniform(-1, 2.5)
    logits = (nrng.standard_normal(n) * spread).astype(np.float32)
    none = np.zeros(n, dtype=bool)
    if rng.random() < 0.6:
        none[rng.randrange(n)] = True
    k = rng.choice([1, 1, 1, 2, 3, 4])
    x = sorted(rng.sample(range(n), rng.randint(0, max(0, n // 3)))) if rng.random() < 0.3 else []
    if len(x) >= n:
        x = x[:-1]
    t = rng.choice([1.0, 1.0, 0.7])
    return logits, none, k, x, t


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    cases, py_lp = [], []
    while len(cases) < 10_000:
        logits, none, k, x, t = case(rng, nrng)
        seq, lp = sample_sequence(logits, none, k, t, 0.0, rng, x or None)
        if not seq:
            continue
        cases.append({"logits": logits.tolist(), "none": none.astype(int).tolist(), "k": k, "t": t, "x": x, "seq": seq})
        py_lp.append(lp)
    dist, exact = [], []
    for _ in range(20):
        logits, none, _, x, t = case(rng, nrng)
        logits = (logits / max(1.0, float(np.abs(logits).max()) / 4)).astype(np.float32)  # keep several options likely
        dist.append({"logits": logits.tolist(), "none": none.astype(int).tolist(), "k": 1, "t": t, "x": x})
        z = logits.astype(np.float64) / t
        z[x] = -np.inf
        p = np.exp(z - z.max())
        exact.append(p / p.sum())
    draws = 100_000
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "in.json", Path(td) / "out.json"
        i.write_text(json.dumps({"cases": cases, "dist": dist, "draws": draws}))
        cp = subprocess.run([str(ROOT / "scripts/traingen_main.sh"), "com.gempukku.swccgo.bench.SamplerCheck", str(i), str(o)],
                            capture_output=True, text=True)
        if cp.returncode:
            print(cp.stdout, cp.stderr)
            sys.exit(1)
        out = json.loads(o.read_text())
    diff = np.abs(np.array(out["logp"]) - np.array(py_lp))
    pvals = []
    for cnt, p in zip(out["counts"], exact):
        cnt = np.array(cnt, dtype=float)
        keep = p > 0
        assert cnt[~keep].sum() == 0, "sampled an excluded option"
        f_exp, f_obs = p[keep] * draws, cnt[keep]
        small = f_exp < 5  # chi-square needs expected counts >= 5: pool the rare options into one bin
        if small.any():
            f_exp = np.append(f_exp[~small], f_exp[small].sum())
            f_obs = np.append(f_obs[~small], f_obs[small].sum())
        pvals.append(float(chisquare(f_obs, f_exp).pvalue))
    t1 = np.array([c["t"] == 1.0 for c in cases])
    res = {"cases": len(cases), "logp_max_abs_diff": float(diff.max()), "logp_mean_abs_diff": float(diff.mean()),
           # training samples at T = 1 (server default); at T != 1 numpy divides float32 logits in float32 (Java: double)
           "logp_max_abs_diff_T1": float(diff[t1].max()), "logp_max_abs_diff_T0.7": float(diff[~t1].max()),
           "logp_gate_1e-5_T1": bool(diff[t1].max() <= 1e-5), "chi2_pvalues": [round(v, 4) for v in pvals],
           "chi2_min_p": min(pvals), "chi2_below_0.01": sum(v < 0.01 for v in pvals)}
    print(json.dumps(res, indent=1))
    out_dir = ROOT / "reports/data/rollout-throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sampler-check-{seed}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
