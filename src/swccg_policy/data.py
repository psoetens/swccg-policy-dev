"""Pack TrainGen shards (gzipped JSONL, StateEncoder v1 format) into flat numpy arrays.

Per split directory the packer writes:
  rec.npy    int32 [N, 10 + 18]  dtype, question template, question blueprint, side, tok_off, tok_n, opt_off, opt_n,
                                 game index, decision index, then the 18 global features
  tok.npy    int16 [T, 8]        kind, blueprint, zone, mine, location slot + 1, attached-to token + 1, count, flags
  tokf.npy   float16 [T, 4]      current power, ability, forfeit, deploy cost (NaN = not applicable)
  opt.npy    int32 [O, 6]        template, first text blueprint, source token + 1, blueprint, value + 1, is_none
  chosen.npy uint8 [O]           1 if the teacher chose the option
  games.json                     game ids in order
Template vocabularies (question and option texts) are built on the training split and reused for the others.

    uv run python -m swccg_policy.data --split data/shards/v1-train --vocab-from-this-split
    uv run python -m swccg_policy.data --split data/shards/v1-eval_seen --vocab data/packed/v1-train/vocab.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).resolve().parents[2]
DTYPES = ["EMPTY", "INTEGER", "MULTIPLE_CHOICE", "ARBITRARY_CARDS", "CARD_ACTION_CHOICE", "ACTION_CHOICE", "CARD_SELECTION"]
DTYPE_INDEX = {d: i for i, d in enumerate(DTYPES)}
N_GLOBAL = 18
REC_W = 10 + N_GLOBAL


def card_index(card_table: str | Path) -> dict[str, int]:
    ids = np.load(card_table)["ids"]
    return {str(b): i for i, b in enumerate(ids)}


def fnum(x):
    return math.nan if x is None else float(x)


def parse_record(r, bp_idx: dict[str, int]):
    """One decoded JSON record -> (rec fields, tokens, token floats, option rows, chosen flags, qtemplate, option templates)."""
    version, game_id, dec, side, dtype, qt, qbps, glob_, toks, opts, chosen = r[:11]
    if version != 1:  # plan remove-old-generations: V1 records only (V3.1 records are parsed by v4_data)
        raise ValueError(f"unsupported observation schema {version}")
    ti = [(t[0], bp_idx.get(t[1], 1), t[2], t[3], t[4] + 1, t[5] + 1, min(t[6], 127), t[7]) for t in toks]
    tf = [(fnum(t[8]), fnum(t[9]), fnum(t[10]), fnum(t[11])) for t in toks]
    ch = set(chosen)
    orow, otmpl, och = [], [], []
    for i, o in enumerate(opts):
        tbp = bp_idx.get(o[1][0], 1) if o[1] else 0
        obp = bp_idx.get(o[3], 1) if o[3] else 0
        val = (o[5] + 1) if o[5] is not None else 0
        orow.append((0, tbp, o[2] + 1, obp, min(val, 32), 1 if o[0] in ("<pass>", "<none>") else 0))
        otmpl.append(o[0])
        och.append(1 if i in ch else 0)
    qbp = bp_idx.get(qbps[0], 1) if qbps else 0
    rec = (DTYPE_INDEX.get(dtype, 0), version, qbp, 0 if side == "DS" else 1, dec, list(glob_))
    return rec, ti, tf, orow, och, qt, otmpl, game_id


def parse_shard(args):
    """Parse one shard into numpy arrays (built in chunks so memory stays at numpy size)."""
    path, card_table = args
    bp_idx = card_index(card_table)
    recs, qts, ots = [], [], []
    tok_parts, tokf_parts, opt_parts, cho_parts = [], [], [], []
    ct, cf, co, cc = [], [], [], []
    n_tok = n_opt = 0
    unknown_bp = collections.Counter()
    skipped = collections.Counter()

    def flush():
        if ct:
            tok_parts.append(np.asarray(ct, dtype=np.int16))
            tokf_parts.append(np.asarray(cf, dtype=np.float16))
        if co:
            opt_parts.append(np.asarray(co, dtype=np.int32))
            cho_parts.append(np.asarray(cc, dtype=np.uint8))
        ct.clear(); cf.clear(); co.clear(); cc.clear()

    with gzip.open(path, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            if r[0] != 1:
                raise ValueError("the BC packer only accepts V1 records")
            if not r[10]:
                skipped["no_chosen"] += 1
                continue
            if len(r[9]) < 2:
                skipped["single_option"] += 1
                continue
            for t in r[8]:
                if t[1] not in bp_idx:
                    unknown_bp[t[1]] += 1
            rec, ti, tf, orow, och, qt, otmpl, gid = parse_record(r, bp_idx)
            recs.append((rec, n_tok, len(ti), n_opt, len(orow), gid))
            ct.extend(ti); cf.extend(tf); co.extend(orow); cc.extend(och)
            n_tok += len(ti)
            n_opt += len(orow)
            qts.append(qt)
            ots.extend(otmpl)
            if len(ct) > 2_000_000:
                flush()
    flush()
    cat = lambda xs, w, dt: np.concatenate(xs) if xs else np.zeros((0, w) if w else (0,), dtype=dt)
    return (path, recs, cat(tok_parts, 8, np.int16), cat(tokf_parts, 4, np.float16), cat(opt_parts, 6, np.int32),
            cat(cho_parts, 0, np.uint8), qts, ots, unknown_bp, skipped)


def build_vocab(strings: list[str], min_count: int) -> dict[str, int]:
    c = collections.Counter(strings)
    vocab = {"<unk>": 0}
    for s, n in c.most_common():
        if n >= min_count:
            vocab[s] = len(vocab)
    return vocab


def count_templates(args):
    path, card_table = args
    q, o = collections.Counter(), collections.Counter()
    with gzip.open(path, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            if r[10] and len(r[9]) >= 2:
                q[r[5]] += 1
                for opt in r[9]:
                    o[opt[0]] += 1
    return q, o


def pack_shard(args):
    """Parse one shard with a fixed vocabulary and write its arrays to chunk files; return sizes and small stats."""
    path, card_table, vocab, chunk_dir, k = args
    _, recs, toks, tokf, opts, ch, qts, ots, unk, sk = parse_shard((path, card_table))
    qv, ov = vocab["q"], vocab["o"]
    if len(opts):
        opts[:, 0] = [ov.get(x, 0) for x in ots]
    local_games: dict[str, int] = {}
    rec = np.zeros((len(recs), REC_W), dtype=np.int32)
    for i, (r, t_off, t_n, o_off, o_n, gid) in enumerate(recs):
        dtype, _, qbp, side, dec, gl = r
        rec[i] = [dtype, qv.get(qts[i], 0), qbp, side, t_off, t_n, o_off, o_n, local_games.setdefault(gid, len(local_games)), dec] + gl
    base = Path(chunk_dir) / f"c{k:04d}"
    for name, arr in (("rec", rec), ("tok", toks), ("tokf", tokf), ("opt", opts), ("chosen", ch)):
        np.save(f"{base}.{name}.npy", arr)
    return k, len(rec), len(toks), len(opts), list(local_games), unk, sk


def pack(split: Path, out: Path, card_table: Path, vocab_path: Path | None, min_count: int, workers: int) -> dict:
    import shutil
    import tempfile
    shards = sorted(glob.glob(str(split / "shard-*.jsonl.gz")))
    with Pool(workers) as pool:
        if vocab_path is None:
            qc, oc = collections.Counter(), collections.Counter()
            for q, o in pool.imap(count_templates, [(s_, str(card_table)) for s_ in shards]):
                qc.update(q)
                oc.update(o)
            mk = lambda c: {"<unk>": 0, **{t: i + 1 for i, (t, n) in enumerate(x for x in c.most_common() if x[1] >= min_count)}}
            vocab = {"q": mk(qc), "o": mk(oc)}
        else:
            vocab = json.load(open(vocab_path))
        out.mkdir(parents=True, exist_ok=True)
        chunk_dir = tempfile.mkdtemp(prefix="pack-", dir=str(out))
        metas = sorted(pool.imap_unordered(pack_shard, [(s_, str(card_table), vocab, chunk_dir, k) for k, s_ in enumerate(shards)]))
    n = sum(m[1] for m in metas)
    T = sum(m[2] for m in metas)
    O = sum(m[3] for m in metas)
    rec = np.lib.format.open_memmap(out / "rec.npy", mode="w+", dtype=np.int32, shape=(n, REC_W))
    tok = np.lib.format.open_memmap(out / "tok.npy", mode="w+", dtype=np.int16, shape=(T, 8))
    tokf = np.lib.format.open_memmap(out / "tokf.npy", mode="w+", dtype=np.float16, shape=(T, 4))
    opt = np.lib.format.open_memmap(out / "opt.npy", mode="w+", dtype=np.int32, shape=(O, 6))
    cho = np.lib.format.open_memmap(out / "chosen.npy", mode="w+", dtype=np.uint8, shape=(O,))
    games: list[str] = []
    unknown, skipped = collections.Counter(), collections.Counter()
    ri = to = oo = 0
    for k, nr, nt, no, local_games, unk, sk in metas:
        base = Path(chunk_dir) / f"c{k:04d}"
        r = np.load(f"{base}.rec.npy")
        r[:, 4] += to
        r[:, 6] += oo
        r[:, 8] += len(games)
        rec[ri:ri + nr] = r
        tok[to:to + nt] = np.load(f"{base}.tok.npy")
        tokf[to:to + nt] = np.load(f"{base}.tokf.npy")
        opt[oo:oo + no] = np.load(f"{base}.opt.npy")
        cho[oo:oo + no] = np.load(f"{base}.chosen.npy")
        games += local_games
        unknown.update(unk)
        skipped.update(sk)
        ri, to, oo = ri + nr, to + nt, oo + no
    for a_ in (rec, tok, tokf, opt, cho):
        a_.flush()
    shutil.rmtree(chunk_dir)
    json.dump(vocab, open(out / "vocab.json", "w"))
    json.dump(games, open(out / "games.json", "w"))
    per_type = collections.Counter(DTYPES[int(d)] for d in np.asarray(rec[:, 0]))
    stats = {"records": n, "tokens": T, "options": O, "tokens_per_record": T / max(n, 1), "options_per_record": O / max(n, 1),
             "games": len(games), "per_type": dict(per_type), "skipped": dict(skipped),
             "unknown_blueprints": dict(unknown.most_common(20)), "unknown_blueprint_tokens": sum(unknown.values()),
             "vocab_q": len(vocab["q"]), "vocab_o": len(vocab["o"])}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="directory with shard-*.jsonl.gz")
    ap.add_argument("--out", help="output directory (default data/packed/<split name>)")
    ap.add_argument("--cards", default=str(ROOT / "data/cards/card_table.npz"))
    ap.add_argument("--vocab", help="vocab.json of the training split (omit to build the vocabulary from this split)")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    split = Path(a.split)
    out = Path(a.out) if a.out else ROOT / "data/packed" / split.name
    stats = pack(split, out, Path(a.cards), Path(a.vocab) if a.vocab else None, a.min_count, a.workers)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
