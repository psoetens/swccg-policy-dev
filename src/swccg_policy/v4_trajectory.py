"""SWTRJ003: int32 token rows, actor/critic uint8 grids and acting-table provenance."""

import gzip
import struct
import zlib

import numpy as np
import orjson

from .trajectory import FRAME, TYPES, _read
from .v4_contract import validate


MAGIC = b"SWTRJ003"
STEP = struct.Struct(">10idq")


class TrajectoryReader:
    """Explicit, validated acting-table registry; no implicit growth or SHA relabeling."""

    def __init__(self, learner_tables, acting_tables=()):
        self.learner = learner_tables
        self.acting = {learner_tables.sha: learner_tables}
        for tables in acting_tables:
            tables.assert_prefix_of(learner_tables)
            self.acting[tables.sha] = tables

    def load(self, path):
        with open(path, "rb") as stream:
            if _read(stream, 8) != MAGIC:
                raise ValueError("SWTRJ003 required")
            _read(stream, 1)
            sha = _read(stream, 32).hex()
        if sha not in self.acting:
            raise ValueError("Acting table SHA is not in the learner's explicit registry")
        # Prefix relation was checked at registration; decoding validates the owned table's IDs.
        acting = self.acting[sha]
        return load_binary(path, acting)


def encode_game(game, tables):
    parts = []
    count = len(game["items"])
    repeats = game.get("repeats", [0] * count)
    ks = game.get("ks", [1] * count)
    sequences = game.get("knowledge_sequences", [0] * count)
    for index, (actor, critic) in enumerate(zip(game["items"], game["critic_items"], strict=True)):
        validate(actor, tables.vocabulary, len(tables.cards.ids), actor=True)
        validate(critic, tables.vocabulary, len(tables.cards.ids))
        if actor.get("table_sha") != tables.sha or critic.get("table_sha") != tables.sha:
            raise ValueError("Step table SHA disagrees with acting policy")
        if not np.array_equal(actor["head"], critic["head"]):
            raise ValueError("Actor/critic head mismatch")
        selected = game["seqs"][index]
        excluded = game["excluded"][index]
        parts.append(STEP.pack(len(actor["tok"]), len(actor["opt"]), len(critic["tok"]),
                               len(actor["kgrid"]), len(critic["kgrid"]), len(selected),
                               -1 if excluded is None else len(excluded), game["policy_versions"][index],
                               repeats[index], ks[index], float(game["old_logp"][index]), sequences[index]))
        for array, dtype in ((actor["head"], "<i4"), (actor["tok"], "<i4"), (actor["tokf"], "<f4"),
                             (actor["opt"], "<i4"), (actor["kgrid"], "u1"),
                             (critic["tok"], "<i4"), (critic["tokf"], "<f4"), (critic["kgrid"], "u1")):
            parts.append(np.asarray(array, dtype=dtype).tobytes())
        parts.append(np.asarray(selected, dtype=">i4").tobytes())
        if excluded is not None:
            parts.append(np.asarray(excluded, dtype=">i4").tobytes())
    metadata = {key: game[key] for key in ("g", "s", "o", "w", "reward", "won", "lf", "opp", "learner")
                if key in game}
    metadata.update(n_steps=count, table_sha=tables.sha, schema_version=4)
    widths = {item["tokf"].shape[1] for item in (*game["items"], *game["critic_items"])}
    if len(widths) != 1:
        raise ValueError("Mixed float widths in one game")
    if widths == {8}:
        metadata["float_width"] = 8
    return orjson.dumps(metadata), b"".join(parts)


def decode_game(raw, metadata, acting):
    position = 0

    def array(dtype, shape):
        nonlocal position
        count = int(np.prod(shape))
        size = count * np.dtype(dtype).itemsize
        if position + size > len(raw):
            raise ValueError("Truncated ABI-4 tensor block")
        result = np.frombuffer(raw, dtype=dtype, count=count, offset=position).reshape(shape)
        position += size
        return result

    result = dict(metadata)
    if result.get("schema_version") != 4 or result.get("table_sha") != acting.sha:
        raise ValueError("Trajectory game schema/table mismatch")
    count = result.pop("n_steps")
    # Plan objective-dojo Amendment 2: tensor ABI 5 records eight floats per token.
    width = result.pop("float_width", 4)
    if width not in (4, 8):
        raise ValueError("Invalid float width")
    if not isinstance(count, int) or not 1 <= count <= len(raw) // (STEP.size + 88):
        raise ValueError("Invalid step count")
    fields = ("items", "critic_items", "types", "policy_versions", "seqs", "excluded",
              "knowledge_sequences", "repeats", "ks", "old_logp")
    result.update({key: [] for key in fields})
    for _ in range(count):
        if position + STEP.size > len(raw):
            raise ValueError("Truncated ABI-4 step header")
        nt, no, nc, nk, nkc, ns, nx, version, repeats, k, logp, sequence = STEP.unpack_from(raw, position)
        position += STEP.size
        if not (0 <= nt <= 1_000_000 and 0 <= nc <= 1_000_000 and 0 <= no <= 100_000
                and 0 <= nk <= nt and 0 <= nkc <= nc and 0 <= ns <= no and -1 <= nx <= no
                and version >= 0 and repeats >= 0 and k >= 0 and np.isfinite(logp)):
            raise ValueError("Invalid ABI-4 step dimensions/sampling fields")
        head = array("<i4", (22,))
        actor = {"tok": array("<i4", (nt, 8)), "tokf": array("<f4", (nt, width)),
                 "opt": array("<i4", (no, 6)), "kgrid": array("u1", (nk, 153))}
        critic = {"tok": array("<i4", (nc, 8)), "tokf": array("<f4", (nc, width)),
                  "kgrid": array("u1", (nkc, 153)), "opt": np.empty((0, 6), np.int32)}
        for item, privileged in ((actor, False), (critic, True)):
            item.update(head=head, schema_version=4, table_sha=acting.sha, privileged=privileged)
            validate(item, acting.vocabulary, len(acting.cards.ids), actor=not privileged)
        selected = array(">i4", (ns,)).tolist()
        excluded = None if nx == -1 else array(">i4", (nx,)).tolist()
        if any(index < 0 or index >= no for index in selected + (excluded or [])):
            raise ValueError("Invalid selected/excluded option")
        values = (actor, critic, TYPES[int(head[0])], version, selected, excluded, sequence, repeats, k, logp)
        for field, value in zip(fields, values, strict=True):
            result[field].append(value)
    if position != len(raw):
        raise ValueError("Trailing ABI-4 trajectory bytes")
    result["old_logp"] = np.asarray(result["old_logp"], np.float32)
    return result


def write_binary(path, games, tables, level=1):
    if not 0 <= level <= 9:
        raise ValueError("Compression level must be 0..9")
    with open(path, "wb") as stream:
        stream.write(MAGIC + bytes([int(level != 0)]) + bytes.fromhex(tables.sha))
        for game in games:
            metadata, raw = encode_game(game, tables)
            stored = raw if level == 0 else gzip.compress(raw, compresslevel=level, mtime=0)
            stream.write(FRAME.pack(len(metadata), len(raw), len(stored), zlib.crc32(raw)))
            stream.write(metadata)
            stream.write(stored)


def load_binary(path, tables, acting_tables=None):
    acting = tables if acting_tables is None else acting_tables
    acting.assert_prefix_of(tables)
    games = []
    with open(path, "rb") as stream:
        if _read(stream, 8) != MAGIC:
            raise ValueError("SWTRJ003 required; older schemas cannot be relabeled")
        codec = _read(stream, 1)[0]
        if codec not in (0, 1) or _read(stream, 32).hex() != acting.sha:
            raise ValueError("Trajectory codec or acting table SHA mismatch")
        while header := stream.read(FRAME.size):
            if len(header) != FRAME.size:
                raise ValueError("Truncated game frame")
            metadata_size, raw_size, stored_size, checksum = FRAME.unpack(header)
            if metadata_size > 1_048_576 or raw_size > 2 ** 31 or stored_size > 2 ** 31:
                raise ValueError("Oversized trajectory frame")
            metadata = orjson.loads(_read(stream, metadata_size))
            stored = _read(stream, stored_size)
            if codec:
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                raw = decoder.decompress(stored, raw_size + 1)
                if not decoder.eof or decoder.unused_data:
                    raise ValueError("Invalid compressed trajectory frame")
            else:
                raw = stored
            if len(raw) != raw_size or zlib.crc32(raw) != checksum:
                raise ValueError("Trajectory length/checksum mismatch")
            games.append(decode_game(raw, metadata, acting))
    return games
