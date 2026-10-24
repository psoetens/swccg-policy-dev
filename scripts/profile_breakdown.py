#!/usr/bin/env python3
"""Break an async-profiler collapsed-stack file (event cpu/itimer or alloc, option `threads`) of a TrainGen run down into
categories, by the first matching rule along each stack (root -> leaf order of the rules below, checked innermost first where
stated). Prints JSON: share of samples per category, and the top leaf frames overall and within the game threads.

    python3 scripts/profile_breakdown.py runs/bench/prof-cpu/profile.collapsed
"""
import collections
import json
import re
import sys

# (category, predicate on the ';'-joined stack). Checked in order; first hit wins.
RULES = [
    ("gc", lambda s: re.search(r"^\[?(GC Thread|G1 |ParGC|VM Thread|G1 Conc|G1 Refine|G1 Service)", s) is not None),
    ("jit", lambda s: re.search(r"^\[?(C1 CompilerThre|C2 CompilerThre)", s) is not None),
    ("policy_socket_io", lambda s: "PolicyAi.send" in s or "PolicyAi.query" in s),
    ("encoder_modifier_queries", lambda s: "StateEncoder.encode" in s and "ModifiersQuerying" in s),
    ("encoder_other", lambda s: "StateEncoder" in s),
    ("policy_ai_other", lambda s: "PolicyAi" in s),
    ("bots_advanced_random_beginner", lambda s: re.search(r"(AdvancedAi|RandomAi|BeginnerAi)\.", s) is not None),
    ("engine_carry_out", lambda s: "carryOutPendingActionsUntilDecisionNeeded" in s),
    ("engine_decision_made", lambda s: "decisionMade" in s),
    ("game_setup", lambda s: "startGame" in s or "DefaultSwccgGame.<init>" in s or "loadDeck" in s),
    ("traingen_other", lambda s: "TrainGen" in s),
]


def main(path):
    cats = collections.Counter()
    leaves = collections.Counter()
    game_leaves = collections.Counter()
    engine_frames = collections.Counter()
    total = 0
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        stack, _, n = line.rpartition(" ")
        n = int(n)
        total += n
        cat = next((c for c, pred in RULES if pred(stack)), "other")
        cats[cat] += n
        frames = stack.split(";")
        leaves[frames[-1]] += n
        if "TrainGen" in stack:
            game_leaves[frames[-1]] += n
            # engine classes on the stack (deduplicated per sample): where rule evaluation spends its time
            seen = set()
            for f in frames:
                m = re.match(r"com/gempukku/swccgo/logic/(\w+/)*(\w+)\.(\w+)", f.replace(".", "/", f.count(".") - 1))
                if m and "querying" in f.lower() and f not in seen:
                    seen.add(f)
                    engine_frames[f] += n
    out = {"total_samples": total,
           "categories": {k: round(v / total, 4) for k, v in cats.most_common()},
           "top_leaves": [(k, round(v / total, 4)) for k, v in leaves.most_common(25)],
           "top_game_thread_leaves": [(k, round(v / total, 4)) for k, v in game_leaves.most_common(25)],
           "top_modifier_query_frames_inclusive": [(k, round(v / total, 4)) for k, v in engine_frames.most_common(20)]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
