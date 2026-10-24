#!/usr/bin/env python3
"""Export the developer distribution from the lab repo (plan dev-distribution).

Copies a curated subset of the lab repo's *working tree* into this repo: code, reusable scripts, tests that run
without lab data, V3.1 deck schedules plus exactly the deck files they reference, and the architecture docs.
Benchmark / probe results, one-off experiment chains and lab-only tooling stay behind.

    python3 tools/export_from_lab.py --lab ~/src/swccg-policy --out . [--dry-run]

Writes tools/export-manifest.txt (every exported path) and tools/export-dropped.txt (tracked lab paths left out:
every dropped script / test by name, other areas as file counts per directory, with the rule that dropped them). Files this repo edited after an export are listed in tools/export-local.txt and
are never overwritten.
"""
import argparse
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

# One-off experiment scripts (plan dev-distribution §0.3); everything else in scripts/ is reusable tooling.
ONE_OFF = [
    "*_chain*.sh", "*_server.sh", "*_local.sh", "night_*", "lr2_*", "n2_chain.sh", "n3_chain.sh", "s24_*",
    "pair_watch.sh", "cc*", "cn_*", "cf_*", "tc_*", "sc_*", "ss_*", "la_*", "lc_chain.sh", "dna_*", "nvf_b_*",
    "objective_*", "mpg_check*", "rr_check.sh", "csr_*", "churn_check.sh", "frs_chain.sh", "et_*", "vc_*.sh",
    "vcf_check.sh", "free_move_*", "slow_*", "rando_*", "pe_*", "h2h_shc_lc.sh", "greedy_baselines.sh",
    "harness_replay.sh", "npe_repro.sh", "v31_after_stage.sh", "v31_pilot_then_train.sh", "option_inputs_ab_*",
    "titles_learner_*", "actor_v*", "context_v5_*", "decision_cap_fixes_runs.sh", "game_stats_runs.sh",
    "learner_*_runs.sh", "concurrency_bench.sh", "hourly_check.sh", "check_knowledge_podrace.sh",
    "analyze_v31_amendment*", "followup_*", "critic_fit*", "critic_ceiling_readout.py", "critic_counterfactuals.py",
    "critic_diag_report.py", "critic_long_report.py", "critic_probes_report.py", "critic_replay_report.py",
    "critic_v2_report.py", "pg_noise.py", "bf16_study.py", "balance_probe*", "identity_probe.py", "combo_*",
    "counter_map.py", "distill_dna.py", "extend_dna.py", "ladder_decide.py", "v6_pilot_check.py",
    "v31_precision_capacity.py", "qnet_train.py", "shaping_extract.py", "dual_pairs.py", "spike_*", "compare_*",
    "check_community_bisco.py", "pack_community_share.py", "csr_count.py", "location_probe.py",
    "check_location_icons.py", "context_ablation.py", "check_v31_input_norms.py", "check_v31_near_ties.py",
    "check_v31_soft_tables.py", "check_v31_template_norms.py", "check_v31_java.py", "collect_padded_curves.py",
    "summarize_v31_bc.py", "browser_*", "brainiac_golden.py", "make_release.py", "scan_untracked_refs.py",
    "prune_runs_list.py", "hostsync.py", "rl_pilot_v*", "rl_watchdog.sh", "rl_curve.sh", "rl_final.sh",
    "bench_server.py", "traingen*.sh", "relativize_deck_paths.py", "night_guard.sh", "mem_guard.sh",
    "strength_watch.sh", "dojo_flip_decay_check.py", "dojo_*_games.sh", "fetch_wiki_decks.py",
    "import_cct_jp_decks.py", "import_decktech_decks.py", "scan_decktech_archive.py", "slow_*.py",
]
# Kept although a pattern above matches: the generic training, probe and release tools.
KEEP_SCRIPTS = {"lr2_arm.sh", "objective_lib.sh", "night_probe.py", "night_wave.sh", "brainiac_server.sh",
                "cf_misplayed_schedule.py",
                # imported / called by kept tools (shaping_readout; native_check, sampler_check, swccg_policy.play)
                "critic_ceiling_readout.py", "traingen.sh", "traingen_main.sh"}
DROP_SCRIPT_DIRS = {"facedown_fixes", "facedown_review", "multiselect_count", "leaderboard", "runs",
                    "__pycache__"}
# V3.1 schedules and pools ("v3b and smaller"); the V1 corpus schedules stay in the lab.
SCHEDULES = [
    "data/schedules/rl_pool_decipher.txt", "data/schedules/rl_pool_decipher_weights.csv",
    "data/schedules/rl_pool_decipher_v2.txt", "data/schedules/rl_pool_decipher_v2_weights.csv",
    "data/schedules/rl_pool_decipher_v3.txt", "data/schedules/rl_pool_decipher_v3_weights.csv",
    "data/schedules/rl_pool_decipher_v3b.txt", "data/schedules/eval_decipher_heldout.tsv",
    "data/schedules/h2h_pool_v3_only.tsv", "reports/data/dojo-sweep/schedule-v3b.tsv",
]
DOCS = [
    "reports/architecture-context.md", "reports/architecture-knowledge-tracker.md", "reports/architecture-rl-loop.md",
    "reports/how-the-policy-network-works.md", "reports/training-configuration.md",
    "reports/knowledge-v3.1-contract.md", "reports/brainiac-v6.md", "reports/engine-on-feat-brainiac.md",
    "reports/preflight.md",
]
# Untracked lab files exported under another path (plan dev-distribution Amendment 1: the V-card decks of plan
# v-cards, not trained on yet); the decks they name are exported like the schedules' decks.
MAPPED = {f"runs/v-cards/schedules/{name}": f"data/schedules/v-cards/{name}"
          for name in ("gather-ds.tsv", "gather-ls.tsv", "heldout.tsv", "heldout-800.tsv")}
OTHER = ["reports/data/v-cards/decks.json", "pyproject.toml", "uv.lock", "data/dojo/rewards.json", "data/brainiac/export-records.jsonl",
         # inputs that kept code reads from reports/data: model configuration (efficient.py names_map, counter_map)
         # and the table / pool pipeline
         "reports/data/card-identity-combos/combo-map.json", "reports/data/links-round2/counter-map.json",
         "reports/data/knowledge-v3.1/card-database-export.json", "reports/data/knowledge-v3.1/decipher-pool.json",
         "reports/data/context-overhaul/context-v5-keys.json", "reports/data/dojo-sweep/schedule-v3.tsv"]
CODE_DIRS = ["src/", "engine/", "native/src/", "tests/java/"]
CODE_FILES = ["native/build.sh", "native/setup.sh", "native/build-v3.1/libswccgpolicy.so"]
DECK_RE = re.compile(r"data/decks/[^\t\n\"']+?\.(?:txt|xml|json)")


def tracked(lab):
    out = subprocess.run(["git", "-C", str(lab), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def script_rule(path):
    parts = path.split("/")
    if len(parts) > 2:
        return "keep" if parts[1] not in DROP_SCRIPT_DIRS else f"script dir {parts[1]}"
    name = parts[1]
    if name in KEEP_SCRIPTS:
        return "keep"
    hit = next((p for p in ONE_OFF if fnmatch.fnmatch(name, p)), None)
    return f"one-off {hit}" if hit else "keep"


def imports_dropped(text, dropped_mods):
    for mod in dropped_mods:
        if re.search(rf"\b(import {mod}\b|from {mod} import|scripts/{mod}\.py)", text):
            return mod
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lab, out = a.lab.resolve(), a.out.resolve()
    files = tracked(lab)
    keep, dropped = [], []
    script_keep = {p for p in files if p.startswith("scripts/") and script_rule(p) == "keep"}
    # a one-off that kept code imports stays (transitively; tests do not pull one-offs back in)
    pulled = {}
    while True:
        dropped_mods = {Path(p).stem for p in files if p.startswith("scripts/") and p.endswith(".py")
                        and p.count("/") == 1 and p not in script_keep}
        new = {}
        for p in sorted(script_keep) + [f for f in files if f.startswith("src/") and f.endswith(".py")]:
            if p.endswith(".py"):
                mod = imports_dropped((lab / p).read_text(errors="replace"), dropped_mods)
                if mod:
                    new.setdefault(f"scripts/{mod}.py", p)
        if not new:
            break
        script_keep |= set(new)
        pulled.update(new)
    for p in files:
        if p.startswith("scripts/"):
            rule = "keep" if p in script_keep else script_rule(p)
        elif p.startswith("tests/") and not p.startswith("tests/java/"):
            mod = imports_dropped((lab / p).read_text(errors="replace"), dropped_mods) if p.endswith(".py") else None
            rule = f"test imports dropped {mod}" if mod else "keep"
            if "leaderboard" in p or "hostsync" in p or "prune_runs" in p:
                rule = "lab-only test"
        elif any(p.startswith(d) for d in CODE_DIRS) or p in CODE_FILES or p in SCHEDULES or p in DOCS or p in OTHER:
            rule = "keep"
        else:
            rule = "not in distribution"
        (keep if rule == "keep" else dropped).append((p, rule))
    keep_paths = [p for p, _ in keep]
    # decks: exactly the files the kept schedules reference
    decks = set()
    for s in SCHEDULES:
        decks.update(DECK_RE.findall((lab / s).read_text()))
    for source in MAPPED:
        decks.update(DECK_RE.findall((lab / source).read_text()))
    for fixture in sorted((out / "tests/fixtures").glob("*.txt")):
        decks.update(DECK_RE.findall(fixture.read_text()))
    missing = sorted(d for d in decks if not (lab / d).is_file())
    if missing:
        raise SystemExit(f"{len(missing)} referenced decks missing, e.g. {missing[:3]}")
    keep_paths += sorted(decks)
    # scripts kept code still names but the rules dropped -> report (not silently added)
    kept_text = "\n".join((lab / p).read_text(errors="replace") for p in keep_paths
                          if p.endswith((".py", ".sh")) and (lab / p).is_file())
    dangling = sorted({Path(p).name for p, r in dropped if p.startswith("scripts/") and "/" not in p[8:]
                       and Path(p).name in kept_text})
    local = set()
    lf = out / "tools/export-local.txt"
    if lf.exists():
        local = {l.strip() for l in lf.read_text().splitlines() if l.strip() and not l.startswith("#")}
    for mod, by in sorted(pulled.items()):
        print(f"  kept {mod}: imported by {by}")
    print(f"keep {len(keep_paths)} files ({len(decks)} decks), drop {len(dropped)}; "
          f"{len(dangling)} dropped scripts still named by kept code")
    if a.dry_run:
        for d in dangling:
            print("  named:", d)
        return
    for p in keep_paths:
        if p in local:
            continue
        dst = out / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lab / p, dst)
    for source, target in MAPPED.items():
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lab / source, out / target)
    keep_paths += list(MAPPED.values())
    (out / "tools/export-manifest.txt").write_text("\n".join(sorted(keep_paths)) + "\n")
    groups = {}
    dropped = [(p, r) for p, r in dropped if p not in decks]
    for p, r in dropped:
        key = p if p.startswith(("scripts/", "tests/")) else "/".join(p.split("/")[:3 if p.startswith(("data/", "reports/data/")) else 1])
        groups.setdefault((key, r), 0)
        groups[(key, r)] += 1
    (out / "tools/export-dropped.txt").write_text(
        "# tracked lab path or directory<TAB>files<TAB>rule that left it out\n"
        + "".join(f"{k}\t{n}\t{r}\n" for (k, r), n in sorted(groups.items())))
    (out / "tools/export-dangling.txt").write_text(
        "# dropped scripts whose name still appears in exported code (mostly comments)\n" + "\n".join(dangling) + "\n")


if __name__ == "__main__":
    main()
