"""Plan objective-learnability: per-objective RL pools and frozen-eval schedules.

Pool: the target side keeps only the RL-pool decks (with their copy multiplicity) that contain the objective;
the other side keeps the full pool. Eval schedule: 128 games, target decks cycled against the held-out decks
of the opposite side (data/schedules/eval_decipher_heldout.tsv), one row per game, target seat fixed.
"""
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = {  # slug: (blueprint id, title, side)
    'mind-what': ('7_138', 'Mind What You Have Learned', 'LS'),
    'carbon-chamber': ('7_296', 'Carbon Chamber Testing', 'DS'),
    'endor-ops': ('8_167', 'Endor Operations', 'DS'),
    'set-your-course': ('111_6', 'Set Your Course For Alderaan', 'DS'),
    'court': ('110_6', 'Court Of The Vile Gangster', 'DS'),
    'this-deal': ('109_12', 'This Deal Is Getting Worse All The Time', 'DS'),
}
GAMES = 128
# Amendment 5 (user 2026-10-04): Carbon Chamber Testing with Jabba's Prize only. Explicit target decks (equal weight):
# swdb-217 from the corpus plus three Decipher-era lists imported by scripts/import_cct_jp_decks.py. The fourth import
# (web-cctjp-hanudel-sabacc) is left out: in the engine it loops to the 6,000-decision cap within 1-11 turns.
EXPLICIT = {
    'cct-jp': ("Carbon Chamber Testing (Jabba's Prize decks)", 'DS', [
        'swdb-217-Jabba_s_prize.xml', 'web-cctjp-oakes-unh-elimination.xml',
        'web-cctjp-ford-dark-deal.xml', 'web-cctjp-wilcox-bastardized.xml'], 4),
    # Amendment 6 (user 2026-10-04): Endor Operations without the PREF2 deck, which has no Establish Secret Base.
    'endor-ops-clean': ('Endor Operations (decks with Establish Secret Base)', 'DS', [
        'swdb-145-Endor_operations.xml', 'swdb-177-Endor_Endor_Operations_Themed.xml',
        'swdb-194-DS_Endor_Biker_Ops.xml', 'swdb-31-Endor_Operations.xml'], 6),
}


def main():
    pool = [line.split('\t') for line in (ROOT / 'data/schedules/rl_pool_decipher.txt').read_text().splitlines()
            if '\t' in line and not line.startswith('#')]
    pool = [(side.strip(), deck.strip()) for side, deck in pool]
    heldout = [line.split('\t') for line in (ROOT / 'data/schedules/eval_decipher_heldout.tsv').read_text().splitlines()
               if line and not line.startswith('#')]
    held = {'DS': sorted({row[0] for row in heldout}), 'LS': sorted({row[1] for row in heldout})}
    out = ROOT / 'data/schedules/objective'
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for slug, (blueprint, title, side) in OBJECTIVES.items():
        marker = f'blueprintId="{blueprint}"'
        target = [(s, d) for s, d in pool if s == side and marker in (ROOT / d).read_text()]
        other = [(s, d) for s, d in pool if s != side]
        (out / f'{slug}.txt').write_text(''.join(f'{s}\t{d}\n' for s, d in target + other))
        decks = sorted({d for _, d in target})
        opponents = held['LS' if side == 'DS' else 'DS']
        rows = []
        for (deck, opponent) in itertools.islice(zip(itertools.cycle(decks), itertools.cycle(opponents)), GAMES):
            rows.append(f'{deck}\t{opponent}' if side == 'DS' else f'{opponent}\t{deck}')
        (out / f'{slug}-eval.tsv').write_text('\n'.join(rows) + '\n')
        summary[slug] = {'title': title, 'side': side.lower(), 'target_decks': len(decks), 'target_lines': len(target),
                         'other_lines': len(other), 'eval_games': len(rows), 'eval_opponent_decks': len(opponents)}
    corpus = Path('data/decks/v4/corpus')  # project-relative, like the pools (plan host-migration Amendment 3)
    for slug, (title, side, names, copies) in EXPLICIT.items():
        target = [(side, str(corpus / name)) for name in names for _ in range(copies)]
        other = [(s, d) for s, d in pool if s != side]
        (out / f'{slug}.txt').write_text(''.join(f'{s}\t{d}\n' for s, d in target + other))
        decks = [str(corpus / name) for name in names]
        opponents = held['LS' if side == 'DS' else 'DS']
        rows = [f'{deck}\t{opponent}' if side == 'DS' else f'{opponent}\t{deck}'
                for deck, opponent in itertools.islice(zip(itertools.cycle(decks), itertools.cycle(opponents)), GAMES)]
        (out / f'{slug}-eval.tsv').write_text('\n'.join(rows) + '\n')
        summary[slug] = {'title': title, 'side': side.lower(), 'target_decks': len(decks), 'target_lines': len(target),
                         'other_lines': len(other), 'eval_games': len(rows), 'eval_opponent_decks': len(opponents)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
