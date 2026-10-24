"""Plan learner-speed G-5: the per-option table (seconds, speed-up, parity, memory) from reports/data/learner-speed.

  .venv/bin/python scripts/learner_speed_report.py   -> reports/data/learner-speed/table.md (included in the report)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/data/learner-speed'
# label: (options, plan item, precision rule of the gradient parity: actor / critic cosine floor)
ROWS = {
    'base': ('today (eager, fp32 critic, TF32 off)', '-', None),
    'tf32': ('critic TF32', 'S-1', 'r9'),
    'compile': ('compiled table builds', 'S-2', (0.99, 0.999)),
    'stream': ('critic stream', 'S-3', (0.999, 0.999)),
    'prefetch': ('overlapped decode', 'S-4', 'identical'),
    'combined': ('S-1 + S-2 + S-3 + S-4', 'all', (0.99, 0.99)),
    'combined-nostream': ('S-1 + S-2 + S-4', '', (0.99, 0.99)),
    'notf32': ('S-2 + S-3 + S-4', 'plan-only best', (0.99, 0.999)),
    'notf32-nostream': ('S-2 + S-4', '', (0.99, 0.999)),
    'single': ('single-GEMM tables', 'S-5 (new)', (0.99, 0.999)),
    'single-compile': ('S-5 + S-2', 'S-5 (new)', (0.99, 0.999)),
    'single-all': ('S-5 + S-2 + S-3 + S-4', 'S-5 (new)', (0.99, 0.999)),
    'single-nocompile': ('S-5 + S-3 + S-4', 'chosen (user)', (0.99, 0.999)),
}


def load(name):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else None


def r9_verdict():
    result = load('grads_r9_tf32.json')
    if result is None:
        return 'R-9: not run', None
    cells = {cell: result['cells'][cell]['tf32:cos']['median'] for cell in ('gae:64', 'gae:256', 'mc:256')}
    ok = all(value >= 0.99 for value in cells.values())
    return 'R-9 ' + ', '.join(f'{cell} {value:.4f}' for cell, value in cells.items()), ok


def parity(label, rule):
    if rule is None:
        return '-', True
    if rule == 'r9':
        text, ok = r9_verdict()
        result = load(f'parity_{label}.json')
        if result:
            text += f"; learner mb critic {result['gradients']['critic:option:cos']['median']:.4f}"
        return text, ok
    if rule == 'identical':
        timing = load(f'time_{label}.json')
        ok = bool(timing and timing['summary'].get('prefetch_identical'))
        return 'identical batches' if ok else 'NOT identical', ok
    result = load(f'parity_{label}.json')
    if result is None:
        return 'not run', False
    grads = result['gradients']
    actor, critic = grads['actor:option:cos'], grads['critic:option:cos']
    ok = actor['median'] >= rule[0] and critic['median'] >= rule[1]
    if 'S-1' in ROWS[label][0]:
        ok = ok and bool(r9_verdict()[1])
    return (f"actor {actor['median']:.4f} (min {actor['min']:.4f}), critic {critic['median']:.4f} "
            f"(min {critic['min']:.4f})"), ok


def main():
    base = load('time_base.json')
    reference = base['summary']['learner_seconds']
    lines = ['| config | plan | values+GAE s | PPO s | decode s | learner s / update | speed-up | peak GiB | parity | ok |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for label, (options, item, rule) in ROWS.items():
        timing = load(f'time_{label}.json')
        if timing is None:
            continue
        summary = timing['summary']
        text, ok = parity(label, rule)
        lines.append(f"| `{label}`: {options} | {item} | {summary['values_gae_seconds']:.2f} | "
                     f"{summary['ppo_seconds']:.2f} | {summary.get('decode_seconds', timing['decode_seconds']):.2f} | "
                     f"**{summary['learner_seconds']:.2f}** | {reference / summary['learner_seconds']:.2f}x | "
                     f"{summary['peak_gib']:.2f} | {text} | {'yes' if ok else 'NO'} |")
    update_lines = ['', '| config | actor delta cos (repeat / option) | critic delta cos (repeat / option) |',
                    '|---|---|---|']
    for label in ROWS:
        result = load(f'parity_{label}.json')
        if result is None:
            continue
        repeat, option = result['update']['repeat'], result['update']['option']
        update_lines.append(f"| `{label}` | {repeat['actor:delta:cos']:.3f} / {option['actor:delta:cos']:.3f} | "
                            f"{repeat['critic:delta:cos']:.3f} / {option['critic:delta:cos']:.3f} |")
    text = '\n'.join(lines + update_lines) + '\n'
    (DATA / 'table.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
