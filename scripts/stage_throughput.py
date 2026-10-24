"""Plan host-migration M-4: a stage run's throughput over its first N seconds (updates, games per second after the stage's
own 180-s warm-up, learner seconds, stale steps, GPU use per GPU, load, free RAM).

  .venv/bin/python scripts/stage_throughput.py runs/dojo/<tag> 620
"""

import json, statistics as st, sys
from pathlib import Path
run = Path(sys.argv[1]); limit = float(sys.argv[2])
rows = [json.loads(l) for l in open(run / 'learner.jsonl')]
t0 = json.load(open(run / 'started.json'))['time']
rows = [r for r in rows if r['time'] - t0 <= limit]
warm = [r for r in rows if r['time'] - t0 >= 180]  # the stage's own discard-first-180-s convention
span = warm[-1]['time'] - warm[0]['time'] if len(warm) > 1 else float('nan')
games = sum(r['games'] for r in warm[1:])
print(json.dumps({
    'updates': len(rows), 'updates_after_180s': len(warm),
    'games_per_s_after_180s': round(games / span, 3) if span == span else None,
    'update_s_median': round(st.median(r['update_seconds'] for r in warm), 2),
    'stale_steps': sum(r['discarded_stale_steps'] for r in rows),
    'steps_used': sum(r['steps'] for r in rows),
    'queue_wait_s_median': round(st.median(r.get('queue_wait_seconds', 0) for r in warm), 2),
}))
res = run / 'resources.json'
if res.exists():
    samples = [s for s in json.load(open(res)) if s['elapsed'] >= 180 and s['elapsed'] <= limit]
    if samples and 'gpus' in samples[0]:
        for i, gpu in enumerate(samples[0]['gpus']):
            print('gpu', gpu['index'], 'util median', st.median(s['gpus'][i]['util_percent'] for s in samples),
                  'mem max MiB', max(s['gpus'][i]['used_mib'] for s in samples))
    elif samples:
        print('gpu util median', st.median(s['gpu_util_percent'] for s in samples), 'mem max', max(s['gpu_used_mib'] for s in samples))
    print('load median', round(st.median(s['load'][0] for s in samples), 1), 'min RAM GB', round(min(s['memory_available'] for s in samples) / 1e9, 1))
