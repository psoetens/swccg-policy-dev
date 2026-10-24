"""Process and resource helpers for the stage launchers (scripts/v31_stage.py, scripts/v31_collect.py). Moved here
unchanged from the removed V2/V3 launcher knowledge_stage.py (plan remove-old-generations).
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from swccg_policy.trajectory import trajectory_files

ROOT=Path(__file__).resolve().parents[1]

def stop(p):
    if p is None or p.poll() is not None:return
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.wait()

def queue_stats(folder):
    count=steps=size=0
    for p in trajectory_files(folder):
        try:
            meta=json.loads(p.with_name(p.name+'.meta.json').read_text());n=p.stat().st_size
        except FileNotFoundError:continue  # learner can atomically consume it between reads
        count+=1;steps+=meta['steps'];size+=n
    return {'files':count,'steps':steps,'bytes':size}

def rss_tree(pid):
    total=0
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):total=int(line.split()[1])*1024
        for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():total+=rss_tree(int(child))
    except (FileNotFoundError,ProcessLookupError):pass
    return total

def gpu_table(text):
    """nvidia-smi rows (used, total, util, temperature) by index, in nvidia-smi's numbering."""
    return [[int(x.strip()) for x in line.split(',')] for line in text.splitlines() if line.strip()]

def resources(gpus=None):
    """Host and GPU sample. The gpu_* keys describe the first of `gpus` (default: nvidia-smi's first GPU, as before);
    with `gpus`, 'gpus' lists each of them (plan host-migration M-3)."""
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value,*_=line.split();mem[key.rstrip(':')]=int(value)*1024
    table=gpu_table(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True))
    vals=table[gpus[0] if gpus else 0]
    sample={'memory_available':mem['MemAvailable'],'swap_free':mem['SwapFree'],'disk_free':shutil.disk_usage(ROOT).free,
            'load':os.getloadavg(),'gpu_used_mib':vals[0],'gpu_total_mib':vals[1],'gpu_util_percent':vals[2],'gpu_temperature':vals[3]}
    if gpus:
        sample['gpus']=[{'index':i,'used_mib':table[i][0],'total_mib':table[i][1],'util_percent':table[i][2]} for i in gpus]
    return sample
