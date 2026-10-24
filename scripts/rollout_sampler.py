#!/usr/bin/env python3
"""Sample CPU use once per second: machine busy fraction (/proc/stat) and CPU seconds of named processes including all their
threads (/proc/PID/stat utime + stime). Writes CSV rows: t, machine_busy, machine_cores, then one cumulative CPU-seconds column
per named process. Stops on SIGINT/SIGTERM. Used by scripts/rollout_bench.sh."""
import argparse
import os
import signal
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")


def machine():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def proc_cpu(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        fields = s[s.rindex(")") + 2:].split()
        return (int(fields[11]) + int(fields[12])) / TICK
    except (FileNotFoundError, ProcessLookupError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pids", required=True, help="name=pid,name=pid")
    ap.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args()
    named = [p.split("=") for p in a.pids.split(",") if p]
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    cores = os.cpu_count()
    with open(a.out, "w") as f:
        f.write("t,machine_busy,machine_cores," + ",".join(n for n, _ in named) + "\n")
        tot0, idle0 = machine()
        while not stop:
            time.sleep(a.every)
            tot, idle = machine()
            busy = 1 - (idle - idle0) / max(1, tot - tot0)
            tot0, idle0 = tot, idle
            f.write(f"{time.time():.3f},{busy:.4f},{cores}," + ",".join(f"{proc_cpu(p):.2f}" for _, p in named) + "\n")
            f.flush()


if __name__ == "__main__":
    sys.exit(main())
