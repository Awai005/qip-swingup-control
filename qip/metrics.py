"""
Evaluation metric from the challenge: time until the state enters the region
|theta_i| <= 0.05 rad, |theta_dot_i| <= 0.05 rad/s, |x| <= 0.5 m and then
stays there for more than 5 s.

Reported twice: with raw angles (strict) and with wrapped angles, because
the PDF does not say whether the evaluator wraps.

Usage on a transitions.txt produced by main.py:
    python -m qip.metrics transitions.txt
"""
import argparse
import re
import numpy as np
from qip.common import HOLD_SEC, in_region
CTRL_MS = 10


def region_stats(states, ctrl_ms=CTRL_MS, wrap_angles=False):
    dt = ctrl_ms / 1000.0
    need = int(round(HOLD_SEC / dt))
    inside = np.array([in_region(s, wrap_angles) for s in states], dtype=bool)

    entry_time, run, best_run, run_start = None, 0, 0, 0
    for k, flag in enumerate(inside):
        if flag:
            if run == 0:
                run_start = k
            run += 1
            best_run = max(best_run, run)
            if run >= need and entry_time is None:
                entry_time = run_start * dt
        else:
            run = 0
    return dict(
        entry_time=entry_time,                     # None = never held for HOLD_SEC
        longest_hold=best_run * dt,
        frac_inside=float(inside.mean()) if len(inside) else 0.0,
    )


def episode_score(states, ctrl_ms=CTRL_MS):
    """Single scalar for model selection: lower is better.

    Success -> entry time (s). Failure -> 100 - longest hold, so partial
    progress still ranks above nothing.
    """
    st = region_stats(states, ctrl_ms, wrap_angles=False)
    if st["entry_time"] is not None:
        return st["entry_time"], st
    return 100.0 - st["longest_hold"], st


def parse_transitions(path):
    """Read transitions.txt -> list of next_states (plus the first state)."""
    states = []
    with open(path) as f:
        for line in f:
            groups = re.findall(r"\(([^)]*)\)", line)
            if len(groups) != 3:
                continue
            s = [float(v) for v in groups[0].split(",")]
            ns = [float(v) for v in groups[2].split(",")]
            if not states:
                states.append(s)
            states.append(ns)
    return states


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="transitions.txt")
    p.add_argument("--ctrl_ms", type=int, default=CTRL_MS)
    a = p.parse_args()
    states = parse_transitions(a.path)
    print(f"{len(states)} states ({len(states) * a.ctrl_ms / 1000:.2f} s)")
    for name, wa in [("raw angles", False), ("wrapped angles", True)]:
        st = region_stats(states, a.ctrl_ms, wrap_angles=wa)
        et = "not reached" if st["entry_time"] is None else f"{st['entry_time']:.2f} s"
        print(f"[{name:14s}] entry+hold: {et} | longest hold {st['longest_hold']:.2f} s "
              f"| inside {100 * st['frac_inside']:.1f}%")
