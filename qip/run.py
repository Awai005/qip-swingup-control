"""
Run the controller in fill_the_code.py for 10 s without the animation (same loop
as main.py), write transitions_headless.txt and print the challenge metrics.

  python -m qip.run                          # real simulator (start ./simulator/base first)
  python -m qip.run --sim model              # identified model, no simulator needed
  python -m qip.run --disturb 6.0,0.1,20     # add 20 N for 0.1 s at t = 6 s (repeatable)
"""
import argparse
import os
import numpy as np
from fill_the_code import Controller, lqr_delta
from qip.common import HANGING, DT, U_MAX
from qip.metrics import region_stats
from qip.model import QIPModel
from qip.sim_client import SimClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rollout(sim="binary", sec=10.0, disturb=(), lqr=None, swingup=None, client=None):
    """Closed-loop run from the hanging start. disturb: (start_s, duration_s, force_N) pulses."""
    kw = {k: v for k, v in (("lqr", lqr), ("swingup", swingup)) if v is not None}
    ctrl = Controller(**kw)
    if sim == "model":
        model = QIPModel(); s = HANGING.copy()
        advance = lambda s, u: model.step(s, u)[0]
    else:
        client = client or SimClient()
        if client.ctrl_ms() != 10:
            raise RuntimeError("start the simulator with --ctrl_ms 10")
        s = client.state("reset")
        advance = lambda s, u: client.state(f"step {u}")
    states, forces = [s.copy()], []
    for k in range(int(round(sec / DT))):
        u = ctrl.compute_force(list(s))
        extra = sum(f for (t0, dur, f) in disturb if t0 <= k * DT < t0 + dur)
        s = advance(s, float(np.clip(u + extra, -U_MAX, U_MAX)))
        states.append(np.array(s)); forces.append(u)
    return np.array(states), np.array(forces), ctrl


def report(states, ctrl):
    raw = region_stats(states, 10, wrap_angles=False)
    wrp = region_stats(states, 10, wrap_angles=True)
    et = lambda d: "not reached" if d["entry_time"] is None else f"{d['entry_time']:.2f} s"
    sw = "never" if ctrl.switch_step is None else f"{ctrl.switch_step * DT:.2f} s"
    final = states[-1]
    return (f"LQR takes over at {sw} | region entered (+5 s hold): raw {et(raw)}, wrapped {et(wrp)} | "
            f"longest hold {raw['longest_hold']:.2f} s | final |x| {abs(final[0]):.3f} m, "
            f"max |angle error| {np.degrees(np.abs(lqr_delta(final, 1e9)[2::2]).max()):.3f} deg")


def fmt(values):
    return ", ".join(f"{v:.9f}" for v in values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", choices=["binary", "model"], default="binary")
    p.add_argument("--sec", type=float, default=10.0)
    p.add_argument("--disturb", action="append", default=[], help="start_s,duration_s,force_N")
    p.add_argument("--out", default="transitions_headless.txt")
    a = p.parse_args()
    pulses = [tuple(float(v) for v in d.split(",")) for d in a.disturb]
    states, forces, ctrl = rollout(a.sim, a.sec, pulses)
    with open(a.out, "w") as f:                       # same format as main.py's transitions.txt
        for k, u in enumerate(forces):
            f.write(f"({fmt(states[k])})({u:.9f})({fmt(states[k + 1])})\n")
    print(f"[{a.sim}] {len(forces)} steps -> {a.out}" + (f" | disturbances {pulses}" if pulses else ""))
    print(report(states, ctrl))


if __name__ == "__main__":
    main()
