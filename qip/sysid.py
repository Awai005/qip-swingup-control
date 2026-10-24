"""
System identification of the black-box simulator.

1. Record transitions (state, force, next state) under random smooth forces.
2. One-step prediction error of the datasheet model (problem statement values).
3. Least-squares fit of the physical model to the recorded transitions:
     link masses  m_i = rho * l_i   (one linear density rho; this structure was
                                     found by first fitting all parameters freely)
     inertias     I_1..I_4 about each CoG,  CoG at l_i / 2,  gravity g
   The cart mass (1.3 kg) and link lengths are taken from the problem statement.
4. One-step prediction error of the identified model on held-out transitions.

    ./simulator/base --ctrl_ms 10                 # terminal 1
    python -m qip.sysid                           # terminal 2 (~2-4 min)
    python -m qip.sysid --data sysid_data.npz     # refit from saved data (no simulator)
"""
import argparse
import json
import os
import time
import numpy as np
from scipy.optimize import least_squares
from qip.common import CART_MASS, LINK_LENGTHS, DATASHEET, U_MAX
from qip.model import QIPModel
from qip.sim_client import SimClient


def collect(episodes=6, steps=500, seed=0):
    """Random smooth forces for 350 steps, then free motion, from the hanging start."""
    c = SimClient(); rng = np.random.default_rng(seed)
    S, U, S2 = [], [], []
    for _ in range(episodes):
        s = c.state("reset"); u = 0.0
        for t in range(steps):
            u = float(np.clip(0.95 * u + rng.normal(0, 8), -U_MAX, U_MAX)) if t < 350 else 0.0
            s2 = c.state(f"step {u}")
            S.append(s); U.append(u); S2.append(s2); s = s2
    c.close()
    return np.array(S), np.array(U), np.array(S2)


def one_step_rms(model, S, U, S2):
    e = model.step(S, U, sub=10) - S2
    return np.sqrt((e ** 2).mean(0))


def build(p):
    rho, I, g = p[0], p[1:5], p[5]
    return QIPModel(CART_MASS, rho * LINK_LENGTHS, LINK_LENGTHS, LINK_LENGTHS / 2, I, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="load transitions instead of recording")
    ap.add_argument("--save", default="sysid_data.npz")
    ap.add_argument("--fit_samples", type=int, default=400)
    a = ap.parse_args()

    if a.data:
        d = np.load(a.data); S, U, S2 = d["S"], d["U"], d["S2"]
    else:
        t0 = time.time(); S, U, S2 = collect()
        np.savez(a.save, S=S, U=U, S2=S2)
        print(f"recorded {len(S)} transitions in {time.time() - t0:.0f} s -> {a.save}")

    rng = np.random.default_rng(1)
    idx = rng.permutation(len(S)); fit, test = idx[:a.fit_samples], idx[-600:]
    scale = np.std(S2 - S, axis=0) + 1e-6
    resid = lambda p: ((build(p).step(S[fit], U[fit], sub=4) - S2[fit]) / scale).ravel()
    p0 = np.r_[1.0, DATASHEET["I"], 9.81]
    t0 = time.time()
    r = least_squares(resid, p0, x_scale="jac", max_nfev=60)
    print(f"fit done in {time.time() - t0:.0f} s")

    sheet = QIPModel(CART_MASS, DATASHEET["m"], LINK_LENGTHS, DATASHEET["c"], DATASHEET["I"], 9.81)
    ident = build(r.x)
    e0 = one_step_rms(sheet, S[test], U[test], S2[test])
    e1 = one_step_rms(ident, S[test], U[test], S2[test])
    res = dict(density=r.x[0], masses=(r.x[0] * LINK_LENGTHS).tolist(), inertias=r.x[1:5].tolist(), g=r.x[5])
    print(f"\nlink density rho = {res['density']:.4f} kg/m  -> masses {np.round(res['masses'], 4)} kg")
    print(f"inertias about CoG {np.round(res['inertias'], 6)} kg m^2, g = {res['g']:.4f} m/s^2")
    print(f"one-step RMS error of the angular rates [rad/s] (held-out transitions):")
    print(f"  datasheet model : {np.round(e0[3::2], 5)}")
    print(f"  identified model: {np.round(e1[3::2], 5)}")
    with open("sysid_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print("saved sysid_result.json (the values used are in qip/common.py)")


if __name__ == "__main__":
    main()
