"""
Design the balancing LQR on the identified model and estimate its region of
attraction (RoA) with the +-60 N limit. Writes lqr.npz into the QIP folder.

    python -m qip.design_lqr                # default design, ~2-4 min
    python -m qip.design_lqr --R 0.1 --q_theta 100

Steps
  1. Linearise the identified model at upright -> A, B (continuous time).
  2. Zero-order-hold discretisation at 10 ms -> Ad, Bd (the controller runs at
     10 ms, so the design is done in discrete time, not continuous LQR).
  3. Discrete LQR: K = (R + Bd'P Bd)^-1 Bd'P Ad, P from the discrete Riccati equation.
  4. Capture test: simulate the saturated LQR on the NONLINEAR model from a
     broad mix of near-upright states and record which settle into the
     scoring region.
  5. Switching levels of V = d'P d, calibrated on that test:
       rho_in  = largest level with >= 99% capture below it   (RL -> LQR)
       rho_out = largest level with >= 90% capture below it   (LQR -> RL)
     Switching in also requires |x| <= x_switch.

  The basin is thin in "bending" directions (links tilted against each other):
  with +-60 N the fast unstable mode (~30 rad/s) leaves little margin. That is
  why the levels are calibrated by simulation rather than taken from a single
  ellipsoid bound.
"""
import argparse
import os
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from fill_the_code import lqr_delta, U_MAX, CTRL_MS
from qip.model import QIPModel
from qip.common import in_region

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def zoh(A, B, dt):
    n = A.shape[0]
    E = np.zeros((n + 1, n + 1)); E[:n, :n] = A; E[:n, n:] = B
    F = expm(E * dt)
    return F[:n, :n], F[:n, n:]


def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def closed_loop(model, K, X, x_cap, steps):
    """Saturated LQR on the nonlinear model. Returns success mask and entry times."""
    first = np.full(len(X), np.inf); alive = np.ones(len(X), bool)
    tail = np.ones(len(X), bool)
    for t in range(steps):
        d = lqr_delta(X, x_cap)
        u = np.clip(-(d @ K.T)[:, 0], -U_MAX, U_MAX)
        X = model.step(X, u)
        bad = ~np.isfinite(X).all(1) | (np.abs(lqr_delta(X, 1e9)[:, 2::2]) > np.pi / 2).any(1)
        alive &= ~bad; X[bad] = 0.0
        inr = in_region(X, wrap_angles=True)
        first[np.isinf(first) & inr & alive] = (t + 1) * CTRL_MS / 1000
        if t >= steps - 100:
            tail &= inr
    return alive & tail, first


def sample_near_upright(n, rng, x_switch):
    """Broad mix of near-upright states: per-sample scale from 0.003 to 0.08 rad,
    independent link angles, angular rates 1-8x the angle scale, cart within x_switch."""
    X = np.zeros((n, 10))
    sc = np.exp(rng.uniform(np.log(0.003), np.log(0.08), n))
    X[:, 2::2] = rng.normal(0, 1, (n, 4)) * sc[:, None]
    X[:, 3::2] = rng.normal(0, 1, (n, 4)) * sc[:, None] * rng.uniform(1, 8, n)[:, None]
    X[:, 0] = rng.uniform(-x_switch, x_switch, n)
    X[:, 1] = rng.normal(0, 0.3, n)
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q_x", type=float, default=10.0)
    p.add_argument("--q_xdot", type=float, default=1.0)
    p.add_argument("--q_theta", type=float, default=10.0)
    p.add_argument("--q_thdot", type=float, default=1.0)
    p.add_argument("--R", type=float, default=0.01)
    p.add_argument("--x_cap", type=float, default=0.5, help="cart position clip inside the LQR [m]")
    p.add_argument("--x_switch", type=float, default=0.5, help="|x| limit for switching to the LQR [m]")
    p.add_argument("--p_in", type=float, default=0.99, help="capture rate required below rho_in")
    p.add_argument("--p_out", type=float, default=0.90, help="capture rate required below rho_out")
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--settle_s", type=float, default=5.0, help="time allowed to settle in the test")
    p.add_argument("--out", default=os.path.join(_ROOT, "lqr.npz"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    model = QIPModel()
    A, B = model.linearize()
    dt = CTRL_MS / 1000
    Ad, Bd = zoh(A, B, dt)
    Q = np.diag([a.q_x, a.q_xdot] + [a.q_theta, a.q_thdot] * 4)
    R = np.array([[a.R]])
    K, P = dlqr(Ad, Bd, Q, R)
    ol = np.sort(np.linalg.eigvals(A).real)[::-1]
    cl = np.abs(np.linalg.eigvals(Ad - Bd @ K)).max()
    print(f"open-loop unstable poles [rad/s]: {np.round(ol[:4], 2)}")
    print(f"closed-loop spectral radius (discrete): {cl:.4f}")
    print(f"K = {np.round(K.ravel(), 2)}")

    # Capture test on the nonlinear model with saturation, then calibrate the
    # switching levels of V = d'P d against the empirical capture rate.
    X = sample_near_upright(a.samples, rng, a.x_switch)
    print(f"simulating {a.samples} near-upright starts for {a.settle_s} s ...")
    ok, first = closed_loop(model, K, X.copy(), a.x_cap, int(a.settle_s / dt))
    d = lqr_delta(X, a.x_cap)
    V = np.einsum("ni,ij,nj->n", d, P, d)
    order = np.argsort(V)
    cum = np.cumsum(ok[order]) / np.arange(1, len(V) + 1)

    def level(p_req):
        good = np.flatnonzero(cum >= p_req)
        return float(V[order][good.max()]) if len(good) else float(V.min())

    rho_in, rho_out = level(a.p_in), level(a.p_out)
    rho_out = max(rho_out, 1.5 * rho_in)
    print(f"\nrho_in  = {rho_in:9.1f}  (capture {100 * ok[V < rho_in].mean():.1f}% of {np.sum(V < rho_in)} samples below it)")
    print(f"rho_out = {rho_out:9.1f}  (capture {100 * ok[V < rho_out].mean():.1f}% below it)")
    inside = V < rho_in
    dth = np.abs(d[inside][:, 2::2]).max(1); dthd = np.abs(d[inside][:, 3::2]).max(1)
    print(f"states below rho_in: largest link angle error median {np.degrees(np.median(dth)):.2f} deg, "
          f"max {np.degrees(dth.max()):.2f} deg; largest link rate median {np.median(dthd):.2f}, max {dthd.max():.2f} rad/s")
    f = first[inside & np.isfinite(first)]
    print(f"time to scoring region from below rho_in: median {np.median(f):.2f} s, 95th pct {np.percentile(f, 95):.2f} s")

    np.savez(a.out, K=K, P=P, rho_in=rho_in, rho_out=rho_out, x_cap=a.x_cap, x_switch=a.x_switch,
             Q=Q, R=R, Ad=Ad, Bd=Bd)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
