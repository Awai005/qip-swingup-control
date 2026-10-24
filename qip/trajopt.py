"""
Swing-up trajectory by direct multiple shooting (CasADi + IPOPT) on the
identified model, plus time-varying LQR (TVLQR) gains to track it.

The evaluation always starts from the same state (hanging, at rest), so one
optimised trajectory is enough. Open loop it would not survive: the chain is
chaotic and the model is not perfect. The TVLQR feedback
        u_k = u*_k - K_k (x_k - x*_k)        (clipped to +-60 N)
keeps the real simulator on the trajectory, and the balancing LQR takes over
at the top. The TVLQR's terminal cost is the balancing LQR's P, so the two
controllers join without a seam.

    pip install casadi                      (training environment only)
    python -m qip.trajopt                    # T = 3.5 s, writes swingup.npz (~2 min)
    python -m qip.trajopt --T 3.0            # faster swing-up, if a solution exists

Problem
    minimise   sum (u_k/60)^2 dt + 1e-3 * sum |theta_dot_k|^2 dt
    subject to x_{k+1} = F(x_k, u_k)        RK4 of the identified model, 4 substeps
               x_0 = hanging at rest
               |u_k| <= 60 N, |x_k| <= x_max
               x_N: all link angles = 0 (raw, so the strict metric is met),
                    all velocities 0, |cart| <= 0.3 m
Solved first with 20 ms intervals (robust), then refined at 10 ms (the control period).
"""
import argparse
import os
import time
import numpy as np
import casadi as ca
from qip.model import QIPModel
from qip.common import HANGING

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U_MAX = 60.0


def step_function(model, dt, sub=4):
    s = ca.SX.sym("s", 10); u = ca.SX.sym("u")

    def accel(s, F):
        th, thd = s[2::2], s[3::2]
        M = ca.SX.zeros(5, 5); M[0, 0] = model.Mt
        for j in range(4):
            M[0, j + 1] = model.a[j] * ca.cos(th[j]); M[j + 1, 0] = M[0, j + 1]
            for k in range(4):
                M[j + 1, k + 1] = model.Jd[j] if j == k else model.B[j, k] * ca.cos(th[j] - th[k])
        rhs = ca.SX.zeros(5)
        rhs[0] = F + sum(model.a[j] * ca.sin(th[j]) * thd[j] ** 2 for j in range(4))
        for j in range(4):
            rhs[j + 1] = model.g * model.a[j] * ca.sin(th[j]) - sum(
                model.B[j, k] * ca.sin(th[j] - th[k]) * thd[k] ** 2 for k in range(4) if k != j)
        return ca.solve(M, rhs)

    def deriv(s, u):
        a = accel(s, u); d = ca.SX.zeros(10); d[0::2] = s[1::2]; d[1::2] = a
        return d

    h = dt / sub; x = s
    for _ in range(sub):
        k1 = deriv(x, u); k2 = deriv(x + h / 2 * k1, u); k3 = deriv(x + h / 2 * k2, u); k4 = deriv(x + h * k3, u)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("F", [s, u], [x])


def solve(model, T, dt, x_max, guess=None, seed=0, verbose=False):
    N = int(round(T / dt)); F = step_function(model, dt)
    opti = ca.Opti()
    X = opti.variable(10, N + 1); U = opti.variable(1, N)
    opti.subject_to(X[:, 0] == HANGING)
    opti.subject_to(X[:, 1:] == F.map(N)(X[:, :-1], U))
    opti.subject_to(opti.bounded(-U_MAX, U, U_MAX))
    opti.subject_to(opti.bounded(-x_max, X[0, :], x_max))
    opti.subject_to(X[2:, N] == 0); opti.subject_to(X[1, N] == 0)
    opti.subject_to(opti.bounded(-0.3, X[0, N], 0.3))
    opti.minimize(ca.sumsqr(U) / U_MAX ** 2 * dt + 1e-3 * ca.sumsqr(X[3::2, :]) * dt)
    if guess is None:
        rng = np.random.default_rng(seed); tt = np.linspace(0, 1, N + 1); g = np.zeros((10, N + 1))
        for i in range(4):
            g[2 + 2 * i] = np.pi * (1 - tt) + rng.normal(0, 0.3) * np.sin(np.pi * tt)
        opti.set_initial(X, g); opti.set_initial(U, rng.normal(0, 5, (1, N)))
    else:
        Xg, Ug = guess
        tg = np.arange(len(Xg)) * (T / (len(Xg) - 1)); t = np.arange(N + 1) * dt
        opti.set_initial(X, np.array([np.interp(t, tg, Xg[:, i]) for i in range(10)]))
        opti.set_initial(U, np.interp(t[:-1], tg[:-1], Ug)[None, :])
    opti.solver("ipopt", {"print_time": False},
                {"max_iter": 3000, "tol": 1e-7, "acceptable_tol": 1e-5,
                 "print_level": 5 if verbose else 0, "print_frequency_iter": 50})
    try:
        sol = opti.solve()
        return True, sol.value(X).T, np.atleast_1d(sol.value(U))
    except RuntimeError:
        return False, opti.debug.value(X).T, np.atleast_1d(opti.debug.value(U))


def tvlqr(model, Xs, Us, P_end, Q, R, eps=1e-6):
    """Backward Riccati recursion along the trajectory (finite-difference linearisation)."""
    N = len(Us); K = np.zeros((N, 10)); P = P_end.copy()
    for t in reversed(range(N)):
        Xb = np.repeat(Xs[t][None], 22, 0); Ub = np.full(22, Us[t])
        for k in range(10):
            Xb[2 * k, k] += eps; Xb[2 * k + 1, k] -= eps
        Ub[20] += eps; Ub[21] -= eps
        Y = model.step(Xb, Ub)
        A = np.stack([(Y[2 * k] - Y[2 * k + 1]) / (2 * eps) for k in range(10)], 1)
        B = ((Y[20] - Y[21]) / (2 * eps))[:, None]
        K[t] = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)[0]
        P = Q + A.T @ P @ (A - B @ K[t][None]); P = 0.5 * (P + P.T)
    return K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, default=3.5, help="swing-up duration [s]")
    p.add_argument("--x_max", type=float, default=2.5, help="cart limit during the swing-up [m]")
    p.add_argument("--seeds", type=int, default=4, help="initial guesses to try for the 20 ms solve")
    p.add_argument("--out", default=os.path.join(_ROOT, "swingup.npz"))
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    model = QIPModel()
    lqr = dict(np.load(os.path.join(_ROOT, "lqr.npz")))

    coarse = None
    for seed in range(a.seeds):
        t0 = time.time()
        ok, X, U = solve(model, a.T, 0.02, a.x_max, seed=seed, verbose=a.verbose)
        print(f"20 ms solve, guess {seed}: {'ok' if ok else 'failed'} ({time.time() - t0:.0f} s)")
        if ok:
            coarse = (X, U); break
    if coarse is None:
        raise SystemExit("no swing-up found; try a longer --T")
    t0 = time.time()
    ok, X, U = solve(model, a.T, 0.01, a.x_max, guess=coarse, verbose=a.verbose)
    print(f"10 ms refinement: {'ok' if ok else 'failed'} ({time.time() - t0:.0f} s)")
    if not ok:
        raise SystemExit("refinement failed")
    K = tvlqr(model, X, U, lqr["P"], lqr["Q"], lqr["R"])
    np.savez(a.out, X=X, U=U, K=K)
    print(f"saved {a.out}: {len(U)} steps ({a.T} s), max |u| {np.abs(U).max():.1f} N, "
          f"max |x| {np.abs(X[:, 0]).max():.2f} m")


if __name__ == "__main__":
    main()
