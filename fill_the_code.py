"""
Controller for the Koh Young AI Competition 2024 QIP task (submission file, numpy only).

    swing-up : trajectory x*_k, u*_k optimised offline from the hanging start
               (swingup.npz), tracked with time-varying LQR:
                   u_k = u*_k - K_k (x_k - x*_k)
    balance  : discrete-time LQR on the identified plant (lqr.npz):
                   u_k = -K d_k
    switch   : to the LQR when the trajectory ends, or earlier (second half of
               the trajectory) once the state is inside the LQR funnel
                   V = d^T P d < rho_in  and  |x| <= x_switch

d is the deviation from upright with each angle taken relative to the nearest
multiple of 2*pi and the cart position clipped to +-x_cap, so the LQR returns
the cart to the centre at a bounded rate.

Files needed next to this one: lqr.npz, swingup.npz.
"""
import os
import numpy as np

U_MAX = 60.0            # force bound [N]
CTRL_MS = 10            # control period both controllers are designed for

_HERE = os.path.dirname(os.path.abspath(__file__))


def lqr_delta(state, x_cap):
    """Deviation from the nearest upright equilibrium, cart position clipped.
    Works on one state (10,) or a batch (N, 10)."""
    S = np.array(np.atleast_2d(state), dtype=np.float64)
    S[:, 2::2] -= 2.0 * np.pi * np.round(S[:, 2::2] / (2.0 * np.pi))
    S[:, 0] = np.clip(S[:, 0], -x_cap, x_cap)
    return S if np.ndim(state) == 2 else S[0]


class Controller:
    def __init__(self, lqr=os.path.join(_HERE, "lqr.npz"), swingup=os.path.join(_HERE, "swingup.npz")):
        """lqr / swingup: file paths or dicts of arrays."""
        q = np.load(lqr) if isinstance(lqr, str) else lqr
        self.K = np.asarray(q["K"], float).reshape(-1)
        self.P = np.asarray(q["P"], float)
        self.rho_in = float(q["rho_in"])
        self.x_cap, self.x_switch = float(q["x_cap"]), float(q["x_switch"])

        t = np.load(swingup) if isinstance(swingup, str) else swingup
        self.Xs, self.Us, self.Ks = (np.asarray(t[k], float) for k in ("X", "U", "K"))

        self.u_min, self.u_max = -U_MAX, U_MAX
        self.mode = "swingup"
        self.switch_step = None       # control step at which the LQR took over
        self._k = 0

    def compute_force(self, state: list) -> float:
        x = np.asarray(state, dtype=np.float64)
        d = lqr_delta(x, self.x_cap)

        if self.mode == "swingup":
            in_funnel = float(d @ self.P @ d) < self.rho_in and abs(x[0]) <= self.x_switch
            if self._k >= len(self.Us) or (in_funnel and self._k > len(self.Us) // 2):
                self.mode, self.switch_step = "lqr", self._k

        if self.mode == "swingup":
            k = self._k
            u = self.Us[k] - float(self.Ks[k] @ (x - self.Xs[k]))
        else:
            u = -float(self.K @ d)

        self._k += 1
        return float(np.clip(u, self.u_min, self.u_max))
