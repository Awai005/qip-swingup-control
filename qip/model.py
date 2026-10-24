"""
Identified plant model (cart + 4 links, absolute angles from upright, force on
the cart), vectorised over a batch of states. Used for the LQR design and as a
fast replica of the black-box simulator for training.
"""
import numpy as np
from qip.common import CART_MASS, LINK_LENGTHS, LINK_MASSES, LINK_COG, LINK_MOI, GRAVITY


class QIPModel:
    def __init__(self, Mc=CART_MASS, m=LINK_MASSES, l=LINK_LENGTHS, c=LINK_COG, I=LINK_MOI, g=GRAVITY):
        m, l, c, I = map(lambda v: np.asarray(v, float), (m, l, c, I))
        above = np.array([m[i + 1:].sum() for i in range(4)])
        self.Mt = Mc + m.sum()
        self.a = m * c + l * above                        # gravity / coupling coefficients
        self.Jd = I + m * c ** 2 + l ** 2 * above         # diagonal inertia terms
        self.B = np.zeros((4, 4))
        for j in range(4):
            for k in range(4):
                if j != k:
                    lo, hi = min(j, k), max(j, k)
                    self.B[j, k] = l[lo] * self.a[hi]
        self.g = g

    def accel(self, S, F):
        """S: (N, 10), F: (N,) -> generalized accelerations (N, 5)."""
        th, thd, N = S[:, 2::2], S[:, 3::2], len(S)
        M = np.empty((N, 5, 5))
        M[:, 0, 0] = self.Mt
        M[:, 0, 1:] = self.a * np.cos(th)
        M[:, 1:, 0] = M[:, 0, 1:]
        D = th[:, :, None] - th[:, None, :]
        M[:, 1:, 1:] = self.B * np.cos(D)
        i = np.arange(4)
        M[:, 1 + i, 1 + i] = self.Jd
        rhs = np.empty((N, 5))
        rhs[:, 0] = F + np.sum(self.a * np.sin(th) * thd ** 2, axis=1)
        rhs[:, 1:] = self.g * self.a * np.sin(th) - np.einsum("njk,nk->nj", self.B * np.sin(D), thd ** 2)
        return np.linalg.solve(M, rhs[..., None])[..., 0]

    def deriv(self, S, F):
        a = self.accel(S, F)
        dS = np.empty_like(S)
        dS[:, 0::2], dS[:, 1::2] = S[:, 1::2], a
        return dS

    def step(self, S, F, dt=0.01, sub=4):
        """RK4 with a zero-order-hold force over one control period."""
        S = np.atleast_2d(np.asarray(S, float)); F = np.broadcast_to(np.asarray(F, float), (len(S),))
        h = dt / sub
        for _ in range(sub):
            k1 = self.deriv(S, F); k2 = self.deriv(S + h / 2 * k1, F)
            k3 = self.deriv(S + h / 2 * k2, F); k4 = self.deriv(S + h * k3, F)
            S = S + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return S

    def linearize(self, eps=1e-6):
        """Continuous-time A, B at the upright equilibrium (state order as the simulator)."""
        x0 = np.zeros((1, 10)); A = np.zeros((10, 10))
        for k in range(10):
            d = np.zeros((1, 10)); d[0, k] = eps
            A[:, k] = (self.deriv(x0 + d, np.zeros(1)) - self.deriv(x0 - d, np.zeros(1)))[0] / (2 * eps)
        B = ((self.deriv(x0, np.array([eps])) - self.deriv(x0, np.array([-eps])))[0] / (2 * eps))[:, None]
        return A, B
