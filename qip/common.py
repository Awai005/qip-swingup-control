"""
Constants and helpers shared by the tools in this package.

Plant parameters: identified from the black-box simulator with qip/sysid.py.
The simulator's links follow mass = 4/3 x length; the inertias below (about each
link's CoG) and g are least-squares fits. These differ from the datasheet in the
problem statement for links 2-4 (link 4 is "not given" there).
"""
import numpy as np

CART_MASS = 1.3
LINK_LENGTHS = np.array([0.15, 0.20, 0.40, 0.80])
LINK_MASSES = 4.0 / 3.0 * LINK_LENGTHS                       # 0.200, 0.267, 0.533, 1.067 kg
LINK_COG = LINK_LENGTHS / 2.0
LINK_MOI = np.array([0.0015, 0.003599, 0.028402, 0.227604])  # kg m^2, about the CoG
GRAVITY = 9.8058

# Datasheet values from the problem statement (link 4 not given; 0.81 kg / 0.1728 assumed)
DATASHEET = dict(m=[0.20, 0.25, 0.43, 0.81], c=[0.075, 0.10, 0.20, 0.40], I=[0.0015, 0.0033, 0.0229, 0.1728])

HANGING = np.array([0.0, 0.0] + [np.pi, 0.0] * 4)           # evaluation start state
DT = 0.01                                                   # control period [s]
U_MAX = 60.0

# Scoring region (problem statement, section 5), held for more than 5 s
REGION_THETA, REGION_THETA_DOT, REGION_X, HOLD_SEC = 0.05, 0.05, 0.5, 5.0


def wrap(angle):
    return np.remainder(np.asarray(angle) + np.pi, 2.0 * np.pi) - np.pi


def in_region(states, wrap_angles=False):
    """states: (10,) or (N, 10). Returns bool or bool array."""
    S = np.atleast_2d(states)
    th = wrap(S[:, 2::2]) if wrap_angles else S[:, 2::2]
    ok = (np.abs(S[:, 0]) <= REGION_X) & (np.abs(th) <= REGION_THETA).all(1) \
        & (np.abs(S[:, 3::2]) <= REGION_THETA_DOT).all(1)
    return ok if np.ndim(states) == 2 else bool(ok[0])
