# Swing-up and balance of a quadruple inverted pendulum

Model-based controller for the **Koh Young AI Competition 2024** task (ICCAS 2024): swing a
quadruple inverted pendulum on a cart up from hanging and balance it, with the cart force
limited to ±60 N and a 10 ms control period.

The controller swings the chain up along an optimised trajectory, tracked with time-varying
LQR, and hands over to a discrete-time LQR at the top. All designs use a plant model that I
identified from the competition's black-box simulator.

**Result on the competition simulator:** the state enters the scoring region at **3.64 s** and
stays there until the end of the 10 s run (6.37 s hold, 5 s required). At run time the
controller needs only numpy.

<!-- Optional: add a screenshot of the main.py animation, e.g. Figures/swingup.png -->

## Contents

- [Quick start](#quick-start)
- [Method](#method)
- [Results](#results)
- [Reproducing the design files](#reproducing-the-design-files)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Lessons from reinforcement learning attempts](#lessons-from-reinforcement-learning-attempts)
- [Future work: learning-based control](#future-work-learning-based-control)

## Quick start

The simulator binary is not part of this repository. Copy `simulator/base` from the
competition package into `simulator/`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
chmod +x simulator/base

# Terminal 1: the simulator
./simulator/base --ctrl_ms 10

# Terminal 2: evaluation run with the animation; writes transitions.txt
python main.py
python -m qip.metrics transitions.txt        # entry time and hold, raw and wrapped angles

# Or a headless run (same loop as main.py) with a summary
python -m qip.run
```

Only one program can talk to the simulator at a time. Each run starts with `reset`, so the
simulator does not need restarting between runs.

## Method

### 1. System identification

The problem statement lists the link parameters, but leaves link 4's mass, CoG and inertia
"not given". Using the listed values (with a guess for link 4), the one-step prediction of
the angular rates was off by up to 0.41 rad/s per 10 ms step. I therefore identified the
plant from data (`qip/sysid.py`).

1. Record 3,000 transitions (state, force, next state) from the simulator under random smooth forces.
2. Fit the cart-and-four-link Lagrangian model to them by least squares on the one-step
   prediction.

A free fit of all parameters showed that the simulator's links follow **mass = 4/3 × length**.
The final fit uses that structure, with one linear density, four inertias and g:

| | Link 1 | Link 2 | Link 3 | Link 4 |
|---|---|---|---|---|
| Mass, problem statement [kg] | 0.200 | 0.250 | 0.430 | not given |
| **Mass, identified [kg]** | **0.200** | **0.267** | **0.533** | **1.067** |
| Inertia about CoG, problem statement [kg m²] | 0.0015 | 0.0033 | 0.0229 | not given |
| **Inertia about CoG, identified [kg m²]** | **0.0015** | **0.0036** | **0.0284** | **0.2276** |

Other values: cart 1.3 kg, CoG at l/2, g = 9.806 m/s², no friction. With the identified model,
the one-step RMS error of the angular rates is 3×10⁻⁵ to 5×10⁻⁴ rad/s, about 1000 times
smaller than with the listed values. The simulator is deterministic.

### 2. Balancing LQR

The model is linearised at upright (`qip/design_lqr.py`). The open-loop unstable poles are at
29.6, 14.3, 6.3 and 3.3 rad/s. The fastest has a 34 ms time constant, so the design is done in
**discrete time**: zero-order-hold discretisation at 10 ms, then the discrete Riccati equation

$$
K = (R + B_d^\top P B_d)^{-1} B_d^\top P A_d, \qquad Q = \mathrm{diag}(10, 1, 10, 1, \dots), \quad R = 0.01 .
$$

The controller acts on the deviation $d$ from the nearest upright equilibrium (angles relative
to the nearest multiple of $2\pi$, cart position clipped to ±0.5 m): $u = -K d$.

**Where the LQR can catch.** I estimated the capture region by simulating the saturated LQR on
the nonlinear model from 3,000 near-upright states, and calibrated a level of the cost-to-go
$V = d^\top P d$:

- Below $\rho_{in} \approx 816$, 99% of states are caught. They reach the scoring region in a
  median 1.5 s.
- The region is thin: states below $\rho_{in}$ have a median largest link-angle error of 0.6°.
  Bending the links against each other excites the fast unstable mode, and ±60 N leaves little
  margin. Tilting the whole chain together is much cheaper.

### 3. Swing-up trajectory

The evaluation always starts from the same state, hanging at rest, so one optimised trajectory
suffices (`qip/trajopt.py`). It is computed by direct multiple shooting with CasADi and IPOPT:

$$
\min_{u_{0:N-1}} \sum_k \Big(\frac{u_k}{60}\Big)^2 \Delta t + 10^{-3} \sum_k \lVert \dot\theta_k \rVert^2 \Delta t
\quad \text{s.t.} \quad
\begin{cases}
x_{k+1} = F(x_k, u_k) & \text{(RK4 of the identified model)}\\
x_0 = \text{hanging at rest}, \quad |u_k| \le 60\ \text{N}, \quad |x_{\text{cart},k}| \le 2.5\ \text{m}\\
x_N:\ \theta_i = 0,\ \dot\theta_i = 0,\ \dot x = 0,\ |x| \le 0.3\ \text{m}
\end{cases}
$$

The horizon is 3.5 s. The problem is solved first with 20 ms intervals, then refined at 10 ms.
The solution uses at most 43 N and 0.99 m of cart travel. The terminal condition uses raw angles
(θᵢ = 0, not 2π), so the strict reading of the scoring region is met.

### 4. Trajectory tracking (time-varying LQR)

Played open loop, the planned forces fail completely on the simulator: the chain is chaotic,
and the small model error grows to tens of radians within 3.5 s. A backward Riccati recursion
along the trajectory, with the balancing LQR's $P$ as terminal cost, gives gains $K_k$ for

$$
u_k = u^*_k - K_k (x_k - x^*_k).
$$

With this feedback, the error at the top is 0.02°.

### 5. Switching

The tracker hands over to the balancing LQR when the trajectory ends, or earlier in its second
half once $V < \rho_{in}$ and $|x| \le 0.5$ m. In the nominal run the handover happens at 2.09 s.

## Results

Competition simulator, 10 s runs from the hanging start (`python -m qip.run`, with `--disturb`
for added force pulses):

| Scenario | LQR takes over | Region entered (raw angles) | Longest hold |
|---|---|---|---|
| Nominal | 2.09 s | 3.64 s | 6.37 s (to end of run) |
| 5 N for 0.1 s at t = 1.5 s (during swing-up) | 2.09 s | 3.66 s | 6.35 s |
| 10 N for 0.1 s at t = 2.5 s (during swing-up) | 2.09 s | 3.44 s | 6.57 s |
| 10–30 N for 0.1–0.2 s at t = 6 s (while balancing) | 2.09 s | recovers in about 1 s | the remaining time is shorter than the 5 s hold |

The identified model reproduces the nominal run on the simulator to the reported precision.

## Reproducing the design files

`lqr.npz` and `swingup.npz` are included. To regenerate them:

```bash
pip install scipy casadi
./simulator/base --ctrl_ms 10     # only needed for sysid
python -m qip.sysid               # identify the plant (~1 min); values are in qip/common.py
python -m qip.design_lqr          # balancing LQR and switching level -> lqr.npz (~1 min)
python -m qip.trajopt             # swing-up trajectory and tracking gains -> swingup.npz (~3-5 min)
python -m qip.trajopt --T 3.0     # try a shorter swing-up
```

## Repository layout

```
fill_the_code.py      Controller (submission file): trajectory tracking + LQR, numpy only
lqr.npz               Balancing LQR: K, P, switching level, cart clip
swingup.npz           Swing-up trajectory x*, u* and tracking gains K_k
main.py, render/      Evaluation script and renderer from the competition package
example.py, *.ipynb   Examples from the competition package
qip/
  common.py           Identified parameters, scoring region
  model.py            Identified plant model (vectorised), linearisation
  sysid.py            System identification from the simulator
  design_lqr.py       Balancing LQR, capture test, switching level
  trajopt.py          Swing-up trajectory optimisation and time-varying LQR gains
  sim_client.py       TCP client for simulator/base
  run.py              Headless 10 s runs, disturbance tests
  metrics.py          Challenge metric (also scores a transitions.txt)
simulator/            Not tracked; place the competition's simulator/base here
```

Submission files: `fill_the_code.py`, `main.py`, `lqr.npz`, `swingup.npz`.

## Limitations

- **The swing-up is planned for one start state.** The tracking feedback handles small
  deviations and pulses. A large disturbance during the swing-up, or a different start, would
  need a new plan.
- **There is no recovery after a fall.** If a disturbance knocks the chain out of the LQR's
  capture region, the LQR cannot swing it back up.
- **The capture region is thin** (sub-degree bending tolerance at ±60 N), so robustness to
  large pulses while balancing is limited by physics, not only by the controller.
- **The evaluation's "temporary uncertainties" are not specified.** I tested force pulses; other
  kinds, such as parameter changes or sensor noise, are untested.
- **Angle convention.** The trajectory ends at raw θ = 0. The LQR holds whichever upright
  equilibrium it catches, which matters only if the evaluator does not wrap angles.

## Lessons from reinforcement learning attempts

Before this solution, I trained SAC from scratch against the simulator. Two failures are worth
recording:

1. **A spinning policy.** With a reward for mean uprightness, the agent learned to rotate the
   chain continuously (about 10 turns per link in 10 s). This collects about 0.4 of the
   maximum reward per step without ever balancing.
2. **A do-nothing policy.** With anti-spin terminations, failure penalties and a squared
   uprightness term, partial swings earned almost nothing (about 0.005 per step for a 30°
   swing), while attempts risked −50. The agent learned to leave the chain hanging, and its
   exploration collapsed.

The underlying problem was exploration, not the algorithm: random exploration practically
never finds a four-link swing-up, and reward shaping only changed which shortcut the agent
found.

## Future work: learning-based control

The working controller can serve as a teacher, removing the exploration problem:

1. **Imitation learning (DART / DAgger).** Distil the tracker and LQR into a neural policy.
   Expert rollouts with injected noise and pulses (DART), or learner-driven rollouts labelled
   by the expert (DAgger), avoid the compounding errors of plain behaviour cloning.
2. **RL fine-tuning with demonstrations.** Initialise SAC from the imitation policy and keep
   expert transitions in the replay buffer (RLPD-style), on the identified model with randomised
   disturbances and parameters. The goal is robustness and recovery from states the plan never
   visits.
3. **Residual RL.** Keep the model-based controller and learn a bounded correction,
   u = u_expert + π(x).

These variants would be compared on nominal entry time, success rate under random disturbance
profiles, ±10% parameter mismatch, and recovery from random starts.

## Acknowledgements and references

The QIP simulator in this work is provided by Koh Young Technology, Inc.

- T. Glück, A. Eder, A. Kugi, "Swing-up control of a triple pendulum on a cart with experimental validation", *Automatica* 49(3), 2013.
- J. Baek, C. Lee, Y. S. Lee, S. Jeon, S. Han, "Reinforcement learning to achieve real-time control of triple inverted pendulum", *Engineering Applications of Artificial Intelligence* 128, 2024.
- C. C. Chung, J. Hauser, "Nonlinear control of a swinging pendulum", *Automatica* 31(6), 1995.
- J. A. E. Andersson et al., "CasADi: a software framework for nonlinear optimization and optimal control", *Mathematical Programming Computation* 11, 2019.
