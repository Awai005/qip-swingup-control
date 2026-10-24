# Quadruple inverted pendulum: swing-up and balance

This repository contains a controller that swings a quadruple inverted pendulum up from
hanging and balances it at the top. The system is a cart with four links in series, driven
by a horizontal force limited to ±60 N, with a 10 ms control period.

The target is the region |θᵢ| ≤ 0.05 rad, |θ̇ᵢ| ≤ 0.05 rad/s and |x| ≤ 0.5 m, held for at
least 5 s. In simulation the pendulum enters that region at 3.64 s and stays there for the
rest of the 10 s run. The controller needs only numpy at run time.

## Running it

The simulator binary is not included in this repository. Place it at `simulator/base`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
chmod +x simulator/base

./simulator/base --ctrl_ms 10          # terminal 1
python main.py                         # terminal 2: animation, writes transitions.txt
python -m qip.metrics transitions.txt  # entry time and hold
```

`python -m qip.run` does the same 10 s run without the animation. Adding `--disturb 6.0,0.1,20`
applies a 20 N pulse for 0.1 s at t = 6 s.

## How it works

1. **System identification.** The published parameter table leaves link 4 out, and the
   listed values for links 2 and 3 do not match the simulator. I recorded 3,000
   transitions from the simulator and fitted the cart and four-link model to them by least
   squares. The simulator's link masses turn out to be 4/3 of their lengths (0.200, 0.267,
   0.533 and 1.067 kg), with inertias about the centre of mass of 0.0015, 0.0036, 0.0284 and
   0.2276 kg m². The identified model predicts one control step with an angular-rate error
   below 5×10⁻⁴ rad/s; the listed parameters give errors of up to 0.4 rad/s.

2. **Balancing LQR.** I linearised the identified model at upright and designed a
   discrete-time LQR at 10 ms. The fastest unstable pole is at 29.6 rad/s, so a continuous
   design would be too coarse at this sample rate. Simulating the saturated LQR from 3,000
   near-upright states shows how narrow its capture region is: the states it reliably catches
   have link-angle errors of around 0.6°.

3. **Swing-up trajectory.** The start state is always the same, so I computed one swing-up
   trajectory offline by direct multiple shooting (CasADi and IPOPT). It brings the chain from
   hanging to upright in 3.5 s, using at most 43 N and 0.99 m of cart travel, and ends at raw
   angles of zero.

4. **Tracking and handover.** On its own, the planned force sequence fails: the chain is
   chaotic and small model errors grow to tens of radians. A time-varying LQR along the
   trajectory corrects this:

   $$
   u_k = \bar{u}_k - K_k (x_k - \bar{x}_k)
   $$

   Here $\bar{x}_k$ and $\bar{u}_k$ are the planned states and forces. The gains $K_k$ come
   from a backward Riccati recursion that ends in the balancing LQR's cost matrix. The
   balancing LQR takes over once the state is inside its capture region; in the nominal run
   that happens at 2.09 s.

## Results

All runs are 10 s in the simulator, from the hanging start.

| Run | Enters target region | Longest hold |
|---|---|---|
| Nominal | 3.64 s | 6.37 s |
| 5 N for 0.1 s at 1.5 s | 3.66 s | 6.35 s |
| 10 N for 0.1 s at 2.5 s | 3.44 s | 6.57 s |
| 10 to 30 N for 0.1 to 0.2 s at 6 s | recovers within about 1 s | too little time left for a 5 s hold |

## Lessons learned

My earlier attempt, [qip_system](https://github.com/Awai005/qip_system), was my entry for
the Koh Young AI Competition. It combined tabular Q-learning for the swing-up with an LQR for
balancing. It did not swing the pendulum up.
Looking back, most of the problems were on the modelling side:

- The Q-table's state bins were too coarse. With bins spanning ±10, every angle in ±π fell
  into one of two bins, so the agent could only see the sign of each variable.
- The LQR was designed on the wrong linear model. The matrices described the hanging-down
  equilibrium, in a different state ordering from the simulator's, so the gain could not
  stabilise the upright position.
- The listed physical parameters are not the simulator's. Identifying the plant from data
  came first; every later design depends on it.
- The balancing controller has to be designed in discrete time at the actual control period.
- A planned trajectory is useless without feedback along it, because the chain is chaotic.
- The LQR's capture region is very thin with a 60 N limit, so the handover has to be precise.
  This is why a trajectory planned and tracked on an accurate model works where trial and error
  did not.

## Limitations and future work

The swing-up is planned for one start state. The tracking feedback absorbs small pulses, but
there is no recovery from a fall or from a very different start. These are where a learned
policy could help. Using this controller as a teacher would avoid the exploration problem,
either through imitation learning with DART or DAgger, or by learning a residual correction on
top of the controller.

## Reproducing the design files

`lqr.npz` and `swingup.npz` are included. To regenerate them:

```bash
pip install scipy casadi
python -m qip.sysid        # needs the simulator running; values go in qip/common.py
python -m qip.design_lqr   # lqr.npz
python -m qip.trajopt      # swingup.npz, a few minutes
```

## Files

```
fill_the_code.py   controller (trajectory tracking + LQR), numpy only
lqr.npz            balancing LQR gain, cost matrix and switching level
swingup.npz        planned trajectory, forces and tracking gains
main.py, render/   run script and renderer provided with the simulator
qip/               identification, LQR design, trajectory optimisation, metrics
simulator/         not tracked; place simulator/base here
```

## Acknowledgement

The QIP simulator in this work is provided by Koh Young Technology, Inc.

T. Glück, A. Eder and A. Kugi, "Swing-up control of a triple pendulum on a cart with
experimental validation", Automatica 49(3), 2013.