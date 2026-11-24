# Gen2 trajectory-spring stability analysis

Notes from 2026-11-23, after the first pick-and-place run went unstable.
Branch `trajectory-spring`. Everything here can be reproduced with the
commands in [Reproducing](#reproducing).

## Summary

- **Cause of the incident:** the run was launched without a config, so the
  trajectory spring ran at the code's defaults (300 N/m, 30 N·s/m, 40 N cap)
  instead of the 60 N/m / 5 N·s/m used in every working test. It oscillated
  at 3.7 Hz until the arm's power was cut.
- **Where our model was wrong:** it had no loop delay. The real loop lags
  about **55–65 ms** (three independent estimates agree). With that delay
  the model reproduces both hardware results; without it, it calls
  everything stable.
- **Safe range (model, both recorded poses):** stiffness **≤ 100 N/m with
  damping 2–10 N·s/m**. 150–200 N/m is marginal (damping 5–10 only); 300 N/m
  is unstable at any damping. The working 60/5 is well inside, but even it
  has only 4–6 dB gain margin in the arm's lightest directions.
- **With delay, more damping can make it worse.** Damping adds phase lead
  but raises the crossover frequency into where the delay's lag is larger.
- **Joint 7 has a discrete-time damping limit** of about 0.19 N·m·s/rad at
  100 Hz; `null_damping` (0.5) is above it and `rot_damping`'s old default
  (0.2) is on it. Friction hides this on the arm.
- **Done:** `gen2_spring.launch.py` now defaults to the 60/5 gains.
  `tools/gen2_stability_model.py` checks gains before they go on the arm.

## 1. The incident

Bag `~/gen2_bags/rosbag2_2026_11_23-18_58_34` (replay:
`rosbag2_2026_11_23-18_58_34_replay.html` next to it).

- Spring node config: `gen2_springs_none.yaml` (the launch default), so no
  `trajectory_spring.*` parameters → code defaults: stiffness 300 N/m,
  damping 30 N·s/m, rot 10 N·m/rad / 0.2, max force 40 N, max moment 5 N·m.
- `pick_place_test.py` step 1 (anchor down toward 8 cm above the pick spot,
  position only) started at 178.5 s. The gripper tracked within 3–7 mm for
  9.5 s.
- At **188.3 s** it went unstable: spring force swinging ±30–40 N
  (e.g. (+14, −23), (−8, +8), (−24, +28), (+25, −31) N), gripper 2–5 cm
  around the anchor. The collision field's first CAUTION (link 5 close)
  fired at the same moment — a likely kick.
- Power cut by the user at about 193 s (SDK error 1015, gripper silent).
- Signature (100 Hz joint states, 188–193 s): **3.7 Hz**, carried by
  **joints 1, 3 and 5**; commanded torque ±14 N·m on joint 1. Before 188 s
  the arm already moved at 3–5 Hz, about 20× smaller: barely stable.
- The relay adds 0.5 ms (spring torque → command), so the delay is not in
  our ROS plumbing.

## 2. Measuring the loop delay

During the oscillation, the measured joint torque (sensors, in
`joint_states.effort`) follows the commanded torque by:

| joint | lag | correlation | command amplitude | measured amplitude |
|---|---|---|---|---|
| 1 | +64 ms | +0.44 | 3.40 N·m | 3.81 N·m |
| 3 | +56 ms | +0.49 | 1.80 N·m | 1.10 N·m |
| 5 | +52 ms | +0.51 | 1.50 N·m | 0.75 N·m |

(`tools/bag_torque_lag.py`, 188.5–192.8 s.) The correlations are moderate
(the sensed torque also contains inertia and friction), but the three joints
agree. Where the delay comes from — bridge loop, USB, the firmware's torque
loop — is not known yet.

## 3. The model, and where it was wrong

Model (`tools/gen2_stability_model.py`): Gen2 + Robotiq URDF, rigid-body
dynamics (pinocchio `aba`, 0.2 ms steps), exact gravity compensation,
viscous joint friction 0.3 N·m·s/rad, Coulomb friction 0.3 N·m (tanh),
springcontroller's own `TrackingPoseSpring` computed at 100 Hz and held for
each 10 ms tick, position only. The arm starts 2 cm from a fixed anchor;
"unstable" = still swinging more than 1 cm after 3 s.

### 3.1 No delay: everything looks stable

At both recorded poses, with posture damping 0.1, both 60/5 and 300/30
settle. That contradicts the arm.

(A first version also blew up for every gain set — joint 7, see §6 — which
turned out to be a real but separate effect.)

### 3.2 Delay sweep (onset pose)

| delay | Coulomb | velocity noise | 60 N/m / 5 | 300 N/m / 30 |
|---|---|---|---|---|
| 0 ms | 0 / 0.3 | 0 / 0.02 rad/s | stable | stable |
| 10 ms | 0 / 0.3 | 0 | stable | stable |
| 20 ms | 0 / 0.3 | 0 / 0.02 | stable | **unstable, 10–11 Hz** (one noisy case stable) |
| 30 ms | 0 / 0.3 | 0 | stable | **unstable, 7.3–7.7 Hz** |
| 40 ms | 0 / 0.3 | 0 | stable | **unstable, 5.7–6.0 Hz** |

Delay alone explains it; friction and noise hardly matter. The oscillation
frequency falls as the delay grows, toward the observed 3.7 Hz at about
55–65 ms — consistent with §2.

### 3.3 Gain map at 60 ms delay

Both recorded poses (onset: low, reaching toward the table; goal start:
higher) give nearly the same pattern:

| stiffness \ damping (N·s/m) | 2 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| 30 N/m | ok | ok | ok | ok | ok |
| 60 N/m | ok | ok | ok | ok | ok |
| 100 N/m | ok | ok | ok | ok | ok |
| 150 N/m | ok | ok | ok | unstable at the goal-start pose (ok at onset) | unstable |
| 200 N/m | unstable | ok | ok | unstable | unstable |
| 300 N/m | unstable | unstable | unstable | unstable | unstable |

## 4. Classical (Bode) view

Linearized about a pose, along one principal direction of the gripper's
motion, the loop is

    L(s) = (k + b·s) · e^(−s·τ) / (m·s² + c·s)

- `k + b·s`: the spring/damper (a PD controller),
- `m`: effective mass at the gripper in that direction,
  from Λ = (J M⁻¹ Jᵀ)⁻¹,
- `c`: joint viscous friction seen at the gripper as damping,
  C = Λ J M⁻¹ (0.3·I) M⁻¹ Jᵀ Λ,
- `τ`: 60 ms measured delay + 5 ms (holding each command for a 10 ms tick).

Phase: the plant gives −90° − atan(m·ω/c) (−180° for a pure mass), the
damping adds atan(b·ω/k), the delay subtracts ω·τ. Unstable when the phase
at gain crossover is below −180°.

**Effective masses** (onset pose; start pose within 0.1 kg):
0.28, 0.33 and 1.88 kg; friction-as-damping 7.1, 4.0 and 36.8 N·s/m. The two
light directions are mostly wrist motion (joints 5–7).

**Phase-crossover frequency** (where a pure-mass loop reaches −180°; a
delay-unstable loop oscillates near it) for 300/30:

| τ | 20 ms | 30 ms | 40 ms | **50 ms** | 60 ms | 65 ms |
|---|---|---|---|---|---|---|
| frequency | 11.4 Hz | 7.2 Hz | 5.0 Hz | **3.7 Hz** | 2.8 Hz | 2.4 Hz |

The observed 3.7 Hz corresponds to τ ≈ 50 ms — a third estimate of the delay.

**Margins at τ = 65 ms** (onset pose; per direction 0.28 / 0.33 / 1.88 kg):

| k / b | phase margin | gain margin | verdict |
|---|---|---|---|
| **60 / 5** (working) | 71° / 34° / 87° | 6.0 / 4.1 / 21 dB | stable, ~2× gain headroom |
| 100 / 5 | 40° / 7° / 80° | 4.2 / 1.1 / 19 dB | marginal in one direction |
| 150 / 5 | 12° / −18° / 71° | 1.3 / −3.0 / 16 dB | unstable in one direction |
| 200 / 10 | −27° / −44° / 69° | −1.9 / −5.0 / 13 dB | unstable |
| **300 / 30** (old default) | −296° / −244° / 69° | −9.2 / −10.8 / 6 dB | very unstable |

Reading it:
- **The light, wrist-dominated directions set the limit.** Up the arm
  (1.9 kg) everything has plenty of margin.
- **Joint friction is essential.** Without `c`, even 60/5 has a negative
  phase margin in the light directions — which the arm contradicts.
- **This single-direction analysis is more pessimistic than the full
  simulation** (150/5: unstable here, ok in §3.3). It ignores cross-coupling
  between directions, the posture term, and treats Coulomb friction as
  viscous. Use it as a conservative bound and the simulation as the better
  estimate; both put the edge around 100–150 N/m.
- **Why more damping hurts here:** `b·s` raises |L| at high frequency, so
  crossover moves up to where ω·τ is larger. At 200 N/m: damping 5–10 stays
  near crossover ~5 Hz, damping 20 pushes it past 10 Hz.

## 5. Recommendations

- **Gains:** stiffness ≤ 100 N/m, damping 2–10 N·s/m (now: 60/5, the
  launch default). Treat 150–200 N/m as experimental.
- **Rotation:** keep `rot_damping` ≤ 0.1; `rot_stiffness` 6 has worked, but
  rotation with axes held has not been modelled yet (see §7).
- **Posture damping:** consider `null_damping` 0.1 (see §6).
- **Before trying new gains on the arm:**
  `gen2_stability_model.py check --stiffness K --damping B` and
  `gen2_stability_model.py margins --gains K:B`.
- **On the arm:** the oscillation watchdog softens the spring and, if that
  fails, disables torque; `tools/stability_step_test.py` walks up the gains
  with 1.5 cm anchor steps, the watchdog armed. Both are described in
  `torque-mode-trajectory-following.md` (Stability section); neither has
  run on the arm yet.

## 6. Joint 7: the discrete damping limit

Joint 7 (last wrist joint, only the gripper beyond it) has inertia
**0.00094 kg·m²**. A damper whose torque is held for a whole tick is
unstable on it once b·Δt/I > 2, i.e. above
b ≈ 2 · 0.00094 / 0.01 ≈ **0.19 N·m·s/rad** at 100 Hz.

- `null_damping` (posture term, 0.5) acts on joint 7 whenever rotation is
  free: above the limit.
- `rot_damping`: the old default 0.2 sits on the limit (matching the earlier
  simulation note that 1.0 caused a wrist limit cycle); 0.1 is below.
- On the arm, joint 7's friction (stick-slip around 0.13 N·m on the wrist)
  masks it — in the incident bag joint 7 didn't move. Delay makes the limit
  lower in practice.

## 7. Limits of this analysis, open questions

- Only two poses, both from one bag; both low-to-mid reach on the table side.
- Position only. Rotation with axes held (the 2026-11-20 wrist swinging,
  2.5–3 s period, moment at the cap) not yet modelled; likely the same
  delay effect on the rotation loop.
- Friction values (viscous 0.3, Coulomb 0.3) are guesses, not identified.
- The delay's source and whether it varies (temperature, load, USB) are
  unknown; one bag, moderate correlation.
- Inertias come from the URDF; the payload and gripper fit (0.56 kg) are
  not checked against it.
- Force saturation (max_force) and the collision repulsion field (which
  fired at the onset) are not in the linear analysis.
- Next on the arm: step responses at 60 → 80 → 100 N/m in x, y, z with the
  watchdog armed (`tools/stability_step_test.py`); compare overshoot and
  ringing frequency with the model, and refine the delay and friction from
  that.

## Reproducing

```
# model (plain Python, springcontroller venv)
PY=~/.springcontroller_venv/bin/python
$PY springcontroller/tools/gen2_stability_model.py map --delay-ms 60
$PY springcontroller/tools/gen2_stability_model.py check --stiffness 60 --damping 5
$PY springcontroller/tools/gen2_stability_model.py margins --gains 60:5,100:5,300:30
$PY -m pytest test/test_gen2_stability_model.py      # pins the two hardware results

# bag measurements (ROS sourced)
BAG=~/gen2_bags/rosbag2_2026_11_23-18_58_34
python3 springcontroller/tools/bag_oscillation.py $BAG 188 193   # 3.7 Hz, joints 1/3/5
python3 springcontroller/tools/bag_torque_lag.py $BAG 188.5 192.8   # 52-64 ms
$PY springcontroller/tools/bag_replay.py $BAG --start 100           # interactive replay
```
