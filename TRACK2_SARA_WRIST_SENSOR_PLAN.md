# Track 2 (deferred): wrist-mounted Bota SensONE, SARA-inspired localization

Written 2026-08-13. **Not implemented.** Deferred because the Bota SensONE
is currently mounted on a different arm in the lab, in use by someone else
for about a month. Also: mounting it right before the Robotiq gripper (the
only place it can go on this arm) means it will only ever see
end-effector-region forces, not general-purpose along-the-arm contact — still
a useful zone, just narrower than the full paper's redundant-sensing design.
Track 1 (software-only residual-vector-matching fix, see
`test_press_to_pin.py` and `press_to_pin.py`'s `_localize_contact`) was
implemented instead and does not depend on any of this.

This document preserves the design work from that planning session so it
doesn't need to be re-derived when the sensor is available again. Read this
alongside [[gen3_press_to_pin_debug]] / the `press_to_pin` code before
picking it back up — some of the "current state" claims below (line numbers,
etc.) may have drifted if Track 1 or other work has touched the same files
since.

Motivating paper: Iskandar, Eiberger, Albu-Schäffer, De Luca, Dietrich,
"Collision Detection, Identification, and Localization on the DLR SARA Robot
with Sensing Redundancy," ICRA 2021, DOI 10.1109/ICRA48506.2021.9561677.
Their method: redundant physical 6-DOF F/T sensors (base + wrist + optional
interface points) modeled as "virtual locked joints" in an extended dynamics
formulation (their Eqs 7-17), then geometric line-of-force ∩ known
link-surface-geometry contact localization (Eqs 18-22).

---

## (a) Sensor integration

**Driver**: use Bota's official `bota_driver` (rokubimini stack,
`rokubimini_msgs`) over the community Humble-only fork
(`idra-lab/bota_ft_sensor_driver`) — it's the vendor-maintained path and
supports both EtherCAT and serial SensONE variants. Confirm which physical
interface this specific unit uses at bring-up time; that only affects the
driver's runtime config, not the topic/message contract. It publishes a
`geometry_msgs/WrenchStamped` (via `rokubimini_msgs`, exact topic name
depends on the driver's YAML config, something like
`/bus0/ft_sensor0/ft_sensor_readings/wrench`) — directly consumable in
`press_to_pin.py` as a plain `geometry_msgs.msg.WrenchStamped` subscription.

**URDF change (required)**: insert a fixed `ft_sensor_link` between
`end_effector_link` and `robotiq_85_base_link`, i.e.:

```
end_effector_link --[fixed, offset = sensor mounting-plate thickness]--> ft_sensor_link --[fixed, zero offset]--> robotiq_85_base_link
```

replacing the current direct zero-offset `robotiq_85_base_joint`. The exact
offset (SensONE's mechanical thickness between its two mounting faces, per
Bota's datasheet) matters for both gravity-comp bookkeeping downstream of the
sensor and this design's own line-of-force geometry — pull it from the
datasheet/CAD, don't estimate it. This lives in `gen3_kinova_flat.urdf`;
confirm whether `gen3_torque_control/urdf/gen3_pinocchio.urdf` /
`gen3_for_pinocchio.xacro` needs the same edit — as of this writing that
model calls `xacro:load_robot` with `gripper=""`, i.e. it's arm-only and
stops before any gripper/sensor geometry, so it likely doesn't need to know
the sensor exists.

**Real hardware risks to flag up front**:
1. Mounting between the Kinova wrist flange and the Robotiq gripper needs a
   mechanical adapter plate matching both bolt patterns — likely sourcing or
   machining, not just a driver config.
2. Added sensor mass/cabling changes true arm dynamics (extra mass, shifted
   CoM, cable drag) — gravity-comp constants and the collision-box AABBs
   (sized from the URDF mesh set) will be stale until the URDF/geometry is
   regenerated to include the sensor.

## (b) The key technical question: predicting the "no-contact" wrench at the sensor

**Don't** read `pin.rnea`'s `data.f` at joint 7 and transform it to the
sensor frame. `press_to_pin`/`gen3_spring.launch.py` already lock the
gripper's joints via `locked_joint_names` → `pin.buildReducedModel`, which
folds *all* mass from `bracelet_link` through the whole gripper into one
composite body attached to joint 7. `data.f[joint7_id]` is therefore the
wrench transmitted to that whole lump — including `bracelet_link`'s and
`end_effector_link`'s own weight, which never actually loads the sensor
(only what's bolted to the sensor's distal face does). Naively transforming
that value to the sensor frame double-counts real structural mass as if it
were loading the sensor — not negligible, `bracelet_link` is a real machined
housing.

**Recommended approach instead** (avoids the double-counting by
construction, rather than trying to get pinocchio to cut mid-composite-body,
which its Python API doesn't cleanly support — there's no public
`JointModelFixed`-as-a-real-joint the way there is for revolute/prismatic):

1. Keep the main 7-DOF dynamics model built from the **arm-only** URDF
   (`gen3_kinova_flat_arm_only.urdf`, which already stops before the
   gripper) for gravity comp / RNEA of the arm's own DOF — this sidesteps the
   double-counting question entirely rather than solving it inside
   pinocchio's model tree.
2. Separately, compute — once, offline, as a hand-derived constant, not
   per-cycle — the lumped `pin.Inertia` of everything distal to the sensor
   plate (sensor body + `robotiq_85_base_link` + all gripper links), summed
   via the parallel-axis theorem in the sensor frame. `pin.Inertia` objects
   support this directly (`Inertia(m, com, I)` plus additive composition
   under a common frame).
3. Per cycle, predict the "no-contact" wrench at the sensor as a
   single-rigid-body Newton-Euler equation using that constant and the
   sensor frame's kinematic state (from the already-current main-arm
   pinocchio data via `pin.getFrameVelocity`/`getFrameAcceleration` at the
   sensor's frame id): `predicted = I_lumped * a_local + v_local x* (I_lumped
   * v_local) - gravity_wrench_local`. Exact spatial-algebra call sequence
   (`pin.Motion`/`pin.Force`/`pin.Inertia` operators) wasn't verified by
   running code — nail it down in a short standalone spike before wiring
   into the node.
4. **Lower-fidelity simplification for a first version** (matches "even in a
   lower-fidelity way" framing from the original ask): only trust this when
   the existing `press.velocity_gate` condition already holds (the system
   already assumes quasi-static motion for residual reasoning elsewhere).
   Under that gate, velocity/acceleration terms vanish and the predicted
   wrench collapses to just the **static weight** of the distal assembly
   resolved into the sensor's current orientation — no noisy
   double-differentiated encoder signals needed. Build and bench-validate
   this simplified version first (see Verification below) before attempting
   the full inertial version.

## (c) External wrench

`external = measured_wrench (from WrenchStamped) - predicted_wrench (from
(b))`, both in the sensor's local frame at the sample time. All the risk of
this step lives in (b) being right.

## (d) Line of force

Paper's Eqs 18-20: given `(f, m)` at the sensor point, `xr =
-pinv(skew(f)) @ m` is a point on the line of action, direction `f/‖f‖`,
`l(α) = xr + α·(f/‖f‖)`. Compute in the sensor's local frame (where `(f,m)`
are already expressed), then transform into world frame using the sensor
frame's current `oMf` (already available from the shared
`URDFArmConfiguration.update()`) before intersecting against link geometry,
since collision-box world placements (`oMg[gid]`) are in world frame.

## (e) Ray-vs-box intersection

New method on `URDFArmConfiguration` (it already owns the collision model
and live `oMg` placements): `intersect_ray_with_links(origin_world,
direction_world, candidate_links=None) -> list[(link_name, t, point_world)]`.
For each geometry object whose `.geometry` is a `coal.Box` and whose parent
link is in `candidate_links` (or all, if `None`): transform the ray into the
box's local frame via `placement.inverse()`, run the standard slab method
against `[-halfSide, +halfSide]` per axis, keep `t >= 0` hits, sort by
distance. Self-contained and easily unit-testable in isolation (synthetic
box + known ray → known intersection point) — build and test this before
wiring into anything else; it has no dependency on the sensor or the
dynamics work in (b)/(c).

Restrict `candidate_links` to the sensor's structurally-reachable set: link
6 (`spherical_wrist_2_link`)/link 7 (`bracelet_link`) area, `end_effector_link`,
the sensor body, gripper/tool links. Not links 1-4 — the sensor structurally
cannot see loads there.

## (f) Ambiguity / disambiguation

When the ray hits more than one candidate link, use Track 1's
`_localize_contact` signature-fit score against the *arm's own joint-torque
residuals* at each geometrically-plausible point as a coarse consistency
filter — discard candidates that are geometrically plausible but torque-fit
poorly. This reuses Track 1's matcher rather than inventing a second scoring
system, and plays the role the paper's second physical sensor would
otherwise play, substituted here with the joint-torque channel that's
already available for free.

## (g) Node integration

- New `~/wrench` subscription (`geometry_msgs/WrenchStamped`, topic as a
  param, e.g. `sensor.wrench_topic`), cached like `_commanded_cb` caches
  commanded torques.
- Gate the whole path behind `press.velocity_gate` (already exists) plus a
  cheap first check that Track 1's coarse joint-residual signal shows any
  wrist-zone activity at all, before doing the full line-of-force/ray work.
- For links in the sensor's reachable set: prefer this path's result when
  confident (non-ambiguous per (f)); fall back to Track 1 when ambiguous or
  the sensor driver isn't running. Links 1-4: always Track 1, unconditionally
  — the sensor can't see them.
- This should be a dispatch wrapper around the existing localization step,
  not a second state machine — `_latch`, hysteresis, hold-time logic stay
  unchanged (already made track-agnostic by Track 1's refactor).
- New `sensor.trust_live` param, default `false` — until validated live (see
  Verification), this path should run in shadow/logging-only mode and never
  itself trigger a real `AddSpring` call.

## Files (when this is picked back up)

| File | Change | Effort |
|---|---|---|
| `bota_driver`/`rokubimini` (external) | Install, configure for the SensONE's actual interface, bring up `WrenchStamped` topic | M (hardware/driver bring-up) |
| `gen3_kinova_flat.urdf` | Insert `ft_sensor_link`/joint with real datasheet offset | S-M (needs the real datasheet dimension + mount CAD) |
| New `springcontroller/springcontroller/springcontroller/sensor_wrench_predictor.py` | Lumped distal-inertia constant + predicted-wrench computation (b)/(c); build/test standalone first | M |
| `urdf_arm_configuration.py` | New `intersect_ray_with_links(...)` (e) | S-M |
| `press_to_pin.py` | New `~/wrench` subscription, dispatch wrapper combining Track 1 + this, new params | M |
| New `test/test_sensor_wrench_predictor.py`, `test/test_ray_box_intersection.py` | Component unit tests, before end-to-end wiring | M |
| `test/replay_rosbag_localization.py` (from Track 1) | Extend to validate against a new labeled rosbag with the sensor mounted, repeating the original J4/J5/J6 protocol | S (extension) |

**Rough total: L**, dominated by (1) hardware mounting/cabling/driver
bring-up (unpredictable — could stall on parts/mechanical adapter
availability) and (2) getting (b)'s predicted-wrench math right and
validated in isolation before trusting it against live contact data — that's
the one piece of real physics risk in the whole design and deserves its own
spike before (d)-(g) are built on top of it.

## Recommended sequencing (when resumed)

1. (e) ray-vs-box intersection — fully independent, unit-testable against
   synthetic geometry, no hardware needed. Can start anytime.
2. (b) predicted-wrench spike, validated per the bench test below, before
   any node wiring.
3. Hardware: URDF edit + sensor mount + driver bring-up, in parallel with
   (2) once the sensor and adapter plate are in hand.
4. Node integration (g), last, once (1), (2), and hardware are each
   independently validated.

## Verification (when resumed)

1. **Bench, no-motion validation first**: arm stationary, sensor mounted, no
   external load — log measured vs. predicted wrench from (b) and confirm
   the difference sits near the sensor's noise floor before trusting it for
   anything. Isolates (b) from all contact-detection logic.
2. **Rosbag replay against labeled ground truth**: collect a new labeled
   rosbag repeating the original J4/J5/J6 press protocol (reusing the
   existing `test_marker` mechanism) with the sensor mounted, replay it
   through `test/replay_rosbag_localization.py`, and compare against the
   original (pre-sensor) dataset's misattribution count.
3. Only after both pass should `sensor.trust_live` flip to `true` for real
   pin creation.
