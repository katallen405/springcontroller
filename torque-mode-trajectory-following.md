# Torque-mode trajectory following (spring tracking)

`virtual_spring_node` can follow a planned joint trajectory *compliantly*: a
6-DOF spring on the gripper is pulled along the plan, so the arm tracks it
but gives way when pushed. This is separate from the stiff, position-mode
FollowJointTrajectory server in `gen2_torque_control`.

Code: `springcontroller/springcontroller/trajectory_spring.py` (math, ROS-free)
and `trajectory_action.py` (action server, parameters). Branch:
`trajectory-spring`. First hardware runs: Gen2, 2026-11-17.

## How it works

Every joint-state update (100 Hz on the Gen2):

1. The plan is sampled at the current *trajectory time* to get planned joint
   angles `q_d` and velocities `q̇_d`.
2. The **anchor** is FK of `q_d`: a position and orientation of
   `j2s7s300_ee_link`. Its velocity (`J·q̇_d`) is used as feed-forward.
3. A spring/damper pulls the gripper toward the anchor:
   - force  `F = K·(x_d − x) + D·(v_d − v)`, capped at `max_force`
   - moment `M = K_rot·log(R_d Rᵀ) + D_rot·(ω_d − ω)`, capped at `max_moment`
   - joint torques `τ = Jᵀ [F; M]`
4. A weak **posture term** pulls the joints toward `q_d` in the null space
   (motions that don't move the gripper). On a 7-DOF arm this is the only
   thing that keeps the elbow on the plan.

Gravity is compensated by the Gen2 firmware (payload set on
`gen2_torque_node`), not by the spring.

## Running a goal

```
ros2 launch springcontroller gen2_spring.launch.py start_bridge:=true \
  config:=$(ros2 pkg prefix springcontroller)/share/springcontroller/config/gen2_trajectory_first_run.yaml \
  load_collision_scene:=true armviz:=true
ros2 service call /gen2_torque_control/enable std_srvs/srv/SetBool "{data: true}"

ros2 action send_goal /spring_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [j2s7s300_joint_1, j2s7s300_joint_2, j2s7s300_joint_3, j2s7s300_joint_4, j2s7s300_joint_5, j2s7s300_joint_6, j2s7s300_joint_7], \
    points: [{positions: [Q1, Q2, Q3, Q4, Q5, Q6, Q7], time_from_start: {sec: 10}}]}}"
```

Watch it:
- `ros2 topic echo /virtual_spring_node/trajectory/state`: state, trajectory
  time, cross-fade weight, position/rotation error, and `springs` — the
  tracking spring's current parameters and weight, and each spring on the
  same link (e.g. `tip_spring`) with the stiffness it pulls with now and
  its full stiffness, so you can watch the hand-over (10 Hz; in the bag).
  The action's own feedback can't carry this: FollowJointTrajectory
  feedback has no text field.
- When a goal ends (success, abort or cancel), its result message ends with
  which springs now hold the gripper, e.g. `Active springs: tracking spring
  (60 N/m, 6 N*m/rad) holding the gripper where it ended; tip_spring (10 N/m)
  faded out until ~/trajectory/release.` Also in the spring node's log.
- `ros2 run tf2_ros tf2_echo j2s7s300_link_base j2s7s300_ee_link`: where the
  gripper actually is, in the same frame and point as the anchor

After it ends, the spring keeps holding the final pose until the next goal or:
```
ros2 service call /virtual_spring_node/trajectory/release std_srvs/srv/Trigger
```

## World-frame goals: `FollowAnchorPath`

`/spring_trajectory_controller/follow_anchor_path`
(`springcontroller_interfaces/action/FollowAnchorPath`) moves the spring's
**anchor** — where its attachment point on the gripper link is pulled to —
through points in any TF frame. No IK: the arm finds its own joints, and the
posture term keeps them near where they started.

- `points` + `times`: where the anchor goes and when (from goal start, > 0,
  increasing). Each segment is a minimum-jerk move, at rest at every point.
  Its peak speed is 1.875 × the segment's average and must stay under
  `max_ee_speed_mps` (0.10 m/s: 0.4 m needs ≥ 7.5 s).
- `header.frame_id`: frame of everything in the goal; empty = the base frame
  (`trajectory_spring.base_frame`, `world`). Other frames are looked up in
  TF once, at goal start (`robot_state_publisher` provides the arm's links).
- **Orientation is free by default** (position only). `constrain_rotation`
  `[x, y, z]` holds the rotation about each axis of the goal frame, turning
  from the current orientation to `orientation` over the whole goal. Free
  axes get no rotational spring and no rotational damping; the error on the
  held axes is exact (swing–twist), not a small-angle approximation.
- Pausing, finishing, the springs reports and the abort messages work as
  for joint-space goals. Feedback here does carry text: `state`,
  `time_from_start`, both errors and `springs` (JSON). The result adds
  `active_springs`.

```
# Anchor to A, position only, 15 s
ros2 action send_goal --feedback /spring_trajectory_controller/follow_anchor_path \
  springcontroller_interfaces/action/FollowAnchorPath \
  "{points: [{x: -0.40, y: 0.00, z: 0.50}], times: [{sec: 15}]}"

# A -> B in a straight line (0.57 m: >= 10.6 s), gripper pointing down
# (tool z = -world z: 180 deg about x), yaw free
ros2 action send_goal --feedback /spring_trajectory_controller/follow_anchor_path \
  springcontroller_interfaces/action/FollowAnchorPath \
  "{points: [{x: 0.00, y: -0.40, z: 0.50}], times: [{sec: 12}], \
    orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}, constrain_rotation: [true, true, false]}"

# A point in another frame: 10 cm along j2s7s300_ee_link's own +z axis from
# where that link is now (looked up once at goal start; check which way its
# z points with tf2_echo or armviz before trying it)
ros2 action send_goal /spring_trajectory_controller/follow_anchor_path \
  springcontroller_interfaces/action/FollowAnchorPath \
  "{header: {frame_id: j2s7s300_ee_link}, points: [{x: 0.0, y: 0.0, z: 0.10}], times: [{sec: 5}]}"
```

Caveats specific to anchor paths:
- **No reachability or collision pre-check.** The anchor goes exactly where
  you say; a point the arm can't reach, or a line through the table, just
  stalls against the arm (pause → abort) or the collision clamp / repulsion.
- **The straight line is the anchor's, not the joints'.** Long moves that
  swing the base a lot may take the elbow somewhere odd; the posture term
  only gently prefers the start joints.
- **Gripper rotation speed isn't checked**, only anchor speed.
- Needs `springcontroller_interfaces` rebuilt; without it the node logs
  `FollowAnchorPath ... (rebuild it)` and serves only the joint-space action.

## Parameters

All `trajectory_spring.*` parameters are read **once at startup**: edit the
config YAML (e.g. `config/gen2_trajectory_first_run.yaml`) and relaunch.
Write them as floats (`60.0`, not `60`).

### Spring strength

| Parameter | Default | First-run config | What it does |
|---|---|---|---|
| `stiffness` | 300 N/m | 60 | Pull toward the anchor position. Higher = tracks closer, feels stiffer when pushed. 40 was too soft (lag + stick-slip): see caveats. |
| `damping` | 30 N·s/m | 5 | Damps the *difference* between planned and actual gripper velocity, so it doesn't drag against the plan. |
| `rot_stiffness` | 10 N·m/rad | 6 | Pull toward the anchor orientation. 2 was too weak for the Gen2 wrist: see caveats. |
| `rot_damping` | 0.2 N·m·s/rad | 0.1 | Keep ≤ 0.2: 1.0 gave a wrist limit cycle in simulation. |
| `max_force` | 40 N | 10 | Cap on the spring force. Also caps how hard it can push you. |
| `max_moment` | 5 N·m | 3 | Cap on the spring moment. |
| `null_stiffness` | 2 N·m/rad | (default) | Posture pull toward the planned joint angles, in motions that don't move the gripper. Raise if joint 3 (elbow roll) ends off the plan. |
| `null_damping` | 0.5 N·m·s/rad | (default) | Damps those motions. |

Rough feel: at 60 N/m a 5 cm error pulls with 3 N. At 6 N·m/rad a 0.3 rad
twist pushes with 1.8 N·m (at 2 N·m/rad it was 0.6 N·m).

### Pausing and finishing: limits on spring pull, not distance

Both are set as **how hard a spring pulls**, so the allowed error follows
from spring strength: distance = force ÷ stiffness, angle = moment ÷
rot_stiffness. A softer spring is automatically more forgiving.

| Parameter | Default | First-run config | What it does |
|---|---|---|---|
| `pause_force_n` | 24 N | 4.8 (= 8 cm at 60 N/m) | Trajectory time stops (the anchor waits) when the **tracking** spring pulls harder than this. At steady state the spring pulls as hard as you push, so in practice: *pushed harder than this*. |
| `pause_moment_nm` | 5 N·m | 3.0 (= 0.5 rad at 6 N·m/rad) | Same, for orientation. |
| (fixed) resume at 60 % | | 2.9 N / 1.8 N·m (4.8 cm / 0.3 rad) | Time restarts when **both** drop below 60 % of the pause limits. |
| `pause_timeout_sec` | 10 s | | Paused longer than this: aborted ("Held off the path…"). |
| `goal_force_tolerance_n` | 4 N | 4 (= **40 cm** via the 10 N/m tip spring) | Done when the spring that **holds the gripper afterwards** would pull at most this. That's the same-link Cartesian spring(s) the trajectory fades out (`tip_spring`, 10 N/m), or the tracking spring if there are none. |
| `goal_moment_tolerance_nm` | 1.2 N·m | 1.2 (= 0.2 rad) | Same for orientation, on the tracking spring's `rot_stiffness` (the tip spring has no orientation). |
| `goal_tolerance_rad` | 0 (off) | 0 | Optional per-joint check on top; ≤ 0 = joints not checked. |
| (fixed) 0.02 rad/s | | | Every joint must also have stopped. |
| `goal_time_tolerance_sec` | 2 s | | How long after the plan's end it may take before aborting. |
| `start_tolerance_rad` | 0.05 rad | | Only if the goal has a point at t = 0: how close that point must be to the arm. |

**40 cm is very loose** — that's what 4 N on a 10 N/m spring means. Lower
`goal_force_tolerance_n` (1 N = 10 cm) or stiffen the tip spring to tighten it.

A goal can still ask for per-joint tolerances itself (`goal_tolerance`:
positive = check that joint, 0 = use `goal_tolerance_rad`, negative = don't
check) and `goal_time_tolerance`. `path_tolerance` in the goal is ignored;
pausing replaces it.

### Speed limits and other settings

| Parameter | Default | First-run config | What it does |
|---|---|---|---|
| `max_ee_speed_mps` | 0.25 m/s | 0.10 | Goals whose planned gripper speed exceeds this are refused; the message gives the speed. |
| (URDF) joint velocity limits | 0.63–0.84 rad/s | | Goals exceeding any joint's limit are refused, with the minimum duration. |
| `crossfade_sec` | 1 s | | Fade between the tip-hold spring and the tracking spring at start and release. |
| `link` | first of `end_effector_link`, `j2s7s300_ee_link`, `tool0` | | Link the spring is attached to. |
| `local_point` | [0, 0, 0] | | Attachment point in that link's frame. |
| `action_name` | `/spring_trajectory_controller/follow_joint_trajectory` | | |
| `anchor_action_name` | `/spring_trajectory_controller/follow_anchor_path` | | World-frame goals. |
| `base_frame` | `world` | | Frame of the arm model; FollowAnchorPath goals are transformed into it. |

### Node-wide settings that affect trajectories

| Parameter | Default | What it does |
|---|---|---|
| `spring_ramp_duration_sec` | 1.5 s | Spring forces ramp in after every torque enable; goals are refused until it finishes. |
| `caution_threshold` | 0.05 m | Below this distance to another link or the table, a small repulsion force pushes them apart. |
| `danger_threshold` | 0.01 m | Below this, spring torque (not gravity comp) is scaled down toward zero. |
| `repulsion_max_force_n` / `repulsion_max_total_torque_nm` | 5 N / 5 N·m | Caps on that repulsion. |
| `joint_limit_margin_rad` / `joint_limit_max_torque_nm` | 0.175 rad / 2 N·m | Push away from joint limits within this margin. |
| `collision_disable_grace_sec` | 30 s | After the collision clamp turns springs off, torque control is disabled if they aren't re-enabled within this time. |

## What can stop a goal

| Message | Meaning | What to do |
|---|---|---|
| `Torque control is not ENABLED -- enable it first.` | Sent before enabling, or torque dropped. | Enable; check for a firmware drop-out (below). |
| `Springs are still ramping in -- try again in a moment.` | Sent within 1.5 s of enable. | Wait. |
| `Planned end-effector speed peaks at … m/s` | Too fast for `max_ee_speed_mps`. | Longer `time_from_start`. |
| `Planned speed of j2s7s300_joint_N peaks at …` | Over a joint's URDF speed limit. | Use at least the duration it gives. |
| `Held off the path for more than 10.0s: spring pulling F N (X cm off) and M N*m (Y rad off); resumes below …` | Paused and the pull never dropped under 60 % of the pause limits. | If you weren't holding it: the spring can't move the arm against friction. M high: raise `rot_stiffness`/`max_moment`. F high: raise `stiffness`/`max_force`. Or raise `pause_force_n` / `pause_moment_nm`. |
| `Not within goal tolerance 2.0s after the plan ended: spring force … (X cm off, tip_spring at 10 N/m; tolerance … cm)` | The holding spring would still pull more than `goal_force_tolerance_n`. | Raise `goal_force_tolerance_n`, or the arm really is far off. |
| `…: spring moment … (Y rad off)` | Orientation off by more than `goal_moment_tolerance_nm` ÷ `rot_stiffness`. | Raise the moment tolerance or `rot_stiffness`. |
| `…: joint N … rad off (goal tolerance …)` | Only if the goal (or `goal_tolerance_rad`) asked for per-joint tolerances. | Loosen them. |
| `…: still moving` | Not stopped (> 0.02 rad/s) yet. | Longer `goal_time_tolerance`. |
| `Stopped mid-trajectory: Torque control is not ENABLED…` | Torque dropped while running, usually the Gen2 firmware. | See firmware drop-outs. |
| `Stopped mid-trajectory: Springs are disabled (~/enable or collision clamp).` | Collision clamp or `~/enable false`. | The log will show `SELF-COLLISION detected…` or `COLLISION WITH SCENE OBJECT` if it was the clamp. Move clear, then `~/enable true`. |
| `Arm is in collision/DANGER -- not starting a trajectory.` | Start pose is under 1 cm from something. | Move clear first. |

After any abort or cancel, the tracking spring holds the arm **where it
stopped**, not where the plan had got to, so nothing snaps. The tip-hold
spring stays faded out until `trajectory/release`.

## Pick and place test (Gen2)

`springcontroller/tools/pick_place_test.py` picks up a small object and puts
it down elsewhere with FollowAnchorPath goals (position only, rotation free).

Setup: launch `gen2_spring.launch.py start_bridge:=true …`, enable torque,
hand on the E-STOP, object on the table (small and light: the payload limit
is `max_payload_kg`, 1 kg). Then:

```
source ~/sandbox/install/setup.bash
~/.springcontroller_venv/bin/python springcontroller/tools/pick_place_test.py
#   --object-mass 0.08          known mass (kg); otherwise it weighs the object
#   --pick X Y Z --place X Y Z  skip teaching (a taught run prints these)
```

1. It checks torque is on and the gripper is activated (it offers to
   activate it: the jaws sweep fully open and closed), and opens the gripper.
2. **Teaching**: springs go off and the arm floats on gravity compensation,
   so hold it. Put the open gripper around the object at grasp height, press
   Enter; put it where the object should go, Enter; hold the arm where it
   should start, Enter. Springs come back on holding where the arm is (they
   re-anchor whenever springs are re-enabled).
3. It asks before starting, then: above the pick spot (8 cm,
   `--approach`) → down (2 cm/s) → checks the gripper is within 1.5 cm
   (`--reach-tol`) → closes → checks the gripper reports an object (not
   "closed empty") → lifts → weighs it (or uses `--object-mass`) and turns
   on **payload compensation** → over to above the place spot (5 cm/s) →
   down → checks → opens → payload off → lifts.

**Payload compensation** (`/virtual_spring_node/set_payload`,
`springcontroller_interfaces/srv/SetPayload`): holds a grasped object's
weight by adding `Jᵀ·(0, 0, m·g)` at the gripper point, ramped in over
0.5 s, alongside gravity compensation (so springs off or the collision clamp
don't drop it). Without it, 1 N of object sags the gripper ~1.7 cm at
60 N/m. Shown in `~/spring_states` (`payload`). Weighing compares the
spring's vertical pull above the pick spot before the grasp and after the
lift; joint friction limits it to roughly ±0.1 kg, so `--object-mass` is
better when known.

If anything fails, or on Ctrl-C, it stops: the current goal is cancelled,
the gripper is left as it is (a held object is never dropped), springs hold
the arm where it is (turned back on if it stopped during teaching), and it
prints how to clear the payload if it was on. Every step is in `/rosout`
(`pick_place: …`), so `tools/bag_replay.py` lists the phases.

Caveats: rotation is free, so the wrist floats (only the posture term holds
it) and the fingers may turn a little; stick-slip can leave the gripper a
centimetre or two short, which the reach check catches rather than closing
in the wrong place.

## Replaying a test from its bag

The launch files' bags record every `/virtual_spring_node/` topic, including
`~/spring_states` (every spring's attachment point and target, 10 Hz) and
the trajectory anchor in `~/trajectory/state`. To see a test again:

```
source ~/sandbox/install/setup.bash
~/.springcontroller_venv/bin/python springcontroller/tools/bag_replay.py \
  ~/gen2_bags/rosbag2_2026_11_20-13_29_35 --start 140      # --robot gen3 for Gen3 bags
# -> writes rosbag2_..._replay.html next to the bag; open it in a browser
```

The page is a 3-D replay (arm, gripper path and pointing axis, anchor path
and target axis, the springs' pulls and lines to their targets, scene boxes)
with a scrub bar and linked charts of error, pull, moment and the wrist
joints, plus the bag's torque / trajectory / collision events. It reads the
bag only. For bags from before 2026-11-20 the anchor is estimated from the
recorded spring torques and the page says so (a median 1.4 cm from the exact
anchor on the 2026-11-20 test). The page loads plotly from a CDN, so it needs
internet.

## Stability: which gains are safe on the Gen2

**What happened (2026-11-23 18:58 bag):** the first pick-and-place run was
launched without a config, so the tracking spring ran at the code's defaults
(300 N/m, 30 N·s/m, 40 N cap). It tracked well for 9.5 s, then oscillated at
**3.7 Hz** (joints 1, 3 and 5; force swinging ±30–40 N; gripper 2–5 cm)
until the arm's power was cut. Since then `gen2_spring.launch.py` passes the
hardware-tested gains (below) as defaults; a config still overrides them.

**Where the model was wrong: delay.** A rigid-body simulation of the Gen2
at 100 Hz calls both 60/5 and 300/30 stable. The real loop lags: the
measured joint torque follows the command by **52–64 ms** (cross-correlation
during the oscillation, joints 1/3/5). With that delay in the model,
`tools/gen2_stability_model.py` reproduces both observations — 60/5 stable,
300/30 unstable — and its oscillation frequency falls toward the observed
3.7 Hz at the same ~60 ms (20 ms → 10 Hz, 40 ms → 5.8 Hz). Friction and
velocity noise barely change the result. With delay, damping can make
things *worse*: it pushes at the wrong time.

**Model map at 60 ms delay** (the two recorded poses agree; `ok` = settles
from a 2 cm offset, position only):

| stiffness \ damping | 2 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| 30 – 100 N/m | ok | ok | ok | ok | ok |
| 150 N/m | ok | ok | ok | unstable (one pose) | unstable |
| 200 N/m | unstable | ok | ok | unstable | unstable |
| 300 N/m | unstable | unstable | unstable | unstable | unstable |

So for now: **stiffness ≤ 100 N/m, damping 2–10 N·s/m.** The working 60/5 is
well inside. 150–200 N/m is marginal and only with damping 5–10.

**Second model finding: joint 7.** The last wrist joint's inertia is
0.00094 kg·m²; a damper held for a 10 ms tick is unstable on it above
b ≈ 2I/Δt ≈ 0.19 N·m·s/rad. `null_damping` (0.5, the posture term) is above
that and `rot_damping`'s old default (0.2) sits on it. Joint 7's friction
hides it on the arm, and delay makes it worse; keep `rot_damping` ≤ 0.1 and
consider `null_damping` 0.1.

**Check gains before trying them on the arm:**
```
~/.springcontroller_venv/bin/python springcontroller/tools/gen2_stability_model.py check --stiffness 100 --damping 5
~/.springcontroller_venv/bin/python springcontroller/tools/gen2_stability_model.py map --delay-ms 60
```
`test/test_gen2_stability_model.py` keeps the model honest against the two
hardware observations.

Full notes, measurements, the classical (Bode) margins and how to reproduce
everything: [docs/gen2_stability_analysis.md](docs/gen2_stability_analysis.md).

### Oscillation watchdog

Runs in the control loop whenever the tracking spring is active. It
high-passes the spring's force and moment and, over a 1 s window, trips when
the RMS exceeds 2.5 N or 0.6 N·m *and* the signal reverses at 1.5–20 Hz.
Any joint faster than 4 rad/s also trips it (runaway backstop; normal use
peaks at about 2.2 rad/s).
- **First trip:** stiffness, damping, rot_stiffness and rot_damping ×0.25,
  hold where the arm is (`Oscillation watchdog: ...` in `/rosout`; a running
  goal aborts).
- **Another trip** after a 2 s grace (0.5 s for a speed trip): torque control
  is disabled through the bridge.
- Softening is undone on release or by setting gains live (below).
- `~/trajectory/state` has a `"watchdog"` field (`force_rms`, `moment_rms`,
  `freq_hz`, `softened`, `trips`).
- Validated: synthetic signals; closed loop on the delayed model (catches
  300/30 within 0.4–0.8 s, then the softened hold settles); every
  2026-11 bag (no trips on the 25 working runs; trips on the incident 5 s
  before the visible oscillation). **Not yet tried live.**

| parameter (`trajectory_spring.`) | default | |
|---|---|---|
| `watchdog_enabled` | true | |
| `watchdog_force_rms_n` / `watchdog_moment_rms_nm` | 2.5 / 0.6 | trip levels |
| `watchdog_window_sec` | 1.0 | |
| `watchdog_soften` | 0.25 | gain factor on the first trip |
| `watchdog_grace_sec` | 2.0 | before a second trip disables torque |
| `watchdog_joint_speed_rad_s` | 4.0 | runaway backstop |

It doesn't see slow swinging below 1.5 Hz (the 2026-11-20 wrist motion was
0.35–0.5 Hz).

### Changing gains live

`stiffness`, `damping`, `rot_stiffness`, `rot_damping`, `max_force`,
`max_moment`, `null_stiffness` and `null_damping` (all `trajectory_spring.`)
can be set while running; they apply on the next control tick, also mid-goal:
```
ros2 param set /virtual_spring_node trajectory_spring.stiffness 80.0
```
Negative or non-numeric values are refused. Other parameters still need a
restart.

### Step test: walking up the gains on the arm

`springcontroller/tools/stability_step_test.py` checks the model's map on
the hardware. For each `stiffness:damping` level (default
`60:5,80:5,100:5,100:10,150:5`) it:
1. runs the stability model at the arm's current pose and **skips levels it
   predicts unstable** (`--allow-predicted-unstable` to try them anyway);
2. sets the gains live, holds the current gripper point;
3. steps the anchor 1.5 cm along +x, back, +y, back, +z, back (0.3 s
   each, under the 0.10 m/s anchor speed limit), watching 2.5 s after each;
4. reports overshoot (% of the step), settling time (within 2 mm), and the
   ringing left in the last second (mm RMS and its frequency).

It asks before each level, stops at the first watchdog trip or aborted goal,
and restores the starting gains at the end (also on Ctrl-C). Results print
as a table and are saved to `~/gen2_bags/step_test_<time>.json`.

```
# robot launched (gen2_spring.launch.py start_bridge:=true), torque enabled,
# arm in an open pose clear of the table, hand on the E-STOP; record a bag
~/.springcontroller_venv/bin/python springcontroller/tools/stability_step_test.py
#   --ladder 60:5,80:5   --axes z   --step 0.01   --no-prompt
```
Its `/rosout` lines start with `step_test:`, so the bag replay lists them as
events. Compare the ringing frequency with the model's (~3.7 Hz at 60 ms
delay); a different frequency means the delay estimate needs refining.

**Not yet covered:** rotation gains with rotation held (the 2026-11-20 wrist
swinging, 2.5–3 s period, may be the same delay effect), more poses, the
delay's source (bridge, USB or firmware torque loop) and whether it varies.
Confirm on the arm with small, watched steps before relying on the map.

## Caveats

- **The Gen2 wrist needs more rotational stiffness than 2 N·m/rad.** All
  three aborts in the 2026-11-17 16:57 bag were rotation errors of 0.3–0.5 rad
  with the position within a few cm: 2 N·m/rad × 0.4 rad ≈ 0.8 N·m doesn't
  overcome the wrist's friction. `rot_stiffness: 6.0` / `max_moment: 3.0`
  fixed it on 2026-11-19 (aborts then were 0.13–0.15 rad). Keep
  `rot_damping` ≤ 0.2.
- **Too soft a position spring lags and stick-slips.** At 40 N/m
  (2026-11-19 15:02 bag) the arm fell steadily behind to 8 cm, then stood
  still at 7.5 cm — 3 N didn't beat friction and the gravity-comp error.
  Joint 5 also **stick-slipped** (the "pulsing"): stuck for 0.6–0.8 s while
  the spring torque built to ~0.13 N·m, lurched ~1.5°, stuck again, about
  once a second. A softer spring makes each lurch bigger. Back at 60 N/m.
  A future fix: per-joint friction feed-forward along the planned velocity.
- **Joints used to be checked one by one at the end** (until 2026-11-19), which
  failed whenever the 7-DOF arm's elbow (joint 3) or wrist (joint 7) settled
  elsewhere with the gripper fine. Now the end is judged by spring pull.
- **Single-point goals move at constant joint speed.** Without
  `velocities` in the points, the plan is interpolated linearly: the anchor
  starts and stops abruptly. MoveIt plans (many points, with velocities)
  are interpolated smoothly (cubic Hermite).
- **Turning a lot of orientation takes time.** The speed check covers
  gripper position and joint speeds, not gripper rotation rate. A move that
  reorients the gripper a lot can still outrun a soft rotation spring and
  pause.
- **Gen2 firmware drop-outs abort trajectories.** The firmware leaves torque
  mode on its own every few minutes (68 s, 132 s, 184 s, 319 s on
  2026-11-17), and immediately (< 0.4 s) when the measured joint torques
  disagree too much with its gravity model. After the 15:46 wrist whip it
  refused for every enable until the arm was moved back. Fix for the
  immediate kind: candle, `zero_torque_sensors`, move back, re-enable.
- **Continuous joints report multi-turn angles** (e.g. joint 5 at 7.73 rad).
  Goals may use any turn; a goal with no t = 0 point starts from the arm's
  pose on the turn nearest the plan. Before commit `7d06e88` it didn't, and
  the wrist whipped a full turn.
- **The link 2 / link 4 gap is small in some poses** (2.6 cm at the pose
  used on 2026-11-17), so the repulsion field pushes on joints 4 and 6 the
  whole time. Pick open poses for tests.
- **One goal at a time.** A second goal is rejected while one runs.
- **MoveIt doesn't switch modes.** Torque control must already be enabled;
  in `gen2_moveit_config` the controller is `spring_trajectory_controller`
  (`default: false`).
- **Not tested on hardware yet:** release cross-fade timing under load,
  MoveIt-generated plans, the Gen3.
