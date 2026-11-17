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
  time, cross-fade weight, position/rotation error (10 Hz; recorded in the bag)
- `ros2 run tf2_ros tf2_echo j2s7s300_link_base j2s7s300_ee_link`: where the
  gripper actually is, in the same frame and point as the anchor

After it ends, the spring keeps holding the final pose until the next goal or:
```
ros2 service call /virtual_spring_node/trajectory/release std_srvs/srv/Trigger
```

## Parameters

All `trajectory_spring.*` parameters are read **once at startup**: edit the
config YAML (e.g. `config/gen2_trajectory_first_run.yaml`) and relaunch.
Write them as floats (`60.0`, not `60`).

### Spring strength

| Parameter | Default | First-run config | What it does |
|---|---|---|---|
| `stiffness` | 300 N/m | 60 | Pull toward the anchor position. Higher = tracks closer, feels stiffer when pushed. |
| `damping` | 30 N·s/m | 5 | Damps the *difference* between planned and actual gripper velocity, so it doesn't drag against the plan. |
| `rot_stiffness` | 10 N·m/rad | 2 | Pull toward the anchor orientation. **2 is too weak for the Gen2 wrist**: see caveats. |
| `rot_damping` | 0.2 N·m·s/rad | 0.1 | Keep ≤ 0.2: 1.0 gave a wrist limit cycle in simulation. |
| `max_force` | 40 N | 10 | Cap on the spring force. Also caps how hard it can push you. |
| `max_moment` | 5 N·m | 1.5 | Cap on the spring moment. |
| `null_stiffness` | 2 N·m/rad | (default) | Posture pull toward the planned joint angles, in motions that don't move the gripper. Raise if joint 3 (elbow roll) ends off the plan. |
| `null_damping` | 0.5 N·m·s/rad | (default) | Damps those motions. |

Rough feel: at 60 N/m a 5 cm error pulls with 3 N. At 2 N·m/rad a 0.3 rad
twist pushes with 0.6 N·m.

### Pausing and finishing

| Parameter | Default | What it does |
|---|---|---|
| `pause_error_m` | 0.08 m | If the gripper is further than this from the anchor, trajectory time stops (the anchor waits for the arm). |
| `pause_error_rad` | 0.5 rad | Same, for orientation. |
| (fixed) resume at 60 % | | Time restarts only when **both** errors are below 60 % of the pause thresholds: 4.8 cm and 0.30 rad by default. |
| `pause_timeout_sec` | 10 s | Paused longer than this: aborted ("Held off the path…"). |
| `goal_tolerance_rad` | 0.05 rad | At the end, **every joint** must be within this of the last point… |
| (fixed) 0.02 rad/s | | …and every joint slower than this. |
| `goal_time_tolerance_sec` | 2 s | How long after the plan's end it may take to get there before aborting. |
| `start_tolerance_rad` | 0.05 rad | Only if the goal has a point at t = 0: how close that point must be to the arm. |

A goal can override the end tolerances itself: `goal_tolerance` per joint
(0 = use the default, negative = don't check that joint) and
`goal_time_tolerance`. `path_tolerance` in the goal is ignored; pausing
replaces it.

### Speed limits and other settings

| Parameter | Default | First-run config | What it does |
|---|---|---|---|
| `max_ee_speed_mps` | 0.25 m/s | 0.10 | Goals whose planned gripper speed exceeds this are refused; the message gives the speed. |
| (URDF) joint velocity limits | 0.63–0.84 rad/s | | Goals exceeding any joint's limit are refused, with the minimum duration. |
| `crossfade_sec` | 1 s | | Fade between the tip-hold spring and the tracking spring at start and release. |
| `link` | first of `end_effector_link`, `j2s7s300_ee_link`, `tool0` | | Link the spring is attached to. |
| `local_point` | [0, 0, 0] | | Attachment point in that link's frame. |
| `action_name` | `/spring_trajectory_controller/follow_joint_trajectory` | | |

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
| `Held off the path for more than 10.0s (end-effector error X m, Y rad)` | Paused and never got back under 60 % of the pause thresholds. | If Y ≥ 0.3 rad: the wrist didn't turn; raise `rot_stiffness`/`max_moment`. If X ≥ 4.8 cm: raise `stiffness`/`max_force`, or something is holding the arm. |
| `Not within goal tolerance 2.0s after the plan ended (worst joint N: … rad)` | Gripper may be fine, but a joint is off. | Joint 3: raise `null_stiffness`. Joint 7: raise `rot_stiffness`. Or loosen that joint's `goal_tolerance` in the goal. |
| `Stopped mid-trajectory: Torque control is not ENABLED…` | Torque dropped while running, usually the Gen2 firmware. | See firmware drop-outs. |
| `Stopped mid-trajectory: Springs are disabled (~/enable or collision clamp).` | Collision clamp or `~/enable false`. | The log will show `SELF-COLLISION detected…` or `COLLISION WITH SCENE OBJECT` if it was the clamp. Move clear, then `~/enable true`. |
| `Arm is in collision/DANGER -- not starting a trajectory.` | Start pose is under 1 cm from something. | Move clear first. |

After any abort or cancel, the tracking spring holds the arm **where it
stopped**, not where the plan had got to, so nothing snaps. The tip-hold
spring stays faded out until `trajectory/release`.

## Caveats

- **The Gen2 wrist needs more rotational stiffness than 2 N·m/rad.** All
  three aborts in the 2026-11-17 16:57 bag were rotation errors of 0.3–0.5 rad
  with the position within a few cm: 2 N·m/rad × 0.4 rad ≈ 0.8 N·m doesn't
  overcome the wrist's friction. Next to try: `rot_stiffness: 6.0`,
  `max_moment: 3.0`, keeping `rot_damping` ≤ 0.2.
- **"Done" is judged per joint, but the spring acts on the gripper.** On a
  7-DOF arm the gripper can be on target with joint 3 off (pushed sideways
  by hand) or joint 7 off (wrist not turned). Only the weak posture term or
  the rotation spring corrects those. Both caused goal-tolerance aborts.
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
