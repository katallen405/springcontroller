# springcontroller

A ROS 2 package implementing virtual spring impedance control for robot arms.
Each spring pulls a point on a robot link toward a fixed target in world space,
producing joint torques via the Jacobian-transpose method. Supports two arms:
UR3e (via `forward_effort_controller`) and Kinova Gen3 (via `gen3_torque_control`).

**Don't confuse these two enable services** — they control different things:
- `/virtual_spring_node/enable` — turns spring *forces* on/off. The arm stays
  under torque control (still gravity-compensated on Gen3), springs just stop
  pulling.
- `/gen3_torque_control/enable` (Gen3 only) — turns torque control itself
  on/off. Disabling this drops the arm back to Kortex position hold; it's
  what `virtual_spring_node`'s fail-safe calls automatically if spring
  computation starts failing repeatedly.

## Quick reference

**UR3e** (hardware already does gravity comp) — `ur3e_spring.launch.py` starts
`virtual_spring_node` and `torque_relay` together:
```bash
ros2 launch springcontroller ur3e_spring.launch.py \
  urdf_path:=/path/to/flat_urdf config:=/path/to/springs.yaml
```

**Kinova Gen3** (via `gen3_torque_control`, no hardware gravity comp):
```bash
# Terminal 1 — Kortex low-level torque interface
ros2 run gen3_torque_control gen3_torque_node

# Terminal 2 — spring controller. enable_torque_control:=true auto-enables
# torque mode ~3s after startup, once virtual_spring_node is confirmed
# publishing valid gravity-compensated torques. Defaults to false: leave it
# off to review things first, then enable manually (see below) when ready.
ros2 launch springcontroller gen3_spring.launch.py enable_torque_control:=true

# Manual enable/disable at any time (this is also what the fail-safe calls
# automatically if spring computation starts failing repeatedly):
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: true}"
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: false}"
```

See [Launch](#launch) below for the full set of arguments (`srdf_path`,
`add_gravity_compensation`, `torque_control_service`, etc), and
[CLI reference](#cli-reference) for the complete set of monitoring, runtime
spring-management, and manual-testing commands.

## Dependencies

- ROS 2 Humble (or later, also tested with Kilted)
- `pinocchio`, `numpy<2`, `meshcat` (only needed for `armviz.py`)

### Python venv

This package needs a `--system-site-packages` venv (to see ROS's own
`rclpy` etc. alongside pip-installed packages like `pinocchio`/`meshcat`).
One venv, shared by the whole project — **not** a generic do-everything ROS
venv — so it's easy to recreate on another machine:

```bash
python3 -m venv --system-site-packages ~/.springcontroller_venv
source ~/.springcontroller_venv/bin/activate
pip install "numpy<2" --force-reinstall
pip install meshcat   # only needed for armviz.py
```

The convention here is `~/.springcontroller_venv`. **If you name yours
differently, you need to update every one of these places** — getting this
wrong is a silent-failure trap (things run against the wrong Python with
confusing missing-module errors), not a loud one:

| File | What to change |
|---|---|
| `springcontroller/virtual_spring_node.py` | shebang (line 1) |
| `springcontroller/equilibrium_mover.py` | shebang (line 1) |
| `test/armviz.py` | shebang (line 1) |
| `launch/gen3_spring.launch.py` | `SPRINGCONTROLLER_VENV_PYTHON` constant |
| `launch/ur3e_spring.launch.py` | `SPRINGCONTROLLER_VENV_PYTHON` constant |

The shebangs only matter if you ever run a script directly (e.g.
`./armviz.py`) — `ros2 run`/`ros2 launch` go through colcon's installed
entry points or the explicit venv path in the launch files instead, so
those are the ones that actually matter for normal use. Quick way to check
you got all of them:
```bash
grep -rn "springcontroller_venv" --include="*.py" .
```


## Build
```bash
cd ~/ros2_ws/src
git clone <this repo>
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select virtual_spring_ros2
source install/setup.bash
```

## Configure springs

Edit `config/springs.yaml` to define your springs:

```yaml
/**:
  ros__parameters:
    spring_names: [tip_spring]
    springs:
      tip_spring:
        link_name:   "tool0"
        local_point: [0.0, 0.0, 0.1]
        target:      [0.5, 0.0, 0.8]
        stiffness:   120.0
        damping:     8.0
        rest_length: 0.0
```

`link_name` must match a frame name in your URDF. See [springs.yaml
structure](#springsyaml-structure) below for a fully worked UR3e and Gen3
example, including `inner_radius`/`outer_radius`.

### Joint springs

A `VirtualSpring` pulls a Cartesian point toward a Cartesian target via the
Jacobian transpose -- which means it can be structurally blind to certain
joint rotations. A joint whose axis passes through the spring's attachment
point (e.g. a wrist joint under a coaxial tool flange) can rotate freely
without moving that point at all, so the spring supplies exactly zero
restoring torque there no matter how far it drifts. `JointSpring` fills
that gap: it pulls one joint directly toward a target angle, independent
of any link's Cartesian pose.

```yaml
/**:
  ros__parameters:
    joint_spring_names: [wrist_center]
    joint_springs:
      wrist_center:
        joint_name: "joint_7"
        stiffness:  2.0   # N*m/rad
        damping:    0.3   # N*m*s/rad
        # target_angle omitted -> defaults to the joint's current angle at
        # load time (a soft "hold here" spring). Set it explicitly for a
        # fixed target instead.
```

Add both `springs` and `joint_springs` blocks to the same config file to
use them together -- they're summed into the same total torque. Runtime
target updates: `~/joint_target/<name>` (`std_msgs/Float64`, radians),
mirroring `~/target/<name>` for Cartesian springs. Add/remove at runtime via
`~/add_joint_spring` (`springcontroller_interfaces/AddJointSpring`) and the
existing `~/remove_spring` (name-based, works for either spring type).

### Orientation springs

A `VirtualSpring` pulls a point toward a target -- it can't express "keep
this face pointed at that point" without also dragging the attachment point
toward it. `OrientationSpring` does the opposite: it aligns a direction
fixed in a link's local frame (a "face normal") with the direction from the
attachment point to a target world point, using only the rotational
Jacobian. It produces pure restoring torque and zero translational force,
and re-aims every cycle from the *current* attachment position, so it keeps
pointing at the target as the arm/block moves rather than holding a
rotation frozen at load time.

```yaml
/**:
  ros__parameters:
    orientation_spring_names: [face_participant]
    orientation_springs:
      face_participant:
        link_name:         "end_effector_link"
        local_point:        [0.0, 0.0, 0.1]   # gripper/block center, link-local (m)
        local_face_normal:  [0.0, 0.0, 1.0]   # the block's working face, link-local
        target:              [0.6, 0.4, 0.5]   # e.g. the participant's measured face position (m)
        stiffness:          2.0                # N*m/rad
        damping:            0.2                # N*m*s/rad
```

Unlike `joint_springs`' `target_angle` or `springs`' `target`, `target` here
has **no safe default** and must be set explicitly -- it's an external
real-world point (e.g. a person's face), not something inferable from the
arm's own pose, so an unconfigured spring fails to load rather than aiming
at whatever the zero/neutral pose happens to imply. Add it to the same
config file as `springs`/`joint_springs` to sum all three into one total
torque. Runtime target updates: `~/target/<name>` (`geometry_msgs/
PointStamped`), same topic Cartesian springs use. Add/remove at runtime via
`~/add_orientation_spring` (`springcontroller_interfaces/AddOrientationSpring`)
and the existing `~/remove_spring`.

## Launch

### Kinova Gen3 (via gen3_torque_control node)

```bash
# Terminal 1 — start the Kinova torque interface
ros2 run gen3_torque_control gen3_torque_node

# Terminal 2 — start the spring controller
ros2 launch springcontroller gen3_spring.launch.py

# Terminal 3 — enable torque mode (or pass enable_torque_control:=true above
# to have the launch file do this automatically ~3s after startup)
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: true}"
```

Optional overrides:
```bash
ros2 launch springcontroller gen3_spring.launch.py \
    urdf_path:=/path/to/flat_urdf.urdf \
    config:=/path/to/gen3_springs.yaml \
    srdf_path:=/path/to/gen3.srdf
```

Launch arguments (all optional, shown with their defaults):

| Argument | Default | Description |
|---|---|---|
| `urdf_path` | `flat_urdf_files/gen3_kinova_flat.urdf` | URDF for pinocchio; includes the gripper so its mass counts toward gravity comp |
| `config` | `config/gen3_springs.yaml` | Springs configuration YAML |
| `srdf_path` | Gen3 MoveIt config SRDF | Excludes adjacent-link pairs (e.g. `base_link`/`shoulder_link`) from self-collision checks |
| `add_gravity_compensation` | `true` | Gen3 has no hardware gravity comp, so this stays on unlike the UR3e |
| `torque_control_service` | `/gen3_torque_control/enable` | `SetBool` service used both to auto-enable torque control and by the fail-safe to disable it again |
| `enable_torque_control` | `false` | If `true`, auto-call `torque_control_service` with `data:true` ~3s after startup |
| `armviz` | `false` | If `true`, also launch `armviz.py` (see [Visualization](#visualization)) |

To use a custom springs config:
```bash
ros2 launch springcontroller gen3_spring.launch.py config:=/path/to/your_springs.yaml
```

Spring link names for the Gen3 7-DOF: `base_link`, `shoulder_link`, `half_arm_1_link`,
`half_arm_2_link`, `forearm_link`, `spherical_wrist_1_link`, `spherical_wrist_2_link`,
`bracelet_link`, `end_effector_link`. Edit `config/gen3_springs.yaml` to configure.

**Gripper links work as attachment points too.** `locked_joint_names` (see
above) locks the Robotiq joints rather than removing them, so
`robotiq_85_base_link`, `robotiq_85_left_finger_tip_link`,
`robotiq_85_right_finger_tip_link`, etc. all remain valid frames — the
gripper just moves rigidly with the wrist as far as the Jacobian/gravity
math is concerned, same as a fixed tool flange. Prefer `end_effector_link`
or `robotiq_85_base_link` (the mounting plate) for a general end-effector
attachment: those don't move when the fingers open/close, so they're exact.
A fingertip link is only as accurate as the gripper's actual open/closed
state matches the neutral pose the model was built with — the locked
joints aren't measured at runtime, so if the real gripper position varies,
a fingertip attachment point will be slightly off.

**Fail-safe:** if spring computation fails repeatedly (e.g. a persistent
self-collision or computation error), `virtual_spring_node` automatically
calls `torque_control_service` with `data:false` once, switching the arm
back to Kortex position hold rather than continuing to trust its own torque
computation. This requires a manual re-enable afterward — it won't
auto-recover on its own.

Topic flow:
```
/joint_states → virtual_spring_node → torque_relay → /kinova/joint_torque_command
```

### UR3e (via forward_effort_controller)

```bash
# ur3e_spring.launch.py starts both virtual_spring_node and torque_relay
# (formerly named virtual_spring.launch.py, and formerly required running
# torque_relay separately with ur3e_relay.yaml -- that's bundled in now).
ros2 launch springcontroller ur3e_spring.launch.py \
    urdf_path:=/path/to/flat_urdf config:=/path/to/springs.yaml
```

Launch arguments:

| Argument | Default | Description |
|---|---|---|
| `urdf_path` | *(required)* | URDF for pinocchio |
| `config` | `config/springs.yaml` | Springs configuration YAML |
| `armviz` | `false` | If `true`, also launch `armviz.py` (see [Visualization](#visualization)) |

`torque_relay`'s joint order (`ur3e_shoulder_pan_joint`, ...) is fixed to
the UR3e's 6 joints -- see `UR3E_JOINT_ORDER` in `ur3e_spring.launch.py` if
you need to change it.

Topic flow:
```
/joint_states → virtual_spring_node → torque_relay → /forward_effort_controller/commands
```

### Other ros2_control robots

Neither `ur3e_spring.launch.py` nor `gen3_spring.launch.py` is generic --
both hardcode their robot's joint order and topic names. For a different
`ros2_control`-based arm, run `virtual_spring_node` and `torque_relay`
separately (or copy `ur3e_spring.launch.py` as a starting template):

```bash
ros2 run springcontroller virtual_spring_node --ros-args \
    -p urdf_path:=/path/to/flat_urdf -p config_path:=/path/to/springs.yaml
```

Separately:
```bash
ros2 launch springcontroller torque_relay.launch.py joint_order:="[joint1, joint2, ...]"
```


## Visualization

`armviz.py` (in `test/`) is a MeshCat-based 3D visualizer -- opens in a
browser, shows the robot plus live spring attachment points, targets, and
the line between them. Not RViz. Needs `meshcat` installed in the venv (see
[Python venv](#python-venv)).

Launch it alongside either robot with `armviz:=true`, or run it standalone:
```bash
ros2 launch springcontroller gen3_spring.launch.py armviz:=true
ros2 launch springcontroller ur3e_spring.launch.py armviz:=true

# standalone
python3 test/armviz.py --urdf /path/to/flat_urdf
```

`--urdf` is required for the standalone form. When launched via `ros2
launch`, it's set to the same `urdf_path` as `virtual_spring_node`, and
remapped `/joint_states` on Gen3 to match (its default of plain
`/joint_states` is already right for UR3e).

Press `f` in its terminal to toggle display of every available attachment
frame as a labeled dot -- useful for picking a `link_name` for a spring.


## Known incomplete features

**Re-centering** (`_check_equilibrium_shift`/`_recenter_thread` in
`virtual_spring_node.py`, params `recentering_threshold_rad` /
`recentering_enabled`). Intent: after any spring change, if the new
equilibrium is far enough from the current pose, automatically switch to
position control, run an `equilibrium_mover` to get there smoothly, then
switch back to effort control and re-enable springs -- avoiding a sudden
kick from springs snapping to a very different target.

Never finished: the `ros2 run` call hardcodes a placeholder package name
(`your_study_pkg`) that was never filled in, and the whole sequence depends
on `ros2_control`'s `controller_manager` with specific controller names
(`scaled_joint_trajectory_controller` / `forward_effort_controller`), which
doesn't apply to Gen3 at all. It's gated off by default
(`recentering_enabled: false`) so a large shift just gets logged instead of
silently disabling your springs -- discovered the hard way when it fired
on a live Gen3 session via a large joint-spring target change, disabled
springs, then failed immediately since Gen3 has no `controller_manager`
(now fixed to always re-enable springs regardless of outcome, but the
underlying sequence still doesn't work for Gen3 and isn't finished for
UR3e).

To finish it: replace `your_study_pkg` with the real package, confirm the
controller names match your `ros2_control` setup, and decide whether Gen3
needs an equivalent (it has no `controller_manager`, so the sequence would
need a different mechanism entirely -- possibly just skipping straight to
gravity-comp-only torques during the transition instead). Or: if this
turns out not to be needed, `recentering_threshold_rad`/
`recentering_enabled`/`_check_equilibrium_shift`/`_recenter_thread` and the
recentering call sites in `_add_spring_cb`/`_add_joint_spring_cb` can be
deleted outright.


## Topics

| Topic | Type | Description |
|---|---|---|
| `~/joint_states` (sub) | `sensor_msgs/JointState` | Arm joint positions + velocities |
| `~/joint_torques` (pub) | `sensor_msgs/JointState` | Spring torques in effort field |
| `~/target/<spring_name>` (sub) | `geometry_msgs/PointStamped` | Move a Cartesian or orientation spring's target at runtime |
| `~/joint_target/<spring_name>` (sub) | `std_msgs/Float64` | Move a joint spring's target angle (rad) at runtime |

## Services

| Service | Type | Description |
|---|---|---|
| `~/enable` | `std_srvs/SetBool` | Enable / disable all springs |
| `~/add_spring` | `springcontroller_interfaces/AddSpring` | Add a Cartesian spring at runtime |
| `~/add_joint_spring` | `springcontroller_interfaces/AddJointSpring` | Add a joint spring at runtime |
| `~/add_orientation_spring` | `springcontroller_interfaces/AddOrientationSpring` | Add an orientation spring at runtime |
| `~/remove_spring` | `springcontroller_interfaces/RemoveSpring` | Remove a spring by name (any type) |


## Using the library directly

```python
from springcontroller.virtual_spring import VirtualSpring
from springcontroller.urdf_arm_configuration import URDFArmConfiguration
import numpy as np

arm = URDFArmConfiguration("/path/to/robot.urdf", q=np.zeros(6))
spring = VirtualSpring(
    link_name="tool0",
    local_attachment_point=np.array([0.0, 0.0, 0.1]),
    target_world_point=np.array([0.5, 0.0, 0.8]),
    stiffness=120.0,
    damping=8.0,
)
torques = spring.compute_torques(arm)
```

## Tests

```bash
colcon test --packages-select virtual_spring_ros2
colcon test-result --verbose
```

## CLI reference

Quick reference for every `virtual_spring_node` command, covering both
supported arms. **Monitoring** and **manual torque test** are robot-specific
(split into UR3e / Gen3 subsections) since the underlying topics and control
stack differ; launching is covered above in [Launch](#launch). Everything
from **Enable / disable all springs** onward operates on
`virtual_spring_node`'s own topics/services (`/virtual_spring_node/...`),
which are identical regardless of which arm is running underneath.

### Monitoring — UR3e

```bash
# Watch torques being published
ros2 topic echo /virtual_spring/joint_torques

# Watch what the relay sends to the controller
ros2 topic echo /forward_effort_controller/commands

# Watch end effector position (from TF)
ros2 run tf2_ros tf2_echo ur3e_base_link ur3e_tool0 | grep -E "At time|Translation"

# List all active parameters
ros2 param list /virtual_spring_node

# Check a specific parameter
ros2 param get /virtual_spring_node springs.tip_spring.stiffness
```

### Monitoring — Kinova Gen3

```bash
# Watch torques being published (note: no remap on Gen3, unlike UR3e above)
ros2 topic echo /virtual_spring_node/joint_torques

# Watch what the relay sends to gen3_torque_control
ros2 topic echo /kinova/joint_torque_command

# Watch raw feedback from the Kortex cyclic loop
ros2 topic echo /joint_states

# Watch torque-control enable state
ros2 topic echo /gen3_torque_control/status

# List all active parameters
ros2 param list /virtual_spring_node

# Check a specific parameter
ros2 param get /virtual_spring_node springs.tip_spring.stiffness
```

### Torque control enable / disable (Gen3 only)

```bash
# Enable torque control (arm starts responding to computed torques)
----DO NOT DO THIS UNLESS GRAVITY COMPENSATION IS ACTIVE----
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: true}"

# Disable torque control (arm drops back to Kortex position hold)
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: false}"

# Clear arm faults
ros2 service call /gen3_torque_control/clear_faults std_srvs/srv/Trigger
```

### Enable / disable all springs

```bash
# Disable
ros2 service call /virtual_spring_node/enable std_srvs/srv/SetBool "{data: false}"

# Enable
ros2 service call /virtual_spring_node/enable std_srvs/srv/SetBool "{data: true}"
```

### Add a spring

```bash
ros2 service call /virtual_spring_node/add_spring \
    springcontroller_interfaces/srv/AddSpring "{
        name: 'my_spring',
        link_name: 'ur3e_tool0',
        local_point: [0.0, 0.0, 0.0],
        target: [0.2, 0.1, 0.4],
        stiffness: 50.0,
        damping: 0.0,
        rest_length: 0.0,
        inner_radius: 0.0,
        outer_radius: 0.0
    }"
```

### Add a joint spring

Pulls a single joint toward a target angle directly -- no Jacobian, so it
supplies restoring torque on rotational axes a Cartesian spring can be
structurally blind to (e.g. a wrist joint coaxial with the spring's
attachment point). See [Joint springs](#joint-springs) above for the full
explanation.

```bash
ros2 service call /virtual_spring_node/add_joint_spring \
    springcontroller_interfaces/srv/AddJointSpring "{
        name: 'wrist_center',
        joint_name: 'joint_7',
        target_angle: 0.0,
        stiffness: 2.0,
        damping: 0.3
    }"
```

### Add an orientation spring

Aligns a link-local face normal with the direction toward a fixed world
point, using only the rotational Jacobian -- pure torque, no positional
pull. See [Orientation springs](#orientation-springs) above for the full
explanation.

```bash
ros2 service call /virtual_spring_node/add_orientation_spring \
    springcontroller_interfaces/srv/AddOrientationSpring "{
        name: 'face_participant',
        link_name: 'end_effector_link',
        local_point: [0.0, 0.0, 0.1],
        local_face_normal: [0.0, 0.0, 1.0],
        target: [0.6, 0.4, 0.5],
        stiffness: 2.0,
        damping: 0.2
    }"
```

### Remove a spring

Works for any spring type -- it's name-based, not type-specific.

```bash
ros2 service call /virtual_spring_node/remove_spring \
    springcontroller_interfaces/srv/RemoveSpring "{name: 'my_spring'}"
```

### Move a spring target at runtime

```bash
ros2 topic pub --once /virtual_spring_node/target/tip_spring \
    geometry_msgs/msg/PointStamped "{
        header: {frame_id: 'world'},
        point: {x: -0.3, y: 0.2, z: 0.5}
    }"
```

### Move a joint spring target at runtime

```bash
ros2 topic pub --once /virtual_spring_node/joint_target/wrist_center \
    std_msgs/msg/Float64 "{data: 0.5}"
```

### Move a spring attachment point at runtime

```bash
ros2 topic pub --once /virtual_spring_node/attachment/tip_spring \
    geometry_msgs/msg/PointStamped "{
        header: {frame_id: 'ur3e_tool0'},
        point: {x: 0.0, y: 0.0, z: 0.05}
    }"
```

### Change spring parameters at runtime

```bash
# Change stiffness
ros2 param set /virtual_spring_node springs.tip_spring.stiffness 100.0

# Change damping
ros2 param set /virtual_spring_node springs.tip_spring.damping 2.0

# Change rest length
ros2 param set /virtual_spring_node springs.tip_spring.rest_length 0.05

# Change target
ros2 param set /virtual_spring_node springs.tip_spring.target "[-0.3, 0.2, 0.4]"
```

> Note: parameter changes update the stored value but do NOT hot-reload the spring
> object itself. Use the `~/target/<name>` topic for live target updates.
> For stiffness/damping changes to take effect, remove and re-add the spring.

### Watch which springs are active

```bash
ros2 topic echo /virtual_spring_node/springs_updated
```

Publishes a JSON array of spring names whenever a spring is added or removed.

### Manually send a torque command (testing)

These bypass `virtual_spring_node` entirely — no gravity comp, no
collision checking, no fail-safe. Only use them with the arm in a
supported/spotted position.

#### UR3e

```bash
# Send zeros to all joints (safe stop -- UR3e's own controller still
# provides gravity comp underneath, so zero here just means "no extra force")
ros2 topic pub --once /forward_effort_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

# Move shoulder lift only
ros2 topic pub --once /forward_effort_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0, -2.0, 0.0, 0.0, 0.0, 0.0]}"
```

#### Kinova Gen3

```bash
# gen3_torque_control must be ENABLED (see "Torque control enable /
# disable" above) for this to have any effect.
#
# ⚠️ Zero here is NOT a safe stop like it is on the UR3e -- gen3_torque_control
# has no hardware gravity comp, so zero torques let the arm fall. To actually
# stop safely, disable torque control instead:
#   ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: false}"
ros2 topic pub --once /kinova/joint_torque_command \
    std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

### springs.yaml structure

UR3e example:

```yaml
/**:
  ros__parameters:
    spring_names: ['tip_spring', 'elbow_spring']
    springs:
      tip_spring:
        link_name:    "ur3e_tool0"
        local_point:  [0.0, 0.0, 0.0]
        target:       [-0.3, 0.2, 0.45]
        stiffness:    55.2
        damping:      0.0
        rest_length:  0.0
        inner_radius: 0.0
        outer_radius: 0.0
      elbow_spring:
        link_name:    "ur3e_forearm_link"
        local_point:  [0.0, 0.0, 0.1]
        target:       [0.1, 0.0, 0.3]
        stiffness:    30.0
        damping:      1.0
        rest_length:  0.0
        inner_radius: 0.0
        outer_radius: 0.0
```

Gen3 example (`gen3_springs.yaml`):

```yaml
/**:
  ros__parameters:
    spring_names:
      - tip_spring
    springs:
      tip_spring:
        link_name:    "bracelet_link"   # wrist/end-effector link on Gen3 7-DOF
        local_point:  [0.0, 0.0, 0.1]   # attachment point in link-local frame (m)
        target:       [0.5, 0.0, 0.5]   # target position in world frame (m)
        stiffness:    50.0              # N/m (start low for safety)
        damping:      5.0               # N·s/m
        rest_length:  0.0               # m (0 = zero-length spring)
        inner_radius: 0.0
        outer_radius: 0.0
```

See [Kinova Gen3 launch](#kinova-gen3-via-gen3_torque_control-node) above
for the full list of valid Gen3 spring link names, including gripper links.

#### Adding a joint spring (e.g. Gen3 wrist)

`joint_7`'s axis runs through `end_effector_link`, so a Cartesian spring
attached there supplies exactly zero restoring torque for that joint's own
rotation -- add a `joint_springs` block to the same config file to give it
one:

```yaml
/**:
  ros__parameters:
    spring_names: [tip_spring]
    springs:
      tip_spring:
        # ... as above ...
    joint_spring_names: [wrist_center]
    joint_springs:
      wrist_center:
        joint_name: "joint_7"
        stiffness:  2.0   # N*m/rad -- start low
        damping:    0.3   # N*m*s/rad
        # target_angle omitted -> defaults to joint_7's current angle at load
```

### ur3e_relay.yaml structure

`ur3e_spring.launch.py` sets these same values inline (`UR3E_JOINT_ORDER`,
`torque_topic`, `command_topic`), so you don't need this file for normal
use. It's kept for running `torque_relay` standalone, outside the launch
file:

```bash
ros2 run springcontroller torque_relay \
    --ros-args --params-file /path/to/ur3e_relay.yaml
```

```yaml
torque_relay:
  ros__parameters:
    command_topic: "/forward_effort_controller/commands"
    torque_topic:  "/virtual_spring/joint_torques"
    joint_order:
      - ur3e_shoulder_pan_joint
      - ur3e_shoulder_lift_joint
      - ur3e_elbow_joint
      - ur3e_wrist_1_joint
      - ur3e_wrist_2_joint
      - ur3e_wrist_3_joint
```

### Gen3 torque_relay configuration

Same story as the UR3e above: `gen3_spring.launch.py` sets `torque_relay`'s
parameters inline (`GEN3_JOINT_ORDER`, `torque_topic`, `command_topic`), so
there's nothing extra to pass on the command line for normal use. If you
need to run `torque_relay` standalone for Gen3 instead of through the
launch file:

```bash
ros2 run springcontroller torque_relay --ros-args \
    -p joint_order:="[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]" \
    -p torque_topic:="/virtual_spring_node/joint_torques" \
    -p command_topic:="/kinova/joint_torque_command"
```
