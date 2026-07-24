# springcontroller

A ROS 2 package implementing virtual spring impedance control for robot arms.
Each spring pulls a point on a robot link toward a fixed target in world space,
producing joint torques via the Jacobian-transpose method.

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
# automatically if spring computation starts failing):
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: true}"
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: false}"
```

See [Launch](#launch) below for the full set of arguments (`srdf_path`,
`add_gravity_compensation`, `torque_control_service`, etc).

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

`link_name` must match a frame name in your URDF. You can list all available
frames with:

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
/kinova/joint_states_lowlevel → virtual_spring_node → torque_relay → /kinova/joint_torque_command
```

### UR3e (via forward_effort_controller)

```bash
# ur3e_spring.launch.py starts both virtual_spring_node and torque_relay
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

Press `f` in its terminal to toggle display of every available attachment
frame as a labeled dot -- useful for picking a `link_name` for a spring.


## Topics

| Topic | Type | Description |
|---|---|---|
| `~/joint_states` (sub) | `sensor_msgs/JointState` | Arm joint positions + velocities |
| `~/joint_torques` (pub) | `sensor_msgs/JointState` | Spring torques in effort field |
| `~/target/<spring_name>` (sub) | `geometry_msgs/PointStamped` | Move a spring's target at runtime |

## Services

| Service | Type | Description |
|---|---|---|
| `~/enable` | `std_srvs/SetBool` | Enable / disable all springs |


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
