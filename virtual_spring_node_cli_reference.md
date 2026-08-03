# virtual_spring_node — CLI quick reference

Covers both supported arms. The **Launching**, **Monitoring**, and **manual
torque test** sections are robot-specific (split into UR3e / Gen3
subsections below) since the underlying topics and control stack differ.
Everything from **Enable / disable all springs** onward operates on
`virtual_spring_node`'s own topics/services (`/virtual_spring_node/...`),
which are identical regardless of which arm is running underneath.

**Don't confuse these two enable services** — they control different things:
- `/virtual_spring_node/enable` — turns spring *forces* on/off. The arm stays
  under torque control (still gravity-compensated on Gen3), springs just stop
  pulling.
- `/gen3_torque_control/enable` (Gen3 only) — turns torque control itself
  on/off. Disabling this drops the arm back to Kortex position hold; it's
  what `virtual_spring_node`'s fail-safe calls automatically if spring
  computation starts failing repeatedly.

---

## Launching — UR3e

```bash
# ur3e_spring.launch.py starts both virtual_spring_node and torque_relay
# (formerly named virtual_spring.launch.py, and formerly required running
# torque_relay separately with ur3e_relay.yaml -- that's bundled in now).
ros2 launch springcontroller ur3e_spring.launch.py \
    urdf_path:="/path/to/robot.urdf" \
    config:="/path/to/springs.yaml"

# armviz:=true also launches the MeshCat visualizer (see below)
ros2 launch springcontroller ur3e_spring.launch.py \
    urdf_path:="/path/to/robot.urdf" armviz:=true
```

## Launching — Kinova Gen3

```bash
# Terminal 1 — Kortex low-level torque interface
ros2 run gen3_torque_control gen3_torque_node

# Terminal 2 — spring controller + torque_relay (gen3_spring.launch.py
# starts both). enable_torque_control:=true auto-enables torque mode ~3s
# after startup, once virtual_spring_node is confirmed publishing valid
# gravity-compensated torques; defaults to false (enable manually instead,
# see "Torque control enable / disable" below).
ros2 launch springcontroller gen3_spring.launch.py enable_torque_control:=true

# Optional overrides:
ros2 launch springcontroller gen3_spring.launch.py \
    urdf_path:=/path/to/flat_urdf.urdf \
    config:=/path/to/gen3_springs.yaml \
    srdf_path:=/path/to/gen3.srdf
```

---

## Monitoring — UR3e

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

## Monitoring — Kinova Gen3

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

---

## Torque control enable / disable (Gen3 only)

```bash
# Enable torque control (arm starts responding to computed torques)
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: true}"

# Disable torque control (arm drops back to Kortex position hold)
ros2 service call /gen3_torque_control/enable std_srvs/srv/SetBool "{data: false}"

# Clear arm faults
ros2 service call /gen3_torque_control/clear_faults std_srvs/srv/Trigger
```

---

## Visualization (armviz)

`test/armviz.py` -- MeshCat-based 3D visualizer (browser, not RViz). Shows
the robot, spring attachment points, targets, and the line between them.
Needs `meshcat` installed in the venv (see the package README's "Python
venv" section).

```bash
# via launch file (either robot)
ros2 launch springcontroller gen3_spring.launch.py armviz:=true
ros2 launch springcontroller ur3e_spring.launch.py armviz:=true

# standalone
python3 test/armviz.py --urdf /path/to/flat_urdf.urdf
```

`--urdf` is required. When launched via `ros2 launch`, it's set to the same
`urdf_path` as `virtual_spring_node`, and remapped `/joint_states` on Gen3
to match (its default of plain `/joint_states` is already right for UR3e).

Press `f` in its terminal to toggle every available attachment frame as a
labeled dot -- handy for picking a `link_name` for a new spring.

---

## Enable / disable all springs

```bash
# Disable
ros2 service call /virtual_spring_node/enable std_srvs/srv/SetBool "{data: false}"

# Enable
ros2 service call /virtual_spring_node/enable std_srvs/srv/SetBool "{data: true}"
```

---

## Add a spring

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

---

## Add a joint spring

Pulls a single joint toward a target angle directly -- no Jacobian, so it
supplies restoring torque on rotational axes a Cartesian spring can be
structurally blind to (e.g. a wrist joint coaxial with the spring's
attachment point). See the README's "Joint springs" section for the full
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

---

## Remove a spring

Works for either spring type -- it's name-based, not type-specific.

```bash
ros2 service call /virtual_spring_node/remove_spring \
    springcontroller_interfaces/srv/RemoveSpring "{name: 'my_spring'}"
```

---

## Move a spring target at runtime

```bash
ros2 topic pub --once /virtual_spring_node/target/tip_spring \
    geometry_msgs/msg/PointStamped "{
        header: {frame_id: 'world'},
        point: {x: -0.3, y: 0.2, z: 0.5}
    }"
```

## Move a joint spring target at runtime

```bash
ros2 topic pub --once /virtual_spring_node/joint_target/wrist_center \
    std_msgs/msg/Float64 "{data: 0.5}"
```

---

## Move a spring attachment point at runtime

```bash
ros2 topic pub --once /virtual_spring_node/attachment/tip_spring \
    geometry_msgs/msg/PointStamped "{
        header: {frame_id: 'ur3e_tool0'},
        point: {x: 0.0, y: 0.0, z: 0.05}
    }"
```

---

## Change spring parameters at runtime

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

---

## Watch which springs are active

```bash
ros2 topic echo /virtual_spring_node/springs_updated
```

Publishes a JSON array of spring names whenever a spring is added or removed.

---

## Manually send a torque command (testing)

These bypass `virtual_spring_node` entirely — no gravity comp, no
collision checking, no fail-safe. Only use them with the arm in a
supported/spotted position.

### UR3e

```bash
# Send zeros to all joints (safe stop -- UR3e's own controller still
# provides gravity comp underneath, so zero here just means "no extra force")
ros2 topic pub --once /forward_effort_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

# Move shoulder lift only
ros2 topic pub --once /forward_effort_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0, -2.0, 0.0, 0.0, 0.0, 0.0]}"
```

### Kinova Gen3

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

---

## springs.yaml structure (UR3e example)

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

## gen3_springs.yaml structure (Gen3 example)

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

Gen3 spring link names: `base_link`, `shoulder_link`, `half_arm_1_link`,
`half_arm_2_link`, `forearm_link`, `spherical_wrist_1_link`,
`spherical_wrist_2_link`, `bracelet_link`, `end_effector_link`.

Gripper links (`robotiq_85_base_link`, `robotiq_85_left_finger_tip_link`,
`robotiq_85_right_finger_tip_link`, etc.) also work — locking a joint keeps
its frame, it just moves rigidly with the wrist. Use `end_effector_link` or
`robotiq_85_base_link` for a general attachment point (exact, unaffected by
finger position); a fingertip link is only as accurate as the gripper's
real open/closed state matches the neutral pose the model was built with,
since the locked joints aren't measured at runtime.

### Adding a joint spring (e.g. Gen3 wrist)

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

---

## ur3e_relay.yaml structure

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

## Gen3 torque_relay configuration

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
