# virtual_spring_node — CLI quick reference

## Launching

```bash
ros2 launch springcontroller virtual_spring.launch.py \
    urdf_path:="/path/to/robot.urdf" \
    config:="/path/to/springs.yaml"

# With torque relay
ros2 run springcontroller torque_relay \
    --ros-args --params-file /path/to/ur3e_relay.yaml
```

---

## Monitoring

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

## Remove a spring

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

```bash
# Send zeros to all joints (safe stop)
ros2 topic pub --once /forward_effort_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

# Move shoulder lift only
ros2 topic pub --once /forward_effort_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0, -2.0, 0.0, 0.0, 0.0, 0.0]}"
```

---

## springs.yaml structure

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

---

## ur3e_relay.yaml structure

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
