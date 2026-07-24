# springcontroller

A ROS 2 package implementing virtual spring impedance control for robot arms.
Each spring pulls a point on a robot link toward a fixed target in world space,
producing joint torques via the Jacobian-transpose method.


## Dependencies

- ROS 2 Humble (or later, also tested with Kilted)
- `pinocchio` —
# Create the venv (allow access to ROS system packages like rclpy)
python3 -m venv --system-site-packages ~/ros_venv
# Activate it
source ~/ros_venv/bin/activate
# get the right numpy
pip install "numpy<2" --force-reinstall


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

# Terminal 3 — enable torque mode
ros2 service call /kinova/torque_control/enable std_srvs/srv/SetBool "{data: true}"
```

To use a custom springs config:
```bash
ros2 launch springcontroller gen3_spring.launch.py config:=/path/to/your_springs.yaml
```

Spring link names for the Gen3 7-DOF: `base_link`, `shoulder_link`, `half_arm_1_link`,
`half_arm_2_link`, `forearm_link`, `spherical_wrist_1_link`, `spherical_wrist_2_link`,
`bracelet_link`, `end_effector_link`. Edit `config/gen3_springs.yaml` to configure.

Topic flow:
```
/kinova/joint_states_lowlevel → virtual_spring_node → torque_relay → /kinova/joint_torque_command
```

### Other robots (generic ros2_control effort controller)

```bash
ros2 launch springcontroller virtual_spring.launch.py urdf_path:=/path_to_flat_urdf config:=/path_to_springs.yaml
```

Separately:
```bash
ros2 launch springcontroller torque_relay.launch.py joint_order:="[joint1, joint2, ...]"
```


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
