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
```
bash
ros2 launch springcontroller virtual_spring.launch.py urdf_path:=/path_to_flat_urdf  config:=/path_to_springs.yaml
```

Separately:
ros2 launch springcontroller torque_relay.launch.py joint_order:="[elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]"


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
