# springcontroller

A ROS 2 package implementing virtual spring impedance control for robot arms.
Each spring pulls a point on a robot link toward a fixed target in world space,
producing joint torques via the Jacobian-transpose method.

##NOTE: the springcontroller_interfaces folder should be moved one level up to the src/ directory for colcon to find it properly, but is in this folder for repository cleanliness reasons

## Dependencies

- ROS 2 Humble (or later)
- `pinocchio` — `sudo apt install ros-humble-pinocchio` or `pip install pin`
- `numpy`

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

```bash
ros2 run springcontroller virtual_spring_node.py --ros-args \
  -p urdf_path:=/path/to/robot.urdf
# Then: ros2 param get /virtual_spring_node link_names
```

## Launch

```bash
ros2 launch springcontroller virtual_spring.launch.py \
  urdf_path:=/path/to/robot.urdf \
  config:=/path/to/springs.yaml
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
