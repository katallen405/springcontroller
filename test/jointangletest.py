#!/home/kat/ros_venv/bin/python3
import math
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import json

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    "/home/kat/workspace/src/ceeorobot_cell/ceeorobot_description/urdf/ceeorobot_flat.urdf"
)
data = model.createData()

pinocchio_joint_index = {
    model.names[i]: i - 1
    for i in range(1, model.njoints)
}

latest_q = pin.neutral(model)
spring_states = {}  # name -> {attachment: [x,y,z], target: [x,y,z]}

def joint_cb(msg):
    global latest_q
    q = pin.neutral(model)
    for name, pos in zip(msg.name, msg.position):
        angle = math.remainder(pos, 2 * math.pi)
        if name in pinocchio_joint_index:
            q[pinocchio_joint_index[name]] = angle
    latest_q = q

def torque_cb(msg):
    """Read attachment points from the torque topic header isn't useful,
    but we can subscribe to springs_updated and attachment separately."""
    pass

rclpy.init()
node = rclpy.create_node('viz_node')
node.create_subscription(JointState, '/joint_states', joint_cb, 10)

# Subscribe to joint_torques to get attachment point positions
def torques_cb(msg):
    pass  # positions come from pinocchio directly below

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# Spring targets — set these to match your springs.yaml
SPRINGS = {
    "tip_spring": {
        "target": np.array([-0.3, 0.2, 0.45]),
        "color": 0xff4444,   # red sphere for target
    }
}

def draw_springs(q):
    """Update spring visualizations in MeshCat."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    for name, spring in SPRINGS.items():
        # Draw target as a red sphere
        target = spring["target"]
        T_target = np.eye(4)
        T_target[:3, 3] = target
        viz.viewer[f"springs/{name}/target"].set_object(
            g.Sphere(0.02),
            g.MeshLambertMaterial(color=0xff2222, opacity=0.8)
        )
        viz.viewer[f"springs/{name}/target"].set_transform(T_target)

        # Draw attachment point as a blue sphere (from FK)
        frame_id = model.getFrameId("ur3e_tool0")
        T_attach = np.eye(4)
        T_attach[:3, :3] = data.oMf[frame_id].rotation
        T_attach[:3, 3] = data.oMf[frame_id].translation
        viz.viewer[f"springs/{name}/attachment"].set_object(
            g.Sphere(0.02),
            g.MeshLambertMaterial(color=0x2222ff, opacity=0.8)
        )
        viz.viewer[f"springs/{name}/attachment"].set_transform(T_attach)

        # Draw line between attachment and target
        vertices = np.array([
            data.oMf[frame_id].translation,
            target
        ]).T  # meshcat wants 3xN
        viz.viewer[f"springs/{name}/line"].set_object(
            g.Line(
                g.PointsGeometry(vertices),
                g.LineBasicMaterial(color=0xffaa00)
            )
        )

try:
    while True:
        rclpy.spin_once(node, timeout_sec=0.01)
        viz.display(latest_q)
        draw_springs(latest_q)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
