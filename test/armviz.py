#!/home/katallen/.springcontroller_venv/bin/python3
import argparse
import math
import re
import json
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g

import rclpy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MeshCat-based 3D visualizer for virtual_spring_node: "
                     "shows the robot, spring attachment points, and targets."
    )
    parser.add_argument(
        "--urdf", required=True,
        help="Path to the robot URDF (flattened, no unresolved xacro) to visualize.",
    )
    # Strip ROS-specific args (--ros-args -r ... -p ... etc.) before parsing
    # our own, so remapping (e.g. /joint_states -> /kinova/joint_states_lowlevel
    # on Gen3) still works cleanly when launched via ros2 launch/run.
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


_args = parse_args()

URDF          = _args.urdf
JOINT_STATES  = "/joint_states"
SPRINGS_TOPIC = "/virtual_spring_node/springs_updated"
SPRING_NODE   = "/virtual_spring_node"

# ---------------------------------------------------------------------------
# Pinocchio setup
# ---------------------------------------------------------------------------

model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF)
data = model.createData()

# Maps joint name -> the actual pinocchio Joint object, so we can write into
# the right slot(s) of q via joint.idx_q. NOT the same as "joint number" --
# continuous joints (e.g. Gen3's joint_1/3/5/7) consume 2 slots in q (cos,
# sin), not 1, so a naive sequential index silently misaligns every joint
# after the first continuous one. That was producing the wildly twisted/
# self-intersecting poses seen in the visualizer.
pinocchio_joints = {
    model.names[i]: model.joints[i]
    for i in range(1, model.njoints)
}

latest_q    = pin.neutral(model)
springs     = {}   # name -> {"target": np.array|None, "link_name": str}
target_subs = {}   # name -> rclpy subscription (kept alive)

show_frames = False  # toggled via 'f' keypress in the terminal

def get_all_frames():
    """Return list of (frame_name, frame_id) for every non-joint frame."""
    return [
        (model.frames[i].name, i)
        for i in range(model.nframes)
        if model.frames[i].type != pin.FrameType.JOINT
    ]

# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def get_param_target(name):
    """Read initial spring target from the spring node's ROS parameters."""
    try:
        out = subprocess.check_output([
            'ros2', 'param', 'get', SPRING_NODE,
            f'springs.{name}.target'
        ], timeout=2).decode()
        nums = re.findall(r'[-\d.]+', out.split(':')[-1])
        if len(nums) == 3:
            return np.array([float(x) for x in nums])
    except Exception as e:
        print(f"[viz] could not read target for '{name}': {e}")
    return None

def get_param_link(name):
    """Read initial link_name from the spring node's ROS parameters."""
    try:
        out = subprocess.check_output([
            'ros2', 'param', 'get', SPRING_NODE,
            f'springs.{name}.link_name'
        ], timeout=2).decode()
        return out.split(':')[-1].strip()
    except Exception as e:
        print(f"[viz] could not read link_name for '{name}': {e}")
        return "ur3e_tool0"

def load_springs_from_params():
    """Bootstrap spring list from ROS params at startup."""
    try:
        out = subprocess.check_output([
            'ros2', 'param', 'get', SPRING_NODE, 'spring_names'
        ], timeout=2).decode()
        # output: "String values are: ['tip_spring', 'elbow_spring']"
        names = re.findall(r"'([^']+)'", out)
        if names:
            print(f"[viz] found springs from params: {names}")
            for name in names:
                if name not in springs:
                    target    = get_param_target(name)
                    link_name = get_param_link(name)
                    springs[name] = {
                        "target":    target,
                        "link_name": link_name,
                    }
                    sub = node.create_subscription(
                        PointStamped,
                        f"/virtual_spring_node/target/{name}",
                        make_target_cb(name),
                        10,
                    )
                    target_subs[name] = sub
                    print(f"[viz] loaded spring: '{name}'  link={link_name}  target={target}")
        else:
            print("[viz] no springs found in params yet")
    except Exception as e:
        print(f"[viz] could not read spring_names: {e}")
# ---------------------------------------------------------------------------
# ROS callbacks
# ---------------------------------------------------------------------------

def joint_cb(msg):
    global latest_q
    q = pin.neutral(model)
    for name, pos in zip(msg.name, msg.position):
        angle = math.remainder(pos, 2 * math.pi)
        joint = pinocchio_joints.get(name)
        if joint is not None:
            if joint.nq == 2:  # continuous joint: stored as (cos, sin)
                q[joint.idx_q]     = math.cos(angle)
                q[joint.idx_q + 1] = math.sin(angle)
            else:
                q[joint.idx_q] = angle
    latest_q = q

def make_target_cb(spring_name):
    """Return a callback that updates the live target for a named spring."""
    def cb(msg):
        if spring_name in springs:
            springs[spring_name]["target"] = np.array([
                msg.point.x, msg.point.y, msg.point.z
            ])
            print(f"[viz] target updated for '{spring_name}': {springs[spring_name]['target']}")
    return cb

def springs_updated_cb(msg):
    """Called when the spring node adds or removes springs."""
    active_names  = set(json.loads(msg.data))
    current_names = set(springs.keys())

    # Add new springs
    for name in active_names - current_names:
        target    = get_param_target(name)
        link_name = get_param_link(name)
        springs[name] = {
            "target":    target,
            "link_name": link_name,
        }
        sub = node.create_subscription(
            PointStamped,
            f"/virtual_spring_node/target/{name}",
            make_target_cb(name),
            10,
        )
        target_subs[name] = sub
        print(f"[viz] tracking spring: '{name}'  link={link_name}  target={target}")

    # Remove old springs
    for name in current_names - active_names:
        springs.pop(name, None)
        target_subs.pop(name, None)
        viz.viewer[f"springs/{name}"].delete()
        print(f"[viz] removed spring: '{name}'")

# ---------------------------------------------------------------------------
# MeshCat drawing
# ---------------------------------------------------------------------------



def draw_springs(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    for name, spring in springs.items():
        target    = spring.get("target")
        link_name = spring.get("link_name")

        # Attachment point from FK
        frame_id = model.getFrameId(link_name)
        if frame_id >= len(data.oMf):
            print(f"[viz] unknown frame '{link_name}' for spring '{name}'")
            continue
        attachment = data.oMf[frame_id].translation.copy()

        # Blue sphere at attachment point
        T_attach = np.eye(4)
        T_attach[:3, 3] = attachment
        viz.viewer[f"springs/{name}/attachment"].set_object(
            g.Sphere(0.05),
            g.MeshLambertMaterial(color=0x0088ff, transparent=False)
        )
        viz.viewer[f"springs/{name}/attachment"].set_transform(T_attach)

        if target is not None:
            # Red sphere at target
            T_target = np.eye(4)
            T_target[:3, 3] = target
            viz.viewer[f"springs/{name}/target"].set_object(
                g.Sphere(0.05),
                g.MeshLambertMaterial(color=0xff0000, transparent=False)
            )
            viz.viewer[f"springs/{name}/target"].set_transform(T_target)

            # Orange line from attachment to target
            vertices = np.column_stack([attachment, target])  # 3x2
            viz.viewer[f"springs/{name}/line"].set_object(
                g.Line(
                    g.PointsGeometry(vertices),
                    g.LineBasicMaterial(color=0xff9900)
                )
            )
        else:
            print(f"[viz] no target yet for spring '{name}' — "
                  f"try: ros2 topic pub --once "
                  f"/virtual_spring_node/target/{name} "
                  f"geometry_msgs/msg/PointStamped "
                  f"'{{header: {{frame_id: world}}, point: {{x: 0.0, y: 0.0, z: 0.5}}}}'")

def draw_frames(q):
    """Draw all available attachment frames as black dots with name labels."""
    if not show_frames:
        viz.viewer["frames"].delete()
        return

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    for frame_name, frame_id in get_all_frames():
        if frame_id >= len(data.oMf):
            continue
        pos = data.oMf[frame_id].translation.copy()

        # Black dot at frame origin
        T = np.eye(4)
        T[:3, 3] = pos
        viz.viewer[f"frames/{frame_name}/dot"].set_object(
            g.Sphere(0.02),
            g.MeshLambertMaterial(color=0x000000, transparent=False)
        )
        viz.viewer[f"frames/{frame_name}/dot"].set_transform(T)

  
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

rclpy.init(args=sys.argv)
node = rclpy.create_node('spring_viz_node')

node.create_subscription(JointState, JOINT_STATES,  joint_cb,           10)
node.create_subscription(String,     SPRINGS_TOPIC, springs_updated_cb, 10)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# if node is already running, bootstrap from parameters
load_springs_from_params()

print("Spring visualizer ready.")
print(f"  Robot URDF:      {URDF}")
print(f"  Joint states:    {JOINT_STATES}")
print(f"  Springs updates: {SPRINGS_TOPIC}")
print("Waiting for springs from virtual_spring_node...")

import threading, sys, tty, termios

def _key_listener():
    """Press 'f' to toggle frame display on/off."""
    global show_frames
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        # stdin isn't a real TTY -- e.g. launched via ros2 launch's
        # ExecuteProcess rather than run interactively. The 'f' toggle just
        # isn't available in that case; nothing else here depends on it.
        print("[viz] stdin isn't a TTY -- 'f' frame-toggle unavailable "
              "(this is normal when launched via ros2 launch).")
        return
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == 'f':
                show_frames = not show_frames
                state = "ON" if show_frames else "OFF"
                sys.stdout.write(f"\r\n[viz] frame display: {state}\r\n")
                if show_frames:
                    print("[viz] available frames:")
                    for name, fid in get_all_frames():
                        sys.stdout.write(f"  - {name}  (id={fid})\r\n")
                    sys.stdout.flush()
            elif ch in ('\x03', 'q'):   # Ctrl-C or q
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

_key_thread = threading.Thread(target=_key_listener, daemon=True)
_key_thread.start()
print("  Press 'f' to toggle available frame display.")

try:
    while True:
        rclpy.spin_once(node, timeout_sec=0.01)
        viz.display(latest_q)
        draw_springs(latest_q)
        draw_frames(latest_q)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
