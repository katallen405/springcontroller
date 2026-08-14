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
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


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
    # our own, so this still works cleanly when launched via ros2 launch/run.
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


_args = parse_args()

URDF          = _args.urdf
JOINT_STATES  = "/joint_states"
SPRINGS_TOPIC = "/virtual_spring_node/springs_updated"
SPRING_NODE   = "/virtual_spring_node"
COLLISION_STATE_TOPIC = "/virtual_spring_node/collision_object_state"

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

# id -> {"object_pose": 4x4 np.ndarray, "primitives": [(shape, dims, relative_pose 4x4)]}
collision_objects = {}

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
    """Bootstrap spring list from ROS params at startup. Returns True if any
    spring names were found (including ones already known), False if the
    param service isn't reachable yet or reports none -- lets the caller
    retry instead of treating "nothing yet" as final."""
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
            return True
        else:
            print("[viz] no springs found in params yet")
            return False
    except Exception as e:
        print(f"[viz] could not read spring_names: {e}")
        return False
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

def _pose_msg_to_matrix(pose):
    """geometry_msgs/Pose -> 4x4 homogeneous transform. Same field-order
    swap as virtual_spring_node's _pose_to_se3: ROS quaternions are
    (x, y, z, w) but pin.Quaternion takes (w, x, y, z)."""
    q = pin.Quaternion(
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
    )
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    return pin.SE3(q, t).homogeneous

def collision_object_cb(msg):
    """Mirrors virtual_spring_node's _collision_object_cb/_apply_collision_object
    ADD/APPEND/MOVE/REMOVE handling, but only to track shape/pose for
    drawing -- ~/collision_object_state (TRANSIENT_LOCAL) is virtual_spring_node
    echoing what it actually applied, from both the live ~/collision_object
    topic and the YAML scene loader, so this stays in sync regardless of
    which path added an object or whether armviz was even running yet."""
    if msg.operation in (CollisionObject.ADD, CollisionObject.APPEND):
        object_pose = _pose_msg_to_matrix(msg.pose)
        primitives = []
        for i, primitive in enumerate(msg.primitives):
            relative_pose = (
                _pose_msg_to_matrix(msg.primitive_poses[i])
                if i < len(msg.primitive_poses)
                else np.eye(4)
            )
            if primitive.type == SolidPrimitive.BOX:
                primitives.append(("box", list(primitive.dimensions[:3]), relative_pose))
            elif primitive.type == SolidPrimitive.CYLINDER:
                # SolidPrimitive convention: dimensions[:2] = [height, radius]
                primitives.append(("cylinder", list(primitive.dimensions[:2]), relative_pose))
            else:
                print(f"[viz] collision object '{msg.id}': unsupported primitive "
                      f"type {primitive.type}, skipping.")
        collision_objects[msg.id] = {"object_pose": object_pose, "primitives": primitives}
        draw_collision_object(msg.id)
        print(f"[viz] collision object '{msg.id}': {len(primitives)} shape(s) tracked.")
    elif msg.operation == CollisionObject.REMOVE:
        collision_objects.pop(msg.id, None)
        viz.viewer[f"collision_objects/{msg.id}"].delete()
        print(f"[viz] collision object '{msg.id}' removed.")
    elif msg.operation == CollisionObject.MOVE:
        entry = collision_objects.get(msg.id)
        if entry is None:
            print(f"[viz] collision object '{msg.id}': MOVE received but not "
                  "currently tracked; ignoring.")
            return
        entry["object_pose"] = _pose_msg_to_matrix(msg.pose)
        draw_collision_object(msg.id)

# ---------------------------------------------------------------------------
# MeshCat drawing
# ---------------------------------------------------------------------------

# meshcat.geometry.Cylinder's axis is local +Y (three.js convention); ROS/
# shape_msgs SolidPrimitive cylinders have their axis along local +Z. This
# extra rotation (90 deg about X) is applied after the object/relative pose
# so the drawn cylinder's axis matches the frame it's actually collision-
# checked against, instead of silently rendering sideways.
_CYLINDER_AXIS_ALIGN = np.eye(4)
_CYLINDER_AXIS_ALIGN[:3, :3] = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
])

_COLLISION_OBJECT_MATERIAL = g.MeshLambertMaterial(
    color=0x808080, opacity=0.5, transparent=True
)

def draw_collision_object(obj_id):
    """(Re)draw every primitive of a tracked collision object at its current
    pose. Event-driven (called from collision_object_cb on ADD/MOVE), unlike
    draw_springs/draw_frames -- scene objects don't move with joint state,
    so there's nothing to refresh in the main display loop."""
    entry = collision_objects.get(obj_id)
    if entry is None:
        return
    for i, (shape, dims, relative_pose) in enumerate(entry["primitives"]):
        path = f"collision_objects/{obj_id}/{i}"
        T = entry["object_pose"] @ relative_pose
        if shape == "box":
            viz.viewer[path].set_object(g.Box(dims), _COLLISION_OBJECT_MATERIAL)
        else:  # cylinder
            height, radius = dims
            viz.viewer[path].set_object(g.Cylinder(height, radius), _COLLISION_OBJECT_MATERIAL)
            T = T @ _CYLINDER_AXIS_ALIGN
        viz.viewer[path].set_transform(T)



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

# TRANSIENT_LOCAL to match virtual_spring_node's ~/collision_object_state
# publisher -- lets armviz pick up the full current scene even if it starts
# after objects were already loaded (e.g. via the YAML scene loader at
# launch), not just ones added/moved/removed while armviz happens to be
# running.
_collision_state_qos = QoSProfile(
    depth=50,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
node.create_subscription(
    CollisionObject, COLLISION_STATE_TOPIC, collision_object_cb, _collision_state_qos
)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# Bootstrap from parameters, retrying for a bit rather than a single
# immediate attempt: gen3_spring.launch.py starts armviz with no ordering
# guarantee against virtual_spring_node (unlike enable_torque_control/
# load_collision_scene, which explicitly wait), so a one-shot check here
# routinely lost the race against virtual_spring_node still declaring its
# spring_names/springs.* parameters -- and since nothing re-announces the
# startup spring afterward otherwise, losing that race meant it silently
# never appeared. Same ~10s timeout convention as wait_and_enable_torque.py.
BOOTSTRAP_TIMEOUT_SEC = 10.0
_bootstrap_deadline = time.time() + BOOTSTRAP_TIMEOUT_SEC
_found_springs = load_springs_from_params()
while not _found_springs and time.time() < _bootstrap_deadline:
    rclpy.spin_once(node, timeout_sec=0.5)
    _found_springs = load_springs_from_params()
if not _found_springs:
    print(f"[viz] no springs found in params after {BOOTSTRAP_TIMEOUT_SEC:.0f}s -- "
          "will still pick up springs added/removed via ~/springs_updated afterward.")

print("Spring visualizer ready.")
print(f"  Robot URDF:        {URDF}")
print(f"  Joint states:      {JOINT_STATES}")
print(f"  Springs updates:   {SPRINGS_TOPIC}")
print(f"  Collision objects: {COLLISION_STATE_TOPIC}")
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
