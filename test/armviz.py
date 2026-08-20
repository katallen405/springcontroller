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
    parser.add_argument(
        "--zmq-url", default=None,
        help="meshcat zmq_url to attach to (e.g. tcp://127.0.0.1:6001). If a "
             "meshcat-server is already listening there, this process attaches "
             "to its scene instead of spawning a new server on a random port -- "
             "needed so other processes (e.g. a UI iframe) can reach a stable, "
             "predictable URL. Default (None) spawns a fresh server on a "
             "random port, as before.",
    )
    parser.add_argument(
        "--open", action="store_true", default=False,
        help="Open a browser tab pointed at the meshcat viewer on startup. "
             "Off by default when attaching to a pinned --zmq-url server "
             "(e.g. one already opened by a UI iframe or another viewer).",
    )
    parser.add_argument(
        "--caution-threshold", type=float, default=0.05,
        help="Must match virtual_spring_node's caution_threshold param -- "
             "draws a translucent gold halo this far outside each collision "
             "object's own surface, showing the repulsion field's reach "
             "(see get_repulsion_torques()). 0 or negative disables the halo.",
    )
    # Strip ROS-specific args (--ros-args -r ... -p ... etc.) before parsing
    # our own, so this still works cleanly when launched via ros2 launch/run.
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


_args = parse_args()

URDF          = _args.urdf
CAUTION_THRESHOLD = _args.caution_threshold
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

# Names already warned about via draw_springs' "no target yet" message --
# draw_springs runs in the main loop at ~20Hz, so without this it printed
# that line ~20x/sec for as long as a spring's target stayed unresolved
# (e.g. no /joint_states publisher running yet to trigger
# virtual_spring_node's target resolution), flooding the log forever
# instead of once. Cleared if a target later resolves and then somehow
# reverts to None, so a genuine regression still gets reported.
_no_target_warned = set()

# Same dedupe, for link_name staying unresolved (e.g. the robot/spring node
# isn't up yet). Distinct from _unknown_frame_warned below: this is the
# expected "not read yet" case, that one is "read a real value, but it's
# not a frame this URDF has" -- worth keeping separate so a genuine bad
# link_name still gets its own message instead of being swallowed here.
_no_link_warned = set()

# Names already warned about via draw_springs' "unknown frame" message --
# same 20Hz flood risk as _no_target_warned above, but for a link_name that
# *did* resolve, just not to a frame this URDF actually has.
_unknown_frame_warned = set()

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

# Cartesian-target spring types live under different parameter prefixes
# (VirtualSpring: springs.<name>.*, OrientationSpring: orientation_springs.
# <name>.*) but both publish on the same /virtual_spring_node/target/<name>
# topic and both have a meaningful attachment point + target to draw, so
# armviz treats them the same everywhere except this prefix lookup.
SPRING_PARAM_PREFIXES = ["springs", "orientation_springs"]

def _get_spring_param(name, key):
    """Try each known spring-type prefix in turn; return the raw `ros2
    param get` output from whichever one actually has this parameter."""
    for prefix in SPRING_PARAM_PREFIXES:
        try:
            return subprocess.check_output([
                'ros2', 'param', 'get', SPRING_NODE,
                f'{prefix}.{name}.{key}'
            ], timeout=5).decode()
        except Exception:
            continue
    return None

def get_param_target(name):
    """Read initial spring target from the spring node's ROS parameters."""
    out = _get_spring_param(name, 'target')
    if out is None:
        print(f"[viz] could not read target for '{name}'")
        return None
    nums = re.findall(r'[-\d.]+', out.split(':')[-1])
    if len(nums) == 3:
        return np.array([float(x) for x in nums])
    return None

def get_param_link(name):
    """Read initial link_name from the spring node's ROS parameters."""
    out = _get_spring_param(name, 'link_name')
    if out is None:
        print(f"[viz] could not read link_name for '{name}'")
        # None, not a hardcoded fallback -- this script is robot-agnostic
        # (used for both UR3e and Gen3), so guessing a specific robot's
        # frame name here is wrong by construction for whichever robot
        # *isn't* that guess. Confirmed live 2026-08-19 on a Gen3 run: a
        # transient timeout on this one read (competing with
        # virtual_spring_node's own heavy startup for CPU, same root cause
        # _track_spring's docstring already describes for target) left
        # link_name permanently stuck at the old "ur3e_tool0" fallback,
        # since -- unlike target -- nothing ever retried it. Callers must
        # treat None as "not resolved yet", same as target=None.
        return None
    return out.split(':')[-1].strip()

def _track_spring(name):
    """Ensure `name` is present in `springs` with a resolved target,
    (re)trying the target fetch if it's still missing from a previous
    attempt. Creates the runtime ~/target/<name> subscription the first
    time `name` is seen. Returns True once the target is known.

    A spring name can become known (e.g. via spring_names or
    ~/springs_updated) before its target/link_name sub-fetches -- separate
    'ros2 param get' subprocesses, each with their own DDS discovery --
    actually succeed, especially right after virtual_spring_node's own
    heavy startup (URDF/collision model load) when it's competing hardest
    for CPU. Previously a spring already present in `springs` was treated
    as fully handled even with target=None, so a single transient failure
    meant it silently never got a target -- confirmed 2026-08-14:
    tip_spring showed up (arm was genuinely pulling toward it) but never
    appeared in meshcat. Calling this again for an already-tracked name is
    what gives a later retry (bootstrap loop, or another ~/springs_updated
    message) a chance to actually recover from that."""
    entry = springs.get(name)
    if entry is None:
        target    = get_param_target(name)
        link_name = get_param_link(name)
        springs[name] = entry = {
            "target":    target,
            "link_name": link_name,
        }
        # TRANSIENT_LOCAL to match virtual_spring_node's ~/target/<name>
        # publisher -- see _latched_qos below for why a plain VOLATILE
        # subscription here misses a target resolved before this
        # subscription existed.
        sub = node.create_subscription(
            PointStamped,
            f"/virtual_spring_node/target/{name}",
            make_target_cb(name),
            _latched_qos,
        )
        target_subs[name] = sub
        print(f"[viz] tracking spring: '{name}'  link={link_name}  target={target}")
    else:
        if entry.get("target") is None:
            entry["target"] = get_param_target(name)
            print(f"[viz] retried target for '{name}': {entry['target']}")
        if entry.get("link_name") is None:
            entry["link_name"] = get_param_link(name)
            print(f"[viz] retried link_name for '{name}': {entry['link_name']}")
    return entry.get("target") is not None and entry.get("link_name") is not None

def load_springs_from_params():
    """Bootstrap spring list from ROS params at startup. Returns True only
    once every known spring name (Cartesian or orientation) also has a
    target -- see _track_spring."""
    any_found = False
    all_resolved = True
    for names_param in ("spring_names", "orientation_spring_names"):
        try:
            out = subprocess.check_output([
                'ros2', 'param', 'get', SPRING_NODE, names_param
            ], timeout=5).decode()
            # output: "String values are: ['tip_spring', 'elbow_spring']"
            names = re.findall(r"'([^']+)'", out)
        except Exception as e:
            print(f"[viz] could not read {names_param}: {e}")
            continue
        if not names:
            print(f"[viz] no springs found in {names_param} yet")
            continue
        any_found = True
        print(f"[viz] found springs from params ({names_param}): {names}")
        # List comprehension, not all(gen) -- must attempt every name
        # (each with its own retry) rather than short-circuiting on the
        # first one still missing a target.
        if not all([_track_spring(name) for name in names]):
            all_resolved = False
    return any_found and all_resolved
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

    # Add new springs, and retry any already-tracked one whose target
    # fetch previously failed and never got picked up again (see
    # _track_spring) -- a later ~/springs_updated message is otherwise the
    # only other chance after the bootstrap retry loop gives up.
    for name in active_names:
        if name not in current_names or springs[name].get("target") is None:
            _track_spring(name)

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

# Same gold as the study UI's .val.CAUTION (springcontroller_ui/web/index.html).
# Wireframe, not a filled translucent volume -- two overlapping transparent
# meshes (this halo and the solid object's own semi-transparent material)
# don't reliably depth-sort in WebGL, so a low-opacity filled halo could
# render behind/be swallowed by the solid object depending on draw order.
# A wireframe has no fill to fight over, so it stays visible regardless.
_CAUTION_HALO_MATERIAL = g.MeshBasicMaterial(
    color=0xd4a017, wireframe=True, opacity=0.7, transparent=True
)

def draw_collision_object(obj_id):
    """(Re)draw every primitive of a tracked collision object at its current
    pose. Event-driven (called from collision_object_cb on ADD/MOVE), unlike
    draw_springs/draw_frames -- scene objects don't move with joint state,
    so there's nothing to refresh in the main display loop.

    Also draws a translucent gold "halo" copy of each primitive, padded
    outward by CAUTION_THRESHOLD -- get_repulsion_torques() (see
    urdf_arm_configuration.py) isn't a spatial vector field sampled at many
    points; at any instant it's a single force computed from whichever point
    on the robot and obstacle are currently nearest each other, direction
    away from the obstacle's nearest point, magnitude ramping from 0 at
    caution_threshold up to full strength at danger_threshold. The one
    stable *shape* in that picture is this offset shell -- the boundary
    where the field can start acting at all -- so that's what's drawn,
    rather than an arrow field that would imply forces exist at points the
    math never actually evaluates.
    """
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

        if CAUTION_THRESHOLD > 0.0:
            halo_path = f"collision_objects/{obj_id}/{i}/halo"
            pad = CAUTION_THRESHOLD
            if shape == "box":
                halo_dims = [d + 2 * pad for d in dims]
                viz.viewer[halo_path].set_object(g.Box(halo_dims), _CAUTION_HALO_MATERIAL)
            else:  # cylinder -- T already includes _CYLINDER_AXIS_ALIGN above
                height, radius = dims
                viz.viewer[halo_path].set_object(
                    g.Cylinder(height + 2 * pad, radius + pad), _CAUTION_HALO_MATERIAL
                )
            # Child of the solid primitive's own path, so it inherits T
            # automatically -- no separate set_transform needed.



def draw_springs(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    for name, spring in springs.items():
        target    = spring.get("target")
        link_name = spring.get("link_name")

        if link_name is None:
            # Not resolved yet (e.g. the spring node isn't up yet) -- an
            # expected, transient state that _track_spring keeps retrying,
            # not a bug. Skip this spring for now, same as an unresolved
            # target below.
            if name not in _no_link_warned:
                _no_link_warned.add(name)
                print(f"[viz] no link_name yet for spring '{name}' -- "
                      f"waiting on virtual_spring_node's parameters")
            continue
        _no_link_warned.discard(name)

        # Attachment point from FK
        frame_id = model.getFrameId(link_name)
        if frame_id >= len(data.oMf):
            if name not in _unknown_frame_warned:
                _unknown_frame_warned.add(name)
                print(f"[viz] unknown frame '{link_name}' for spring '{name}'")
            continue
        _unknown_frame_warned.discard(name)
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
            _no_target_warned.discard(name)
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
        elif name not in _no_target_warned:
            _no_target_warned.add(name)
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

# TRANSIENT_LOCAL to match virtual_spring_node's ~/springs_updated,
# ~/target/<name>, and ~/collision_object_state publishers -- a subscriber
# only gets a TRANSIENT_LOCAL publisher's already-published history if it
# is *also* TRANSIENT_LOCAL (DDS durability replay is per-subscription, not
# automatic). With plain VOLATILE subscriptions here, a late-starting
# armviz.py never received the one-shot springs_updated/target broadcasts
# virtual_spring_node already sent before it connected, and fell back to
# polling `ros2 param get` (load_springs_from_params' bootstrap retry loop)
# -- or, if that also lost the race, spammed "no target yet" forever.
_latched_qos = QoSProfile(
    depth=50,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)
node.create_subscription(String, SPRINGS_TOPIC, springs_updated_cb, _latched_qos)
node.create_subscription(
    CollisionObject, COLLISION_STATE_TOPIC, collision_object_cb, _latched_qos
)

viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=_args.open, zmq_url=_args.zmq_url)
viz.loadViewerModel()

# Bootstrap from parameters, retrying for a bit rather than a single
# immediate attempt: gen3_spring.launch.py starts armviz with no ordering
# guarantee against virtual_spring_node (unlike enable_torque_control/
# load_collision_scene, which explicitly wait), so a one-shot check here
# routinely lost the race against virtual_spring_node still declaring its
# spring_names/springs.* parameters -- and since nothing re-announces the
# startup spring afterward otherwise, losing that race meant it silently
# never appeared. 20s (vs. the ~10s convention elsewhere, e.g.
# wait_and_enable_torque.py) because load_springs_from_params() itself
# retries per-name target/link_name lookups too -- each is its own
# subprocess 'ros2 param get' call (own DDS discovery each time), so a
# single bootstrap round can burn several seconds on its own, especially
# competing with virtual_spring_node's own heavy startup for CPU.
BOOTSTRAP_TIMEOUT_SEC = 20.0
_bootstrap_deadline = time.time() + BOOTSTRAP_TIMEOUT_SEC
_found_springs = load_springs_from_params()
while not _found_springs and time.time() < _bootstrap_deadline:
    rclpy.spin_once(node, timeout_sec=0.5)
    _found_springs = load_springs_from_params()
if not _found_springs:
    print(f"[viz] springs not fully resolved from params after "
          f"{BOOTSTRAP_TIMEOUT_SEC:.0f}s (see [viz] messages above for which "
          "name/field) -- will still pick up further changes via "
          "~/springs_updated afterward.")

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
