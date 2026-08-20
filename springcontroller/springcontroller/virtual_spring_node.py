#!/home/katallen/.springcontroller_venv/bin/python3
"""
virtual_spring_node.py

ROS 2 node that maintains a collection of virtual springs and publishes
the resulting joint torques on each JointState update from the arm.

Subscriptions
-------------
~/joint_states  (sensor_msgs/JointState)
    Current joint positions and velocities from the arm.

~/target/<spring_name>  (geometry_msgs/PointStamped)
    Move a Cartesian or orientation spring's target at runtime.

~/attachment/<spring_name>  (geometry_msgs/PointStamped)
    Move a Cartesian or orientation spring's attachment point at runtime.

~/joint_target/<joint_spring_name>  (std_msgs/Float64)
    Move a JointSpring's target angle (rad) at runtime.

<collision_object_topic>  (moveit_msgs/CollisionObject)
    Default ~/collision_object. ADD/APPEND/REMOVE/MOVE scene collision
    objects for the self-collision hard clamp -- same wire format MoveIt's
    own planning scene uses.

<torque_status_topic>  (std_msgs/String)
    Only subscribed if the parameter is non-empty. "ENABLED"/"DISABLED",
    e.g. from gen3_torque_control -- drives the spring-force ramp-in on
    each torque-enable.

/robot_description  (std_msgs/String, TRANSIENT_LOCAL)
    Startup only: preferred URDF source (falls back to the urdf_path
    parameter if nothing arrives within a few seconds); subscription is
    torn down immediately after.

Publications
------------
~/joint_torques  (sensor_msgs/JointState)
    Effort field carries the summed virtual-spring torques.

~/target/<spring_name>  (geometry_msgs/PointStamped)
    Broadcasts a Cartesian or orientation spring's current target -- once
    at load (if configured), on resolving a deferred one against the arm's
    real pose (see springs.<n>.target / orientation_springs.<n>.target
    below) on the first /joint_states callback, and on add_spring/
    add_orientation_spring -- and again on every runtime update via the
    subscription above.

~/springs_updated (std_msgs/String)
    Publishes the current spring name list (JSON, any type) whenever it
    changes via add_spring/add_joint_spring/add_orientation_spring/
    remove_spring (supports spring_viz, maybe later UI).

~/safety_status (std_msgs/String)
    Current self/scene-collision state, prefixed "SAFE"/"DANGER"/
    "COLLISION" -- published every /joint_states callback, same cadence as
    the collision check itself. Lets a consumer (e.g.
    wait_and_enable_torque.py) check the resting pose before torque-enable
    without duplicating the collision model.

Services
--------
~/enable   (std_srvs/SetBool)
    Enable or disable all springs at once.

~/set_gravity_compensation  (std_srvs/SetBool)
    Toggle add_gravity_compensation at runtime.

~/set_repulsion_enabled  (std_srvs/SetBool)
    Toggle the repulsion_enabled field (see get_repulsion_torques) at
    runtime -- ros2 param set has no effect on it, same as every other
    parameter here; this is the only way to flip it without restarting.

~/update_collision_thresholds  (springcontroller_interfaces/UpdateCollisionThresholds)
    Live-tune danger_threshold and caution_threshold together (rejects the
    request if caution_threshold isn't strictly greater than
    danger_threshold). Same "ros2 param set has no effect" reasoning as
    set_repulsion_enabled above.

~/add_spring  (springcontroller_interfaces/AddSpring)
    Add a Cartesian spring.

~/add_joint_spring  (springcontroller_interfaces/AddJointSpring)
    Add a joint spring.

~/add_orientation_spring  (springcontroller_interfaces/AddOrientationSpring)
    Add an orientation spring.

~/remove_spring (springcontroller_interfaces/RemoveSpring)
    Remove a spring (any type, name-based) from the torque calculations.

~/load_collision_scene  (springcontroller_interfaces/LoadCollisionScene)
    Load a static scene-collision YAML (see config/*_collision_scene.yaml)
    through the same path <collision_object_topic> uses.


Parameters
----------
urdf_path : str
    Fallback URDF/XACRO path, used only if /robot_description doesn't
    arrive within a few seconds of startup.
config_path : str
    Path to the springs YAML file (checked at startup; populates the
    springs.<n>.* / joint_springs.<n>.* / orientation_springs.<n>.*
    parameters below).
publish_rate_hz : double
    Declared but currently unused -- torques publish once per
    /joint_states callback, not on a timer.
plot_output_path : str
    Where plot_spring_extensions() saves its figure on shutdown.
plot_on_shutdown : bool
    If true, plot spring extensions/torques when the node shuts down.
add_gravity_compensation : bool
    Add pinocchio gravity torques to the published command -- true for
    arms whose own controller doesn't already do it (e.g. Gen3 via
    gen3_torque_control), false where it does (e.g. UR3e).
recentering_threshold_rad : double
    Equilibrium-shift threshold (see _check_equilibrium_shift) above which
    re-centering triggers, if recentering_enabled.
recentering_enabled : bool
    Off by default -- the re-centering sequence needs ros2_control's
    controller_manager and a real equilibrium_mover package, neither of
    which apply to Gen3 (and it's unfinished even for UR3e). When off, a
    large shift is only logged, springs are left alone.
srdf_path : str
    Optional SRDF path excluding known-adjacent-link pairs from the
    self-collision model (pin.removeCollisionPairs); without it, adjacent
    links will false-positive as self-collisions.
danger_threshold : double
    Self-collision distance (m) below which torques start scaling down.
locked_joint_names : string[]
    Joints excluded from the arm model's active DOF.
torque_disable_service : str
    SetBool service called to take the arm out of torque control if
    spring computation fails repeatedly. Empty (default) disables this
    fail-safe.
collision_object_topic : str
    Topic for live scene collision objects (default ~/collision_object).
torque_status_topic : str
    "ENABLED"/"DISABLED" status topic driving the spring-force ramp-in.
    Empty (default) disables ramping entirely -- spring_scale stays 1.0.
spring_ramp_duration_sec : double
    Ramp duration for spring-force torque after torque-enable. Gravity
    comp is never ramped (see compute_total_torques).
collision_check_interval_sec : double
    Throttle interval for the self-collision check; the torque-scaling
    decision still runs every cycle, against the most recent result.
collision_log_path : str
    CSV log of collision-check events, flushed incrementally. Empty
    disables logging.
spring_names, springs.<n>.* : per-VirtualSpring config (see
    config/springs.yaml) -- link_name, local_point, target, stiffness,
    damping, rest_length, inner_radius, outer_radius. `target` left
    unconfigured (NaN sentinel) defers to the arm's real position on the
    first /joint_states callback rather than resolving against the URDF's
    zero/neutral pose at __init__ time.
joint_spring_names, joint_springs.<n>.* : per-JointSpring config --
    joint_name, target_angle, stiffness, damping. `target_angle` left
    unconfigured defers the same way, to the joint's real current angle.
orientation_spring_names, orientation_springs.<n>.* : per-OrientationSpring
    config -- link_name, local_point, local_face_normal, target, stiffness,
    damping. `target` has no safe default (it's an external point, e.g. a
    person's face, not inferable from the arm's pose) and must be set
    explicitly.
"""

import csv
import math
import os

#from typeguard import value

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np
import pinocchio as pin
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PointStamped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import SetBool
from std_msgs.msg import String, Float64, Bool, Float64MultiArray
import os
import threading

from springcontroller.virtual_spring import (
    VirtualSpring, JointSpring, OrientationSpring, SpringCollection,
)
from springcontroller.urdf_arm_configuration import URDFArmConfiguration
import yaml
from springcontroller_interfaces.srv import (
    AddSpring, RemoveSpring, AddJointSpring, AddOrientationSpring,
    LoadCollisionScene, CheckCollisionAtAngles, UpdateSpring,
    UpdateCollisionThresholds,
)


import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _pose_to_se3(pose: Pose) -> pin.SE3:
    """geometry_msgs/Pose -> pin.SE3. Note the field-order swap: ROS
    quaternions are (x, y, z, w) but pin.Quaternion takes (w, x, y, z)."""
    q = pin.Quaternion(
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
    )
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    return pin.SE3(q, t)


def fetch_robot_description(node, timeout_sec: float = 3.0) -> str | None:
    """
    Try to get the URDF from /robot_description (published by robot_state_publisher,
    used by RViz). Returns the XML string, or None if nothing arrives in time.

    Uses TRANSIENT_LOCAL durability so we receive the message even if we
    subscribe after it was published.
    """
    description = None

    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    def cb(msg: String) -> None:
        nonlocal description
        description = msg.data

    sub = node.create_subscription(String, "/robot_description", cb, qos)
    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while description is None and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return description


class VirtualSpringNode(Node):

    def __init__(self):
        super().__init__("virtual_spring_node")

        # Parameters
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("config_path", "")
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("plot_output_path", "/home/kat/spring_extensions.png")
        self.declare_parameter("plot_on_shutdown", False)
        self.declare_parameter("add_gravity_compensation", False)
        self.declare_parameter("recentering_threshold_rad", 0.3)
        # Off by default: the re-centering sequence (switch to position
        # control, run equilibrium_mover, switch back) depends on
        # ros2_control's controller_manager and a real equilibrium_mover
        # package -- neither applies to Gen3, and even on UR3e it currently
        # points at a placeholder package name ("your_study_pkg") that was
        # never filled in. Until that's actually finished and verified,
        # this stays opt-in so a large equilibrium shift never silently
        # disables springs on a robot/setup that can't complete the
        # sequence. When off, the shift is still computed and logged.
        self.declare_parameter("recentering_enabled", False)
        self.declare_parameter("srdf_path", "")
        # 1cm, not 5cm: 5cm was confirmed too conservative for close-quarters
        # work 2026-08-14/2026-08-19 (see gen3_collision_clamp_debug project
        # memory) -- scaling starting a full 5cm out interfered with normal
        # spring operation well before anything was actually at risk.
        self.declare_parameter("danger_threshold", 0.01)
        # Low-strength repulsion field around scene collision objects --
        # separate from (and additive on top of) the danger_threshold clamp
        # above, which only ever removes spring torque. Off by default until
        # verified on hardware: this is a new autonomous force, and per
        # collision_recovery_torque_ramp project memory, new autonomous
        # collision-adjacent behavior on this arm goes through an explicit
        # opt-in before it runs unattended.
        self.declare_parameter("repulsion_enabled", False)
        # Must be > danger_threshold -- the ramp runs from caution_threshold
        # (force starts at 0) down to danger_threshold (force saturates at
        # repulsion_max_force_n). 5cm was the project's old danger_threshold
        # default before it was tightened to 1cm for close-quarters spring
        # work (see gen3_collision_clamp_debug project memory) -- reused here
        # as a reasonable "starting to notice an obstacle" radius for a much
        # gentler force than the clamp ever applied.
        self.declare_parameter("caution_threshold", 0.05)
        # Small Cartesian force (N) applied at the near-collision point,
        # mapped to joint torques via Jacobian-transpose -- see
        # get_repulsion_torques(). Chosen to stay well under
        # gen3_torque_control's torque_limit_nm even at the smallest (wrist)
        # joints' typical lever arms; raise cautiously and re-check against
        # that limit if this default proves too weak on hardware.
        self.declare_parameter("repulsion_max_force_n", 5.0)
        # repulsion_max_force_n bounds each *pair*'s own Cartesian force,
        # not the resulting joint torque -- a contact far out on the
        # gripper already has a large lever arm, and several simultaneous
        # contacts (e.g. a multi-geometry gripper wrapping a curved
        # obstacle) sum on top of that. Confirmed live 2026-08-19: a
        # gripper closing around a cylinder scene object produced summed
        # |tau| of 20-36 N*m, well past "low-strength" and close to the
        # wrist joints' torque_limit_nm=9. 5.0 sits a bit above what a
        # single max-force contact alone typically produces near the
        # shoulder (so a normal single-contact case isn't neutered) while
        # strongly bounding the multi-contact pileup that caused that --
        # tune down further if it still feels strong with several
        # simultaneous contacts.
        self.declare_parameter("repulsion_max_total_torque_nm", 5.0)
        self.declare_parameter("locked_joint_names", [""])
        self.declare_parameter("torque_disable_service", "")
        # If springs stay auto-disabled by the collision clamp this long
        # without an explicit ~/enable(true), escalate to a full torque
        # disable (via torque_disable_service) instead of leaving the arm on
        # gravity-comp-only indefinitely. Gravity comp alone isn't a true
        # hold -- it slowly drifts (see torque_ramp_gravity_comp_lesson
        # project memory) -- so an operator who clears a collision, then
        # walks to the UI to re-enable, can come back to find the arm has
        # drifted right back into what it just cleared. Long enough to not
        # fire during a normal "move it clear, walk over, click enable"
        # cycle; short enough to bound how long unattended drift can run.
        self.declare_parameter("collision_disable_grace_sec", 30.0)
        self.declare_parameter("collision_object_topic", "~/collision_object")
        # Ramp only spring-force torque in over spring_ramp_duration_sec
        # after torque control actually gets enabled on the arm (tracked
        # via torque_status_topic, e.g. gen3_torque_control's
        # /gen3_torque_control/status "ENABLED"/"DISABLED"). Gravity comp
        # is deliberately never ramped -- see compute_total_torques.
        # Empty topic (default) disables this entirely: spring_scale stays
        # 1.0 always, matching prior behavior (e.g. UR3e, which has no
        # such status topic).
        self.declare_parameter("torque_status_topic", "")
        self.declare_parameter("spring_ramp_duration_sec", 1.5)
        # get_collision_status() offline-profiled at ~126ms/call
        # (2026-08-07) -- called every _joint_state_cb, it was blocking the
        # control loop down to ~8-21Hz instead of the intended ~100Hz,
        # the likely dominant cause of the day's command-staleness/fault
        # incidents (far more than the network-flakiness theory chased
        # earlier). Self-collision doesn't need 100Hz freshness to still
        # catch a developing collision well before contact, so it's
        # throttled to collision_check_interval_sec instead of running
        # every cycle; the torque-scaling decision still applies on every
        # cycle, just using the most recent check (up to that interval
        # stale) rather than skipping the safety logic between checks.
        self.declare_parameter("collision_check_interval_sec", 0.05)
        self._collision_check_interval = (
            self.get_parameter("collision_check_interval_sec").get_parameter_value().double_value
        )
        self._last_collision_check_time = None
        self._last_collision_status = None
        self._repulsion_enabled = self.get_parameter("repulsion_enabled").get_parameter_value().bool_value
        self._caution_threshold = self.get_parameter("caution_threshold").get_parameter_value().double_value
        self._repulsion_max_force_n = self.get_parameter("repulsion_max_force_n").get_parameter_value().double_value
        self._repulsion_max_total_torque_nm = (
            self.get_parameter("repulsion_max_total_torque_nm").get_parameter_value().double_value
        )
        # Cached alongside _last_collision_status -- recomputed at the same
        # throttled cadence (collision_check_interval_sec) since it reuses
        # the same pin.computeDistances results get_collision_status() just
        # populated. None until repulsion_enabled and the first throttled
        # check has run.
        self._last_repulsion_torques = None
        # Whether springs are currently disabled *because the hard collision
        # clamp disabled them* (as opposed to an operator explicitly calling
        # ~/enable(false) themselves) -- an edge-trigger guard so the
        # in_collision branch below only calls _set_springs_enabled(False)
        # once per collision episode, not every cycle. Deliberately requires
        # an explicit ~/enable(true) to clear (see _enable_cb) -- no
        # automatic recovery. A same-day-earlier version of this ramped
        # spring torque back in automatically over spring_ramp_duration_sec
        # as soon as the pair cleared the danger zone; confirmed live
        # 2026-08-19 that this let an orientation spring left ~180deg off
        # (arm manually walked out of a collision while its own torque was
        # suppressed) re-engage at full strength the instant the pair
        # cleared, producing a large, axis-sensitive correction torque that
        # drove joint 4 into a fault trying to close on link 5 -- automatic
        # recovery from a hard collision clamp is exactly the failure mode
        # to avoid; re-enabling is now a deliberate operator action.
        self._springs_auto_disabled = False
        # Timestamp of when the collision clamp most recently auto-disabled
        # springs (None while not auto-disabled) -- drives the grace-period
        # escalation to a full torque disable, see collision_disable_grace_sec.
        self._springs_auto_disabled_since = None
        self._collision_disable_grace_sec = (
            self.get_parameter("collision_disable_grace_sec").get_parameter_value().double_value
        )
        # Timestamp of the most recent explicit ~/enable(true) call -- ramps
        # spring torque back in smoothly over spring_ramp_duration_sec from
        # THAT moment (same duration/curve as the torque-enable ramp below,
        # just keyed off this event too), so a manual re-enable after a
        # collision-triggered disable doesn't itself snap to full strength
        # instantly.
        self._springs_reenabled_since = None
        self._add_grav_comp = self.get_parameter("add_gravity_compensation").get_parameter_value().bool_value
        self._recentering_threshold = self.get_parameter("recentering_threshold_rad").get_parameter_value().double_value
        self._recentering_enabled = self.get_parameter("recentering_enabled").get_parameter_value().bool_value
        self._recentering_in_progress = False

        # Fail-safe: on repeated spring-computation failure, switch the arm
        # out of torque control entirely (e.g. back to Kortex position hold)
        # rather than trust our own possibly-broken torque computation.
        # Left unset (default "") for arms like the UR3e where torque mode
        # is driven by ros2_control rather than a SetBool enable service.
        torque_disable_service = (
            self.get_parameter("torque_disable_service").get_parameter_value().string_value
        )
        self._torque_disable_client = None
        if torque_disable_service:
            self._torque_disable_client = self.create_client(
                SetBool, torque_disable_service
            )
        self._failsafe_triggered = False
        # Own one-shot guard for the collision-clamp grace-period escalation
        # (see _trigger_collision_disable_escalation) -- reset per episode
        # in _set_springs_enabled, unlike _failsafe_triggered above.
        self._collision_escalation_triggered = False

        # Collision-event log: written incrementally (flushed every row, not
        # just on clean shutdown) so the real closest_pair/in_collision/
        # in_danger sequence survives even a hard crash or e-stop power
        # loss -- needed to root-cause the rclpy throttled-logger ValueError
        # seen live 2026-08-05 (couldn't be reproduced from a guessed
        # sequence; need the real one). Empty string disables it.
        self.declare_parameter(
            "collision_log_path", os.path.expanduser("~/springcontroller_collision_log.csv")
        )
        collision_log_path = os.path.expanduser(
            self.get_parameter("collision_log_path").get_parameter_value().string_value
        )
        self._collision_log_file = None
        self._collision_log_writer = None
        self._collision_log_last_flush = -1.0
        # Manual time-gates for the collision/danger log lines below, in
        # place of rclpy's throttle_duration_sec= kwarg -- that mechanism's
        # per-call-site filter cache has been observed live (2026-08-05,
        # 2026-08-13) to raise "Requested logging filters cannot be changed
        # between calls" on effectively every call at this control-loop
        # rate, defeating the throttle entirely and logging at full ~100Hz
        # instead of the intended rate. A simple elapsed-time comparison
        # (same pattern as _collision_log_last_flush above) sidesteps that
        # cache altogether.
        self._collision_warn_log_last = -1.0
        self._repulsion_warn_log_last = -1.0
        if collision_log_path:
            try:
                write_header = not os.path.isfile(collision_log_path)
                self._collision_log_file = open(collision_log_path, "a", newline="")
                self._collision_log_writer = csv.writer(self._collision_log_file)
                if write_header:
                    self._collision_log_writer.writerow([
                        "elapsed_s", "kind", "in_collision", "in_danger",
                        "link_a", "link_b", "min_distance", "scale_factor",
                        "log_call_failed",
                    ])
                    self._collision_log_file.flush()
            except OSError as e:
                self.get_logger().error(f"Could not open collision log at {collision_log_path}: {e}")
                self._collision_log_file = None
                self._collision_log_writer = None

        # Check config file exists before doing anything else
        config_path = os.path.expanduser(
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        self._config_path = config_path

        if config_path:
            if not os.path.isfile(config_path):
                self.get_logger().fatal(
                    f"Config file not found: '{config_path}'\n"
                    f"Check the path passed to config:= in your launch command."
                )
                raise RuntimeError(f"Config file not found: {config_path}")
            self.get_logger().info(f"Config file found: {config_path}")

            # Actually load the YAML and set parameters
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Navigate to ros__parameters if using standard ROS2 YAML layout
            params = config
            if "/**" in params:
                params = params["/**"]["ros__parameters"]
            elif list(params.keys())[0].endswith("ros__parameters"):
                params = list(params.values())[0]
            
            # Set each parameter so _load_springs_from_params can find them
            for key, value in self._flatten_dict(params).items():
                try:
                    self.declare_parameter(key, value)
                except rclpy.exceptions.ParameterAlreadyDeclaredException:
                    self.set_parameters([rclpy.parameter.Parameter(key, value=value)])
        else:
            self.get_logger().warn(
                "No config_path set -- spring definitions must be "
                "provided via ROS parameters directly."
            )

        # Load URDF — prefer /robot_description topic, fall back to urdf_path param
        self.get_logger().info("Loading URDF...")
        try:
            self._arm = self._load_arm()
        except Exception as e:
            self.get_logger().fatal(f"Failed to load URDF: {e}")
            raise RuntimeError(f"Failed to load URDF: {e}") from e

        self._joint_order = None

        self.get_logger().info(
            f"URDF loaded. {self._arm.n_dof} DOF. "
            f"Available frames:\n  " + "\n  ".join(sorted(self._arm.link_names))
        )
        self.get_logger().info(
            f"nq={self._arm.n_q}, nv={self._arm.n_dof}\n  " +
            "\n  ".join(f"{i}: {self._arm.model.names[i]}" for i in range(self._arm.model.njoints))
            )

        # Jacobian Debugging
        for name in self._arm.link_names:
            T = self._arm.get_link_transform(name)
            self.get_logger().debug(f"{name}: {T[:3, 3]}")

        # Spring collection
        self._springs = SpringCollection()

        # Created here (ahead of the rest of the "Publishers" block below)
        # so _load_springs_from_params can publish the initial spring set
        # once it's done loading -- every other path that changes
        # self._springs (_add_spring_cb/_add_joint_spring_cb/_remove_spring_cb)
        # already does this; the initial config-driven load was the one
        # silent exception, which meant a consumer that only learns about
        # springs via this topic (e.g. armviz.py, if its own one-shot
        # parameter bootstrap already lost the startup race) could
        # permanently miss whatever was loaded from config at startup.
        # TRANSIENT_LOCAL: this is "current full spring list", republished
        # in full on every add/remove, not a stream -- a late subscriber
        # (e.g. armviz.py) should see the latest list immediately instead
        # of falling back to polling `ros2 param get` for up to 20s (see
        # armviz.py's BOOTSTRAP_TIMEOUT_SEC) or missing it outright.
        springs_updated_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._springs_updated_pub = self.create_publisher(
            String, "~/springs_updated", springs_updated_qos
        )

        # TRANSIENT_LOCAL for the same reason as springs_updated above --
        # this is current state, not a stream, so a UI that connects (or
        # reconnects/reloads) after the last enable/disable event should see
        # the real current value immediately rather than showing "unknown"
        # until the next state change happens to occur. Confirmed live
        # 2026-08-20: the study UI's springs-state had no authoritative
        # source at all before this -- it only ever showed whatever the
        # browser's own last ~/enable call optimistically claimed, so a
        # fresh page load (or one where that call was never made this
        # session) just stayed "unknown" forever, even though the real
        # springs were actively enabled or disabled the whole time.
        self._springs_enabled_pub = self.create_publisher(
            Bool, "~/springs_enabled", springs_updated_qos
        )

        self._load_springs_from_params()
        self._publish_springs_updated()
        self._publish_springs_enabled()

        # Maps a CollisionObject.id to [(internal_object_id, relative_pose), ...]
        # -- one entry per primitive -- so REMOVE/MOVE messages, which only
        # carry the top-level id and (for MOVE) a new base pose, can be
        # fanned out to the right environment objects. relative_pose is each
        # primitive's pose relative to the CollisionObject's own pose, kept
        # around so MOVE can recompute world poses without needing the
        # primitives resent (its contract leaves them empty).
        self._collision_object_frames: dict[str, list[tuple[str, "pin.SE3"]]] = {}

        # Publishers
        self._torque_pub = self.create_publisher(
            JointState, "~/joint_torques", 10
        )
        # Repulsion-only component, separate from the combined ~/joint_torques
        # output -- there was previously no way to see this in a rosbag, only
        # a throttled (0.5s) terminal WARN with just the total norm. Same
        # JointState/effort shape as ~/joint_torques so existing tooling
        # (PlotJuggler, etc.) handles it the same way. Always published
        # (zeros when repulsion_enabled is off or nothing's within
        # caution_threshold) for a continuous signal to plot against, rather
        # than a sparse one that only appears while nonzero.
        self._repulsion_torque_pub = self.create_publisher(
            JointState, "~/repulsion_torques", 10
        )

        # (~/springs_updated publisher created earlier, above the
        # SpringCollection/_load_springs_from_params setup -- see there.)

        # Lets external tools (e.g. wait_and_enable_torque.py's pre-flight
        # check) see the current self/scene-collision state without
        # duplicating the URDF/pinocchio collision model themselves --
        # published every _joint_state_cb regardless of torque-enable
        # state, same as the collision check itself.
        self._safety_status_pub = self.create_publisher(String, "~/safety_status", 10)
        # World-frame witness points [ax,ay,az,bx,by,bz] for the same
        # closest_pair ~/safety_status describes, for visualizing the actual
        # detected closest distance/location (e.g. a line in armviz) instead
        # of inferring it from geometry alone -- confirmed live 2026-08-20
        # this was a real gap when a stale caution-halo size made a
        # perfectly correct CAUTION reading look wrong at a glance. Empty
        # data means no collision model loaded / nothing to report, same
        # convention as ~/safety_status's "no collision pairs reported".
        self._closest_points_pub = self.create_publisher(
            Float64MultiArray, "~/closest_collision_points", 10
        )

        # Per-spring ~/target/<name> publishers (broadcast side -- see the
        # subscription of the same name below for the input/override side).
        self._target_pubs: dict[str, "rclpy.publisher.Publisher"] = {}

        # Per-spring ~/target/<name> publishers (broadcast side -- see the
        # subscription of the same name below for the input/override side).
        self._target_pubs: dict[str, "rclpy.publisher.Publisher"] = {}

        # Subscriptions
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )

        self._add_spring_srv = self.create_service(
            AddSpring, "~/add_spring", self._add_spring_cb
)

        self._remove_spring_srv = self.create_service(
            RemoveSpring, "~/remove_spring", self._remove_spring_cb
)
        # Removal/update are shared: both are name-based and don't care
        # about spring type, so they work for joint/orientation springs too.
        self._update_spring_srv = self.create_service(
            UpdateSpring, "~/update_spring", self._update_spring_cb
        )
        self._add_joint_spring_srv = self.create_service(
            AddJointSpring, "~/add_joint_spring", self._add_joint_spring_cb
        )
        self._add_orientation_spring_srv = self.create_service(
            AddOrientationSpring, "~/add_orientation_spring", self._add_orientation_spring_cb
        )

        # Scene collision objects (safety hard-clamp): same wire format
        # MoveIt's own planning scene uses, so an existing perception/MoveIt
        # publisher interoperates without any extra glue. ADD/APPEND/REMOVE/
        # MOVE are handled the same way MoveIt itself interprets them.
        collision_object_topic = (
            self.get_parameter("collision_object_topic").get_parameter_value().string_value
        )
        self._collision_object_sub = self.create_subscription(
            CollisionObject, collision_object_topic, self._collision_object_cb, 10
        )
        self.get_logger().info(
            f"Listening for scene collision objects on {collision_object_topic}"
        )

        # Echoes every applied ADD/APPEND/MOVE/REMOVE (from both the live
        # topic above and the YAML scene loader) so a late-joining consumer
        # (e.g. armviz.py) can reconstruct current scene state without
        # sharing this node's in-process URDFArmConfiguration. TRANSIENT_LOCAL
        # with enough depth to cover a scene's whole operation history lets a
        # subscriber that joins after the fact replay it and end up in the
        # same state -- same trick as fetch_robot_description() above, just
        # with more than 1 message of history since scene state isn't a
        # single latest-value topic.
        collision_state_qos = QoSProfile(
            depth=50,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._collision_object_state_pub = self.create_publisher(
            CollisionObject, "~/collision_object_state", collision_state_qos
        )

        self._load_collision_scene_srv = self.create_service(
            LoadCollisionScene, "~/load_collision_scene", self._load_collision_scene_cb
        )
        self._check_collision_at_srv = self.create_service(
            CheckCollisionAtAngles, "~/check_collision_at", self._check_collision_at_cb
        )

        # Spring-force ramp-in, keyed to the arm's actual torque-enable
        # event rather than this node's own startup -- virtual_spring_node
        # is typically already running (and steady) for seconds before
        # torque control is enabled on the arm, so a ramp based on our own
        # start_time wouldn't line up with the real transient.
        self._spring_ramp_duration = (
            self.get_parameter("spring_ramp_duration_sec").get_parameter_value().double_value
        )
        self._torque_enabled_since = None
        self._last_torque_status = None
        torque_status_topic = (
            self.get_parameter("torque_status_topic").get_parameter_value().string_value
        )
        self._torque_status_topic = torque_status_topic
        # Liveness of gen3_torque_control (or equivalent) itself, distinct
        # from _last_torque_status above: that cache only updates on an
        # ENABLED/DISABLED *transition* (kinova_torque_control_node
        # publishes ~/status on state change, not periodically), so it
        # goes stale silently if the node dies -- e.g. it could sit
        # showing "ENABLED" forever after a crash, with nothing in this
        # node or the UI any the wiser. count_publishers() reflects live
        # DDS discovery instead, so it reports 0 the instant the
        # publisher's process actually disappears, whether or not a
        # status message was ever published around that time.
        #
        # Checked on its own timer, NOT from _joint_state_cb: on Gen3,
        # kinova_torque_control_node is itself the source of /joint_states
        # (republished under its own -r joint_states:=... remap), so
        # _joint_state_cb stops firing the instant that node dies -- a
        # check placed there would need the dead node's own messages to
        # notice it's dead. Confirmed live 2026-08-19: with the check
        # inline in _joint_state_cb, killing kinova_torque_control froze
        # virtual_spring_node's whole control loop (including
        # _publish_safety_status) along with it, so neither the liveness
        # flag nor safety_status ever updated again.
        self._torque_control_connected = None  # None until first checked
        if torque_status_topic:
            self.create_subscription(
                String, torque_status_topic, self._torque_status_cb, 10
            )
            self.create_timer(1.0, self._check_torque_control_connected)
            self.get_logger().info(
                f"Ramping spring force over {self._spring_ramp_duration:.2f}s on each "
                f"torque-enable, keyed off {torque_status_topic}."
            )
        # Stored on self, not just a local -- _add_spring_cb/
        # _add_orientation_spring_cb need this same profile for springs
        # added at runtime (see their own create_publisher calls), not
        # just the ones loaded here at startup.
        self._target_qos = target_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Per-spring target update subscriptions
        for spring in self._springs:
            if isinstance(spring, VirtualSpring):
                topic = f"~/target/{spring.name}"
                self.create_subscription(
                    PointStamped,
                    topic,
                    lambda msg, s=spring: self._target_cb(msg, s),
                    10,
                )
                self.get_logger().info(f"Listening for target updates on {topic}")

                # and for the attachment points
                topic = f"~/attachment/{spring.name}"
                self.create_subscription(
                    PointStamped,
                    topic,
                    lambda msg, s=spring: self._attachment_cb(msg, s),
                    10,
                )

                # TRANSIENT_LOCAL: current target is state, not a stream --
                # published once on resolve/update, same reasoning as
                # ~/springs_updated above. Without this, armviz.py only
                # learns the real target if its subscription happened to
                # already exist at the moment _joint_state_cb resolved it;
                # confirmed 2026-08-19 as the direct cause of armviz's
                # perpetual "no target yet" spam on late startup.
                self._target_pubs[spring.name] = self.create_publisher(
                    PointStamped, f"~/target/{spring.name}", target_qos
                )
                # Springs with a target left unconfigured in YAML don't have
                # a real value yet -- _target_pending is still True, and
                # target_world_point is just the zero placeholder set in
                # _load_one_spring (see the NaN-sentinel comment there).
                # Broadcasting that now would show a wrong target briefly;
                # _joint_state_cb publishes the real one once resolved.
                if not getattr(spring, "_target_pending", False):
                    self._publish_target(spring)
            elif isinstance(spring, JointSpring):
                topic = f"~/joint_target/{spring.name}"
                self.create_subscription(
                    Float64,
                    topic,
                    lambda msg, s=spring: self._joint_target_cb(msg, s),
                    10,
                )
                self.get_logger().info(f"Listening for target updates on {topic}")
            elif isinstance(spring, OrientationSpring):
                # Reuses VirtualSpring's target/attachment topics and
                # callbacks -- both duck-type on target_world_point /
                # local_attachment_point, which OrientationSpring has too.
                topic = f"~/target/{spring.name}"
                self.create_subscription(
                    PointStamped,
                    topic,
                    lambda msg, s=spring: self._target_cb(msg, s),
                    10,
                )
                self.get_logger().info(f"Listening for target updates on {topic}")

                topic = f"~/attachment/{spring.name}"
                self.create_subscription(
                    PointStamped,
                    topic,
                    lambda msg, s=spring: self._attachment_cb(msg, s),
                    10,
                )

                # Unlike VirtualSpring, an OrientationSpring's target is
                # never pending (_load_one_orientation_spring requires it
                # explicitly) -- always safe to broadcast immediately.
                # TRANSIENT_LOCAL for the same reason as VirtualSpring's
                # publisher above -- a late subscriber (e.g. armviz.py)
                # needs the already-published target replayed to it.
                self._target_pubs[spring.name] = self.create_publisher(
                    PointStamped, f"~/target/{spring.name}", target_qos
                )
                self._publish_target(spring)

        # Services
        self._enable_srv = self.create_service(
            SetBool, "~/enable", self._enable_cb
        )

        self.get_logger().info(
            f"VirtualSpringNode ready. {len(self._springs)} spring(s) loaded."
        )

        self.get_logger().info(f"nq={self._arm.n_q}, nv={self._arm.n_dof}")
        self.get_logger().info(f"Joint names: {self._arm.joint_names}")



        # Storage: {spring_name: {'times': [], 'extensions': []}}
        self.spring_data = collections.defaultdict(lambda: {'times': [], 'extensions': [], 'torques':[]})
        self.start_time = None

        # Diagnostic: /joint_states is meant to arrive ~every 10ms
        # (feedback_rate_hz=100 on the gen3_torque_control side). Matches
        # kinova_torque_control_node's cmd_stale_warn_sec instrumentation
        # (2026-08-07) -- lets us tell whether a command-staleness episode
        # traces back to /joint_states itself arriving late, vs. this
        # node's own processing/publish being slow, instead of guessing.
        self._last_js_recv_time = None
        self._js_gap_warned = False
        self._js_gap_warn_sec = 0.02
        # Warn-once-until-clears latch (same pattern as _js_gap_warned above)
        # for the /joint_states-too-short check below -- a flat per-message
        # warn there filled the terminal buffer solid within ~1s the one
        # time it fired for real (2026-08-19, gripper controller's state
        # interface not yet up during startup's first few messages).
        self._js_short_warned = False


        # Expose a service to toggle at runtime without restarting
        self._grav_comp_srv = self.create_service(
            SetBool, "~/set_gravity_compensation", self._set_grav_comp_cb
        )
        # Same reasoning as gravity comp above -- repulsion_enabled is read
        # once into self._repulsion_enabled at startup like any other
        # declare_parameter default, and this node has no
        # add_on_set_parameters_callback anywhere, so a live
        # `ros2 param set /virtual_spring_node repulsion_enabled true`
        # reports success (the parameter server accepts it) but the running
        # control loop never re-reads it -- confirmed live 2026-08-19 as the
        # cause of repulsion silently never activating. A dedicated service
        # is the only way to flip it without a full node restart.
        self._repulsion_enable_srv = self.create_service(
            SetBool, "~/set_repulsion_enabled", self._set_repulsion_enabled_cb
        )
        # Live status, same reasoning/QoS as ~/springs_enabled -- confirmed
        # live 2026-08-20 that without this, there's no way for a UI (or
        # anyone else) to tell whether repulsion is actually on short of
        # remembering their own last ~/set_repulsion_enabled call. Directly
        # caused a whole confused debugging session: repulsion was believed
        # enabled (from an earlier session) but had never actually been
        # turned on this run, and nothing surfaced that mismatch anywhere.
        self._repulsion_enabled_pub = self.create_publisher(
            Bool, "~/repulsion_enabled_status", springs_updated_qos
        )
        self._repulsion_enabled_pub.publish(Bool(data=self._repulsion_enabled))
        # Live [danger_threshold, caution_threshold, repulsion_max_force_n],
        # same TRANSIENT_LOCAL reasoning as the two publishers above --
        # confirmed live 2026-08-20 that without this, armviz's caution-halo
        # padding is set once from the --caution-threshold launch arg and
        # never updates again, so live-tuning caution_threshold via the UI
        # (e.g. to 0.73m) silently left the drawn halo at its stale launch
        # value while the actual triggering distance was the new one --
        # very confusing to look at (halo looked ~7cm larger than the
        # object, but CAUTION was firing at 73cm).
        self._collision_thresholds_pub = self.create_publisher(
            Float64MultiArray, "~/collision_thresholds_status", springs_updated_qos
        )
        self._publish_collision_thresholds()
        # danger_threshold lives on self._arm (baked in at URDF load time,
        # see _load_arm below); caution_threshold is a plain node attribute
        # like repulsion_max_force_n. Same "no add_on_set_parameters_callback
        # anywhere" reasoning as set_repulsion_enabled above -- a live
        # ros2 param set would report success and do nothing.
        self._update_collision_thresholds_srv = self.create_service(
            UpdateCollisionThresholds, "~/update_collision_thresholds",
            self._update_collision_thresholds_cb,
        )
    def _load_arm(self) -> URDFArmConfiguration:
        """
        Load URDF from /robot_description topic if available
        (published by robot_state_publisher; what RViz uses),
        otherwise fall back to the urdf_path parameter.
        """
        srdf_path = self.get_parameter("srdf_path").get_parameter_value().string_value
        danger_threshold = self.get_parameter("danger_threshold").get_parameter_value().double_value
        # declare_parameter needs a non-empty list to infer the array type;
        # [""] is the "no locked joints" sentinel, so filter it back out.
        locked_joint_names = [
            name for name in
            self.get_parameter("locked_joint_names").get_parameter_value().string_array_value
            if name
        ]

        urdf_xml = fetch_robot_description(self)
        if urdf_xml is not None:
            self.get_logger().info("Loaded URDF from /robot_description topic.")
            return URDFArmConfiguration.from_xml_string(
                urdf_xml, danger_threshold=danger_threshold, srdf_path=srdf_path,
                locked_joint_names=locked_joint_names,
            )

        urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        if not urdf_path:
            raise RuntimeError(
                "No /robot_description topic found and urdf_path parameter not set."
            )

        self.get_logger().warn(
            f"No /robot_description topic found; falling back to file: {urdf_path}"
        )
        return URDFArmConfiguration.from_urdf(
            urdf_path, danger_threshold=danger_threshold, srdf_path=srdf_path,
            locked_joint_names=locked_joint_names,
        )


    def _set_grav_comp_cb(self, request, response):
        self._add_grav_comp = request.data
        self.get_logger().info(
            f"Gravity compensation {'enabled' if request.data else 'disabled'}"
        )
        response.success = True
        response.message = f"add_gravity_compensation = {request.data}"
        return response

    def _set_repulsion_enabled_cb(self, request, response):
        self._repulsion_enabled = request.data
        if not request.data:
            self._last_repulsion_torques = None
        self._repulsion_enabled_pub.publish(Bool(data=request.data))
        self.get_logger().info(
            f"Repulsion field {'enabled' if request.data else 'disabled'}"
        )
        response.success = True
        response.message = f"repulsion_enabled = {request.data}"
        return response

    def _update_collision_thresholds_cb(self, request, response):
        # caution_threshold must stay strictly greater than danger_threshold
        # -- get_repulsion_torques() divides by (caution_threshold -
        # danger_threshold) (guarded there against exactly 0, but a
        # negative/inverted span would silently produce a nonsense ramp
        # rather than an error, so reject it here instead).
        if request.danger_threshold <= 0.0:
            response.success = False
            response.message = "danger_threshold must be > 0."
            return response
        if request.caution_threshold <= request.danger_threshold:
            response.success = False
            response.message = "caution_threshold must be > danger_threshold."
            return response
        if request.repulsion_max_force_n <= 0.0:
            response.success = False
            response.message = "repulsion_max_force_n must be > 0."
            return response
        self._arm.danger_threshold = request.danger_threshold
        self._caution_threshold = request.caution_threshold
        self._repulsion_max_force_n = request.repulsion_max_force_n
        # Mirror into the parameter namespace, same as _update_spring_cb --
        # all three names are already declared (see __init__), so no
        # _declare_or_ignore needed here.
        self.set_parameters([
            rclpy.parameter.Parameter(
                "danger_threshold", rclpy.parameter.Parameter.Type.DOUBLE,
                request.danger_threshold,
            ),
            rclpy.parameter.Parameter(
                "caution_threshold", rclpy.parameter.Parameter.Type.DOUBLE,
                request.caution_threshold,
            ),
            rclpy.parameter.Parameter(
                "repulsion_max_force_n", rclpy.parameter.Parameter.Type.DOUBLE,
                request.repulsion_max_force_n,
            ),
        ])
        self._publish_collision_thresholds()
        self.get_logger().info(
            f"Collision thresholds updated: danger_threshold="
            f"{request.danger_threshold:.4f}m, caution_threshold="
            f"{request.caution_threshold:.4f}m, repulsion_max_force_n="
            f"{request.repulsion_max_force_n:.2f}N"
        )
        response.success = True
        response.message = (
            f"danger_threshold={request.danger_threshold:.4f}m, "
            f"caution_threshold={request.caution_threshold:.4f}m, "
            f"repulsion_max_force_n={request.repulsion_max_force_n:.2f}N"
        )
        return response

    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        """Flatten nested dict to dot-separated keys."""
        result = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_dict(v, full_key))
            else:
                result[full_key] = v
        return result

    def _load_one_spring(self, name: str) -> VirtualSpring:
        """Load a single spring by name from current parameters. Raises on error."""
        prefix = f"springs.{name}"
        self._declare_or_ignore(f"{prefix}.link_name",   "")
        self._declare_or_ignore(f"{prefix}.local_point", [0.0, 0.0, 0.0])
        # target's default is a NaN sentinel, not [0,0,0]: lets us tell
        # "target left unconfigured" apart from "target deliberately set to
        # the origin". (local_point above doesn't need this -- [0,0,0] is
        # already a meaningful default, the link's own local origin.)
        self._declare_or_ignore(
            f"{prefix}.target", [float("nan"), float("nan"), float("nan")]
        )
        self._declare_or_ignore(f"{prefix}.stiffness",   0.0)
        self._declare_or_ignore(f"{prefix}.damping",     0.0)
        self._declare_or_ignore(f"{prefix}.rest_length", 0.0)
        self._declare_or_ignore(f"{prefix}.inner_radius", 0.0)
        self._declare_or_ignore(f"{prefix}.outer_radius", 0.0)


        def _get(key, default=None, _prefix=prefix):
            val = self.get_parameter(f"{_prefix}.{key}").value
            return default if val is None else val

        link_name = _get("link_name")
        self._arm.validate_link_name(link_name)  # raises ValueError if bad

        local_point = np.array(_get("local_point", [0, 0, 0]), dtype=float)
        target = np.array(_get("target", [0, 0, 0]), dtype=float)
        target_pending = bool(np.any(np.isnan(target)))
        if target_pending:
            # No target configured: resolve to the attachment point's
            # current world position (mirrors how JointSpring, below,
            # defaults target_angle to the joint's current angle -- via the
            # same deferred-resolve mechanism, for the same reason) so the
            # spring starts at zero extension/force instead of pulling
            # toward a stale fixed point that may be far from wherever the
            # arm actually is this session. A fixed [0.5, 0, 0.5] target
            # sitting 29cm from the real starting pose contributed to a
            # live incident 2026-08-07.
            #
            # Can't resolve it here yet, though: _load_one_spring() runs
            # during __init__, before any real /joint_states message has
            # ever arrived, so self._arm is still at the URDF's zero/neutral
            # configuration -- not the arm's actual pose. Resolving "current
            # position" against that stale zero pose is what caused a
            # second live incident the same day (arm lurched toward the
            # near-vertical "candlestick" position, which is where the
            # zero configuration happens to put end_effector_link).
            # Use a zero placeholder and resolve for real on the first
            # _joint_state_cb, once self._arm reflects reality.
            target = np.zeros(3)

        spring = VirtualSpring(
            link_name=link_name,
            local_attachment_point=local_point,
            target_world_point=target,
            stiffness=float(_get("stiffness", 0.0)),
            damping=float(_get("damping", 0.0)),
            rest_length=float(_get("rest_length", 0.0)),
            name=name,
            inner_radius=float(_get("inner_radius",0.0)),
            outer_radius=float(_get("outer_radius",0.0)),
    )
        spring._target_pending = target_pending
        return spring

    def _load_one_joint_spring(self, name: str) -> JointSpring:
        """Load a single joint spring by name from current parameters. Raises on error."""
        prefix = f"joint_springs.{name}"
        self._declare_or_ignore(f"{prefix}.joint_name",   "")
        # NaN is a sentinel for "not set in config": defaults to the
        # joint's actual current angle (a soft hold-here spring) rather
        # than requiring every joint spring to specify a target. Can't
        # resolve that here, though -- same reason as VirtualSpring's
        # target below (see _load_one_spring): this method runs during
        # __init__, before any real /joint_states message has arrived, so
        # self._arm is still at the URDF's zero/neutral configuration, not
        # the arm's actual pose. Use a zero placeholder and resolve for
        # real on the first _joint_state_cb, once self._arm reflects
        # reality -- resolving against the stale zero pose here is what
        # caused a live incident for VirtualSpring, 2026-08-07.
        self._declare_or_ignore(f"{prefix}.target_angle", float("nan"))
        self._declare_or_ignore(f"{prefix}.stiffness",    0.0)
        self._declare_or_ignore(f"{prefix}.damping",      0.0)

        def _get(key, default=None, _prefix=prefix):
            val = self.get_parameter(f"{_prefix}.{key}").value
            return default if val is None else val

        joint_name = _get("joint_name")
        self._arm.validate_joint_name(joint_name)  # raises ValueError if bad

        target_angle = float(_get("target_angle", float("nan")))
        target_pending = math.isnan(target_angle)
        if target_pending:
            # Can't resolve "current angle" here yet -- this method runs
            # during __init__, before any real /joint_states message has
            # arrived, so self._arm is still at the URDF's zero/neutral
            # configuration, not the joint's actual angle. Resolving
            # against that stale pose is the same mechanism as the
            # VirtualSpring incident target's comment above describes --
            # use a placeholder and resolve for real on the first
            # _joint_state_cb, once self._arm reflects reality.
            target_angle = 0.0

        spring = JointSpring(
            joint_name=joint_name,
            target_angle=target_angle,
            stiffness=float(_get("stiffness", 0.0)),
            damping=float(_get("damping", 0.0)),
            name=name,
        )
        spring._target_pending = target_pending
        return spring

    def _load_one_orientation_spring(self, name: str) -> OrientationSpring:
        """Load a single orientation spring by name from current parameters. Raises on error."""
        prefix = f"orientation_springs.{name}"
        self._declare_or_ignore(f"{prefix}.link_name", "")
        self._declare_or_ignore(f"{prefix}.local_point", [0.0, 0.0, 0.0])
        self._declare_or_ignore(f"{prefix}.local_face_normal", [0.0, 0.0, 1.0])
        # No NaN-sentinel default-to-current-pose here, unlike
        # target_angle above: a joint spring's "hold the current angle"
        # default and a Cartesian spring's "start at the current position"
        # default are both inferable from the arm's own state, but this
        # spring's target is an external real-world point (e.g. the study
        # participant's face) that has no relationship to the arm's pose
        # -- there's no safe value to fall back on, so it's required.
        self._declare_or_ignore(
            f"{prefix}.target", [float("nan"), float("nan"), float("nan")]
        )
        self._declare_or_ignore(f"{prefix}.stiffness", 0.0)
        self._declare_or_ignore(f"{prefix}.damping", 0.0)

        def _get(key, default=None, _prefix=prefix):
            val = self.get_parameter(f"{_prefix}.{key}").value
            return default if val is None else val

        link_name = _get("link_name")
        self._arm.validate_link_name(link_name)  # raises ValueError if bad

        target = np.array(_get("target", [float("nan")] * 3), dtype=float)
        if np.any(np.isnan(target)):
            raise ValueError(
                f"Orientation spring '{name}': 'target' must be set "
                f"explicitly (e.g. the participant's measured face "
                f"position) -- there's no safe default to infer it from."
            )

        spring = OrientationSpring(
            link_name=link_name,
            local_attachment_point=np.array(_get("local_point", [0, 0, 0]), dtype=float),
            local_face_normal=np.array(_get("local_face_normal", [0, 0, 1]), dtype=float),
            target_world_point=target,
            stiffness=float(_get("stiffness", 0.0)),
            damping=float(_get("damping", 0.0)),
            name=name,
        )
        return spring

    def _torque_status_cb(self, msg: String) -> None:
        is_enabled = (msg.data == "ENABLED")
        was_enabled = (self._last_torque_status == "ENABLED")
        if is_enabled and not was_enabled:
            self._torque_enabled_since = self.get_clock().now()
            self.get_logger().info(
                f"Torque control enabled -- ramping spring force in over "
                f"{self._spring_ramp_duration:.2f}s (gravity comp unramped)."
            )
        elif not is_enabled:
            self._torque_enabled_since = None
        self._last_torque_status = msg.data

    def _check_torque_control_connected(self) -> None:
        """
        Runs on its own 1Hz timer -- see the comment on
        self._torque_control_connected in __init__ for why this can't live
        in _joint_state_cb. Edge-triggered log so a sustained outage
        doesn't spam; on the falling edge, also republishes ~/safety_status
        immediately (using whatever collision status was last computed)
        rather than waiting for _joint_state_cb, since on Gen3 that may
        never fire again once kinova_torque_control is the thing that died.
        """
        was_connected = self._torque_control_connected
        self._torque_control_connected = (
            self.count_publishers(self._torque_status_topic) > 0
        )
        if was_connected and not self._torque_control_connected:
            self.get_logger().error(
                f"{self._torque_status_topic}'s publisher has disappeared "
                f"-- gen3_torque_control (or equivalent) is no longer "
                f"running. Torque commands published from here are no "
                f"longer being applied to the arm."
            )
            self._publish_safety_status(self._last_collision_status)
        elif self._torque_control_connected and was_connected is False:
            self.get_logger().info(f"{self._torque_status_topic}'s publisher is back.")

    def _joint_state_cb(self, msg: JointState) -> None:
        now = self.get_clock().now()
        if self._last_js_recv_time is not None:
            gap = (now - self._last_js_recv_time).nanoseconds / 1e9
            if gap > self._js_gap_warn_sec and not self._js_gap_warned:
                self.get_logger().warn(
                    f"/joint_states gap {gap * 1000:.1f}ms exceeds "
                    f"{self._js_gap_warn_sec * 1000:.0f}ms warn threshold "
                    f"(expected ~10ms at 100Hz)."
                )
                self._js_gap_warned = True
            elif gap <= self._js_gap_warn_sec:
                self._js_gap_warned = False
        self._last_js_recv_time = now

    # Build joint reordering maps on first message
        if self._joint_order is None:
            
            pinocchio_names = self._arm.joint_names  # list in pinocchio order
            ros_names = list(msg.name) # from the UR3e these come in in a bizarre order, match names to catch issues
            self.get_logger().info(f"pinocchio_names {pinocchio_names} ros_names {ros_names}")
            self.get_logger().info(f"ros_message{msg}")
            try:
                # For each pinocchio joint, find its index in the ROS message
                self._joint_order = [ros_names.index(n) for n in pinocchio_names]
                self.get_logger().info(f"Joint order map (ROS->Pinocchio): {list(zip(pinocchio_names, self._joint_order))}")
                # For each ROS joint, find its index in the pinocchio torques array
                # -- skips ros_names with no pinocchio counterpart (e.g. a
                # gripper joint reported on /joint_states but locked out of
                # the active-DOF model) instead of raising, since a name
                # arriving that's outside pinocchio_names doesn't mean this
                # message is unusable, only that torque_order has nothing to
                # say about that particular ROS joint.
                ros_names_mapped = [n for n in ros_names if n in pinocchio_names]
                self._torque_order = [pinocchio_names.index(n) for n in ros_names_mapped]
                self.get_logger().info(f"Torque order map (Pinocchio->ROS): {list(zip(ros_names_mapped, self._torque_order))}")
            except ValueError as e:
                self.get_logger().error(f"Joint name mismatch: {e}. ROS names: {ros_names}, Pinocchio names: {pinocchio_names}")
                return

        # self._joint_order holds fixed indices into msg.position/msg.velocity
        # (computed once above from the first message's joint names), so a
        # message with *extra* trailing joints beyond the arm's n_dof --
        # e.g. a locked gripper joint reported on /joint_states -- is fine;
        # only a message too short to cover every index in self._joint_order
        # is actually unusable. Latched (see _js_short_warned) instead of
        # warning every callback -- this fired at full topic rate 2026-08-19
        # (gripper controller's state interface not up yet for the first
        # message, present by the next one) and filled the terminal buffer.
        if len(msg.position) <= max(self._joint_order):
            if not self._js_short_warned:
                self.get_logger().warn(
                    f"/joint_states has {len(msg.position)} joints, too few for "
                    f"the arm joint indices this node resolved on startup "
                    f"(need > {max(self._joint_order)}). Skipping until it recovers."
                )
                self._js_short_warned = True
            return
        elif self._js_short_warned:
            self.get_logger().info("/joint_states length recovered.")
            self._js_short_warned = False

        # Reorder positions and velocities to match Pinocchio's joint ordering
        pos = np.array(msg.position)
        vel = np.array(msg.velocity) if msg.velocity else np.zeros(len(msg.position))
        q_arm = pos[self._joint_order]
        qdot  = vel[self._joint_order]
        
        try:
            # Initialize start time on first callback
            if self.start_time is None:
                self.start_time = self.get_clock().now()

            # update the time
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                
            
            # update the joint angles from the arm
            self._arm.update_from_angles(q_arm, qdot)

            # Resolve any springs whose target was left unconfigured: now
            # that self._arm reflects a real measured pose (not the URDF's
            # zero/neutral configuration it started at -- see
            # _load_one_spring / _load_one_joint_spring), set target =
            # current position/angle so the spring starts at zero
            # extension/force instead of pulling toward wherever the zero
            # pose happens to put it. OrientationSpring never sets
            # _target_pending (its target has no safe default and is
            # required at load time), so it never matches either branch
            # here.
            for spring in self._springs:
                if isinstance(spring, VirtualSpring) and getattr(spring, "_target_pending", False):
                    T = self._arm.get_link_transform(spring.link_name)
                    p_local_h = np.append(spring.local_attachment_point, 1.0)
                    spring.target_world_point = (T @ p_local_h)[:3]
                    spring._target_pending = False
                    self.get_logger().info(
                        f"Spring '{spring.name}': target resolved to current "
                        f"attachment position ({spring.target_world_point[0]:.3f}, "
                        f"{spring.target_world_point[1]:.3f}, {spring.target_world_point[2]:.3f})."
                    )
                    # Without this, nothing ever tells external observers
                    # (e.g. armviz) what the resolved target actually is --
                    # they only learn of a target via this same topic, and
                    # a pending one has no explicit config to have already
                    # broadcast at load time.
                    self._publish_target(spring)
                    # The topic publish above only reaches whoever's already
                    # subscribed at this exact moment (armviz.py, for one,
                    # isn't -- it's still busy loading its own URDF/meshcat
                    # this early). The springs.<name>.target *parameter*
                    # still holds the NaN-sentinel placeholder otherwise --
                    # confirmed 2026-08-14: armviz's `ros2 param get`-based
                    # bootstrap retried for the full 20s and never saw
                    # anything but NaN, since nothing here ever wrote the
                    # resolved value back to it. Late subscribers/pollers
                    # need this to reflect ground truth same as the topic.
                    self.set_parameters([
                        rclpy.parameter.Parameter(
                            f"springs.{spring.name}.target",
                            rclpy.parameter.Parameter.Type.DOUBLE_ARRAY,
                            spring.target_world_point.tolist(),
                        )
                    ])
                elif isinstance(spring, JointSpring) and getattr(spring, "_target_pending", False):
                    joint_index = self._arm.joint_names.index(spring.joint_name)
                    spring.target_angle = self._arm.get_joint_angle(joint_index)
                    spring._target_pending = False
                    self.get_logger().info(
                        f"Joint spring '{spring.name}': target_angle resolved "
                        f"to current angle ({spring.target_angle:.4f} rad)."
                    )
                    # Same reasoning as VirtualSpring above -- keep the
                    # parameter in sync with the resolved value, not just
                    # the in-memory spring object.
                    self.set_parameters([
                        rclpy.parameter.Parameter(
                            f"joint_springs.{spring.name}.target_angle",
                            rclpy.parameter.Parameter.Type.DOUBLE,
                            float(spring.target_angle),
                        )
                    ])

            # compute the actual torques generated by the full set of springs
            # if self._add_grav_comp is true, include calculations for gravity compensation
            spring_scale = 1.0
            if self._spring_ramp_duration > 0.0:
                # Take the *minimum* of both ramps -- torque-enable's and a
                # manual springs re-enable's -- so whichever happened most
                # recently governs. Same duration/curve for both, just two
                # different trigger events feeding the same ramp shape.
                if self._torque_enabled_since is not None:
                    ramp_elapsed = (
                        (self.get_clock().now() - self._torque_enabled_since).nanoseconds / 1e9
                    )
                    spring_scale = min(spring_scale, max(0.0, ramp_elapsed / self._spring_ramp_duration))
                if self._springs_reenabled_since is not None:
                    reenable_elapsed = (
                        (self.get_clock().now() - self._springs_reenabled_since).nanoseconds / 1e9
                    )
                    spring_scale = min(spring_scale, max(0.0, reenable_elapsed / self._spring_ramp_duration))
            torques = self._springs.compute_total_torques(
                self._arm, self._add_grav_comp, spring_scale
            )

            # Gravity comp must never be scaled by anything except the
            # add_gravity_compensation on/off switch itself -- see
            # torque_ramp_gravity_comp_lesson (a real arm drop toward the
            # table, 2026-08-05). compute_total_torques() already keeps it
            # out of spring_scale for that reason; split it out here too so
            # the collision clamp below can't reintroduce the same bug by
            # scaling/zeroing it along with the spring torque.
            gravity_torques = (
                self._arm.get_gravity_torques() if self._add_grav_comp
                else np.zeros_like(torques)
            )
            spring_only_torques = torques - gravity_torques

            # Self-collision safety scaling (throttled -- see
            # collision_check_interval_sec)
            if (self._last_collision_check_time is None or
                    (now - self._last_collision_check_time).nanoseconds / 1e9
                    >= self._collision_check_interval):
                self._last_collision_status = self._arm.get_collision_status()
                self._last_repulsion_torques = (
                    self._arm.get_repulsion_torques(
                        self._caution_threshold, self._repulsion_max_force_n,
                        self._repulsion_max_total_torque_nm,
                    ) if self._repulsion_enabled else None
                )
                self._last_collision_check_time = now
            collision = self._last_collision_status
            self._publish_safety_status(collision)
            if collision is not None:
                a, b = collision.closest_pair
                involves_scene_object = (
                    self._arm.is_environment_object(a)
                    or self._arm.is_environment_object(b)
                )
                kind = "COLLISION WITH SCENE OBJECT" if involves_scene_object else "SELF-COLLISION"
                log_call_failed = False
                if collision.in_collision:
                    #no spamming about collisions
                    # irrelevant if springs are off
                    if not self._springs_auto_disabled:
                        #only if torque is enabled, irrelevant if off
                        if self._last_torque_status == "ENABLED":
                            self.get_logger().error(
                                f"{kind} detected between {a} and {b}!"
                                f"(Zeroing spring torque )"
                                f"(gravity comp held)."
                            )
                        self._set_springs_enabled(
                            False,
                            reason=f"{kind} detected between {a} and {b}",
                            auto=True,
                        )
                    # Zero spring torque for *this* cycle too -- the
                    # disable above only takes effect on the next
                    # compute_total_torques() call. Never touch gravity
                    # comp, even at the hard clamp: dropping it right when
                    # two links are closest would let the arm sag/fall at
                    # the worst possible moment instead of holding position.
                    torques = gravity_torques
                                
                elif collision.in_danger:
                    label = "Near scene-object collision" if involves_scene_object else "Near self-collision"
                    if (self._last_torque_status == "ENABLED"
                            and elapsed - self._collision_warn_log_last >= 0.5):
                        self.get_logger().warn(
                            f"{label}: {a} / {b}, "
                            f"dist={collision.min_distance:.3f}m, "
                            f"scale={collision.scale_factor:.2f}"
                        )
                        self._collision_warn_log_last = elapsed
                    # Scale spring torque only -- gravity comp stays at full
                    # magnitude (same reasoning as the in_collision branch
                    # above). Scaling it down here was the likely cause of a
                    # sag/recover physical limit cycle: reduced gravity comp
                    # near the boundary lets the arm sag, sag changes the
                    # geometry, which can pull it back out of danger and
                    # restore full gravity comp, arm lifts back in, repeat.
                    # Naturally contributes ~0 if springs are currently
                    # auto-disabled from a prior in_collision episode, same
                    # as everywhere else -- nothing to scale.
                    torques = gravity_torques + collision.scale_factor * spring_only_torques

                if self._collision_log_writer is not None:
                    self._collision_log_writer.writerow([
                        f"{elapsed:.4f}", kind, collision.in_collision, collision.in_danger,
                        a, b, f"{collision.min_distance:.4f}", f"{collision.scale_factor:.4f}",
                        log_call_failed,
                    ])
                    # Flushing every cycle (was: every single row, up to
                    # 100Hz) is a plausible direct cause of the periodic
                    # ~50-90ms torque-command latency seen live 2026-08-07
                    # -- throttled to 2x/sec instead. Crash-safety cost is
                    # capped at <0.5s of data loss, an acceptable tradeoff
                    # against not stalling the control loop on disk I/O
                    # every cycle.
                    if elapsed - self._collision_log_last_flush >= 0.5:
                        self._collision_log_file.flush()
                        self._collision_log_last_flush = elapsed

            # Repulsion field: unconditionally additive, outside every
            # branch above (in_collision / in_danger / clear) and
            # independent of spring enable state -- see repulsion_enabled's
            # declaration. Applied after gravity comp is already folded into
            # `torques`, as a pure additional term, never a multiplier --
            # keeps torque_ramp_gravity_comp_lesson's invariant intact.
            if self._repulsion_enabled and self._last_repulsion_torques is not None:
                torques = torques + self._last_repulsion_torques
                repulsion_out = self._last_repulsion_torques
                repulsion_norm = float(np.linalg.norm(self._last_repulsion_torques))
                if (repulsion_norm > 1e-6 and self._last_torque_status == "ENABLED"
                        and elapsed - self._repulsion_warn_log_last >= 0.5):
                    self.get_logger().warn(
                        f"Repulsion field active: |tau|={repulsion_norm:.3f} N*m"
                    )
                    self._repulsion_warn_log_last = elapsed
            else:
                repulsion_out = np.zeros(self._arm.n_dof)
            repulsion_msg = JointState()
            repulsion_msg.header.stamp = self.get_clock().now().to_msg()
            repulsion_msg.name = self._arm.joint_names
            repulsion_msg.effort = repulsion_out.tolist()
            self._repulsion_torque_pub.publish(repulsion_msg)

            # Grace-period escalation: springs auto-disabled by the
            # collision clamp, still not manually re-enabled, and it's been
            # too long -- see collision_disable_grace_sec's declaration.
            # Runs every cycle regardless of *current* collision/danger
            # state, since the whole point is catching the case where the
            # pair cleared a while ago (in_collision/in_danger both false
            # again) but the operator hasn't reached ~/enable yet.
            if self._springs_auto_disabled and self._springs_auto_disabled_since is not None:
                grace_elapsed = (
                    (now - self._springs_auto_disabled_since).nanoseconds / 1e9
                )
                if grace_elapsed >= self._collision_disable_grace_sec:
                    self._trigger_collision_disable_escalation(grace_elapsed)

            # Record spring state for plot_spring_extensions() (on-shutdown plot)
            for spring in self._springs:
                if spring._last_state is not None:
                    self.spring_data[spring.name]['times'].append(elapsed)
                    self.spring_data[spring.name]['extensions'].append(spring._last_state.extension)
                    self.spring_data[spring.name]['torques'].append(spring._last_state.torques)
        except Exception as e:
            # Fail safe rather than simply not publishing: if we just
            # returned here, torque_relay/gen3_torque_control would keep
            # applying the last torque command forever with no further
            # correction as the arm's actual pose drifts away from whatever
            # that command was valid for.
            #
            # Zero torque is only safe here if the hardware itself does
            # gravity compensation (add_gravity_compensation=False, e.g.
            # UR3e). On arms without that (add_gravity_compensation=True,
            # e.g. Gen3 via gen3_torque_control) a bare zero command lets
            # the arm fall, so fall back to gravity-only torques (drop the
            # spring forces, keep holding against gravity) using the same
            # policy the normal path already uses; only fall back further
            # to a true zero if even that computation fails.
            self.get_logger().error(f"Spring computation failed: {e}")
            try:
                torques = (
                    self._arm.get_gravity_torques()
                    if self._add_grav_comp
                    else np.zeros(self._arm.n_dof)
                )
            except Exception as grav_e:
                self.get_logger().error(
                    f"Gravity-only fallback also failed: {grav_e}. Publishing zero torques."
                )
                torques = np.zeros(self._arm.n_dof)

            self._trigger_torque_disable_failsafe()

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name   = self._arm.joint_names
        out.effort = torques.tolist()
        self._torque_pub.publish(out)

    def _trigger_torque_disable_failsafe(self) -> None:
        """
        On repeated spring-computation failure, switch the arm out of
        torque control entirely via torque_disable_service (if configured)
        rather than keep trusting our own torque computation. Fires once
        per node lifetime (not once per episode, unlike
        _trigger_collision_disable_escalation below) -- a control loop
        that just proved its own math unreliable shouldn't get to decide
        again later in the same run that it's safe to trust itself.
        """
        if self._failsafe_triggered:
            return
        self._failsafe_triggered = True
        self._call_torque_disable_service("spring computation failing repeatedly")

    def _trigger_collision_disable_escalation(self, grace_elapsed: float) -> None:
        """
        Springs have stayed auto-disabled by the collision clamp for too
        long without an explicit ~/enable(true) -- see
        collision_disable_grace_sec. Escalate to a full torque disable so
        gravity-comp-only drift (torque_ramp_gravity_comp_lesson) can't
        slowly walk the arm back into what it just cleared while the
        operator is still walking to the UI.

        Own one-shot guard, separate from _failsafe_triggered: this one
        re-arms on every fresh auto-disable episode (reset in
        _set_springs_enabled) so it protects the *next* collision too, not
        just the first one this session -- unlike the spring-computation
        failsafe, "the operator was slow to reach the UI once" says
        nothing about whether the same thing should be caught again later.
        """
        if self._collision_escalation_triggered:
            return
        self._collision_escalation_triggered = True
        self._call_torque_disable_service(
            f"springs stayed auto-disabled by a collision for "
            f"{grace_elapsed:.1f}s without being re-enabled -- "
            "gravity-comp-only drift could walk the arm back into what it "
            "just cleared"
        )

    def _call_torque_disable_service(self, reason: str) -> None:
        """Shared by both fail-safes above: fire-and-forget disable call
        via torque_disable_service, once the caller's own one-shot guard
        has already been checked/set."""
        if self._torque_disable_client is None:
            return
        if not self._torque_disable_client.service_is_ready():
            self.get_logger().error(
                "Fail-safe: torque_disable_service not available -- "
                "cannot switch out of torque control automatically!"
            )
            return
        self.get_logger().error(
            f"Fail-safe: {reason} -- "
            "disabling torque control via "
            f"{self._torque_disable_client.srv_name}."
        )
        future = self._torque_disable_client.call_async(SetBool.Request(data=False))
        future.add_done_callback(self._torque_disable_failsafe_done_cb)

    def _torque_disable_failsafe_done_cb(self, future) -> None:
        try:
            result = future.result()
            self.get_logger().error(f"Fail-safe torque disable result: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Fail-safe torque disable call failed: {e}")

    def _target_cb(self, msg: PointStamped, spring: VirtualSpring) -> None:
        p = msg.point
        new_target = np.array([p.x, p.y, p.z])
        # Skip if this is our own echo (see the re-publish below) -- the
        # np.allclose() bounds the recursion to one bounce: the republished
        # message comes back through this same callback with the target
        # already applied, so the second time through this is a no-op.
        if np.allclose(spring.target_world_point, new_target):
            return
        spring.move_target(new_target)
        self.get_logger().debug(
            f"Updated target for '{spring.name}': "
            f"[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]"
        )
        # Re-broadcast on the same topic so a subscriber whose QoS the
        # ORIGINAL publisher doesn't satisfy still sees the update --
        # confirmed live 2026-08-19: retargeting a spring from the UI
        # (rosbridge-created publisher, default/VOLATILE durability) never
        # reached armviz, whose ~/target/<name> subscription is
        # TRANSIENT_LOCAL (needed separately so a late-joining armviz sees
        # an already-resolved target -- see _publish_target's call sites).
        # This node's own publisher on this same topic IS TRANSIENT_LOCAL,
        # so re-publishing here is what actually reaches it. This was
        # previously "deliberately not republished" on the assumption that
        # every subscriber -- including armviz -- already sees the
        # original message directly; that assumption held before armviz's
        # subscription needed to be TRANSIENT_LOCAL.
        self._publish_target(spring)

    def _publish_target(self, spring) -> None:
        """Broadcast a Cartesian-target spring's (VirtualSpring or
        OrientationSpring) current target on ~/target/<name>.

        Only call this for a target this node set on its own initiative
        (load-time, or resolving a pending one against the arm's real
        pose) -- never from _target_cb, which would create a publish/
        receive self-loop on the same topic.
        """
        pub = self._target_pubs.get(spring.name)
        if pub is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.point.x, msg.point.y, msg.point.z = (
            float(v) for v in spring.target_world_point
        )
        pub.publish(msg)

    def _attachment_cb(self, msg: PointStamped, spring: VirtualSpring) -> None:
        p = msg.point
        spring.local_attachment_point = np.array([p.x, p.y, p.z])

    def _joint_target_cb(self, msg: Float64, spring: JointSpring) -> None:
        spring.move_target(msg.data)
        self.get_logger().debug(
            f"Updated target for '{spring.name}': {msg.data:.4f} rad"
        )

    # ------------------------------------------------------------------
    # Scene collision objects
    # ------------------------------------------------------------------

    def _apply_collision_object(
        self, msg: CollisionObject, exclude_links: list[str] | None = None
    ) -> int:
        """
        Register (or replace) every primitive of a CollisionObject as an
        environment collision object on the arm. Shared by the live
        ~/collision_object topic and the YAML scene loader -- one code path
        for ADD/APPEND. Only BOX/CYLINDER primitives are supported (meshes
        and planes are silently skipped with a warning).

        exclude_links (YAML-only -- moveit_msgs/CollisionObject has no field
        for it) skips creating collision pairs against those URDF links for
        every primitive of this object, e.g. the base link against the
        table it's bolted to. See add_environment_object's docstring.

        Returns the number of primitives registered.
        """
        object_pose = _pose_to_se3(msg.pose)
        n_primitives = len(msg.primitives)
        frames: list[tuple[str, pin.SE3]] = []

        for i, primitive in enumerate(msg.primitives):
            relative_pose = (
                _pose_to_se3(msg.primitive_poses[i])
                if i < len(msg.primitive_poses)
                else pin.SE3.Identity()
            )

            if primitive.type == SolidPrimitive.BOX:
                shape = "box"
                dims = list(primitive.dimensions[:3])
            elif primitive.type == SolidPrimitive.CYLINDER:
                shape = "cylinder"
                height, radius = primitive.dimensions[:2]
                dims = [radius, height]
            else:
                self.get_logger().warn(
                    f"CollisionObject '{msg.id}': unsupported primitive "
                    f"type {primitive.type} (only BOX/CYLINDER are "
                    "supported), skipping."
                )
                continue

            sub_id = msg.id if n_primitives == 1 else f"{msg.id}::{i}"
            self._arm.add_environment_object(
                sub_id, shape, dims, object_pose * relative_pose,
                exclude_links=exclude_links,
            )
            frames.append((sub_id, relative_pose))

        self._collision_object_frames[msg.id] = frames
        self._collision_object_state_pub.publish(msg)
        return len(frames)

    def _collision_object_cb(self, msg: CollisionObject) -> None:
        if msg.operation in (CollisionObject.ADD, CollisionObject.APPEND):
            count = self._apply_collision_object(msg)
            self.get_logger().info(
                f"Collision object '{msg.id}': {count} shape(s) registered."
            )
        elif msg.operation == CollisionObject.REMOVE:
            for sub_id, _ in self._collision_object_frames.pop(msg.id, []):
                self._arm.remove_environment_object(sub_id)
            self._collision_object_state_pub.publish(msg)
            self.get_logger().info(f"Collision object '{msg.id}' removed.")
        elif msg.operation == CollisionObject.MOVE:
            frames = self._collision_object_frames.get(msg.id)
            if not frames:
                self.get_logger().warn(
                    f"Collision object '{msg.id}': MOVE received but the "
                    "object isn't currently registered; ignoring."
                )
                return
            object_pose = _pose_to_se3(msg.pose)
            for sub_id, relative_pose in frames:
                self._arm.move_environment_object(sub_id, object_pose * relative_pose)
            self._collision_object_state_pub.publish(msg)
        else:
            self.get_logger().warn(
                f"Collision object '{msg.id}': unknown operation "
                f"{msg.operation}, ignoring."
            )

    def _load_collision_scene_cb(
        self, request: LoadCollisionScene.Request, response: LoadCollisionScene.Response
    ) -> LoadCollisionScene.Response:
        """
        Load a static collision scene from YAML (see config/*_collision_scene.yaml
        for the schema) -- meant to be called once from a launch file so a
        scene can be pre-defined, but callable at any time to swap it.
        Builds one CollisionObject per entry and feeds it through the same
        _apply_collision_object path the live topic uses.
        """
        yaml_path = os.path.expanduser(request.yaml_path)
        try:
            with open(yaml_path, "r") as f:
                scene = yaml.safe_load(f) or {}
        except OSError as e:
            response.success = False
            response.message = f"Could not read '{yaml_path}': {e}"
            response.count = 0
            return response
        except yaml.YAMLError as e:
            response.success = False
            response.message = f"Malformed YAML in '{yaml_path}': {e}"
            response.count = 0
            return response

        total = 0
        try:
            for object_id, entry in (scene.get("objects") or {}).items():
                obj = CollisionObject()
                obj.id = object_id
                obj.operation = CollisionObject.ADD
                obj.header.frame_id = entry.get("frame_id", "")

                primitive = SolidPrimitive()
                shape = entry["shape"]
                if shape == "box":
                    primitive.type = SolidPrimitive.BOX
                elif shape == "cylinder":
                    primitive.type = SolidPrimitive.CYLINDER
                else:
                    raise ValueError(f"unknown shape '{shape}' for object '{object_id}'")
                primitive.dimensions = [float(d) for d in entry["dimensions"]]
                obj.primitives = [primitive]

                pose_entry = entry.get("pose", {})
                position = pose_entry.get("position", [0.0, 0.0, 0.0])
                orientation = pose_entry.get("orientation", [0.0, 0.0, 0.0, 1.0])
                obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = position
                (obj.pose.orientation.x, obj.pose.orientation.y,
                 obj.pose.orientation.z, obj.pose.orientation.w) = orientation

                exclude_links = entry.get("exclude_links") or None
                total += self._apply_collision_object(obj, exclude_links=exclude_links)
        except (KeyError, ValueError, TypeError) as e:
            response.success = False
            response.message = f"Error parsing '{yaml_path}': {e}"
            response.count = total
            return response

        response.success = True
        response.message = f"Loaded {total} collision shape(s) from '{yaml_path}'."
        response.count = total
        self.get_logger().info(response.message)
        return response

    def _check_collision_at_cb(
        self, request: CheckCollisionAtAngles.Request, response: CheckCollisionAtAngles.Response
    ) -> CheckCollisionAtAngles.Response:
        """
        Pre-flight collision check for a hypothetical joint configuration --
        e.g. before commanding a real move there -- against the currently
        loaded model (self-collision pairs, SRDF exclusions, and any scene
        objects from ~/load_collision_scene or ~/collision_object). Does not
        touch the live tracked pose; see get_collision_status_for_angles().
        """
        angles = list(request.joint_angles_rad)
        if len(angles) != self._arm.n_dof:
            response.in_collision = True  # fail closed on a malformed request
            response.in_danger = True
            response.min_distance = float("nan")
            response.link_a = ""
            response.link_b = ""
            self.get_logger().warn(
                f"check_collision_at: expected {self._arm.n_dof} joint_angles_rad, "
                f"got {len(angles)}. Reporting in_collision=True (fail closed)."
            )
            return response

        status = self._arm.get_collision_status_for_angles(angles)
        if status is None:
            # No collision model loaded -- nothing to check against, so
            # there's nothing to reject on. Matches get_collision_status()'s
            # own None-means-unavailable convention.
            response.in_collision = False
            response.in_danger = False
            response.min_distance = float("inf")
            response.link_a = ""
            response.link_b = ""
            return response

        response.in_collision = status.in_collision
        response.in_danger = status.in_danger
        response.min_distance = status.min_distance
        response.link_a, response.link_b = status.closest_pair
        return response

    def _declare_or_ignore(self, name, default):
        try:
            self.declare_parameter(name, default)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

    def _set_springs_enabled(self, enabled: bool, reason: str = "", auto: bool = False) -> None:
        """
        Shared enable/disable path for every spring -- used by both the
        ~/enable service (operator-driven) and the collision hard clamp
        (auto-driven, see _joint_state_cb's in_collision branch). Having
        one mechanism instead of two means the collision clamp's "zero
        spring torque" state is the same, visible, queryable state an
        operator's ~/enable(false) call produces -- not a separate
        untracked override.

        auto=True marks this as the collision clamp auto-disabling (sets
        _springs_auto_disabled so _joint_state_cb doesn't re-call this
        every single cycle while still in collision, and so ~/enable can
        tell an auto-disable apart from a manual one). auto=False (the
        ~/enable service path) always clears that flag and, when
        re-enabling, starts the re-enable ramp -- see
        _springs_reenabled_since's declaration.
        """
        for spring in self._springs:
            spring.enabled = enabled
        if auto:
            self._springs_auto_disabled = not enabled
            if not enabled:
                # Fresh episode -- start its own grace period and let it
                # escalate again even if a *previous* episode already did.
                self._springs_auto_disabled_since = self.get_clock().now()
                self._collision_escalation_triggered = False
            else:
                self._springs_auto_disabled_since = None
        else:
            self._springs_auto_disabled = False
            self._springs_auto_disabled_since = None
            if enabled:
                self._springs_reenabled_since = self.get_clock().now()
        state = "enabled" if enabled else "disabled"
        suffix = f" ({reason})" if reason else ""
        self.get_logger().info(f"Springs {state}{suffix}.")
        self._publish_springs_enabled()

    def _enable_cb(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        # Enabling while still in_collision is pointless -- _joint_state_cb's
        # collision clamp will auto-disable again on its very next cycle
        # (see _set_springs_enabled's auto=True path), so this call would
        # otherwise report success and "All springs enabled." while the
        # UI's live springs-state immediately flips back to disabled a
        # moment later -- confusing/misleading (confirmed live 2026-08-20:
        # the msg-box's stale "enabled" text sat there green while the
        # status row correctly showed disabled-by-collision). Report the
        # truth up front instead of a success that won't hold.
        if request.data and self._last_collision_status is not None \
                and self._last_collision_status.in_collision:
            a, b = self._last_collision_status.closest_pair
            response.success = False
            response.message = (
                f"Refusing to enable: still in collision ({a}/{b}) -- "
                "would be immediately auto-disabled again. Move clear first."
            )
            return response
        self._set_springs_enabled(request.data, reason="via ~/enable service")
        state = "enabled" if request.data else "disabled"
        response.success = True
        response.message = f"All springs {state}."
        return response

    # ------------------------------------------------------------------
    # Runtime re-centering
    # ------------------------------------------------------------------

    def _check_equilibrium_shift(self) -> None:
        """
        Called after any spring change. Always solves for the new
        equilibrium and logs the shift. If it exceeds
        recentering_threshold_rad AND recentering_enabled is true, triggers
        a re-centering cycle:
          disable springs → switch to position controller →
          run equilibrium_mover → switch back to effort → re-enable springs

        recentering_enabled defaults to false (see its declare_parameter
        call) since this cycle depends on ros2_control's controller_manager
        and a real equilibrium_mover package -- neither applies to Gen3,
        and it's not yet finished even for UR3e. With it off, a large shift
        is logged but springs are left alone.
        """
        if self._recentering_in_progress:
            self.get_logger().warn(
                "Re-centering already in progress; ignoring equilibrium check."
            )
            return

        n_dof = self._arm.n_dof
        # NOT self._arm.joint_positions -- that's pinocchio's raw internal q
        # vector (nq-length), which for any robot with continuous joints
        # (e.g. Gen3's 4) is longer than n_dof since each needs 2 slots
        # (cos, sin). update_from_angles/residual below need one angle per
        # joint (n_dof-length), which get_joint_angle decodes correctly.
        q_current = np.array([self._arm.get_joint_angle(i) for i in range(n_dof)])

        from scipy.optimize import fsolve

        def residual(angles):
            self._arm.update_from_angles(angles, np.zeros(n_dof))
            return (self._springs.compute_total_torques(self._arm, False)
                    + self._arm.get_gravity_torques())

        q_star, info, ier, _ = fsolve(residual, q_current, full_output=True)

        # Restore current state after solve
        self._arm.update_from_angles(q_current, np.zeros(n_dof))

        if ier != 1:
            self.get_logger().error(
                "Equilibrium solve did not converge after spring change. "
                "Leaving arm in current configuration."
            )
            return

        # Wrap to (-pi, pi] before measuring the shift: for continuous
        # joints, angles feed into a cos/sin encoding (see update_from_angles)
        # that's exactly 2*pi-periodic, so fsolve is free to converge on an
        # equilibrium many whole revolutions away from q_current -- an
        # equally valid root of the periodic residual, but a meaningless
        # "shift" if measured on the raw unwrapped difference.
        delta = q_star - q_current
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        shift = np.max(np.abs(delta))
        self.get_logger().info(
            f"New equilibrium shift: {np.degrees(shift):.1f} deg "
            f"(threshold: {np.degrees(self._recentering_threshold):.1f} deg)"
        )

        if shift < self._recentering_threshold:
            self.get_logger().info("Shift within threshold — no re-centering needed.")
            return

        if not self._recentering_enabled:
            self.get_logger().warn(
                f"Large equilibrium shift ({np.degrees(shift):.1f} deg) but "
                "recentering_enabled is false -- not auto-disabling springs. "
                "Set recentering_enabled:=true once the re-centering sequence "
                "(controller_manager + equilibrium_mover) is actually set up "
                "for this robot."
            )
            return

        self.get_logger().warn(
            f"Large equilibrium shift ({np.degrees(shift):.1f} deg). "
            "Initiating re-centering."
        )
        self._recentering_in_progress = True

        for spring in self._springs:
            spring.enabled = False

        threading.Thread(
            target=self._recenter_thread,
            args=(q_star,),
            daemon=True,
        ).start()

    def _recenter_thread(self, q_star: np.ndarray) -> None:
        """
        Background thread: switch to position controller, run equilibrium_mover,
        switch back to effort controller, re-enable springs.

        Springs are always re-enabled on the way out, success or failure --
        re-enabling itself is never unsafe (a spring computing torques that
        don't reach the motors because the wrong controller is active is a
        no-op, not a hazard), whereas leaving them silently disabled forever
        is the real footgun. A failure is still logged fatally, since it
        means the controller switch may need manual attention -- just not
        left in a state where the operator has no idea springs are off.
        """
        import subprocess

        try:
            self.get_logger().info("Re-centering: switching to position controller.")
            subprocess.run([
                "ros2", "service", "call",
                "/controller_manager/switch_controller",
                "controller_manager_msgs/srv/SwitchController",
                "{activate_controllers: [scaled_joint_trajectory_controller], "
                "deactivate_controllers: [forward_effort_controller], "
                "strictness: 2}",
            ], check=True)

            self.get_logger().info("Re-centering: running equilibrium_mover.")
            subprocess.run([
                "ros2", "run", "your_study_pkg", "equilibrium_mover",
                "--ros-args",
                "-p", f"config_path:={self._config_path}",
                "-p", "velocity_scaling:=0.1",
                "-p", "accel_scaling:=0.1",
            ], check=True)

            self.get_logger().info("Re-centering: switching back to effort controller.")
            subprocess.run([
                "ros2", "service", "call",
                "/controller_manager/switch_controller",
                "controller_manager_msgs/srv/SwitchController",
                "{activate_controllers: [forward_effort_controller], "
                "deactivate_controllers: [scaled_joint_trajectory_controller], "
                "strictness: 2}",
            ], check=True)

            self.get_logger().info("Re-centering complete.")

        except subprocess.CalledProcessError as e:
            self.get_logger().fatal(
                f"Re-centering failed: {e}. The controller switch may be "
                "left in a partial state -- check controller_manager "
                "manually. Springs are being re-enabled regardless; "
                "if the wrong controller is active they'll be no-ops "
                "until you fix the controller state."
            )

        finally:
            for spring in self._springs:
                spring.enabled = True
            self._recentering_in_progress = False

    # ------------------------------------------------------------------
    # add_spring callback
    # ------------------------------------------------------------------

    def _add_spring_cb(
        self, request: AddSpring.Request, response: AddSpring.Response
) -> AddSpring.Response:
        name = request.name.strip()
        if not name:
            response.success = False
            response.message = "Spring name must not be empty."
            return response

        if any(s.name == name for s in self._springs):
            response.success = False
            response.message = f"Spring '{name}' already exists."
            return response

        try:
            self._arm.validate_link_name(request.link_name)
            spring = VirtualSpring(
                name=name,
                link_name=request.link_name,
                local_attachment_point=np.array(request.local_point),
                target_world_point=np.array(request.target),
                stiffness=request.stiffness,
                damping=request.damping,
                rest_length=request.rest_length,
                inner_radius=request.inner_radius,
                outer_radius=request.outer_radius,
            )

            
        except ValueError as e:
            response.success = False
            response.message = str(e)
            return response

        # Mirror the parameter namespace so external tools (e.g. visualisation)
        # can see this spring the same way as one loaded from YAML
        prefix = f"springs.{name}"
        params = {
            f"{prefix}.link_name":   (rclpy.parameter.Parameter.Type.STRING,       request.link_name),
            f"{prefix}.local_point": (rclpy.parameter.Parameter.Type.DOUBLE_ARRAY, list(request.local_point)),
            f"{prefix}.target":      (rclpy.parameter.Parameter.Type.DOUBLE_ARRAY, list(request.target)),
            f"{prefix}.stiffness":   (rclpy.parameter.Parameter.Type.DOUBLE,       request.stiffness),
            f"{prefix}.damping":     (rclpy.parameter.Parameter.Type.DOUBLE,       request.damping),
            f"{prefix}.rest_length": (rclpy.parameter.Parameter.Type.DOUBLE,       request.rest_length),
            f"{prefix}.inner_radius": (rclpy.parameter.Parameter.Type.DOUBLE,       request.inner_radius),
            f"{prefix}.outer_radius": (rclpy.parameter.Parameter.Type.DOUBLE,       request.outer_radius),
            
}
        for key, (ptype, value) in params.items():
            self._declare_or_ignore(key, value)
            self.set_parameters([rclpy.parameter.Parameter(key, ptype, value)])
            
        # Also add the name to spring_names so it shows up if someone lists params
        current_names = list(
            self.get_parameter("spring_names").get_parameter_value().string_array_value
    )
        if name not in current_names:
            self.set_parameters([
                rclpy.parameter.Parameter("spring_names",
                                          rclpy.Parameter.Type.STRING_ARRAY,
                                          current_names + [name])
            ])

        spring = VirtualSpring(
            name=name,
            link_name=request.link_name,
            local_attachment_point=np.array(request.local_point),
            target_world_point=np.array(request.target),
            stiffness=request.stiffness,
            damping=request.damping,
            rest_length=request.rest_length,
        )

        self._springs.add(spring)
        for topic, cb in [
            (f"~/target/{name}",     lambda msg, s=spring: self._target_cb(msg, s)),
            (f"~/attachment/{name}", lambda msg, s=spring: self._attachment_cb(msg, s)),
    ]:
            self.create_subscription(PointStamped, topic, cb, 10)
        # TRANSIENT_LOCAL (self._target_qos), matching the startup-time
        # publishers above -- a plain depth=10 default here silently gave
        # springs added at runtime a VOLATILE publisher, mismatching
        # armviz's TRANSIENT_LOCAL subscription. Confirmed live 2026-08-19:
        # "New publisher discovered ... offering incompatible QoS ...
        # DURABILITY" the moment a spring was added from the UI.
        self._target_pubs[name] = self.create_publisher(
            PointStamped, f"~/target/{name}", self._target_qos
        )
        self._publish_target(spring)

        response.success = True
        response.message = f"Spring '{name}' added."
        response.id = len(self._springs) - 1
        self.get_logger().info(response.message)
        self._publish_springs_updated()
        self._check_equilibrium_shift()
        return response

    def _add_joint_spring_cb(
        self, request: AddJointSpring.Request, response: AddJointSpring.Response
    ) -> AddJointSpring.Response:
        name = request.name.strip()
        if not name:
            response.success = False
            response.message = "Joint spring name must not be empty."
            return response

        if any(s.name == name for s in self._springs):
            response.success = False
            response.message = f"Spring '{name}' already exists."
            return response

        try:
            self._arm.validate_joint_name(request.joint_name)
            spring = JointSpring(
                name=name,
                joint_name=request.joint_name,
                target_angle=request.target_angle,
                stiffness=request.stiffness,
                damping=request.damping,
            )
        except ValueError as e:
            response.success = False
            response.message = str(e)
            return response

        # Mirror the parameter namespace so external tools (e.g. visualisation)
        # can see this spring the same way as one loaded from YAML
        prefix = f"joint_springs.{name}"
        params = {
            f"{prefix}.joint_name":   (rclpy.parameter.Parameter.Type.STRING, request.joint_name),
            f"{prefix}.target_angle": (rclpy.parameter.Parameter.Type.DOUBLE, request.target_angle),
            f"{prefix}.stiffness":    (rclpy.parameter.Parameter.Type.DOUBLE, request.stiffness),
            f"{prefix}.damping":      (rclpy.parameter.Parameter.Type.DOUBLE, request.damping),
        }
        for key, (ptype, value) in params.items():
            self._declare_or_ignore(key, value)
            self.set_parameters([rclpy.parameter.Parameter(key, ptype, value)])

        # Also add the name to joint_spring_names so it shows up if someone lists params
        current_names = list(
            self.get_parameter("joint_spring_names").get_parameter_value().string_array_value
        )
        if name not in current_names:
            self.set_parameters([
                rclpy.parameter.Parameter("joint_spring_names",
                                          rclpy.Parameter.Type.STRING_ARRAY,
                                          current_names + [name])
            ])

        self._springs.add(spring)
        topic = f"~/joint_target/{name}"
        self.create_subscription(
            Float64, topic, lambda msg, s=spring: self._joint_target_cb(msg, s), 10
        )

        response.success = True
        response.message = f"Joint spring '{name}' added."
        response.id = len(self._springs) - 1
        self.get_logger().info(response.message)
        self._publish_springs_updated()
        self._check_equilibrium_shift()
        return response

    def _add_orientation_spring_cb(
        self, request: AddOrientationSpring.Request, response: AddOrientationSpring.Response
    ) -> AddOrientationSpring.Response:
        name = request.name.strip()
        if not name:
            response.success = False
            response.message = "Orientation spring name must not be empty."
            return response

        if any(s.name == name for s in self._springs):
            response.success = False
            response.message = f"Spring '{name}' already exists."
            return response

        try:
            self._arm.validate_link_name(request.link_name)
            spring = OrientationSpring(
                name=name,
                link_name=request.link_name,
                local_attachment_point=np.array(request.local_point),
                local_face_normal=np.array(request.local_face_normal),
                target_world_point=np.array(request.target),
                stiffness=request.stiffness,
                damping=request.damping,
            )
        except ValueError as e:
            response.success = False
            response.message = str(e)
            return response

        # Mirror the parameter namespace so external tools (e.g. visualisation)
        # can see this spring the same way as one loaded from YAML
        prefix = f"orientation_springs.{name}"
        params = {
            f"{prefix}.link_name":         (rclpy.parameter.Parameter.Type.STRING,       request.link_name),
            f"{prefix}.local_point":       (rclpy.parameter.Parameter.Type.DOUBLE_ARRAY, list(request.local_point)),
            f"{prefix}.local_face_normal": (rclpy.parameter.Parameter.Type.DOUBLE_ARRAY, list(request.local_face_normal)),
            f"{prefix}.target":            (rclpy.parameter.Parameter.Type.DOUBLE_ARRAY, list(request.target)),
            f"{prefix}.stiffness":         (rclpy.parameter.Parameter.Type.DOUBLE,       request.stiffness),
            f"{prefix}.damping":           (rclpy.parameter.Parameter.Type.DOUBLE,       request.damping),
        }
        for key, (ptype, value) in params.items():
            self._declare_or_ignore(key, value)
            self.set_parameters([rclpy.parameter.Parameter(key, ptype, value)])

        # Also add the name to orientation_spring_names so it shows up if someone lists params
        current_names = list(
            self.get_parameter("orientation_spring_names").get_parameter_value().string_array_value
        )
        if name not in current_names:
            self.set_parameters([
                rclpy.parameter.Parameter("orientation_spring_names",
                                          rclpy.Parameter.Type.STRING_ARRAY,
                                          current_names + [name])
            ])

        self._springs.add(spring)
        for topic, cb in [
            (f"~/target/{name}",     lambda msg, s=spring: self._target_cb(msg, s)),
            (f"~/attachment/{name}", lambda msg, s=spring: self._attachment_cb(msg, s)),
        ]:
            self.create_subscription(PointStamped, topic, cb, 10)
        # TRANSIENT_LOCAL (self._target_qos), matching the startup-time
        # publishers above -- a plain depth=10 default here silently gave
        # springs added at runtime a VOLATILE publisher, mismatching
        # armviz's TRANSIENT_LOCAL subscription. Confirmed live 2026-08-19:
        # "New publisher discovered ... offering incompatible QoS ...
        # DURABILITY" the moment a spring was added from the UI.
        self._target_pubs[name] = self.create_publisher(
            PointStamped, f"~/target/{name}", self._target_qos
        )
        self._publish_target(spring)

        response.success = True
        response.message = f"Orientation spring '{name}' added."
        response.id = len(self._springs) - 1
        self.get_logger().info(response.message)
        self._publish_springs_updated()
        self._check_equilibrium_shift()
        return response

    def _load_springs_from_params(self) -> None:
        self._declare_or_ignore("spring_names", [""])
        spring_names = (
            self.get_parameter("spring_names")
            .get_parameter_value()
            .string_array_value
        )
        self.get_logger().info(f"Spring names from params: {list(spring_names)}")
        for name in spring_names:
            if not name:
                continue
            try:
                spring = self._load_one_spring(name)
            except (ValueError, RuntimeError) as e:
                self.get_logger().fatal(f"Spring '{name}' failed to load: {e}")
                raise
            self._springs.add(spring)
            self.get_logger().info(f"Loaded spring: {spring}")

        self._declare_or_ignore("joint_spring_names", [""])
        joint_spring_names = (
            self.get_parameter("joint_spring_names")
            .get_parameter_value()
            .string_array_value
        )
        self.get_logger().info(f"Joint spring names from params: {list(joint_spring_names)}")
        for name in joint_spring_names:
            if not name:
                continue
            try:
                joint_spring = self._load_one_joint_spring(name)
            except (ValueError, RuntimeError) as e:
                self.get_logger().fatal(f"Joint spring '{name}' failed to load: {e}")
                raise
            self._springs.add(joint_spring)
            self.get_logger().info(f"Loaded joint spring: {joint_spring}")

        self._declare_or_ignore("orientation_spring_names", [""])
        orientation_spring_names = (
            self.get_parameter("orientation_spring_names")
            .get_parameter_value()
            .string_array_value
        )
        self.get_logger().info(f"Orientation spring names from params: {list(orientation_spring_names)}")
        for name in orientation_spring_names:
            if not name:
                continue
            try:
                orientation_spring = self._load_one_orientation_spring(name)
            except (ValueError, RuntimeError) as e:
                self.get_logger().fatal(f"Orientation spring '{name}' failed to load: {e}")
                raise
            self._springs.add(orientation_spring)
            self.get_logger().info(f"Loaded orientation spring: {orientation_spring}")

    def _remove_spring_cb(
        self, request: RemoveSpring.Request, response: RemoveSpring.Response
) -> RemoveSpring.Response:
        name = request.name.strip()

        spring = next((s for s in self._springs if s.name == name), None)
        if spring is None:
            response.success = False
            response.message = f"Spring '{name}' not found."
            return response

        self._springs.remove(name)

        # Which names-list parameter and per-spring parameter keys to clean
        # up depends on the spring's type -- each type lives under its own
        # prefix and its own <type>_names list (spring_names / joint_spring_
        # names / orientation_spring_names). Previously this always assumed
        # VirtualSpring's springs.<name>.* prefix and only ever edited
        # spring_names, so removing a JointSpring left it in
        # joint_spring_names (it would reload on restart) and its
        # joint_springs.<name>.* parameters undeclared -- never actually
        # exercised until OrientationSpring made the type-specific cleanup
        # unavoidable to get right for a third type too.
        if isinstance(spring, VirtualSpring):
            names_param = "spring_names"
            prefix = f"springs.{name}"
            keys = [
                f"{prefix}.link_name", f"{prefix}.local_point", f"{prefix}.target",
                f"{prefix}.stiffness", f"{prefix}.damping", f"{prefix}.rest_length",
                f"{prefix}.inner_radius", f"{prefix}.outer_radius",
            ]
        elif isinstance(spring, JointSpring):
            names_param = "joint_spring_names"
            prefix = f"joint_springs.{name}"
            keys = [
                f"{prefix}.joint_name", f"{prefix}.target_angle",
                f"{prefix}.stiffness", f"{prefix}.damping",
            ]
        else:  # OrientationSpring
            names_param = "orientation_spring_names"
            prefix = f"orientation_springs.{name}"
            keys = [
                f"{prefix}.link_name", f"{prefix}.local_point",
                f"{prefix}.local_face_normal", f"{prefix}.target",
                f"{prefix}.stiffness", f"{prefix}.damping",
            ]

        current_names = list(
            self.get_parameter(names_param).get_parameter_value().string_array_value
        )
        self.set_parameters([
            rclpy.parameter.Parameter(
                names_param,
                rclpy.parameter.Parameter.Type.STRING_ARRAY,
                [n for n in current_names if n != name]
            )
        ])

        for key in keys:
            try:
                self.undeclare_parameter(key)
            except rclpy.exceptions.ParameterNotDeclaredException:
                pass

        response.success = True
        response.message = f"Spring '{name}' removed."
        self.get_logger().info(response.message)
        self._publish_springs_updated()
        return response

    def _update_spring_cb(
        self, request: UpdateSpring.Request, response: UpdateSpring.Response
    ) -> UpdateSpring.Response:
        """
        Adjust an existing spring's stiffness/damping in place -- unlike
        add_spring, which refuses if the name already exists, so this is
        the only way to change these without remove-then-re-add (which
        would also momentarily drop the spring's force entirely). Doesn't
        change the spring's target/attachment/geometry, just k and b.
        """
        name = request.name.strip()
        spring = next((s for s in self._springs if s.name == name), None)
        if spring is None:
            response.success = False
            response.message = f"Spring '{name}' not found."
            return response

        if request.stiffness < 0:
            response.success = False
            response.message = f"stiffness must be >= 0, got {request.stiffness}."
            return response
        if request.damping < 0:
            response.success = False
            response.message = f"damping must be >= 0, got {request.damping}."
            return response

        spring.stiffness = request.stiffness
        spring.damping = request.damping

        # Mirror into the parameter namespace, same as add_spring -- keeps
        # it in sync with the live spring instead of going stale. Prefix
        # depends on spring type (same dispatch as _remove_spring_cb) --
        # hardcoding springs.<name> here would silently write a bogus
        # parameter under the wrong prefix for a JointSpring/
        # OrientationSpring while leaving its real one stale.
        if isinstance(spring, VirtualSpring):
            prefix = f"springs.{name}"
        elif isinstance(spring, JointSpring):
            prefix = f"joint_springs.{name}"
        else:  # OrientationSpring
            prefix = f"orientation_springs.{name}"
        for key, value in [
                (f"{prefix}.stiffness", request.stiffness),
                (f"{prefix}.damping", request.damping),
        ]:
            self._declare_or_ignore(key, value)
            self.set_parameters([
                rclpy.parameter.Parameter(key, rclpy.parameter.Parameter.Type.DOUBLE, value)
            ])

        response.success = True
        response.message = (
            f"Spring '{name}' updated: stiffness={request.stiffness}, damping={request.damping}."
        )
        self.get_logger().info(response.message)
        self._publish_springs_updated()
        return response

    def _publish_springs_updated(self) -> None:
        import json
        msg = String()
        msg.data = json.dumps([s.name for s in self._springs])
        self._springs_updated_pub.publish(msg)

    def _publish_springs_enabled(self) -> None:
        # All springs are always enabled/disabled together (see
        # _set_springs_enabled) -- any one of them reflects the aggregate
        # state, defaulting True if there are no springs loaded at all yet.
        # SpringCollection supports iteration/len but not indexing (no
        # __getitem__) -- confirmed live 2026-08-20, self._springs[0]
        # crashed the node on startup with "not subscriptable".
        first = next(iter(self._springs), None)
        enabled = first.enabled if first is not None else True
        self._springs_enabled_pub.publish(Bool(data=enabled))

    def _publish_collision_thresholds(self) -> None:
        # [danger_threshold, caution_threshold, repulsion_max_force_n], in
        # that fixed order -- consumed by armviz to keep the caution-halo
        # padding in sync with live updates (see ~/collision_thresholds_status
        # in __init__ for why this exists).
        self._collision_thresholds_pub.publish(Float64MultiArray(data=[
            self._arm.danger_threshold, self._caution_threshold,
            self._repulsion_max_force_n,
        ]))

    def _publish_safety_status(self, collision) -> None:
        """
        Broadcast the current self/scene-collision state on
        ~/safety_status, prefixed "SAFE"/"CAUTION"/"DANGER"/"COLLISION" so a
        plain prefix check is enough for a consumer (e.g.
        wait_and_enable_torque.py's pre-flight check, which treats anything
        not starting with "SAFE" as unsafe) to act on it without parsing
        distances or link names.

        CAUTION mirrors the repulsion field's caution_threshold (see
        get_repulsion_torques) -- reported whenever the closest pair is
        inside that radius but not yet in_danger, regardless of whether
        repulsion_enabled is actually on, since it's useful early warning
        either way.
        """
        msg = String()
        if collision is None:
            msg.data = "SAFE (no collision pairs reported)"
        else:
            a, b = collision.closest_pair
            if collision.in_collision:
                msg.data = f"COLLISION: {a}/{b} dist={collision.min_distance:.4f}m"
            elif collision.in_danger:
                msg.data = (
                    f"DANGER: {a}/{b} dist={collision.min_distance:.4f}m "
                    f"scale={collision.scale_factor:.2f}"
                )
            elif collision.min_distance < self._caution_threshold:
                msg.data = f"CAUTION: {a}/{b} dist={collision.min_distance:.4f}m"
            else:
                msg.data = f"SAFE (closest: {a}/{b} dist={collision.min_distance:.4f}m)"
        # Append, never prepend -- consumers (e.g. wait_and_enable_torque.py)
        # only check the SAFE/DANGER/COLLISION prefix, so this can't break
        # that contract.
        if self._springs_auto_disabled:
            msg.data += " (springs disabled by collision -- re-enable via ~/enable)"
        if self._torque_status_topic and self._torque_control_connected is False:
            msg.data += (
                " (gen3_torque_control UNREACHABLE -- torque commands are "
                "not being applied; torque-enabled status shown elsewhere "
                "may be stale)"
            )
        self._safety_status_pub.publish(msg)

        points_msg = Float64MultiArray()
        if collision is not None:
            points_msg.data = (
                list(collision.closest_point_a) + list(collision.closest_point_b)
            )
        self._closest_points_pub.publish(points_msg)

    def destroy_node(self) -> None:
        if self._collision_log_file is not None:
            self._collision_log_file.close()
        super().destroy_node()

    def plot_spring_extensions(self):
        num_springs = len(self.spring_data)
        if num_springs > 0:
            fig, axes = plt.subplots(num_springs, 2, figsize=(14, 4 * num_springs), squeeze=False)
        else:
            self.get_logger().info("No spring data recorded; nothing to plot.")
            return

        for row, (name, data) in enumerate(self.spring_data.items()):
            # --- Left column: extension ---
            axes[row, 0].plot(data['times'], data['extensions'])
            axes[row, 0].set_title(f"{name} — extension")
            axes[row, 0].set_xlabel('Time (s)')
            axes[row, 0].set_ylabel('Extension (m)')
            axes[row, 0].grid(True)

            # --- Right column: torques (one line per DOF) ---
            torques_over_time = np.array(data['torques'])  # shape: (n_timesteps, n_dof)
            for dof_idx in range(torques_over_time.shape[1]):
                joint_name = self._arm.joint_names[dof_idx]
                axes[row, 1].plot(data['times'], torques_over_time[:, dof_idx], label=joint_name)
            axes[row, 1].set_title(f"{name} — torques")
            axes[row, 1].set_xlabel('Time (s)')
            axes[row, 1].set_ylabel('Torque (Nm)')
            axes[row, 1].legend(fontsize='small')
            axes[row, 1].grid(True)

        plt.suptitle('Spring Data Over Time', fontsize=14)
        plt.tight_layout()
        plot_path = self.get_parameter("plot_output_path").get_parameter_value().string_value
        fig.savefig(plot_path, dpi=150)
        self.get_logger().info(f"Plot saved to {plot_path}")
        # Deliberately no plt.show(): this runs in main()'s shutdown path on
        # a headless background node -- a blocking GUI call with no timeout
        # there would hang shutdown indefinitely (no display, or a window
        # that never gets closed) needing a SIGQUIT to escape, the same
        # class of bug fixed in gen3_torque_control's shutdown path
        # 2026-08-19. savefig() above is enough to inspect the plot after
        # the fact.

def main(args=None):
    rclpy.init(args=args)
    node = VirtualSpringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.get_parameter("plot_on_shutdown").get_parameter_value().bool_value:
            node.plot_spring_extensions()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
