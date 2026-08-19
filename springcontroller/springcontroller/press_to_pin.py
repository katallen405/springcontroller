#!/usr/bin/env python3
"""
press_to_pin.py

ROS 2 node that lets a person add a virtual spring by physically pressing
on a link of the arm. When a sustained press is detected, a spring is
created (via the virtual_spring_node add_spring service) attaching that
link to its current position in space, with stiffness proportional to the
force being applied at the moment the press latches.

How it works
------------
1. External torque residual.  On every JointState update:

       tau_ext = tau_measured - tau_gravity(model) - tau_commanded - baseline

   where tau_commanded is the most recent output of virtual_spring_node
   (the arm is under effort control, so measured efforts contain the
   commanded spring torques) and `baseline` is a slowly adapting per-joint
   exponential moving average that absorbs friction, current-to-torque
   scale error, and any static model mismatch. The baseline freezes while
   a press is suspected or the arm is moving, so the press itself is not
   absorbed into it.

2. Contact localization.  A force applied to link i produces torque only
   on joints 1..i; joints distal to the contact see nothing, and the
   torque grows toward the base via the longer lever arm. Because of that
   amplification, picking "the most distal joint whose own residual
   exceeds threshold" is unreliable in practice: a real press near a
   distal link can clear threshold on a more proximal joint while the
   distal joint's own, smaller-but-real share never clears the same
   threshold, producing a confidently wrong (too-proximal) localization.
   Instead, every link is scored by how well a single point force at that
   link's origin, fit by least squares, reconstructs the *whole* observed
   residual vector (see _localize_contact); the best-fitting link wins,
   and a near-tie between two non-adjacent links suppresses the press
   rather than guessing.

3. Force estimate and stiffness.  The applied force is estimated from the
   residual via the contact link's translational Jacobian:

       F = pinv(Jv^T) @ tau_ext        (evaluated at the link origin)

   and stiffness is  k = clip(k_per_newton * ||F||, k_min, k_max).
   Since the true contact point along the link is unknown, ||F|| is an
   approximation — good enough for a proportional design knob.

4. Trigger state machine (per node, not per joint):

       ARMED       press candidate must persist on the same joint for
                   press_hold_time before latching
       LATCH       read link world position, call add_spring
                   (target = current position, rest_length = 0)
       REFRACTORY  no new pins until all residuals fall below
                   release_fraction * threshold for rearm_time

Subscriptions
-------------
/joint_states                        (sensor_msgs/JointState)
    Positions, velocities, and efforts from the arm.
<commanded_torque_topic>             (sensor_msgs/JointState)
    Commanded spring torques from virtual_spring_node
    (default /virtual_spring/joint_torques).

Publications
------------
~/press_events   (std_msgs/String, JSON)
    One message per detected press: link, force estimate, stiffness,
    spring name. Intended for GUI / logging consumers.
~/residuals      (sensor_msgs/JointState)
    Debug: the current external-torque residual per joint (effort field).
    Useful for tuning thresholds with rqt_plot / PlotJuggler.

Services
--------
~/enable   (std_srvs/SetBool)
    Enable or disable press detection (enabled by default; see the
    start_enabled parameter to start disabled).

Service clients
---------------
<add_spring_service>   (springcontroller_interfaces/AddSpring)
    Default /virtual_spring_node/add_spring.

Parameters
----------
urdf_path : str
    Fallback URDF path if /robot_description is not being published.
add_spring_service : str
    Name of the AddSpring service (default /virtual_spring_node/add_spring).
commanded_torque_topic : str
    Topic carrying commanded spring torques (default
    /virtual_spring/joint_torques).
subtract_model_gravity : bool
    Subtract pinocchio gravity torques from measured efforts (default
    true). On arms whose reported efforts already exclude gravity, set
    false — the baseline will absorb small leftovers either way.
start_enabled : bool
    Whether press detection (and therefore pin creation) is active at
    startup (default true). When false, the node still comes up and
    ~/enable can be called to turn it on later.
thresholds.<joint_name> : double
    Per-joint press threshold in the units of the effort field. Any joint
    without an explicit threshold uses thresholds.default.
thresholds.default : double
    Fallback threshold (default 1.5).
press.hold_time : double
    Seconds a press must persist before a spring is created (default 0.25).
press.release_fraction : double
    Residuals must fall below release_fraction * threshold to re-arm
    (default 0.5, i.e. hysteresis).
press.rearm_time : double
    Seconds residuals must stay below the release level before a new
    press can be detected (default 0.5).
press.velocity_gate : double
    Max |qdot| (rad/s) above which detection is suspended entirely —
    fast motion makes the quasi-static residual meaningless (default 1.0).
ambiguity.margin : double
    Minimum gap (as a fraction of ||residual||) the best-fitting link's
    reconstruction error must have over the second-best before a press is
    latched, when the two best candidates are different, non-adjacent
    links. Below this margin the press is treated as ambiguous and
    suppressed rather than guessed at (default 0.15).
baseline.alpha : double
    EMA coefficient per update for the residual baseline (default 0.005).
baseline.freeze_fraction : double
    Baseline stops adapting on a joint whose raw residual exceeds
    freeze_fraction * threshold (default 0.5).
baseline.velocity_freeze : double
    Baseline stops adapting when any |qdot| exceeds this (default 0.05).
stiffness.per_newton : double
    N/m of spring stiffness per newton of estimated press force
    (default 20.0).
stiffness.min / stiffness.max : double
    Clamp on created spring stiffness (defaults 10.0 / 300.0).
spring.damping : double
    Damping for created springs (default 5.0).
contact_links.<joint_name> : str
    Optional override mapping a joint to the link frame used for the
    spring attachment. By default the child BODY frame of the joint in
    the pinocchio model is used.
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

import pinocchio as pin

from springcontroller.urdf_arm_configuration import URDFArmConfiguration
from springcontroller_interfaces.srv import AddSpring


# ---------------------------------------------------------------------------
# URDF fetch (same approach as virtual_spring_node)
# ---------------------------------------------------------------------------

def fetch_robot_description(node: Node, timeout_sec: float = 3.0) -> str | None:
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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

# Detection state machine states
ARMED = "ARMED"
REFRACTORY = "REFRACTORY"


class PressToPin(Node):

    def __init__(self):
        super().__init__("press_to_pin")

        # ---------------- parameters ----------------
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("add_spring_service",
                               "/virtual_spring_node/add_spring")
        self.declare_parameter("commanded_torque_topic",
                               "/virtual_spring/joint_torques")
        self.declare_parameter("subtract_model_gravity", True)
        self.declare_parameter("start_enabled", True)

        self.declare_parameter("thresholds.default", 1.5)

        self.declare_parameter("press.hold_time", 0.25)
        self.declare_parameter("press.release_fraction", 0.5)
        self.declare_parameter("press.rearm_time", 0.5)
        self.declare_parameter("press.velocity_gate", 1.0)

        self.declare_parameter("ambiguity.margin", 0.15)

        self.declare_parameter("baseline.alpha", 0.005)
        self.declare_parameter("baseline.freeze_fraction", 0.5)
        self.declare_parameter("baseline.velocity_freeze", 0.05)

        self.declare_parameter("stiffness.per_newton", 20.0)
        self.declare_parameter("stiffness.min", 10.0)
        self.declare_parameter("stiffness.max", 300.0)
        self.declare_parameter("spring.damping", 5.0)

        p = lambda name: self.get_parameter(name).value
        self._subtract_gravity = bool(p("subtract_model_gravity"))
        self._hold_time = float(p("press.hold_time"))
        self._release_fraction = float(p("press.release_fraction"))
        self._rearm_time = float(p("press.rearm_time"))
        self._velocity_gate = float(p("press.velocity_gate"))
        self._ambiguity_margin = float(p("ambiguity.margin"))
        self._baseline_alpha = float(p("baseline.alpha"))
        self._baseline_freeze_fraction = float(p("baseline.freeze_fraction"))
        self._baseline_velocity_freeze = float(p("baseline.velocity_freeze"))
        self._k_per_newton = float(p("stiffness.per_newton"))
        self._k_min = float(p("stiffness.min"))
        self._k_max = float(p("stiffness.max"))
        self._spring_damping = float(p("spring.damping"))

        # ---------------- model ----------------
        self.get_logger().info("Loading URDF...")
        self._arm = self._load_arm()
        self._joint_names = self._arm.joint_names
        n = self._arm.n_dof
        self.get_logger().info(
            f"URDF loaded, {n} DOF: {self._joint_names}"
        )

        # Per-joint thresholds: thresholds.<joint> overrides thresholds.default
        default_thresh = float(p("thresholds.default"))
        self._thresholds = np.full(n, default_thresh)
        for i, name in enumerate(self._joint_names):
            key = f"thresholds.{name}"
            self.declare_parameter(key, default_thresh)
            self._thresholds[i] = float(self.get_parameter(key).value)
        self.get_logger().info(
            "Press thresholds: "
            + ", ".join(f"{jn}={t:.2f}"
                        for jn, t in zip(self._joint_names, self._thresholds))
        )

        # Joint -> contact link mapping (child BODY frame of each joint,
        # overridable via contact_links.<joint_name>)
        self._contact_links = self._build_contact_link_map()
        self.get_logger().info(
            "Contact links: "
            + ", ".join(f"{jn}->{ln}"
                        for jn, ln in zip(self._joint_names,
                                          self._contact_links))
        )

        # ---------------- detection state ----------------
        self._enabled = bool(p("start_enabled"))
        self._joint_order: list[int] | None = None
        self._baseline = np.zeros(n)
        self._baseline_initialized = False
        self._commanded = np.zeros(n)
        self._commanded_received = False
        self._state = ARMED
        self._candidate_link: str | None = None
        self._candidate_since: rclpy.time.Time | None = None
        self._quiet_since: rclpy.time.Time | None = None
        self._pin_counter = 0
        self._pending_request = False

        # ---------------- ROS interfaces ----------------
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self._cmd_sub = self.create_subscription(
            JointState,
            str(p("commanded_torque_topic")),
            self._commanded_cb,
            10,
        )
        self._event_pub = self.create_publisher(String, "~/press_events", 10)
        self._residual_pub = self.create_publisher(
            JointState, "~/residuals", 10
        )
        self._enable_srv = self.create_service(
            SetBool, "~/enable", self._enable_cb
        )

        service_name = str(p("add_spring_service"))
        self._add_spring_client = self.create_client(AddSpring, service_name)
        self.get_logger().info(
            f"PressToPin ready. Waiting for {service_name}..."
        )
        if not self._add_spring_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warning(
                f"{service_name} not available yet — will keep trying when "
                "a press is detected."
            )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_arm(self) -> URDFArmConfiguration:
        urdf_xml = fetch_robot_description(self)
        if urdf_xml is not None:
            self.get_logger().info("Loaded URDF from /robot_description topic.")
            return URDFArmConfiguration.from_xml_string(urdf_xml)

        urdf_path = self.get_parameter("urdf_path").value
        if not urdf_path:
            raise RuntimeError(
                "No /robot_description topic found and urdf_path not set."
            )
        self.get_logger().warning(
            f"No /robot_description topic; falling back to file: {urdf_path}"
        )
        return URDFArmConfiguration.from_urdf(urdf_path)

    def _build_contact_link_map(self) -> list[str]:
        """
        For each actuated joint, find the link frame a press on that joint's
        segment should attach to: the first BODY frame whose parent joint is
        that joint. Overridable per joint via contact_links.<joint_name>.
        """
        model = self._arm.model
        links: list[str] = []
        for i, joint_name in enumerate(self._joint_names):
            joint_id = model.getJointId(joint_name)
            default_link = ""
            for frame in model.frames:
                if (frame.type == pin.FrameType.BODY
                        and frame.parentJoint == joint_id):
                    default_link = frame.name
                    break
            key = f"contact_links.{joint_name}"
            self.declare_parameter(key, default_link)
            link = str(self.get_parameter(key).value)
            if not link:
                raise RuntimeError(
                    f"No BODY frame found for joint '{joint_name}' and no "
                    f"contact_links.{joint_name} override provided."
                )
            self._arm.validate_link_name(link)
            links.append(link)
        return links

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _enable_cb(self, request: SetBool.Request,
                   response: SetBool.Response) -> SetBool.Response:
        self._enabled = request.data
        # Reset transient state so a stale candidate can't fire on re-enable
        self._state = ARMED
        self._candidate_link = None
        self._candidate_since = None
        self._quiet_since = None
        response.success = True
        response.message = (
            f"Press-to-pin {'enabled' if request.data else 'disabled'}."
        )
        self.get_logger().info(response.message)
        return response

    def _commanded_cb(self, msg: JointState) -> None:
        """Cache the most recent commanded spring torques, by name."""
        name_to_effort = dict(zip(msg.name, msg.effort))
        self._commanded = np.array(
            [name_to_effort.get(n, 0.0) for n in self._joint_names]
        )
        self._commanded_received = True

    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.effort:
            self.get_logger().warning(
                "JointState has no effort field — press detection needs "
                "joint efforts.",
                throttle_duration_sec=10.0,
            )
            return

        # Don't seed the baseline (or run detection) until we've seen a
        # real commanded-torque sample. Without this, the baseline can
        # latch in with self._commanded still at its zero default if the
        # first /joint_states callback beats the first commanded-torque
        # callback, so the whole gravity-comp torque shows up as a
        # "residual" the moment the real commanded value arrives.
        if not self._commanded_received:
            self.get_logger().info(
                "Waiting for a commanded-torque sample before arming "
                "press detection...",
                throttle_duration_sec=2.0,
            )
            return

        # Build the ROS->pinocchio reorder map on first message
        if self._joint_order is None:
            try:
                ros_names = list(msg.name)
                self._joint_order = [ros_names.index(n)
                                     for n in self._joint_names]
            except ValueError as e:
                self.get_logger().error(f"Joint name mismatch: {e}")
                return

        if len(msg.position) < self._arm.n_dof:
            return

        pos = np.array(msg.position)[self._joint_order]
        vel = (np.array(msg.velocity)[self._joint_order]
               if msg.velocity else np.zeros(self._arm.n_dof))
        eff = np.array(msg.effort)[self._joint_order]

        self._arm.update_from_angles(pos, vel)

        # ---------------- residual ----------------
        raw = eff - self._commanded
        if self._subtract_gravity:
            raw = raw - self._arm.get_gravity_torques()

        if not self._baseline_initialized:
            self._baseline = raw.copy()
            self._baseline_initialized = True
            return

        residual = raw - self._baseline

        # Baseline adaptation: freeze on joints that look pressed, and
        # freeze everywhere while the arm is moving.
        moving = np.max(np.abs(vel)) > self._baseline_velocity_freeze
        if not moving:
            quiet = (np.abs(residual)
                     < self._baseline_freeze_fraction * self._thresholds)
            a = self._baseline_alpha
            self._baseline[quiet] += a * (raw[quiet] - self._baseline[quiet])

        self._publish_residuals(msg, residual)

        if not self._enabled or self._pending_request:
            return

        # Suspend detection entirely during fast motion (recentering, etc.)
        if np.max(np.abs(vel)) > self._velocity_gate:
            self._candidate_link = None
            self._candidate_since = None
            return

        self._run_state_machine(residual)

    # ------------------------------------------------------------------
    # Detection state machine
    # ------------------------------------------------------------------

    def _run_state_machine(self, residual: np.ndarray) -> None:
        now = self.get_clock().now()
        over = np.abs(residual) > self._thresholds
        release = (np.abs(residual)
                   < self._release_fraction * self._thresholds)

        if self._state == REFRACTORY:
            if release.all():
                if self._quiet_since is None:
                    self._quiet_since = now
                elif (now - self._quiet_since).nanoseconds / 1e9 >= self._rearm_time:
                    self._state = ARMED
                    self._quiet_since = None
                    self.get_logger().info("Re-armed.")
            else:
                self._quiet_since = None
            return

        # ARMED
        if not over.any():
            self._candidate_link = None
            self._candidate_since = None
            return

        localized = self._localize_contact(residual)
        if localized is None:
            # Ambiguous this cycle -- don't let a stale candidate keep
            # counting toward hold_time; a later, confident cycle has to
            # hold for the full duration on its own.
            self._candidate_link = None
            self._candidate_since = None
            return

        joint_idx, link, point, _force = localized

        if link != self._candidate_link:
            self._candidate_link = link
            self._candidate_since = now
            return

        held = (now - self._candidate_since).nanoseconds / 1e9
        if held >= self._hold_time:
            self._latch(joint_idx, link, point, residual)

    def _localize_contact(
        self, residual: np.ndarray
    ) -> tuple[int, str, np.ndarray, np.ndarray] | None:
        """
        Pick which link's contact-point candidate best explains the FULL
        observed residual vector, replacing the old "most distal joint over
        threshold" heuristic (see the module docstring for why that broke
        down: measured live, 9 of 11 labeled test presses near distal
        joints/links were misattributed to more proximal ones).

        For every link, evaluate the translational Jacobian at that link's
        origin (the same point _latch already used for its force estimate),
        fit a 3D force there by least squares against the *whole* residual
        vector, and score the link by the leftover reconstruction error
        `||residual - Jv^T @ force||`. A link that isn't the true contact
        can only ever explain the residual components its own Jacobian
        columns reach (columns for joints distal to that link are
        structurally zero at the link's own origin), so the true contact
        link's fit is generally the best one -- it's the only candidate
        whose columns cover the joints actually carrying torque.

        Returns (joint_idx, link_name, local_point, force) for the winning
        candidate. Returns None when the two best-fitting candidates are
        different, non-adjacent links and neither clearly beats the other
        (gap smaller than ambiguity.margin * ||residual||) -- in that case
        the caller should keep waiting rather than guess. Adjacent-link
        near-ties are not treated as ambiguous: a press near a link
        boundary legitimately fits both of its neighbors similarly well,
        and either is a reasonable pin location.

        Exact ties are common at one specific joint, not just noise: a
        candidate link whose frame origin sits exactly on its own joint's
        axis (true of every contact link in this model, by construction --
        see _build_contact_link_map) has a translational Jacobian that's
        identically zero in that joint's own column, so a residual with
        nothing on more proximal joints reconstructs equally well (both
        exactly, via the zero-force solution) from that link or from any
        more proximal one whose Jacobian is *also* all-zero there (e.g.
        joint_1's own link, which is always fully degenerate). Ties are
        broken toward the more distal candidate, matching the old
        heuristic's bias and avoiding a same-fit link that's structurally
        blind to real presses.
        """
        scored = []
        for j, link in enumerate(self._contact_links):
            point = np.zeros(3)
            J = self._arm.get_jacobian(link, point)
            Jv = J[:3, :]
            force = np.linalg.pinv(Jv.T, rcond=1e-3) @ residual
            predicted = Jv.T @ force
            err = float(np.linalg.norm(residual - predicted))
            scored.append((err, j, link, point, force))

        scored.sort(key=lambda t: (t[0], -t[1]))
        best_err, best_j, best_link, best_point, best_force = scored[0]

        if len(scored) > 1:
            second_err, second_j, second_link, _, _ = scored[1]
            if second_link != best_link and abs(second_j - best_j) > 1:
                norm = float(np.linalg.norm(residual)) or 1e-9
                if (second_err - best_err) < self._ambiguity_margin * norm:
                    self.get_logger().warning(
                        f"Ambiguous press: '{best_link}' (err={best_err:.3f}) "
                        f"vs '{second_link}' (err={second_err:.3f}) too "
                        "close -- suppressing pin.",
                        throttle_duration_sec=1.0,
                    )
                    return None

        return best_j, best_link, best_point, best_force

    def _latch(self, joint_idx: int, link: str, point: np.ndarray,
               residual: np.ndarray) -> None:
        # Force estimate: translational Jacobian at the winning candidate
        # point (link origin, currently -- see _localize_contact). Columns
        # for joints distal to the contact are irrelevant (their residuals
        # are ~0), so the pseudoinverse over the full Jacobian is fine.
        J = self._arm.get_jacobian(link, point)
        Jv = J[:3, :]
        force = np.linalg.pinv(Jv.T, rcond=1e-3) @ residual
        force_mag = float(np.linalg.norm(force))

        stiffness = float(
            np.clip(self._k_per_newton * force_mag, self._k_min, self._k_max)
        )

        target = self._arm.get_link_transform(link)[:3, 3]

        self._pin_counter += 1
        name = f"pin_{link}_{self._pin_counter}"

        self.get_logger().info(
            f"Press on {self._joint_names[joint_idx]} -> link '{link}': "
            f"|F|~{force_mag:.1f}, k={stiffness:.1f}, "
            f"target={np.round(target, 3).tolist()}"
        )

        req = AddSpring.Request()
        req.name = name
        req.link_name = link
        req.stiffness = stiffness
        req.damping = self._spring_damping
        req.rest_length = 0.0
        req.local_point = point.tolist()
        req.target = target.tolist()
        req.inner_radius = 0.0
        req.outer_radius = 0.0

        self._pending_request = True
        future = self._add_spring_client.call_async(req)
        future.add_done_callback(
            lambda fut, n=name, l=link, f=force_mag, k=stiffness:
            self._add_spring_done(fut, n, l, f, k)
        )

        self._state = REFRACTORY
        self._candidate_link = None
        self._candidate_since = None
        self._quiet_since = None

    def _add_spring_done(self, future, name: str, link: str,
                         force_mag: float, stiffness: float) -> None:
        self._pending_request = False
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"add_spring call failed: {e}")
            return

        if not result.success:
            self.get_logger().error(
                f"add_spring rejected '{name}': {result.message}"
            )
            return

        self.get_logger().info(f"Spring '{name}' pinned.")
        event = String()
        event.data = json.dumps({
            "spring": name,
            "link": link,
            "force": round(force_mag, 2),
            "stiffness": round(stiffness, 2),
        })
        self._event_pub.publish(event)

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def _publish_residuals(self, src: JointState,
                           residual: np.ndarray) -> None:
        out = JointState()
        out.header.stamp = src.header.stamp
        out.name = self._joint_names
        out.effort = residual.tolist()
        self._residual_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PressToPin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
