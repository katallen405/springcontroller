#!/home/katallen/.springcontroller_venv/bin/python3
"""
virtual_spring_node.py

ROS 2 node that maintains a collection of virtual springs and publishes
the resulting joint torques on each JointState update from the arm.

Subscriptions
-------------
~/joint_states  (sensor_msgs/JointState)
    Current joint positions and velocities from the arm.

Publications
------------
~/joint_torques  (sensor_msgs/JointState)
    Effort field carries the summed virtual-spring torques.

~/target/<spring_name>  (geometry_msgs/PointStamped)
    Update a spring's target at runtime.

~/springs_updated (String)
    Publishes when the list of springs is changed with add_spring or remove_spring (supports spring_viz, maybe later UI)

Services
--------
~/enable   (std_srvs/SetBool)
    Enable or disable all springs at once.

~/add_spring
    Add a spring (springcontroller_interfaces/AddSpring)

~/remove_spring (springcontroller_interfaces/RemoveSpring)
    Remove a spring from the torque calculations


Parameters
----------
urdf_path   : str  -- path to the robot URDF or XACRO file
config_path : str  -- path to the springs YAML file (checked at startup)
springs.<n>.*     -- per-spring config (see config/springs.yaml)
"""

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
from std_msgs.msg import String, Float64
import os
import threading

from springcontroller.virtual_spring import VirtualSpring, JointSpring, SpringCollection
from springcontroller.urdf_arm_configuration import URDFArmConfiguration
import yaml
from springcontroller_interfaces.srv import (
    AddSpring, RemoveSpring, AddJointSpring, LoadCollisionScene,
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
        self.declare_parameter("danger_threshold", 0.05)
        self.declare_parameter("locked_joint_names", [""])
        self.declare_parameter("torque_disable_service", "")
        self.declare_parameter("collision_object_topic", "~/collision_object")
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
        self._load_springs_from_params()

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

        self._springs_updated_pub = self.create_publisher(String,
                                            "~/springs_updated", 10)

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
        # Removal is shared: RemoveSpring is name-based and doesn't care
        # about spring type, so it works for joint springs too.
        self._add_joint_spring_srv = self.create_service(
            AddJointSpring, "~/add_joint_spring", self._add_joint_spring_cb
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

        self._load_collision_scene_srv = self.create_service(
            LoadCollisionScene, "~/load_collision_scene", self._load_collision_scene_cb
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
            elif isinstance(spring, JointSpring):
                topic = f"~/joint_target/{spring.name}"
                self.create_subscription(
                    Float64,
                    topic,
                    lambda msg, s=spring: self._joint_target_cb(msg, s),
                    10,
                )
                self.get_logger().info(f"Listening for target updates on {topic}")

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


        # Expose a service to toggle at runtime without restarting
        self._grav_comp_srv = self.create_service(
            SetBool, "~/set_gravity_compensation", self._set_grav_comp_cb
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
        self._declare_or_ignore(f"{prefix}.target",      [0.0, 0.0, 0.0])
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

        spring = VirtualSpring(
            link_name=link_name,
            local_attachment_point=np.array(_get("local_point", [0, 0, 0])),
            target_world_point=np.array(_get("target", [0, 0, 0])),
            stiffness=float(_get("stiffness", 0.0)),
            damping=float(_get("damping", 0.0)),
            rest_length=float(_get("rest_length", 0.0)),
            name=name,
            inner_radius=float(_get("inner_radius",0.0)),
            outer_radius=float(_get("outer_radius",0.0)),
    )
        return spring

    def _load_one_joint_spring(self, name: str) -> JointSpring:
        """Load a single joint spring by name from current parameters. Raises on error."""
        prefix = f"joint_springs.{name}"
        self._declare_or_ignore(f"{prefix}.joint_name",   "")
        # NaN is a sentinel for "not set in config" -- defaults to the
        # joint's current angle at load time (a soft hold-here spring)
        # rather than requiring every joint spring to specify a target.
        self._declare_or_ignore(f"{prefix}.target_angle", float("nan"))
        self._declare_or_ignore(f"{prefix}.stiffness",    0.0)
        self._declare_or_ignore(f"{prefix}.damping",      0.0)

        def _get(key, default=None, _prefix=prefix):
            val = self.get_parameter(f"{_prefix}.{key}").value
            return default if val is None else val

        joint_name = _get("joint_name")
        self._arm.validate_joint_name(joint_name)  # raises ValueError if bad

        target_angle = float(_get("target_angle", float("nan")))
        if math.isnan(target_angle):
            joint_index = self._arm.joint_names.index(joint_name)
            target_angle = self._arm.get_joint_angle(joint_index)
            self.get_logger().info(
                f"Joint spring '{name}': no target_angle configured, "
                f"defaulting to current angle ({target_angle:.4f} rad)."
            )

        spring = JointSpring(
            joint_name=joint_name,
            target_angle=target_angle,
            stiffness=float(_get("stiffness", 0.0)),
            damping=float(_get("damping", 0.0)),
            name=name,
        )
        return spring
    def _joint_state_cb(self, msg: JointState) -> None:
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
                self._torque_order = [pinocchio_names.index(n) for n in ros_names]
                self.get_logger().info(f"Torque order map (Pinocchio->ROS): {list(zip(ros_names, self._torque_order))}")
            except ValueError as e:
                self.get_logger().error(f"Joint name mismatch: {e}. ROS names: {ros_names}, Pinocchio names: {pinocchio_names}")
                return

        if len(msg.position) != self._arm.n_dof:
            self.get_logger().warn(
                f"Expected {self._arm.n_dof} joints, got {len(msg.position)}. Skipping."
        )
            return

        # Reorder positions and velocities to match Pinocchio's joint ordering
        pos = np.array(msg.position)
        vel = np.array(msg.velocity) if msg.velocity else np.zeros(self._arm.n_dof)
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

            # compute the actual torques generated by the full set of springs
            # if self._add_grav_comp is true, include calculations for gravity compensation
            torques = self._springs.compute_total_torques(self._arm, self._add_grav_comp)

            # Self-collision safety scaling
            collision = self._arm.get_collision_status()
            if collision is not None:
                a, b = collision.closest_pair
                involves_scene_object = (
                    self._arm.is_environment_object(a)
                    or self._arm.is_environment_object(b)
                )
                kind = "COLLISION WITH SCENE OBJECT" if involves_scene_object else "SELF-COLLISION"
                if collision.in_collision:
                    self.get_logger().error(
                        f"{kind} detected between {a} and {b}! Zeroing torques.",
                        throttle_duration_sec=1.0,
                    )
                    torques = np.zeros_like(torques)
                elif collision.in_danger:
                    label = "Near scene-object collision" if involves_scene_object else "Near self-collision"
                    self.get_logger().warn(
                        f"{label}: {a} / {b}, "
                        f"dist={collision.min_distance:.3f}m, "
                        f"scale={collision.scale_factor:.2f}",
                        throttle_duration_sec=0.5,
                    )
                    torques *= collision.scale_factor

            # logging to the terminal for debugging and to plot later
            for spring in self._springs:
                if spring._last_state is not None:
                    """
                    self.get_logger().info(
                        f"{spring.name} attachment: {spring._last_state.world_attachment_point}",
                        throttle_duration_sec=1.0
                )

                    self.get_logger().info(
                        f"{spring.name} displacement: {spring._last_state.displacement} "
                        f"extension: {spring._last_state.extension:.4f}m",
                        throttle_duration_sec=1.0
                    )

"""
                    # Append to time-series
                    self.spring_data[spring.name]['times'].append(elapsed)
                    self.spring_data[spring.name]['extensions'].append(spring._last_state.extension)
                    self.spring_data[spring.name]['torques'].append(spring._last_state.torques)

                    #self.get_logger().info(f"Total torques: {torques}", throttle_duration_sec=1.0)
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

        # Jacobian Debugging
        #for name in self._arm.link_names:
        #    T = self._arm.get_link_transform(name)
        #    print(f"{name}: {T[:3, 3]}")

    def _trigger_torque_disable_failsafe(self) -> None:
        """
        On repeated spring-computation failure, switch the arm out of
        torque control entirely via torque_disable_service (if configured)
        rather than keep trusting our own torque computation. Fires once;
        requires a manual re-enable afterward rather than auto-recovering,
        since a control loop that just proved unreliable shouldn't decide
        on its own when it's safe to resume.
        """
        if self._torque_disable_client is None or self._failsafe_triggered:
            return
        self._failsafe_triggered = True
        if not self._torque_disable_client.service_is_ready():
            self.get_logger().error(
                "Fail-safe: torque_disable_service not available -- "
                "cannot switch out of torque control automatically!"
            )
            return
        self.get_logger().error(
            "Fail-safe: spring computation failing repeatedly -- "
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
        spring.move_target(np.array([p.x, p.y, p.z]))
        self.get_logger().debug(
            f"Updated target for '{spring.name}': "
            f"[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]"
        )

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

    def _declare_or_ignore(self, name, default):
        try:
            self.declare_parameter(name, default)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

    def _enable_cb(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        for spring in self._springs:
            spring.enabled = request.data
        state = "enabled" if request.data else "disabled"
        response.success = True
        response.message = f"All springs {state}."
        self.get_logger().info(response.message)
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

        shift = np.max(np.abs(q_star - q_current))
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
    # add_spring callback — unchanged except for _check_equilibrium_shift call
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

    def _remove_spring_cb(
        self, request: RemoveSpring.Request, response: RemoveSpring.Response
) -> RemoveSpring.Response:
        name = request.name.strip()

        if not any(s.name == name for s in self._springs):
            response.success = False
            response.message = f"Spring '{name}' not found."
            return response
        
        self._springs.remove(name)
        
        # Remove from spring_names parameter
        current_names = list(
            self.get_parameter("spring_names").get_parameter_value().string_array_value
        )
        self.set_parameters([
            rclpy.parameter.Parameter(
                "spring_names",
                rclpy.parameter.Parameter.Type.STRING_ARRAY,
                [n for n in current_names if n != name]
            )
        ])

        # Undeclare the spring's parameter namespace
        prefix = f"springs.{name}"
        for key in [
                f"{prefix}.link_name",
                f"{prefix}.local_point",
                f"{prefix}.target",
                f"{prefix}.stiffness",
                f"{prefix}.damping",
                f"{prefix}.rest_length",
                f"{prefix}.inner_radius",
                f"{prefix}.outer_radius",
        ]:
            try:
                self.undeclare_parameter(key)
            except rclpy.exceptions.ParameterNotDeclaredException:
                pass

        response.success = True
        response.message = f"Spring '{name}' removed."
        self.get_logger().info(response.message)
        self._publish_springs_updated()
        return response

    def _publish_springs_updated(self) -> None:
        import json
        msg = String()
        msg.data = json.dumps([s.name for s in self._springs])
        self._springs_updated_pub.publish(msg)

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
        self.get_logger().info(f"about to save")
        plot_path = self.get_parameter("plot_output_path").get_parameter_value().string_value
        fig.savefig(plot_path, dpi=150)
        self.get_logger().info(f"Plot saved to {plot_path}")
        plt.show()
        
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
