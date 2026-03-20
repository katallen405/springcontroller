#!/usr/bin/env python3
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

import os

#from typeguard import value

from geometry_msgs import msg
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool

from springcontroller.virtual_spring import VirtualSpring, SpringCollection
from springcontroller.urdf_arm_configuration import URDFArmConfiguration
import yaml
from springcontroller_interfaces.srv import AddSpring, RemoveSpring
from std_msgs.msg import String

class VirtualSpringNode(Node):

    def __init__(self):
        super().__init__("virtual_spring_node")

        # Parameters
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("config_path", "")
        self.declare_parameter("publish_rate_hz", 100.0)

        urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        if not urdf_path:
            self.get_logger().fatal("Parameter 'urdf_path' must be set.")
            raise RuntimeError("urdf_path not set")

        # Check config file exists before doing anything else
        config_path = os.path.expanduser(
            self.get_parameter("config_path").get_parameter_value().string_value
        )
        if config_path:
            if not os.path.isfile(config_path):
                self.get_logger().fatal(
                    f"Config file not found: '{config_path}'\n"
                    f"Check the path passed to config:= in your launch command."
                )
                raise RuntimeError(f"Config file not found: {config_path}")
            self.get_logger().info(f"Config file found: {config_path}")
        
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

        self._urdf_path = urdf_path

        # Eagerly load URDF so link names can be validated at startup
        self.get_logger().info(f"Loading URDF from: {urdf_path}")
        try:
            self._arm = URDFArmConfiguration.from_urdf(self._urdf_path)
        except Exception as e:
            self.get_logger().fatal(f"Failed to load URDF '{urdf_path}': {e}")
            raise RuntimeError(f"Failed to load URDF: {e}") from e

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
            print(f"{name}: {T[:3, 3]}")
        
        # Spring collection
        self._springs = SpringCollection()
        self._load_springs_from_params()

        # Publishers
        self._torque_pub = self.create_publisher(
            JointState, "~/joint_torques", 10
        )
        from std_msgs.msg import String
        self._springs_updated_pub = self.create_publisher(String,
                                            "~/springs_updated", 10)

        # Subscriptions
        self._js_sub = self.create_subscription(
            JointState, "~/joint_states", self._joint_state_cb, 10
        )

        self._add_spring_srv = self.create_service(
            AddSpring, "~/add_spring", self._add_spring_cb
)

        self._remove_spring_srv = self.create_service(
            RemoveSpring, "~/remove_spring", self._remove_spring_cb
)
        # Per-spring target update subscriptions
        for spring in self._springs:
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
            
        # Services
        self._enable_srv = self.create_service(
            SetBool, "~/enable", self._enable_cb
        )

        self.get_logger().info(
            f"VirtualSpringNode ready. {len(self._springs)} spring(s) loaded."
        )

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
    
    def _joint_state_cb(self, msg: JointState) -> None:
        if len(msg.position) != self._arm.n_dof:
            self.get_logger().warn(
                f"Expected {self._arm.n_dof} joints, got {len(msg.position)}. Skipping."
            )
            return
        q_arm = np.array(msg.position)
        qdot = np.array(msg.velocity) if msg.velocity else np.zeros(self._arm.n_dof)
        try:
            self._arm.update_from_angles(q_arm, qdot)
            self.get_logger().info(f"Springs in collection: {len(self._springs)}, ids: {[s.name for s in self._springs]}")
            torques = self._springs.compute_total_torques(self._arm)
            self.get_logger().info(f"Total torques: {torques}")
        except Exception as e:
            self.get_logger().error(f"Spring computation failed: {e}")
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name   = list(msg.name)
        out.effort = torques.tolist()
        self._torque_pub.publish(out)
    
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
            f"{prefix}.inner_radius": (rclpy.parameter.Parameter.Type.DOUBLE,       request.rest_length),
            f"{prefix}.outer_radius": (rclpy.parameter.Parameter.Type.DOUBLE,       request.rest_length),
            
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

def main(args=None):
    rclpy.init(args=args)
    node = VirtualSpringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
