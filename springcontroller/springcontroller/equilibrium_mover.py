#!/home/kat/ros_venv/bin/python3
"""
equilibrium_mover.py

Startup node that:
  1. Loads URDF from /robot_description topic (or urdf_path fallback)
  2. Loads spring config from config_path YAML
  3. Waits for first /joint_states message to use as solver seed
  4. Numerically solves for the spring equilibrium joint configuration
     using the exact same code path as virtual_spring_node
  5. Moves to that configuration:
       has_moveit=true  → plans and executes via MoveIt (slow, collision-free)
       has_moveit=false → sends JointTrajectory directly (for fake 2dof arm)
  6. Publishes True on ~/equilibrium_reached and exits 0 on success,
     or exits 1 on any failure

Run before starting virtual_spring_node. study_launch.py wires up the
OnProcessExit handler to switch controllers and start virtual_spring_node
only after this exits 0.

Parameters
----------
config_path             : str   -- path to springs YAML (same as virtual_spring_node)
urdf_path               : str   -- fallback URDF path if /robot_description unavailable
srdf_path               : str   -- path to SRDF for collision pair filtering (optional)
move_group_name         : str   -- MoveIt move group, e.g. "ur_manipulator"
has_moveit              : bool  -- false for 2dof fake arm, true otherwise
velocity_scaling        : float -- MoveIt velocity scaling (0.0-1.0, default 0.1)
accel_scaling           : float -- MoveIt acceleration scaling (0.0-1.0, default 0.1)
solver_tol              : float -- equilibrium residual tolerance (default 1e-6)
add_gravity_compensation: bool  -- for logging/consistency with virtual_spring_node;
                                   gravity is always included in the equilibrium solve
danger_threshold        : float -- collision danger zone in metres (default 0.05)
"""

import sys
import os
import time
import yaml
import numpy as np
from scipy.optimize import fsolve

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from springcontroller.virtual_spring import VirtualSpring, SpringCollection
from springcontroller.urdf_arm_configuration import URDFArmConfiguration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_robot_description(node, timeout_sec: float = 3.0) -> str | None:
    """
    Try to get the URDF from /robot_description (published by
    robot_state_publisher; what RViz uses). Returns the XML string,
    or None if nothing arrives within timeout_sec.

    Uses TRANSIENT_LOCAL durability so we receive the message even if
    we subscribe after it was published.
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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class EquilibriumMoverNode(Node):

    def __init__(self):
        super().__init__("equilibrium_mover")

        # ----------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------
        self.declare_parameter("config_path", "")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("srdf_path", "")
        self.declare_parameter("move_group_name", "ur_manipulator")
        self.declare_parameter("has_moveit", True)
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("accel_scaling", 0.1)
        self.declare_parameter("solver_tol", 1e-6)
        self.declare_parameter("add_gravity_compensation", False)
        self.declare_parameter("danger_threshold", 0.05)

        self._config_path   = os.path.expanduser(
            self.get_parameter("config_path").value
        )
        self._move_group    = self.get_parameter("move_group_name").value
        self._has_moveit    = self.get_parameter("has_moveit").value
        self._vel_scale     = self.get_parameter("velocity_scaling").value
        self._accel_scale   = self.get_parameter("accel_scaling").value
        self._tol           = self.get_parameter("solver_tol").value
        self._add_grav      = self.get_parameter("add_gravity_compensation").value

        if not self._config_path:
            self.get_logger().fatal("config_path parameter must be set.")
            raise RuntimeError("config_path not set")
        if not os.path.isfile(self._config_path):
            self.get_logger().fatal(
                f"Config file not found: {self._config_path}"
            )
            raise RuntimeError(f"Config file not found: {self._config_path}")

        self._done_pub = self.create_publisher(Bool, "~/equilibrium_reached", 1)

    # ------------------------------------------------------------------
    # Step 1: Load URDF
    # ------------------------------------------------------------------

    def _load_arm(self) -> URDFArmConfiguration:
        """
        Load URDF from /robot_description topic if available,
        otherwise fall back to the urdf_path parameter.
        """
        srdf_path       = self.get_parameter("srdf_path").value
        danger_threshold = self.get_parameter("danger_threshold").value

        urdf_xml = fetch_robot_description(self)
        if urdf_xml is not None:
            self.get_logger().info("Loaded URDF from /robot_description topic.")
            return URDFArmConfiguration.from_xml_string(
                urdf_xml,
                srdf_path=srdf_path,
                danger_threshold=danger_threshold,
            )

        urdf_path = self.get_parameter("urdf_path").value
        if not urdf_path:
            raise RuntimeError(
                "No /robot_description topic found and urdf_path not set."
            )
        self.get_logger().warn(
            f"No /robot_description topic; falling back to file: {urdf_path}"
        )
        return URDFArmConfiguration.from_urdf(
            urdf_path,
            srdf_path=srdf_path,
            danger_threshold=danger_threshold,
        )

    # ------------------------------------------------------------------
    # Step 2: Build spring collection from YAML
    # No torque cap — we want the true equilibrium, not the clamped one.
    # ------------------------------------------------------------------

    def _build_springs(self, arm: URDFArmConfiguration) -> SpringCollection:
        with open(self._config_path) as f:
            config = yaml.safe_load(f)

        params = config
        if "/**" in params:
            params = params["/**"]["ros__parameters"]
        elif params and list(params.keys())[0].endswith("ros__parameters"):
            params = list(params.values())[0]

        springs = SpringCollection()

        for name in params.get("spring_names", []):
            if not name:
                continue
            s = params.get("springs", {}).get(name, {})
            spring = VirtualSpring(
                name=name,
                link_name=s["link_name"],
                local_attachment_point=np.array(s.get("local_point", [0, 0, 0])),
                target_world_point=np.array(s.get("target", [0, 0, 0])),
                stiffness=float(s.get("stiffness", 0.0)),
                damping=0.0,          # zero velocity at equilibrium
                rest_length=float(s.get("rest_length", 0.0)),
                inner_radius=float(s.get("inner_radius", 0.0)),
                outer_radius=float(s.get("outer_radius", 0.0)),
            )
            springs.add(spring)
            self.get_logger().info(f"Loaded spring for solver: {spring}")

        return springs

    # ------------------------------------------------------------------
    # Step 3: Get starting joint state from /joint_states
    # ------------------------------------------------------------------

    def _wait_for_joint_state(
        self, arm: URDFArmConfiguration
    ) -> np.ndarray:
        """
        Block until the first /joint_states message arrives.
        Returns joint angles in pinocchio order.
        """
        self.get_logger().info("Waiting for first /joint_states message...")
        q0 = None

        def cb(msg: JointState) -> None:
            nonlocal q0
            if q0 is not None:
                return
            pinocchio_names = arm.joint_names
            ros_names = list(msg.name)
            try:
                order = [ros_names.index(n) for n in pinocchio_names]
                q0 = np.array(msg.position)[order]
            except ValueError as e:
                self.get_logger().error(f"Joint name mismatch: {e}")

        sub = self.create_subscription(JointState, "/joint_states", cb, 10)
        while q0 is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        self.get_logger().info(f"Initial joint state: {np.round(q0, 3)}")
        return q0

    # ------------------------------------------------------------------
    # Step 4: Solve for equilibrium
    # Uses the exact same compute_total_torques / get_gravity_torques
    # code path as the live controller — deadbands, rest lengths, etc.
    # are all accounted for automatically.
    # ------------------------------------------------------------------

    def _solve_equilibrium(
        self,
        arm: URDFArmConfiguration,
        springs: SpringCollection,
        q0: np.ndarray,
    ) -> np.ndarray:
        n_dof = arm.n_dof

        def residual(angles: np.ndarray) -> np.ndarray:
            arm.update_from_angles(angles, np.zeros(n_dof))
            # Always include gravity in the solve regardless of
            # add_gravity_compensation — gravity is always present.
            tau_springs = springs.compute_total_torques(
                arm, add_gravity_compensation=False
            )
            tau_gravity = arm.get_gravity_torques()
            return tau_springs + tau_gravity

        self.get_logger().info(
            f"Solving for equilibrium from q0={np.round(q0, 3)}"
        )
        q_star, info, ier, msg = fsolve(residual, q0, full_output=True)

        residual_norm = np.linalg.norm(info["fvec"])
        self.get_logger().info(
            f"Solver: ier={ier}, residual norm={residual_norm:.2e}"
        )

        if ier != 1 or residual_norm > self._tol:
            raise RuntimeError(
                f"Equilibrium solve did not converge. "
                f"residual={residual_norm:.2e}, tol={self._tol}, msg={msg}"
            )

        self.get_logger().info(
            f"Equilibrium config: {np.round(q_star, 4)}"
        )
        return q_star

    # ------------------------------------------------------------------
    # Step 5a: Move via MoveIt (real arms)
    # ------------------------------------------------------------------

    def _move_to_config_moveit(
        self, q_star: np.ndarray, joint_names: list[str]
    ) -> None:
        from moveit.planning import MoveItPy
        from moveit.core.robot_state import RobotState

        self.get_logger().info("Initialising MoveItPy...")
        moveit = MoveItPy(node_name="equilibrium_mover_moveit")
        move_group = moveit.get_planning_component(self._move_group)

        robot_model = moveit.get_robot_model()
        target_state = RobotState(robot_model)
        for name, angle in zip(joint_names, q_star):
            target_state.set_joint_positions({name: angle})

        move_group.set_goal_state(robot_state=target_state)
        move_group.set_max_velocity_scaling_factor(self._vel_scale)
        move_group.set_max_acceleration_scaling_factor(self._accel_scale)

        self.get_logger().info(
            f"Planning to equilibrium "
            f"(vel={self._vel_scale}, accel={self._accel_scale})..."
        )
        plan_result = move_group.plan()
        if not plan_result:
            raise RuntimeError("MoveIt planning failed.")

        self.get_logger().info("Plan succeeded. Executing...")
        moveit.execute(plan_result.trajectory, controllers=[])
        self.get_logger().info("Execution complete.")

    # ------------------------------------------------------------------
    # Step 5b: Move via direct JointTrajectory (fake 2dof arm)
    # ------------------------------------------------------------------

    def _move_to_config_direct(
        self, q_star: np.ndarray, joint_names: list[str]
    ) -> None:
        pub = self.create_publisher(
            JointTrajectory,
            "/fake_position_controller/joint_trajectory",
            1,
        )

        msg = JointTrajectory()
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = q_star.tolist()
        point.time_from_start = Duration(sec=5)  # slow and safe
        msg.points = [point]

        self.get_logger().info(
            f"Sending joint trajectory to "
            f"/fake_position_controller/joint_trajectory ..."
        )
        # Publish several times briefly to ensure delivery
        for _ in range(10):
            pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

        # Wait for the arm to finish moving (time_from_start + buffer)
        self.get_logger().info("Waiting for motion to complete...")
        time.sleep(6.0)
        self.get_logger().info("Direct joint command complete.")

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            arm     = self._load_arm()
            springs = self._build_springs(arm)
            q0      = self._wait_for_joint_state(arm)
            q_star  = self._solve_equilibrium(arm, springs, q0)

            if self._has_moveit:
                self._move_to_config_moveit(q_star, arm.joint_names)
            else:
                self.get_logger().info(
                    "has_moveit=false — commanding joints directly."
                )
                self._move_to_config_direct(q_star, arm.joint_names)

            msg = Bool()
            msg.data = True
            self._done_pub.publish(msg)
            self.get_logger().info(
                "Equilibrium reached. Spring controller may now start."
            )
            sys.exit(0)

        except Exception as e:
            self.get_logger().fatal(f"EquilibriumMover failed: {e}")
            sys.exit(1)


def main(args=None):
    rclpy.init(args=args)
    node = EquilibriumMoverNode()
    node.run()


if __name__ == "__main__":
    main()
