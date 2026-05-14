#!/home/kat/ros_venv/bin/python3
"""
equilibrium_mover.py

Startup node that:
  1. Loads the same URDF + spring config as virtual_spring_node
  2. Numerically solves for the spring equilibrium joint configuration
  3. Plans and executes a slow MoveIt trajectory to that configuration
  4. Publishes success so the launch system can proceed to start the spring controller

Run this before starting virtual_spring_node. In your launch file, use
an event handler to start the spring controller only after this node
exits with code 0.

Parameters
----------
urdf_path        : str   -- path to URDF (same as virtual_spring_node)
config_path      : str   -- path to springs YAML (same as virtual_spring_node)
move_group_name  : str   -- MoveIt move group, e.g. "ur_manipulator"
velocity_scaling : float -- how gently to move (0.0–1.0, default 0.1)
accel_scaling    : float -- acceleration scaling (0.0–1.0, default 0.1)
solver_tol       : float -- residual tolerance for equilibrium solve (default 1e-6)
"""

import sys
import os
import yaml
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import OptimizeResult

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

import pinocchio as pin
from springcontroller.virtual_spring import VirtualSpring, SpringCollection
from springcontroller.urdf_arm_configuration import URDFArmConfiguration

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


class EquilibriumMoverNode(Node):

    def __init__(self):
        super().__init__("equilibrium_mover")

        # ----------------------------------------------------------------
        # Parameters — mirror the ones virtual_spring_node uses so you can
        # pass the same launch arguments to both nodes.
        # ----------------------------------------------------------------
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("config_path", "")
        self.declare_parameter("move_group_name", "ur_manipulator")
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("accel_scaling", 0.1)
        self.declare_parameter("solver_tol", 1e-6)
        self.declare_parameter("add_gravity_compensation", False)

        self._urdf_path = self.get_parameter("urdf_path").value
        self._config_path = os.path.expanduser(self.get_parameter("config_path").value)
        self._move_group_name = self.get_parameter("move_group_name").value
        self._vel_scale = self.get_parameter("velocity_scaling").value
        self._accel_scale = self.get_parameter("accel_scaling").value
        self._tol = self.get_parameter("solver_tol").value
        self._add_grav = self.get_parameter("add_gravity_compensation").value

        if not self._urdf_path:
            self.get_logger().fatal("urdf_path must be set.")
            raise RuntimeError("urdf_path not set")
        if not os.path.isfile(self._config_path):
            self.get_logger().fatal(f"Config file not found: {self._config_path}")
            raise RuntimeError(f"Config file not found: {self._config_path}")

        # Publishes True on success so other nodes/launch actions can listen
        self._done_pub = self.create_publisher(Bool, "~/equilibrium_reached", 1)

    # ------------------------------------------------------------------
    # Step 1: Build the same arm + spring objects as virtual_spring_node
    # ------------------------------------------------------------------

    def _build_arm_and_springs(self) -> tuple[URDFArmConfiguration, SpringCollection]:
        arm = URDFArmConfiguration.from_urdf(self._urdf_path)

        with open(self._config_path) as f:
            config = yaml.safe_load(f)

        # Unwrap standard ROS2 YAML layout if needed
        params = config
        if "/**" in params:
            params = params["/**"]["ros__parameters"]
        elif params and list(params.keys())[0].endswith("ros__parameters"):
            params = list(params.values())[0]

        springs = SpringCollection()  # no torque cap — we want the true equilibrium

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
                damping=0.0,          # zero velocity at equilibrium → damping irrelevant
                rest_length=float(s.get("rest_length", 0.0)),
                inner_radius=float(s.get("inner_radius", 0.0)),
                outer_radius=float(s.get("outer_radius", 0.0)),
            )
            springs.add(spring)
            self.get_logger().info(f"Loaded spring for solver: {spring}")

        return arm, springs

    # ------------------------------------------------------------------
    # Step 2: Solve for equilibrium
    # ------------------------------------------------------------------

    def _solve_equilibrium(
        self,
        arm: URDFArmConfiguration,
        springs: SpringCollection,
        q0: np.ndarray,
    ) -> np.ndarray:
        """
        Find q* such that spring torques + gravity torques = 0.

        Uses the exact same compute_total_torques / get_gravity_torques path
        as the live controller, so deadbands, rest lengths, etc. are all
        accounted for automatically.
        """
        n_dof = arm.n_dof

        def residual(angles: np.ndarray) -> np.ndarray:
            arm.update_from_angles(angles, np.zeros(n_dof))
            # Spring torques only (no gravity comp flag — we add gravity explicitly)
            tau_springs = springs.compute_total_torques(arm, add_gravity_compensation=False)
            tau_gravity = arm.get_gravity_torques()
            return tau_springs + tau_gravity

        self.get_logger().info(f"Solving for equilibrium from q0={np.round(q0, 3)}")
        q_star, info, ier, msg = fsolve(residual, q0, full_output=True)

        residual_norm = np.linalg.norm(info["fvec"])
        self.get_logger().info(
            f"Solver finished. ier={ier}, residual norm={residual_norm:.2e}, msg={msg}"
        )

        if ier != 1 or residual_norm > self._tol:
            raise RuntimeError(
                f"Equilibrium solve did not converge. "
                f"residual={residual_norm:.2e}, tol={self._tol}, msg={msg}"
            )

        self.get_logger().info(f"Equilibrium config: {np.round(q_star, 4)}")
        return q_star

    # ------------------------------------------------------------------
    # Step 3: Move there with MoveIt
    # ------------------------------------------------------------------

    def _move_to_config(self, q_star: np.ndarray, joint_names: list[str]) -> None:
        """Plan and execute a slow trajectory to q_star via MoveIt."""
        self.get_logger().info("Initialising MoveItPy...")
        moveit = MoveItPy(node_name="equilibrium_mover_moveit")
        move_group = moveit.get_planning_component(self._move_group_name)

        # Build target RobotState
        robot_model = moveit.get_robot_model()
        target_state = RobotState(robot_model)
        for name, angle in zip(joint_names, q_star):
            target_state.set_joint_positions({name: angle})

        move_group.set_goal_state(robot_state=target_state)
        move_group.set_max_velocity_scaling_factor(self._vel_scale)
        move_group.set_max_acceleration_scaling_factor(self._accel_scale)

        self.get_logger().info(
            f"Planning to equilibrium config (vel={self._vel_scale}, "
            f"accel={self._accel_scale})..."
        )
        plan_result = move_group.plan()

        if not plan_result:
            raise RuntimeError("MoveIt planning failed — cannot move to equilibrium.")

        self.get_logger().info("Plan succeeded. Executing...")
        moveit.execute(plan_result.trajectory, controllers=[])
        self.get_logger().info("Execution complete.")

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Called once after the node is constructed. Exits the process."""
        try:
            arm, springs = self._build_arm_and_springs()

            # Use the neutral (zero) config as the solver seed.
            # Swap this for the current joint state if you can read it
            # before MoveIt is up — see note below.
            q0 = np.zeros(arm.n_dof)
            q_star = self._solve_equilibrium(arm, springs, q0)

            self._move_to_config(q_star, arm.joint_names)

            msg = Bool()
            msg.data = True
            self._done_pub.publish(msg)
            self.get_logger().info("Equilibrium reached. Spring controller may now start.")
            sys.exit(0)

        except Exception as e:
            self.get_logger().fatal(f"EquilibriumMover failed: {e}")
            sys.exit(1)


def main(args=None):
    rclpy.init(args=args)
    node = EquilibriumMoverNode()
    node.run()   # blocks until done, then exits


if __name__ == "__main__":
    main()
