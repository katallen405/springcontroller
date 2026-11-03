"""
test_trajectory_action_node.py

virtual_spring_node's FollowJointTrajectory server, closed loop through ROS:
the test publishes /clock (the node runs on sim time) and /joint_states
from a pinocchio forward-dynamics simulation of the Gen2 arm, integrating
whatever the node publishes on ~/joint_torques (gravity compensation
included). No hardware.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
import rclpy
from builtin_interfaces.msg import Duration, Time
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from springcontroller.urdf_arm_configuration import URDFArmConfiguration
from springcontroller.virtual_spring_node import VirtualSpringNode

PKG = Path(__file__).resolve().parents[1] / "springcontroller"
URDF = str(PKG / "flat_urdf_files" / "gen2_robotiq_flat.urdf")
SRDF = str(PKG / "config" / "gen2_robotiq.srdf")
LOCKED = [
    "gripper_finger1_joint", "gripper_finger2_joint",
    "gripper_finger1_inner_knuckle_joint", "gripper_finger2_inner_knuckle_joint",
    "gripper_finger1_finger_tip_joint", "gripper_finger2_finger_tip_joint",
]
HOME = np.array([4.71, 2.84, 0.0, 0.75, 4.62, 4.48, 4.88])
DT = 0.01
SUBSTEPS = 10


def _stamp(t):
    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def _dur(t):
    return Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


class Harness:
    def __init__(self):
        # use_sim_time is switched on only after construction: startup's
        # fetch_robot_description() times its wait with the node clock, which
        # under sim time wouldn't advance until this harness publishes /clock.
        args = ["--ros-args",
                "-p", f"urdf_path:={URDF}",
                "-p", f"srdf_path:={SRDF}",
                "-p", f"locked_joint_names:=[{','.join(LOCKED)}]",
                "-p", "add_gravity_compensation:=true",
                "-p", "plot_on_shutdown:=false",
                "-p", "joint_limit_repulsion_enabled:=false",
                "-p", "repulsion_enabled:=false"]
        rclpy.init(args=args)
        self.node = VirtualSpringNode()
        self.node.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        self.io = rclpy.create_node("traj_sim")
        self.arm = URDFArmConfiguration.from_urdf(URDF, locked_joint_names=LOCKED)
        self.model = self.arm._model
        self.data = self.model.createData()
        self.names = self.arm.joint_names
        self.q = HOME.copy()
        self.v = np.zeros(7)
        self.t = 1.0
        self.tau = None
        self.state = None
        self.clock_pub = self.io.create_publisher(Clock, "/clock", 10)
        self.js_pub = self.io.create_publisher(JointState, "/joint_states", 10)
        self.io.create_subscription(JointState, "/virtual_spring_node/joint_torques",
                                    self._tau_cb, 10)
        self.io.create_subscription(String, "/virtual_spring_node/trajectory/state",
                                    lambda m: setattr(self, "state", m.data), 10)
        self.client = ActionClient(self.io, FollowJointTrajectory,
                                   "/spring_trajectory_controller/follow_joint_trajectory")
        self.ex = MultiThreadedExecutor()
        self.ex.add_node(self.node)
        self.ex.add_node(self.io)

    def _tau_cb(self, msg):
        self.tau = np.array(msg.effort)
        self._got = True

    def step(self):
        self.t += DT
        self.clock_pub.publish(Clock(clock=_stamp(self.t)))
        self.js_pub.publish(JointState(name=list(self.names), position=list(self.q),
                                       velocity=list(self.v), effort=[0.0] * 7))
        self._got = False
        for _ in range(40):
            self.ex.spin_once(timeout_sec=0.005)
            if self._got:
                break
        tau = self.tau if self.tau is not None else np.zeros(7)
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            a = pin.aba(self.model, self.data, self.arm._angles_to_q(self.q), self.v,
                        tau - 0.5 * self.v)
            self.v = self.v + a * h
            self.q = self.q + self.v * h

    def run(self, seconds, until=None):
        for _ in range(int(round(seconds / DT))):
            self.step()
            if until is not None and until():
                return True
        return False

    def close(self):
        self.node.destroy_node()
        self.io.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.close()


def make_goal(q0, dq, duration=3.0, n=31):
    t = np.linspace(0.0, duration, n)
    s = t / duration
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    dblend = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [f"j2s7s300_joint_{i + 1}" for i in range(7)]
    for ti, b, db in zip(t, blend, dblend):
        goal.trajectory.points.append(JointTrajectoryPoint(
            positions=list(q0 + b * dq), velocities=list(db * dq), time_from_start=_dur(ti)))
    return goal


def send(h, goal):
    assert h.client.wait_for_server(timeout_sec=5.0)
    gf = h.client.send_goal_async(goal)
    h.run(2.0, until=gf.done)
    assert gf.done()
    handle = gf.result()
    assert handle.accepted
    return handle, handle.get_result_async()


def test_spring_trajectory_succeeds_closed_loop(h):
    assert h.run(1.0) is False and h.tau is not None   # node is publishing torques
    q0 = h.q.copy()
    dq = np.array([0.3, -0.15, 0.0, 0.2, 0.0, -0.2, 0.0])
    handle, rf = send(h, make_goal(q0, dq))
    assert h.run(10.0, until=rf.done)
    res = rf.result()
    assert res.status == 4, res.result.error_string                 # SUCCEEDED
    assert res.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
    np.testing.assert_allclose(h.q, q0 + dq, atol=0.06)
    h.run(0.3)                       # ~/trajectory/state publishes at 10 Hz
    assert '"HOLDING"' in h.state
    assert np.max(np.abs(h.v)) < 0.02                                # settled, no wrist limit cycle


def test_spring_trajectory_cancel(h):
    h.run(1.0)
    handle, rf = send(h, make_goal(h.q.copy(), np.array([0.4, 0, 0, 0, 0, 0, 0]), duration=5.0))
    h.run(1.0)
    cf = handle.cancel_goal_async()
    h.run(2.0, until=cf.done)
    assert h.run(3.0, until=rf.done)
    assert rf.result().status == 5                                   # CANCELED
    q_stop = h.q.copy()
    h.run(1.0)
    np.testing.assert_allclose(h.q, q_stop, atol=0.03)               # holds where it stopped


def test_wrong_joints_aborted(h):
    h.run(0.5)
    goal = make_goal(h.q.copy(), np.zeros(7))
    goal.trajectory.joint_names[0] = "bogus"
    handle, rf = send(h, goal)
    assert h.run(2.0, until=rf.done)
    assert rf.result().status == 6
    assert rf.result().result.error_code == FollowJointTrajectory.Result.INVALID_JOINTS


def test_too_fast_rejected(h):
    h.run(0.5)
    handle, rf = send(h, make_goal(h.q.copy(), np.array([1.2, 0, 0, 0, 0, 0, 0]),
                                   duration=0.5))
    assert h.run(2.0, until=rf.done)
    assert rf.result().status == 6 and "end-effector speed" in rf.result().result.error_string
