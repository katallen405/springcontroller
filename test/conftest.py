import os
import random
import sys

# Test this source tree, not a springcontroller already installed in
# install/ (a sourced workspace puts that on PYTHONPATH ahead of the CWD).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "springcontroller_ui"))
sys.path.insert(0, os.path.join(_ROOT, "springcontroller"))

# Isolate the in-process ROS tests (test_trajectory_action_node.py) on a
# private DDS domain: they publish /joint_states and /clock, which on the
# default domain would feed a real virtual_spring_node / arm running on this
# machine. Must be set before rclpy.init(). Linux-safe domain IDs: 1..101.
os.environ["ROS_DOMAIN_ID"] = str(random.Random(os.getpid()).randint(60, 101))
os.environ.pop("ROS_LOCALHOST_ONLY", None)
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
