import os
import random

# These tests create real OrchestrationNodes with rclpy.init(). On the
# default ROS domain they would see -- and could call services on -- a live
# robot stack running on this machine (torque enable/disable, abort_move,
# spring services). On 2026-11-19 the springcontroller node tests did reach
# a live Gen2 stack this way (torque was off, nothing moved). Put every test
# in this directory on a private, localhost-only DDS domain, set before any
# rclpy.init(). Same scheme as ../../test/conftest.py.
os.environ["ROS_DOMAIN_ID"] = str(random.Random(os.getpid()).randint(60, 101))
os.environ.pop("ROS_LOCALHOST_ONLY", None)
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
