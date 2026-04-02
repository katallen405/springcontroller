#!/home/kat/ros_venv/bin/python3
"""
fk_debug.py

Compare Pinocchio FK against expected UR3e DH parameters at zero config.
Run with: python fk_debug.py

Then compare output against:
  ros2 run tf2_ros tf2_echo ur3e_base_link ur3e_<link> 
with the real robot also at zero/home config.
"""

import pinocchio as pin
import numpy as np

URDF = "/home/kat/workspace/src/ceeorobot_cell/ceeorobot_description/urdf/ceeorobot_flat.urdf"

model = pin.buildModelFromUrdf(URDF)
data = model.createData()

# Zero config
q = pin.neutral(model)
print(f"Testing at q = {q}\n")

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

print(f"{'Frame':<40} {'x':>8} {'y':>8} {'z':>8}")
print("-" * 68)
for frame in model.frames:
    fid = model.getFrameId(frame.name)
    T = data.oMf[fid]
    x, y, z = T.translation

    rpy = pin.rpy.matrixToRpy(T.rotation)
    rpy_deg = np.degrees(rpy)
    print(f"{frame.name:<40} {x:>8.4f} {y:>8.4f} {z:>8.4f}  {rpy_deg}")
