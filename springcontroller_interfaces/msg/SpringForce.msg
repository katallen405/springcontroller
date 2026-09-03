# One spring's current contribution, as reported on ~/spring_forces (see
# SpringForces.msg for the array wrapper actually published on that topic).
#
# kind selects how to interpret force/magnitude:
#   "force"  (VirtualSpring) -- force is a real 3D force in N, magnitude = |force|
#   "moment" (PoseSpring)    -- force is a real 3D moment in N*m, magnitude = |force|
#   "torque" (JointSpring)   -- scalar torque (N*m) about the joint's own axis;
#                                force is always [0, 0, 0] here (no Cartesian vector)
#
# distance is a linear distance in meters: for a VirtualSpring, the
# Cartesian distance from the attachment point to target_world_point; for a
# PoseSpring, the distance from the attachment point to position_center
# (PoseSpringState.position_distance in virtual_spring.py). For a
# JointSpring it's still radians (angular error about the joint axis) --
# there's no linear distance for a single joint.
#
# angle_offset_rad is only meaningful for a PoseSpring: the angular error
# (rad) between the link's current face normal and the desired look-at
# direction (PoseSpringState.extension in virtual_spring.py). 0 for every
# other kind.

string name
string kind
float64 magnitude
float64[3] force
float64 distance
float64 angle_offset_rad
