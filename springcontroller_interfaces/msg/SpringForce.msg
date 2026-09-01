# One spring's current contribution, as reported on ~/spring_forces (see
# SpringForces.msg for the array wrapper actually published on that topic).
#
# kind selects how to interpret force/magnitude:
#   "force"  (VirtualSpring)     -- force is a real 3D force in N, magnitude = |force|
#   "moment" (OrientationSpring) -- force is a real 3D moment in N*m, magnitude = |force|
#   "torque" (JointSpring)       -- scalar torque (N*m) about the joint's own axis;
#                                    force is always [0, 0, 0] here (no Cartesian vector)
#
# distance is this spring's current error from its target: meters for a
# VirtualSpring (Cartesian distance from the attachment point to
# target_world_point), radians for an OrientationSpring/JointSpring (angular
# error) -- matching the existing duck-typed `_last_state.extension`
# convention in virtual_spring.py.

string name
string kind
float64 magnitude
float64[3] force
float64 distance
