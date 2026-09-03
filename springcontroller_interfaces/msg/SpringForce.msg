# One spring's current contribution, as reported on ~/spring_forces (see
# SpringForces.msg for the array wrapper actually published on that topic).
# Grouped for readability (name/kind, then "moment", then "position") --
# still a flat message, not nested sub-messages, so it's just field order
# below, not real structure.
#
# kind selects how to interpret force/magnitude:
#   "force"  (VirtualSpring) -- force is a real 3D force in N, magnitude = |force|
#   "moment" (PoseSpring)    -- force is a real 3D moment in N*m, magnitude = |force|
#   "torque" (JointSpring)   -- scalar torque (N*m) about the joint's own axis;
#                                force is always [0, 0, 0] here (no Cartesian vector)

string name
string kind

# "moment" group -- magnitude/force apply to every kind (see kind's own
# comment above for what they mean per kind, not just for a PoseSpring's
# actual moment); angle_offset_rad is PoseSpring-only, the angular error
# (rad) between the link's current face normal and the desired look-at
# direction (PoseSpringState.extension in virtual_spring.py). 0 for every
# other kind.
float64 magnitude
float64[3] force
float64 angle_offset_rad

# "position" group -- PoseSpring-only: the restoring force (N) from
# position_stiffness pulling the attachment point back toward
# position_center once it's outside position_radius
# (PoseSpringState.position_force in virtual_spring.py) -- distinct from
# force/magnitude above, which are the rotational moment, not a Cartesian
# force. [0, 0, 0]/0 for every other kind, and for a PoseSpring with
# position_stiffness=0 or currently inside position_radius (a true dead
# zone -- see PoseSpring's docstring).
float64 position_force_magnitude
float64[3] position_force

# distance is a linear distance in meters: for a VirtualSpring, the
# Cartesian distance from the attachment point to target_world_point; for a
# PoseSpring, the distance from the attachment point to position_center
# (PoseSpringState.position_distance in virtual_spring.py). For a
# JointSpring it's still radians (angular error about the joint axis) --
# there's no linear distance for a single joint. Doesn't fit either group
# above (applies to every kind, not just a PoseSpring) -- kept last.
float64 distance
