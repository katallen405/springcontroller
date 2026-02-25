

### set up static springs
# each spring can only pull/push the robot in one cartesian DoF

# pulling the eef into the workspace linearly (3 springs)

# torsional springs rotating the eef into the allowable orientations (3 springs)

# pushing the robot out of the space where the human is (3 springs)


####

# New springs based on a correction
# level 1: tell the robot we are making a correction
# when the robot is in floating mode, push it away as a correction
# Measure the torques on the joints and ID joint torques
# that exceed a threshold
# Use the position of the joints that exceeded the torque threshold to
# set a new zero point for the position of that **link** with a spring holding it there
# Set the K value of of the spring to (some scalar times) the
# torque that it was pushed away with (arbitrary, proxy for importance?)

# level 2:  automatically detect a correction

##### Torque control based on springs

# LOOP until joint_torque_commands == 0
# Get the current position of each link of the arm (FK)

# Calculate forces from springs attached to links and EEF

# calculate torque on each joint due to forces from springs

# send equal and opposite torques to joints to move the joint torques to 0




