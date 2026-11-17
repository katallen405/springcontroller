# Lab TODO: compliant trajectory following (written 2026-11-17)

Background: on 2026-10-29 we planned three things (plan file
`~/.claude/plans/i-don-t-have-access-streamed-mccarthy.md`):
1. Gen3 effort/torque mode in a `ros2_kortex` fork
2. a position-mode FollowJointTrajectory (FJT) server in `gen2_torque_control`
3. **springcontroller follows a trajectory with a moving end-effector spring**

Plan 3 is the compliant one, and it is **already written**: branch
`trajectory-spring` (and `trajectory-spring-merged`). **Nothing in 2 or 3 has
had a successful hardware run yet.**

## 1. Gen2 position-mode FJT (`gen2_torque_control`, branch `fjt-start-and-speed`)

- [ ] **Decide whether to merge it.** It had one hardware run (2026-11-05). That
      run aborted with every joint already within 0.5°, because the velocity
      reading is noisy. Commit `8b030bd` fixed that (it now judges "settled" from
      positions), adds starting from the current pose, and rejects too-fast
      plans. **That fix has not been run on the arm.**
- [ ] A merge into main is staged but **not committed**: worktree
      `gen2_torque_control/.claude/worktrees/merge-fjt`, branch `merge-fjt`.
      The conflicts (gripper/corrections/diagnostics vs trajectory params in
      `controller.py`, the node, the launch file, README) are resolved by keeping
      both sides. To finish: review `git diff --cached`, then
      `git commit` and `git checkout main && git merge --ff-only merge-fjt`.
- [ ] Tests on the staged merge: 175 passed. Errors in `test_fjt_action.py`
      came from its "other ROS nodes visible" guard, because something else was
      running. Re-run with nothing else up. The `test_flake8` / `test_pep257`
      failures (2630 style findings): check whether main already fails them
      the same way.
- [ ] Hardware check (torque **disabled**, hand on E-STOP):
      `ros2 run gen2_torque_control gen2_send_joint_goal --deg --delta 7=10`.
      Expect SUCCEEDED. Then try a bigger multi-joint move.
- [ ] Then MoveIt end to end through `gen2_moveit_config` (local-only git repo,
      no remote yet; consider pushing it somewhere).

## 2. Compliant following in springcontroller (branch `trajectory-spring`)

How it works: `virtual_spring_node` serves
`/spring_trajectory_controller/follow_joint_trajectory`. A 6-DOF end-effector
spring's anchor is moved along FK(planned q), with the planned velocity as
feed-forward. A weak nullspace pull toward the planned joint angles keeps the
elbow on MoveIt's plan. If the arm is held back > 8 cm, the anchor waits
(pauses) instead of running away. Code: `trajectory_spring.py`,
`trajectory_action.py`. Docs: README section "Follow a planned trajectory".

- [ ] **Jacobian sign fix (`fc084c2`) is not on main.** It flips the sign in
      `get_jacobian`'s point shift (`urdf_arm_configuration.py`), so it changes
      how *every* spring with a `local_point` offset behaves, not just the new
      one. Study data before it used the old sign; tag `pre-jacobian-fix` marks
      that. Decide: merge it on its own first (branch `jacobian-sign-fix`), and
      feel a Gen3/Gen2 spring with `local_point [0,0,0.1]`. It should feel more
      damped, not less.
- [ ] Bring `trajectory-spring` up to date (40 commits behind main): merge
      main into it, run `pytest test/` from the repo root.
- [ ] Gen2 hardware, first run (torque **enabled**, springs ramped, E-STOP):
  - [ ] low gains first, e.g. `trajectory_spring.stiffness` ~100 N/m (default
        300), keep `rot_damping` ≤ 0.2. 1.0 caused a wrist limit cycle in sim.
  - [ ] a small move: does it follow? Then push it mid-move: does it give way,
        pause, and resume? Hold it > 10 s: it should abort and re-seed without
        snapping.
  - [ ] at the end it should HOLD the final pose until
        `ros2 service call /virtual_spring_node/trajectory/release std_srvs/srv/Trigger`.
        Check the tip-hold spring cross-fades back smoothly.
  - [ ] watch `~/trajectory/state` and record a bag.
- [ ] MoveIt: execute with controller `spring_trajectory_controller`
      (`default: false` in `gen2_moveit_config/config/moveit_controllers.yaml`).

## 3. Open design question

- [ ] Keep the EE spring (current design), or switch to / add joint-space
      springs that stream `q_planned(t)` to `~/joint_target/<name>`?
  - EE spring: compliance at the hand is the same in every direction (K in N/m),
    which matches how people push the arm and the existing tip-hold spring. It
    needs the nullspace term and a correct Jacobian.
  - Joint springs: simpler and no Jacobian, but how stiff the hand feels
    depends on the pose and the direction of the push. Also `JointSpring`
    damping is `−b·θ̇`, so it drags against planned motion unless it becomes
    `−b·(θ̇ − θ̇_planned)`.
  - Suggestion: try the EE spring on hardware first; it's already built.

## 4. Gen3 (not mine to test)

- [ ] Colleague: `ros2_kortex` fork `effort-torque-mode`, instructions in
      `kortex_driver/TESTING_EFFORT_MODE.md`. Never run on an arm.
