# ros2_kortex coexistence — open architecture question

Written 2026-08-13. **Not decided, not implemented.** Notes from a design
discussion, kept here so the launch-file research doesn't need to be
re-derived when this gets picked back up.

## The question

`gen3_torque_control`'s `kinova_torque_control_node.py` (sibling package,
`/home/katallen/sandbox/src/gen3_torque_control/gen3_torque_control/`) is a
from-scratch Kortex low-level torque driver — it opens its own direct
TCP (session mgmt, port 10000) + UDP (BaseCyclic realtime, port 10001)
connection to the arm. It does not wrap or call into `ros2_kortex`.

Open question: should this node keep growing into a full replacement for
`ros2_kortex` (e.g. publishing its own `/robot_description`), or should
`ros2_kortex` run alongside it, with responsibilities split between them?
Right now the two would fight over the arm if run naively — see below for
exactly where.

## What ros2_kortex's real launch actually starts

Traced from `ros2_kortex`'s `kortex_bringup/launch/gen3.launch.py` →
`kortex_control.launch.py` (`/home/katallen/sandbox/src/ros2_kortex/`),
as of 2026-08-13:

1. **`ros2_control_node`** (controller_manager) — loads `robot_description`
   + `ros2_controllers.yaml`, and internally loads the
   `kortex_driver/KortexMultiInterfaceHardware` hardware-interface plugin
   (`kortex_description/arms/gen3/7dof/urdf/kortex.ros2_control.xacro`).
   That plugin opens its own TCP (`port`, default 10000) + UDP realtime
   (`port_realtime`, default 10001) Kortex session — the same kind of
   session `kinova_torque_control_node` opens itself.
   **This is the actual collision point if both run at once** — not
   `robot_state_publisher`.
2. **`robot_state_publisher`** — runs `xacro` on `description_file`
   (default `kinova.urdf.xacro`, or `gen3.xacro` via the top-level launch)
   at launch time and publishes `/robot_description` + TF. Architecturally
   independent of the hardware plugin/session — no arm I/O at all.
3. **`joint_state_broadcaster`** spawner (active) — reads state interfaces
   off the hardware plugin, publishes `/joint_states`.
4. **`joint_trajectory_controller`** spawner (active by default via
   `robot_controller` arg) — writes command interfaces.
5. **`twist_controller`** spawner — started `--inactive`.
6. **`robotiq_gripper_controller`** spawner — only if `gripper` arg is set.
7. **`fault_controller`** spawner — only if `use_internal_bus_gripper_comm`
   is true.
8. **`rviz2`** — optional (`launch_rviz`), delayed until after
   `joint_state_broadcaster` finishes spawning.

## Implication

`robot_state_publisher` is decoupled from the Kortex session in
`ros2_kortex`'s own design. Running *just* `robot_state_publisher`
alongside `kinova_torque_control_node` (skip `ros2_control_node` and the
controller spawners entirely) should be safe and would give springcontroller
nodes a real `/robot_description` publisher instead of relying on the
`urdf_path` fallback (see `fetch_robot_description()` in `press_to_pin.py`,
`virtual_spring_node.py`).

The thing that must never run at the same time as
`kinova_torque_control_node` is `ros2_control_node` with the
`KortexMultiInterfaceHardware` plugin active — that's a second process
opening a competing low-level Kortex session, which is a much sharper
conflict than the docstring's existing "leave the driver in
SINGLE_LEVEL_SERVOING" caveat covers (that caveat is about
`SwitchController`-level mode conflicts on a *shared* driver session, not
two independent sessions racing the actuators directly).

## Not yet decided

- Whether to add a `robot_state_publisher` node to `gen3_spring.launch.py`
  (small, standard, decoupled from the hardware question).
- Whether `kinova_torque_control_node` should keep diverging from
  `ros2_kortex` or whether some of `ros2_kortex`'s non-hardware pieces
  (state broadcasting conventions, etc.) are worth adopting/interoperating
  with directly.
