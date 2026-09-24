# springcontroller_ui

Browser-based control panel for running the Gen3 spring-controller study:
mode-switch/e-stop, gripper control, study-start-pose capture/move, spring
reset, an add/remove-springs panel with an embedded meshcat view, and the
per-participant **workspace calibration** flow that turns two tape-measure
numbers into that participant's condition YAML files.

This package never touches hardware directly (see [How it fits together](#how-it-fits-together)) --
it drives `gen3_torque_control` and `virtual_spring_node`, which do.

## How it fits together

```
Browser (web/index.html)
   |  rosbridge websocket (ROS services/topics over JSON)
   v
study_control_panel_node (orchestration_node.py)
   |                              |
   | move_to_joint_angles,        | add_spring / remove_spring /
   | enable/disable torque,       | get_parameters
   | abort_move                   v
   v                        virtual_spring_node (springcontroller package)
gen3_torque_control                 |
(kinova_torque_control_node)        v
   |                          collision checking, spring torques
   v
Kinova Gen3 hardware
```

Three plain-software pieces are launched together by
`launch/study_control_panel.launch.py`:

- **rosbridge_websocket** -- lets the browser page call ROS services/topics
  directly (`web/index.html` is a single static page with no build step).
- **`study_control_panel_node`** (`springcontroller_ui/orchestration_node.py`)
  -- the only piece with real logic. It exists for operations a browser
  client can't safely do as a 1:1 rosbridge passthrough: sequencing
  (abort-then-disable for e-stop), interlocks (refuse to enable torque on a
  stale/unsafe safety status), forward kinematics (for the spring-editor
  panel and workspace calibration), and local file state (study-start
  preset, per-participant condition YAML, CSV logs).
- **`web/ui_server.py`** -- a bare `http.server` static file server for
  `web/index.html`. No framework, no build step.

`study_control_panel_node` never launches `gen3_torque_control` or
`virtual_spring_node` itself, and its own moves never go through
`ros2_kortex`/`kortex_bringup` -- that driver and `gen3_torque_control` both
open independent Kortex sessions and fight over the arm-global servoing mode
if run against real hardware at the same time. "Position control" in this
package means "torque disabled, arm holding position via
`SINGLE_LEVEL_SERVOING`," driven through `gen3_torque_control`'s own
`move_to_joint_angles` topic (collision-gated against
`virtual_spring_node`'s `~/check_collision_at`), not a separate
`ros2_control` controller.

## Study conditions

Each participant runs three conditions, written as separate YAML files that
`gen3_spring.launch.py` (in the `springcontroller` package) loads:

| Condition | File | What it is |
|---|---|---|
| KT | `KT.yaml` | No torque controller -- just `participant_id`/`condition_name` for rosbag routing. |
| Position | `position.yaml` | One `tip_spring` pulling the arm toward the workspace center, plus a damping-only `joint_7` spring. |
| Pose | `pose.yaml` | One pose spring (`face_participant`) that both orients a link to face the participant's eyes AND passively bounds position (a dead zone around the same center, ramped rather than pulled). |

All three are written together by one **Finalize study conditions** click,
from the same underlying center/measurements -- see
[Finalizing](#finalizing-writing-the-yaml-files) below.

## Workspace calibration flow

This is the "Study workspace calibration" panel. It turns two participant
body measurements into a 3D point (the **workspace center** -- the point
condition 1's spring pulls the hand toward, and condition 2's dead-zone is
centered on) plus a face target (where condition 2's pose spring aims). The
underlying pure-Python math lives in `springcontroller_ui/study_workspace_config.py`
(no ROS imports -- testable directly, see `test/test_study_workspace_config.py`).

The flow is **human-in-the-loop**: the math only ever produces a *candidate*.
Nothing is written to disk, and no spring pulls with real force, until an
operator has visually confirmed (and optionally physically adjusted) the
candidate in meshcat.

### 1. Preview workspace center

Operator enters two tape-measure numbers and clicks **Preview workspace
center**:

- **Eye height (cm)** -- seated eye height above the table.
- **Arm length (cm)** -- elbow to fingertip.

The node computes a candidate center (`compute_candidate_center`) and places
a real (non-zero stiffness/damping) demo spring called `workspace_demo` at
that point, so the operator can see it pull the arm there once torque
control is enabled.

#### The math

The candidate center's **x/y** describe a resting hand position out in
front of the seat; **z** aims for eye level without leaving a fixed 30°
"looking down at your hands" cone.

```
x = seat_x                                  (fixed per-lab-setup constant, see below)
y = seat_y + arm_length_m + 0.10            (arm's reach across the table, +10cm clearance)
z = max( eye_height_m / 2,
         eye_height_m - arm_length_m * tan(30 deg) )
```

- `seat_x`/`seat_y` (params `workspace_seat_x`/`workspace_seat_y`) are the
  participant midline's pre-set world-frame seat position -- a physical
  constant of the lab setup (where the chair is bolted relative to the
  table/robot), not derived from anything else. Update these params if the
  seat mark ever moves.
- `y` reaches across the table's short axis (+y) by the participant's own
  arm length, plus a fixed 10cm margin so the target isn't flush against
  their fingertip's absolute limit.
- `z` is the **higher** (closer to eye level) of two candidates:
  - `eye_height / 2` -- halfway between the tabletop (world `z = 0`) and eye
    height.
  - `eye_height - arm_length * tan(30 deg)` -- the height at which a point
    `arm_length` away is exactly 30° below eye level (the eye-level cone
    clamp, `EYE_ANGLE_DEG` in `study_workspace_config.py`).

  Taking the max of these keeps the candidate from drifting far below eye
  level for participants with long arms/low eye height, while still
  reducing to a sensible "halfway up" default otherwise.

The UI also shows this breakdown as plain text (`describe_candidate_center`)
right in the workspace panel's status message, so the operator doesn't have
to trust the numbers blindly.

### 2. Push and confirm

With torque control enabled, `workspace_demo` pulls the arm toward the
candidate. The operator can **physically push** the arm to a different spot
(e.g. to avoid an obstacle, or match what actually feels reachable for this
participant) and click **Capture current location as workspace target** to
lock `workspace_demo`'s target onto wherever the arm now is -- reading it
back via forward kinematics (`~/get_link_pose`), not by re-running the
candidate-center math. This is the human-in-the-loop override: whatever
point is live on `workspace_demo` when the operator clicks **Finalize** is
what actually gets written, whether or not it matches the original
candidate.

### 3. Preview eye location (independent of the center)

**Preview eye location** places a second, force-free (0 stiffness/damping)
marker spring, `workspace_demo_eye`, at condition 2's pose-spring target --
purely visual, no arm movement. This point is **not** derived from the
reach-center at all:

```
eye_x = seat_x
eye_y = seat_y + EYE_TARGET_Y_OFFSET_CM (20 cm, toward the table)
eye_z = eye_height_m
```

(`compute_eye_location` in `study_workspace_config.py`.) The reach-center is
where the participant's *hand* goes; this is where their *face* is -- two
different points on the participant's body that happen to share the same
seat position. It only depends on eye height and seat position, never on
arm length or wherever the reach-center ended up after step 2.

### 4. Finalizing (writing the YAML files)

**Finalize study conditions** reads `workspace_demo`'s *current live target*
(not a re-computed candidate) and, together with the eye location computed
fresh from the current eye-height field, writes:

- `position.yaml` -- `tip_spring` targeting the confirmed center, plus a
  damping-only `joint_7` spring (arrests wrist-roll drift that `tip_spring`
  has no authority over, since its attachment point sits on `joint_7`'s own
  rotation axis).
- `pose.yaml` -- one pose spring (`face_participant`) whose `target` is the
  eye location and whose `position_center`/`position_radius` reuse the
  *same* confirmed center/radius as `position.yaml`'s `tip_spring` -- so the
  two conditions agree on where the arm belongs by construction, not by an
  operator copying numbers between two fields. `pose.yaml` carries no
  separate position-pulling spring: the dead zone itself stands in for one.
- `KT.yaml` -- no springs at all, just participant/condition metadata for
  rosbag routing.

All three share one fixed dead-zone radius, `SHARED_SPRING_RADIUS_M` (7cm,
`inner_radius=0`, `outer_radius=0.07`) -- not computed per-participant, so
every condition agrees on the same reach tolerance regardless of a given
participant's measurements.

A **warning** (non-blocking -- shown but doesn't stop the write) fires if
the *actual* confirmed center ends up more than 30° from eye level, since an
operator's physical push in step 2 can move it back out of the cone the
candidate-center math originally aimed to stay inside.

Finalize refuses to overwrite an existing participant's files unless the
operator explicitly clicks through **Overwrite participant** or changes the
ID and clicks **Save with new participant ID** -- there's no silent
clobber. A row is also appended to `measurements.csv` in the data directory
(default `~/gen3_study_data`) with the confirmed center, measurements, and
any warnings, for later comparison against rosbag/robot data.

### Condition order

Each participant is also assigned a randomized order to run the three
conditions in (`assign_condition_order`), independent of the workspace math
above -- see the "Study order" control in the sidebar's session panel. Order
assignment is block-randomized (every 6 consecutive participants get each of
the 3! = 6 orderings exactly once) and idempotent (looking up an
already-assigned participant ID returns the same order again). It's purely
advisory to the experimenter, who runs the conditions manually -- it doesn't
drive any robot behavior.

## Other panels

- **Torque control** -- enable/disable with a safety-status interlock
  (refuses to enable if `virtual_spring_node`'s `safety_status` isn't
  `SAFE*`, unless "Override safety interlock" is checked), plus a soft
  e-stop that aborts any in-flight move before disabling torque.
- **Gripper control** -- opens/closes the gripper, automatically toggling
  torque control off/on around the move if it was on.
- **Session timer & event log** -- a countdown timer plus a timestamped CSV
  event log (`event_log.csv` per participant), independent of the workspace
  calibration flow.
- **Collision thresholds** -- live-tunable `danger_threshold`/
  `caution_threshold`/`repulsion_max_force_n` on `virtual_spring_node`, plus
  the repulsion-field on/off toggle.
- **Position mode controls** -- capture/move to a saved "study start"
  joint-angle preset, and a raw move-to-joint-angles box. Both go through
  the same collision-gated `move_to_joint_angles` path described above.
- **Meshcat view** -- an embedded iframe onto `armviz`'s live 3D view
  (`gen3_spring.launch.py armviz:=true`), reloadable in place since the
  iframe otherwise goes stale across `armviz` restarts.
- **Add / remove springs** -- direct 1:1 passthrough to
  `virtual_spring_node`'s `add_spring`/`remove_spring`/`add_pose_spring`
  services, for ad hoc spring setups outside the study flow.

## Launching

```bash
ros2 launch springcontroller_ui study_control_panel.launch.py
```

Starts rosbridge, `study_control_panel_node`, and the static UI server
together (default `http://localhost:8090/`). This does **not** start
`gen3_torque_control` or `gen3_spring.launch.py` (virtual_spring_node) --
those still need to be launched separately, with the joint-states remap the
panel's hint text and the top-level `springcontroller` package README
describe, before torque control or the workspace calibration flow will do
anything.

## Tests

```bash
colcon test --packages-select springcontroller_ui
```

`test/test_study_workspace_config.py` covers the workspace-center/eye-location
math directly (no rclpy needed); the other `test/test_*.py` files cover
`orchestration_node.py`'s interlocks, FK helper, move-to-joint-angles
gating, and tip-spring retargeting.
