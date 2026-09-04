# posture_analysis

Post-hoc posture extraction for the `gen3_study` HRI study: pulls video out
of a rosbag2 (mcap) recording and estimates whether the participant is
hunching forward or leaning to a side while working with the robot arm.

This is separate from the live study code (`~/sandbox/src/springcontroller`)
on purpose -- it only ever reads finished bags, never runs during a session.

## Why this uses both cameras, and for different things

The rig has two cameras at opposite corners of the table, and they turned
out to be good at very different things -- not mainly because of height, but
because of where they're aimed:

- `/camera` (across the table, **face camera**): even though it's mounted
  higher, it sits far enough back that the downward angle to the
  participant's face is shallow. Measured face-detection rate, conditioned
  on the participant actually being present: **62%**. This script uses it
  for `head_pitch_deg`/`head_roll_deg` via MediaPipe's Face Landmarker --
  the **primary** continuous hunch/lean signal (pitch = looking/leaning
  down, roll = head tilted toward a shoulder).
- `/camera2` (overhead task view, **shoulder camera**): mounted close to,
  almost directly above, the workspace -- great for watching hands/task, bad
  for faces (13% face-detection rate under the same test; mostly shows the
  crown of the head). This script only uses its shoulder landmarks, as a
  **secondary cross-check**:
    - `shoulder_line_deg` -- lateral-lean cross-check (tilt of the shoulder line)
    - `shoulder_dist_px` -- forward-hunch cross-check (shoulder pixel spacing;
      increases as the torso comes closer to/lower under this camera)

The two camera streams aren't synchronized, so each shoulder-camera frame is
paired with the nearest-in-time face-camera result (within `--max-pair-gap`,
default 0.2s); see `face_dt_sec` in the output CSV to check pairing quality.

**Upper body only, always.** The desk permanently occludes the hips, legs,
and feet in the shoulder camera's view. MediaPipe's full-body Pose Landmarker
doesn't know that -- it still reports hip/knee/ankle landmarks with high
confidence, hallucinating a plausible lower body onto the desk/robot arm
(see `sample_frame_annotated.jpg`: the "hip" points land on the desk,
nowhere near the person). This tool never uses those landmarks, from either
camera.

All deviations (`*_dev_deg`, `*_dev_px`) are relative to a per-recording,
per-signal baseline (median of the first few seconds after that signal is
first detected), because each camera's own mounting angle/position is fixed
but unknown to the script -- absolute values aren't physically meaningful on
their own here. These are proxies for relative change within one recording,
not calibrated biomechanical angles. Always check the annotated overlay
video before trusting the numbers.

**Camera placement is adjustable in Z only** (the rig can't move cameras in
X/Y). The face camera has since been lowered closer to eye level to push its
detection rate higher; the shoulder camera can't come down further --
lowering it risks the robot arm hitting it.

**Also found (now fixed):** `/camera` had 0 recorded messages in several
early bags -- only camera2 was being captured. That was a loose USB
extension cable on the rig, not a launch-file bug
(`gen3_spring.launch.py` in springcontroller was already wired correctly for
both cameras) -- a fully unseated USB connector doesn't enumerate at all, so
the node never sees a device. springcontroller's test procedure now checks
both cameras' cabling before a recording. If you hit a bag with 0 messages
on either camera topic again, check `ls /dev/v4l/by-id/` and `journalctl -k`
for that camera's serial before suspecting this code or the launch file.

## Setup

```bash
python3 -m venv --system-site-packages ~/.posture_venv
source ~/.posture_venv/bin/activate
pip install --upgrade pip
pip install mediapipe "numpy<2"   # numpy<2 needed for ABI compat with
                                   # cv_bridge/matplotlib from the ROS install

mkdir -p ~/.posture_venv/models
curl -sSL -o ~/.posture_venv/models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
curl -sSL -o ~/.posture_venv/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

(`--system-site-packages` is required so the venv can see `rosbag2_py` and
`cv_bridge` from the ROS install.)

## Usage

```bash
source /opt/ros/kilted/setup.bash
source ~/.posture_venv/bin/activate

python3 extract_posture.py <path/to/bag_dir> --out-video <out>.mp4
python3 plot_posture.py <path/to/bag_dir>_posture.csv
```

`extract_posture.py` writes `<bag_dir>_posture.csv` next to the bag by
default (override with `--out-csv`). Two optional overlay videos, each
showing its own camera's footage annotated with what was detected from it,
so you're checking numbers against the frames that actually produced them:
- `--out-video` -- shoulder points + the live metric readout, over the
  shoulder camera's footage
- `--out-face-video` -- face mesh + pitch/roll/yaw readout, over the face
  camera's own footage (use this one to judge head-pose numbers)

Useful flags:
- `--face-topic` / `--shoulder-topic` -- override which topic plays which
  role (defaults: `/camera` for face, `/camera2` for shoulders)
- `--every N` -- process every Nth frame of each camera (speed up a long recording)
- `--max-pair-gap N` -- max seconds apart when pairing a shoulder frame with a face result (default 0.2s)
- `--baseline-seconds N` -- length of each signal's baseline window (default 3s)

If a bag only has one camera recorded, the other signal's columns just come
back all NaN/not-detected -- the script doesn't require both.

## Notes / data hygiene

Extracted CSVs, overlay videos, and any sample frames are participant data
derivatives and are not committed to this repo (see `.gitignore`) -- only
the code and models-download instructions are backed up here. The
downloaded `.task` model files also aren't committed; re-run the `curl`
commands above to regenerate them on a new machine.
