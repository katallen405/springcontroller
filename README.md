# posture_analysis

Post-hoc posture extraction for the `gen3_study` HRI study: pulls the
`/camera2/image_raw_throttled/compressed` video topic out of a rosbag2 (mcap)
recording and estimates whether the participant is hunching forward or
leaning to a side while working with the robot arm.

This is separate from the live study code (`~/sandbox/src/springcontroller`)
on purpose -- it only ever reads finished bags, never runs during a session.

## Why this isn't a standard full-body pose pipeline

The camera is mounted overhead/behind the participant, looking down at the
desk. Two consequences that shaped this tool:

- **Upper body only.** The desk always hides the hips, legs, and feet.
  MediaPipe's full-body Pose Landmarker doesn't know that -- it still reports
  hip/knee/ankle landmarks with high confidence, hallucinating a plausible
  lower body onto the desk/robot arm (see `sample_frame_annotated.jpg`: the
  "hip" points land on the desk, nowhere near the person). This tool never
  uses those landmarks.
- **Face detection is unreliable from this angle.** MediaPipe's Face
  Landmarker (used for head pitch/roll) only fired on ~2% of frames in
  testing, including 0/5 on frames where a human can plainly see the
  participant's face -- the detector is trained on roughly frontal/eye-level
  faces and doesn't recognize one foreshortened this steeply from above.
  Head-pose output is included as a bonus/spot-check signal, not a
  continuous one.

Given that, the primary continuous signal comes from the shoulders (visible
essentially any time the participant is at the desk):

| Column                  | What it means                                                        |
|--------------------------|-----------------------------------------------------------------------|
| `shoulder_line_dev_deg`  | Lateral-lean proxy: tilt of the shoulder line vs. this recording's baseline |
| `shoulder_dist_dev_px`   | Forward-hunch proxy: change in shoulder pixel spacing vs. baseline (under this overhead mount, leaning closer/lower makes the shoulders appear farther apart) |
| `head_pitch_dev_deg`     | Sparse. + = head tipped down (looking/leaning down)                  |
| `head_roll_dev_deg`      | Sparse. Head tilted toward a shoulder                                 |

All deviations are relative to a per-recording baseline (median of the first
few seconds after the participant is detected at the desk), because the
camera's own mounting angle is fixed but unknown to the script -- absolute
angles aren't physically meaningful on their own here. These are proxies for
relative change within one recording, not calibrated biomechanical angles.
Always check the annotated overlay video before trusting the numbers.

**If you want real forward-flexion angles**, reposition the camera to the
side, at roughly seated head/shoulder height, in a 3/4 or profile view
(rather than overhead), and keep it level (not rolled). That should make
face detection reliable enough for continuous head-pose tracking, and turns
forward hunching into a angle change instead of a rough distance proxy.
Worth adding as a second dedicated camera rather than replacing the current
task-view one.

**Also found:** `/camera/image_raw_throttled/compressed` (the other camera)
had 0 recorded messages in every bag checked so far -- only camera2 was
actually being captured. This turned out to be a loose USB extension cable
on that camera, not a launch-file bug (`gen3_spring.launch.py` was already
wired correctly for both cameras) -- a fully unseated USB connector doesn't
enumerate at all, so the node never sees a device. springcontroller's test
procedure now checks both cameras' cabling before a recording, so new bags
should have both feeds; if you hit a bag with 0 messages on either camera
topic again, check `ls /dev/v4l/by-id/` and `journalctl -k` for that
camera's serial before suspecting this code or the launch file.

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
default (override with `--out-csv`). `--out-video` is optional and draws the
shoulder points, face mesh (when detected), and a live metric readout over
the source video -- use it to sanity-check the numbers, not just skim the
CSV.

Useful flags:
- `--every N` -- process every Nth frame (speed up a long recording)
- `--baseline-seconds N` -- length of the baseline window (default 3s)
- `--topic` -- override the video topic (default `/camera2/image_raw_throttled/compressed`)

## Notes / data hygiene

Extracted CSVs, overlay videos, and any sample frames are participant data
derivatives and are not committed to this repo (see `.gitignore`) -- only
the code and models-download instructions are backed up here. The
downloaded `.task` model files also aren't committed; re-run the `curl`
commands above to regenerate them on a new machine.
