# posture_analysis

Post-hoc posture extraction for the `gen3_study` HRI study: pulls video out
of a rosbag2 (mcap) recording and estimates whether the participant is
hunching forward or leaning to a side while working with the robot arm.

This is separate from the live study code (`~/sandbox/src/springcontroller`)
on purpose -- it only ever reads finished bags, never runs during a session.

## Quick Reference

```bash
python3 extract_posture.py <path/to/bag_dir> --out-video <out>.mp4
python3 classify_ergonomics.py <path/to/bag_dir>_posture.csv
python3 plot_posture.py <path/to/bag_dir>_posture.csv \
  --episodes-csv <path/to/bag_dir>_posture_episodes.csv
python3 compare_sessions.py <path/to/directory/of/posture/csvs>
```

## Usage

```bash
source /opt/ros/kilted/setup.bash
source ~/.posture_venv/bin/activate

python3 extract_posture.py <path/to/bag_dir> --out-video <out>.mp4
python3 classify_ergonomics.py <path/to/bag_dir>_posture.csv
python3 plot_posture.py <path/to/bag_dir>_posture.csv \
  --episodes-csv <path/to/bag_dir>_posture_episodes.csv
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

## Classifying hunch/lean exposure

`classify_ergonomics.py` turns a `*_posture.csv` into a heuristic
**threshold + time-in-zone** exposure summary -- not an adapted RULA-style
composite score. These are camera-derived deviation proxies, not
goniometric angles, so a heavier clinical scoring model would claim more
precision than the input actually has. Duration matters as much as peak
magnitude, so the unit of analysis is contiguous "episodes" in a zone, not
just per-frame flags.

```bash
python3 classify_ergonomics.py <bag_dir>_posture.csv \
  [--event-log <bag_dir>/event_log.csv]
```

Per frame, a zone (`hunch` / `lean` / `hunch+lean` / `neutral` / unclassified)
is picked using whichever signal is available:

- **Primary (face)**: `head_pitch_dev_deg > --hunch-threshold-deg` (default
  20deg, an adapted reference point from the commonly-cited >20deg
  neck-flexion risk cutoff in ergonomics literature -- **not** a validated
  cutoff for this rig) and/or `abs(head_roll_dev_deg) > --lean-threshold-deg`
  (default 15deg).
- **Fallback (shoulder)**, only when the face wasn't detected: since
  `shoulder_dist_dev_px`/`shoulder_line_dev_deg` are in pixels, not degrees,
  they can't share the face signal's literature-derived cutoff. Instead
  they're checked against a **session-relative percentile**
  (`--shoulder-percentile`, default 75th) of that same session's own values.
  Every row records which source (`face` vs `shoulder_fallback`) produced
  its label.
- Frames with neither signal are left unclassified and excluded from the
  hunch/lean/neutral percentages (reported separately as a % of wall-clock
  time, so low coverage stays visible instead of silently counting as
  "neutral").

Contiguous same-zone frames are merged into episodes, bridging momentary
(`--gap-tolerance`, default 1.0s) unclassified gaps so one real episode
isn't split by a single missed detection. Episodes at or above
`--sustained-seconds` (default 4.0s) are flagged `sustained`.

Outputs (next to the input CSV, or via `--out-zones`/`--out-episodes`/
`--out-summary`):
- `*_zones.csv` -- per-frame zone + source
- `*_episodes.csv` -- one row per episode (zone, source, start/end/duration,
  sustained); feed this into `plot_posture.py --episodes-csv ...` to shade
  hunch/lean episodes on the time-series plot
- `*_ergo_summary.csv` -- one row of exposure stats for the whole session,
  plus one row per trial if `--event-log` is given (matched by `Trial
  start`/`Trial end` timestamps against the study's `event_log.csv`)

As with everything else in this repo: **always spot-check a flagged
"sustained" episode against `--out-face-video`/`--out-video` before trusting
it.** The default thresholds are starting points, not validated for this
rig -- revisit them once there's a session with a human-rated ground truth
to compare against.

## Comparing exposure across sessions

```bash
python3 compare_sessions.py <path/to/directory/of/posture/csvs> \
  [--pattern '*_position_posture.csv']
```

`compare_sessions.py` batch-runs `classify_ergonomics.py`'s own per-session
classification (same thresholds, same session-relative shoulder-fallback
percentiles -- just computed once per file instead of once) over every
`*_posture.csv` in a directory, then plots a grouped bar chart of
hunch/lean/neutral/unclassified time side by side across sessions, one
group per file. `--pattern` narrows to one condition, e.g.
`*_position_posture.csv` for just the `position` condition in a directory
that also has `*_KT_posture.csv` and `*_pose_posture.csv`.

As with the single-session summary, `hunch`/`lean`/`neutral` are % of that
session's *classified* time and `unclassified` is % of *wall-clock* time --
different denominators, plotted together for comparison but not meant to be
read as parts of one 100% bar. A session with an unusually short
green/red/purple bar and a tall gray one just has poor coverage, not
necessarily good posture -- check `pct_unclassified_of_wallclock` in the
output CSV before drawing conclusions from it.

Writes `exposure_comparison.png` and `exposure_comparison_summary.csv`
(one row per session) next to the input files by default (override with
`--out`/`--out-csv`). It does not write each session's own
`*_zones.csv`/`*_episodes.csv`/`*_ergo_summary.csv` -- run
`classify_ergonomics.py` directly on a session that looks like an outlier
here to get those, and to spot-check it against its overlay video.

## Two cameras for different things

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

## Notes / data hygiene

Extracted CSVs, overlay videos, and any sample frames are participant data
derivatives and are not committed to this repo (see `.gitignore`) -- only
the code and models-download instructions are backed up here. The
downloaded `.task` model files also aren't committed; re-run the `curl`
commands above to regenerate them on a new machine.
