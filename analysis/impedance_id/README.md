# impedance_id

Measures each robot's own impedance at the tip (step 1 of the CQimpedance
plan) and its stable spring range (step 2), from rosbags. Works for the
Gen2, Gen3 and UR3e.

- **Plan and background:** `~/Dropbox/Research/CQimpedance/PLAN_steps1-3.md`
- **Safety:** only reads bags. Starts no ROS nodes, publishes nothing.

## What it gives you

- **Per robot × pose × axis:**
  - apparent mass **M**
  - damping **b**
  - Coulomb friction **Fc**
  - breakaway force **Fs**
  - leftover bias force **F_bias** (the no-spring drift)
- **UR3e only:** a check of the force-free method against the wrist F/T sensor.
- **Per robot × axis × D:** the highest stable test-spring stiffness, next to the passivity prediction.

## Workflow

Run with the springcontroller venv python, with the workspace sourced.

1. **Record the tests.**
   - Gen2/Gen3: the normal launch already records the topics needed.
   - UR3e: record by hand:
     ```
     ros2 bag record /joint_states /force_torque_sensor_broadcaster/wrench \
         /virtual_spring_node/joint_torques /rosout
     ```
   - **End every test with ~1 s hands-off.** On the UR3e this stretch zeros the wrist sensor.
2. **Overview:** plots the tip motion and drafts a segments CSV, one row per burst of motion.
   ```
   python3 analysis/impedance_id/impedance_id.py overview BAG --robot ur3e --out impedance_results
   ```
3. **Label the CSV.** Fill in `pose`, `test`, `axis`, `force_n` (gauge reading), `k_test`, `d_test`. Fix start/end from the plot, and delete rows that aren't tests.
4. **Fit:** writes `results.csv`, plus one check plot per segment in `plots/`. **Look at the plots:** a bad fit still prints confident numbers.
   ```
   python3 analysis/impedance_id/impedance_id.py fit impedance_results/BAG_segments.csv --out impedance_results
   ```
5. **Summary:** prints the tables and writes `step1_summary.csv` and `step2_summary.csv`.
   ```
   python3 analysis/impedance_id/impedance_id.py summary impedance_results/results.csv
   ```

## The tests (segment `test` column)

| test | what you do | what it gives |
|---|---|---|
| `drift` | springs off, let go, hands off | initial accel a0 → F_bias = M·a0; terminal speed |
| `breakaway` | pull with slowly rising force until it moves | Fs: gauge peak (`force_n`), or the UR3e sensor |
| `pull` | steady pull at one force (repeat at 2-3 forces) | b and Fc from force vs. steady speed |
| `oscillation` | one soft test spring (`k_test`), displace ~3 cm, release | M = k/ωn², b = 2ζ√(kM) − d_test. **No force sensor needed** |
| `stability` | step-2 sweep point (`k_test`, `d_test`): displace ~2 cm, release | settled / chatter / unstable |

## Things to know

- **Axes:** `x`/`y`/`z` are in the robot's base frame.
  - `auto` uses the main direction the tip moved.
  - Segments within 20° of each other are grouped together.
  - UR3e: tip position comes from UR's DH parameters, in UR's `base` frame (ROS `base`, which is `base_link` rotated 180° about z).
- **One axis at a time:** each segment is fit as a 1-DOF system along its axis.
  - `off_axis` > 0.3 gives a warning: the motion wandered, so treat that fit with suspicion.
  - Keep excursions to a few cm.
- **UR3e wrench:** check its frame and sign once.
  - Pull along +x with the gauge and see whether `F_measured` is positive.
  - If not, use `--wrench-sign -1`, and/or `--wrench-frame base`.
- **Fc vs. F_bias (F/T fit):** they can only be told apart when the arm moved both ways in a segment. One-way pulls report a combined `ft_const` instead.
- **Fast transients:** smoothing (`--sg-window-ms`, default 150) smears sharp events.
  - The F/T fit skips one window after a release.
  - Very stiff or very fast oscillations may need a shorter window.
- **Stability sweep:**
  - Joint states arrive at ~100 Hz, so buzzing above 50 Hz shows up at a wrong (aliased) frequency.
  - It's still caught, because the motion doesn't decay.
  - Chatter in the commanded spring torques (`/virtual_spring_node/joint_torques`) is checked too.
- **Passivity prediction:** `K_max = 2(b − |D|)/T` (Colgate & Schenkel) is optimistic: 1-DOF, no delay. The sweep decides.

## Tests

```
python3 -m pytest test/test_impedance_id.py
```

- Every fit is checked against a simulated arm with known M, b, Fc and F_bias.
- The end-to-end test writes a small synthetic UR3e bag (file only, no ROS graph) and runs `overview`, `fit` and `summary` on it.
