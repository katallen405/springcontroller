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

### Quick reference

| test | what you do | what it gives |
|---|---|---|
| `drift` | springs off, let go, hands off | bias force F_bias, terminal drift speed |
| `breakaway` | pull with slowly rising force until it moves | breakaway force Fs |
| `pull` | steady pull at one force (repeat at 2-3 forces) | damping b and sliding friction Fc |
| `oscillation` | one soft test spring, displace ~3 cm, release | mass M and damping b. **No force sensor needed** |
| `stability` | step-2 sweep point: displace ~2 cm, release | settled / chatter / unstable |

### The model behind all of them

Along one tip axis, the arm acts like a block on a table:

```
M·a  +  b·v  +  Fc·sign(v)  +  K·x   =   F_hand  +  F_bias
mass    damping  sliding        test       your      leftover force
                 friction       spring     pull      (gravity-comp error)
```

- **M:** how hard it is to start and stop. Includes the arm's own inertia, plus motor and gearbox inertia felt through the joints.
- **b:** resistance that grows with speed (like moving through honey).
- **Fc:** a constant drag whenever it's sliding, the same at any speed.
- **Fs:** the larger force needed to get it unstuck from rest (stiction). Only matters when it's stopped.
- **F_bias:** what makes the arm drift with no springs and no hands. Gravity compensation isn't perfect.
- **K:** a spring the test itself turns on (oscillation and stability only).

Each test is designed so that only a few of these terms matter, letting the script solve for them one at a time.

**All tests:**
- Run with springs off except the one the test asks for. Gravity comp stays in its normal mode.
- Keep motion to a few cm, and roughly along one axis.
- End with ~1 s hands-off.

### `drift`: what the arm does on its own

- **What you do:** hold the arm still at the test pose, let go, keep hands off for a few seconds. Then catch it, before it hits anything.
- **What's happening physically:**
  - With no spring and no hand, only F_bias pushes the arm.
  - Right at release the speed is zero, so damping isn't acting yet. The bias force just fights the arm's mass and sliding friction.
  - The arm speeds up, and damping grows with speed. Eventually damping plus friction balance the bias and the arm coasts at a constant speed.
- **What the script measures:** it fits the textbook "push a mass against damping" curve.
  - Speed rises like `v_terminal · (1 − e^(−t/τ))`, with `τ = M/b`.
  - From that: the initial acceleration `a0` and the final drift speed `v_terminal`.
- **How that becomes F_bias (in `summary`):**
  - `F_bias = M·a0 + Fc`, using M from the oscillation test and Fc from the pull test.
  - Cross-check: `F_bias ≈ b·v_terminal + Fc`.
  - If the arm doesn't move at all, F_bias is smaller than the breakaway force at that pose. That's a useful result on its own.
- **Why it matters:** F_bias is the "the arm wants to wander" feeling. Every spring has to fight it, and a weak spring loses.
- **Check plot:** a curve that bends upward, then straightens into a constant slope. If it keeps curving, the bias changes along the path. Use a shorter segment.

### `breakaway`: how hard to push to get it unstuck

- **What you do:** pull on the tip with the force gauge, increasing the force slowly (over a few seconds) until the arm starts moving. Note the gauge's peak in `force_n`.
- **What's happening physically:**
  - While it's stuck, static friction in the joints and gearboxes matches your pull exactly, so nothing moves.
  - At Fs, friction can't hold any more. The arm slips, and friction drops to the smaller sliding value Fc.
- **What the script measures:**
  - The moment the tip starts moving.
  - Gen2/Gen3: Fs is your gauge reading; the script just confirms it really broke free.
  - UR3e: the wrist sensor's force at that moment.
- **Do both directions:** F_bias helps you in one direction and fights you in the other.
  - `Fs ≈ (F₊ + F₋)/2`
  - `F_bias ≈ (F₋ − F₊)/2`, a free second estimate of the bias.
  - `summary` takes the median of the gauge readings (the average, when there are two).
- **Why it matters:** this is the "sticky" feeling at the start of a fine move, the worst kind of resistance for precision. It also sets a floor: a spring gentler than Fs over your tolerance can't move the arm at all.

### `pull`: damping and sliding friction

- **What you do:** pull the tip with the gauge at a steady force for ~1 s, so it moves at a steady speed. Repeat at 2-3 different forces, e.g. 3, 5 and 8 N. Note each force in `force_n`.
- **What's happening physically:**
  - At steady speed there's no acceleration, so the mass term drops out. Your pull is balanced by damping plus sliding friction: `F = b·v + Fc`.
  - Harder pulls make it go faster, with damping taking up the difference.
- **What the script measures:**
  - The steady-speed stretch of each pull, and the speed there.
  - In `summary`, a straight line through (speed, force) for all the pulls at that pose and axis:
    - slope = **b**
    - intercept = **Fc** (the force needed just to keep it creeping).
- **Do both directions** if you can: F_bias adds to one direction's pulls and subtracts from the other's, so mixing directions averages it out of Fc.
- **Why it matters:** b is the "honey" feel. The stability limit in step 2 also depends on it: the robot's own damping is what lets a stiff spring stay stable.
- **Check plot:** the velocity panel should have a flat plateau. If there isn't one, pull longer or more steadily.

### `oscillation`: mass and damping without a force sensor

- **What you do:**
  - Turn on ONE soft position spring at the tip (e.g. 50-100 N/m) with no damping. Put its stiffness in `k_test` (and its damping in `d_test` if it has any).
  - Push the tip ~3 cm off the spring's target, hold still, let go, hands off.
- **What's happening physically:**
  - The arm plus the spring is a mass on a spring: it bounces back and forth.
  - **How fast it bounces tells you the mass:** a heavier arm swings slower against the same spring, `ωn = √(K/M)`. Since you chose K, the frequency gives M.
  - **How fast the swings die out tells you the damping:** the damping ratio ζ comes from how much each swing shrinks. With M and K known, `b = 2ζ√(K·M)`.
  - The spring is the known force, so no force sensor is needed. That's why this test is the main one on the Gen2 and Gen3.
- **What the script measures:**
  - The release moment.
  - A damped-sine fit to everything after it.
  - If the arm creeps back without overshooting (too much damping or friction), a "two decaying exponentials" fit instead.
  - Then `M = K/ωn²` and `b_robot = 2ζ√(K·M) − d_test`. The commanded damping is subtracted, leaving the robot's own.
- **The `decay` hint:**
  - Damping shrinks each swing by the same **fraction** → `viscous`.
  - Sliding friction shrinks each swing by the same **amount** → `coulomb`.
  - If it says `coulomb`, friction dominates at this pose. Trust b less, and lean on the pull test.
- **Why a soft spring:**
  - It bounces slowly, so the ~10 ms control-loop delay barely matters.
  - A stiff spring makes the delay itself look like extra mass or less damping.
- **Check plot:**
  - The red dashed fit should sit on the blue data, with `r2` above ~0.98.
  - Fewer than ~2 swings → mass is less certain. Try a slightly stiffer spring or a bigger push.

### `stability`: how stiff can the spring go (step 2)

- **What you do:** the same as `oscillation`, but sweeping the spring stiffness up (K ×1.4 each step) at a few damping values. One segment per (K, D) point, each with a ~2 cm push and release. Stop a D row at the first point that buzzes or keeps swinging.
- **What's happening physically:**
  - The virtual spring is computed from joint states ~100 times a second.
  - Between updates the arm keeps moving, but the spring force is a few ms out of date. A stale spring force pushes slightly *in step with* the motion instead of against it, which acts like **negative damping**, growing with K.
  - While the robot's own damping b wins, swings die out. When K gets high enough that the stale-force effect beats b, the swings grow or settle into a buzz.
  - Colgate & Schenkel's passivity bound puts that edge at roughly `K_max ≈ 2(b − D)/T`, with T the update period. Real loops have extra delay, so the true edge is lower.
- **What the script measures:**
  - **decay_ratio:** swing size 2.0-2.5 s after release, divided by swing size in the first 0.5 s. Below 0.2 counts as settled.
  - **hf_fraction:** share of the tip's motion above 10 Hz.
  - **tau_hf_fraction:** the same for the commanded spring torques, if recorded. That's chatter, which often shows in the commands before you see it in motion.
  - **stable:** settled, and neither chatter fraction above 0.3.
  - In `summary`, for each D: the highest stable K, the first unstable K, a working limit at half the highest stable K, and the passivity prediction from step-1's b.
- **Why it matters:** every task's K from step 3 has to sit inside this range on that robot, or it'll buzz.
- **Check plot:** settled points look like `oscillation` plots. Unstable ones keep swinging or show a fuzzy, fast wiggle.

### UR3e bonus: the direct force fit (`ft_*` columns)

- **What's happening:** the UR3e's wrist sensor measures the force at the tip directly. So the whole equation above can be solved at once from every sample, by least squares: M, b, Fc and F_bias together, no assumptions about which terms are small.
- **What it's for:** checking the methods above.
  - `summary` compares the oscillation test's M and b (no sensor) against this fit.
  - Within 20% → the force-free method can be trusted on the Gen2 and Gen3, which have no sensor.
- **Limits:**
  - Fc and F_bias can only be separated when the arm moved both ways in a segment. One-way pulls give one combined `ft_const`.
  - The first smoothing window after a release is skipped: smoothing smears the sudden change there, and that would bias the fit.

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

## Automated tests (pytest)

```
python3 -m pytest test/test_impedance_id.py
```

These check the *script*, not a robot. Nothing joins the ROS graph.

**How they work:**
1. A small simulator (`simulate()` in the test file) plays the role of the arm: the equation above, with M, b, Fc, F_bias and K values the test picks.
   - It steps the physics at 10 kHz, including stick-slip friction.
   - It samples at 100 Hz like the real joint states, and adds 0.1 mm of sensor noise.
2. Each test "runs" one of the robot tests on that simulated arm: hold then release, a steady pull, a slow force ramp, and so on.
3. It checks that the script gets back the values that went in.

**What's covered:**

| Group | Checks that… |
|---|---|
| Oscillation | M within 3% and b within 10% from a bouncing arm; commanded damping is subtracted correctly; heavily damped (no-overshoot) arms are fit too; the viscous/coulomb hint tells the two apart |
| Pull | three pulls give b within 5% and Fc within 0.2 N; one pull alone refuses to fit a line |
| Force fit | with the sensor, M, b, Fc, F_bias all come back when the arm moves both ways; one-way motion merges Fc and F_bias instead of guessing; the test spring is removed correctly |
| Drift | initial acceleration and terminal speed come back; M·a0 + Fc gives F_bias with friction present; a still arm reports "didn't move" |
| Breakaway | the sensor's force at the moment of slip matches Fs |
| Stability | a decaying swing is stable; a growing one isn't; a 30 Hz buzz in the torques is caught |
| Kinematics | UR3e tip position at all-zero joints matches UR's published numbers; Gen2/Gen3 URDFs load, and turning the base joint keeps tip height and radius |
| Summary | grouping, medians, F_bias, K_max per D and the passivity prediction come out right from a hand-made results table; nearby "auto" axes group together |
| End to end | writes a fake UR3e bag (moving joints plus a wrist wrench in the tool frame), then runs `overview`, `fit` and `summary` exactly as you would. Checks the numbers come back and the force-free vs. sensor cross-check says "yes" |

**What they can't tell you:** whether a real arm behaves like the simple model. That's what the check plots, the `r2` values and the UR3e cross-check are for.
