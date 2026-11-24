"""
test_impedance_id.py

Tests for analysis/impedance_id: every fit is checked against a simulated
1-DOF arm with known mass, damping, friction and bias. The end-to-end
test writes a small synthetic UR3e bag to a temp folder (file I/O only:
no ROS nodes, nothing joins the ROS graph) and runs fit + summary on it.
"""

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "impedance_id"))
import idbag  # noqa: E402
import idfit  # noqa: E402
import impedance_id  # noqa: E402

DT = 0.01  # 100 Hz joint states
RNG = np.random.default_rng(0)


def simulate(M, b, K=0.0, Fc=0.0, Fb=0.0, F=lambda t: 0.0, x0=0.0, hold_until=0.0,
             T=6.0, dt_sim=1e-4, noise=1e-4):
    """
    M x'' + b x' + Fc sign(x') + K x = F(t) + Fb, with stick-slip friction
    (static = Coulomb). Before hold_until the arm is held still at x0.
    Returns 100 Hz samples (t, x_noisy, x_true, F_samples).
    """
    n = int(round(T / dt_sim))
    every = int(round(DT / dt_sim))
    x, v = x0, 0.0
    ts, xs, fs = [], [], []
    for i in range(n):
        t = i * dt_sim
        if i % every == 0:
            ts.append(t)
            xs.append(x)
            fs.append(F(t))
        if t < hold_until:
            continue
        Fa = F(t) + Fb - K * x - b * v
        if abs(v) < 1e-6:
            if abs(Fa) <= Fc:
                v = 0.0
                continue
            acc = (Fa - Fc * np.sign(Fa)) / M
        else:
            acc = (Fa - Fc * np.sign(v)) / M
        v_new = v + acc * dt_sim
        if v != 0 and np.sign(v_new) != np.sign(v) and Fc > 0:
            v_new = 0.0       # friction can stop it, not reverse it
        v = v_new
        x += v * dt_sim
    t = np.array(ts)
    x_true = np.array(xs)
    return t, x_true + RNG.normal(0, noise, len(t)), x_true, np.array(fs)


def derivs(x):
    return idfit.sg_derivs(x, DT)


# ── Oscillation (step 1 test 4) ───────────────────────────────────────────────

def test_oscillation_recovers_mass_and_damping():
    M, b, K = 2.0, 4.0, 100.0
    t, x, _, _ = simulate(M, b, K, x0=0.03, hold_until=0.5)
    r = idfit.fit_oscillation(t, x)
    assert r["model"] == "underdamped"
    assert r["r2"] > 0.99
    md = idfit.oscillation_to_mass_damping(r["wn"], r["zeta"], K)
    assert md["M"] == pytest.approx(M, rel=0.03)
    assert md["b_robot"] == pytest.approx(b, rel=0.10)
    assert r["release_s"] == pytest.approx(0.5, abs=0.03)


def test_oscillation_subtracts_commanded_damping():
    M, b, K, D = 2.0, 4.0, 100.0, 3.0
    t, x, _, _ = simulate(M, b + D, K, x0=0.03, hold_until=0.5)
    r = idfit.fit_oscillation(t, x)
    md = idfit.oscillation_to_mass_damping(r["wn"], r["zeta"], K, D)
    assert md["b_total"] == pytest.approx(b + D, rel=0.08)
    assert md["b_robot"] == pytest.approx(b, rel=0.15)


def test_overdamped_oscillation_fit():
    M, b, K = 2.0, 60.0, 100.0
    t, x, _, _ = simulate(M, b, K, x0=0.03, hold_until=0.5, T=5.0)
    r = idfit.fit_oscillation(t, x)
    assert r["model"] == "overdamped"
    assert r["zeta"] > 1
    md = idfit.oscillation_to_mass_damping(r["wn"], r["zeta"], K)
    assert md["M"] == pytest.approx(M, rel=0.15)
    assert md["b_robot"] == pytest.approx(b, rel=0.15)


def test_decay_type_hint():
    viscous = 3.0 * 0.7 ** np.arange(6)
    coulomb = 3.0 - 0.4 * np.arange(6)
    assert idfit.decay_type(viscous) == "viscous"
    assert idfit.decay_type(coulomb) == "coulomb"
    assert idfit.decay_type([1.0, 0.5]) == "unclear"


def test_decay_type_from_simulated_coulomb_friction():
    t, x, _, _ = simulate(2.0, 0.0, 100.0, Fc=0.15, x0=0.03, hold_until=0.5, noise=2e-5)
    r = idfit.fit_oscillation(t, x)
    assert r["decay"] == "coulomb"


# ── Pull (test 3) and the force-speed line ────────────────────────────────────

def test_pulls_give_damping_and_coulomb_friction():
    M, b, Fc = 2.0, 8.0, 1.5
    forces, speeds = [], []
    for Fp in (3.0, 5.0, 8.0):
        t, x, _, _ = simulate(M, b, Fc=Fc, F=lambda tt, Fp=Fp: Fp if 0.5 <= tt < 4.0 else 0.0,
                              T=5.0)
        _, v, _ = derivs(x)
        r = idfit.fit_pull(t, v)
        assert r["v_steady"] == pytest.approx((Fp - Fc) / b, rel=0.03)
        forces.append(Fp)
        speeds.append(r["v_steady"])
    line = idfit.fit_force_speed_line(forces, speeds)
    assert line["b"] == pytest.approx(b, rel=0.05)
    assert line["Fc"] == pytest.approx(Fc, abs=0.2)
    assert line["r2"] > 0.99


def test_force_speed_line_needs_two_speeds():
    assert np.isnan(idfit.fit_force_speed_line([3.0], [0.1])["b"])


# ── Direct regression with a force sensor ─────────────────────────────────────

def test_regress_dynamics_recovers_all_parameters():
    # pushed back and forth, so Coulomb friction and bias are separable
    M, b, Fc, Fb = 2.0, 6.0, 1.0, 0.5
    F = lambda t: 6.0 * np.sin(2 * np.pi * 0.5 * t) if t > 0.3 else 0.0  # noqa: E731
    t, x, _, f = simulate(M, b, Fc=Fc, Fb=Fb, F=F, T=8.0)
    xs, v, a = derivs(x)
    r = idfit.regress_dynamics(f, v, a, x=xs)
    assert r["two_sided"]
    assert r["M"] == pytest.approx(M, rel=0.10)
    assert r["b"] == pytest.approx(b, rel=0.10)
    assert r["Fc"] == pytest.approx(Fc, abs=0.3)
    assert r["F_bias"] == pytest.approx(Fb, abs=0.3)


def test_regress_dynamics_one_way_merges_friction_and_bias():
    M, b, Fc, Fb = 2.0, 6.0, 1.0, 0.5
    F = lambda t: 4.0 + 3.0 * np.sin(2 * np.pi * 0.8 * t) if t > 0.3 else 0.0  # noqa: E731
    t, x, _, f = simulate(M, b, Fc=Fc, Fb=Fb, F=F, T=6.0)
    xs, v, a = derivs(x)
    r = idfit.regress_dynamics(f, v, a, x=xs)
    assert not r["two_sided"]
    assert np.isnan(r["Fc"]) and np.isnan(r["F_bias"])
    assert r["const"] == pytest.approx(Fc - Fb, abs=0.3)
    assert r["M"] == pytest.approx(M, rel=0.10)
    assert r["b"] == pytest.approx(b, rel=0.10)


def test_regress_dynamics_removes_test_spring():
    M, b, K = 2.0, 4.0, 100.0
    t, x, _, f = simulate(M, b, K, x0=0.03, hold_until=0.5)
    xs, v, a = derivs(x)
    # hands-off oscillation: measured external force is zero after release
    r = idfit.regress_dynamics(f, v, a, x=xs, k_test=K, mask=t > 0.6)
    assert r["M"] == pytest.approx(M, rel=0.10)
    assert r["b"] == pytest.approx(b, rel=0.25)


# ── Drift (test 1) and breakaway (test 2) ─────────────────────────────────────

def test_drift_initial_acceleration_and_terminal_speed():
    M, b, Fb = 2.0, 4.0, 1.0
    t, x, _, _ = simulate(M, b, Fb=Fb, hold_until=0.5, T=4.0)
    _, v, _ = derivs(x)
    r = idfit.fit_drift(t, x, v)
    assert r["moved"]
    assert r["a0"] == pytest.approx(Fb / M, rel=0.2)
    assert r["v_terminal"] == pytest.approx(Fb / b, rel=0.05)
    assert r["terminal_reached"]


def test_drift_no_motion():
    t = np.arange(0, 3, DT)
    x = RNG.normal(0, 1e-5, len(t))
    _, v, _ = derivs(x)
    assert not idfit.fit_drift(t, x, v)["moved"]


def test_breakaway_force_from_sensor():
    Fs = 1.5
    t, x, _, f = simulate(2.0, 5.0, Fc=Fs, F=lambda tt: tt, T=3.0)  # 1 N/s ramp
    _, v, _ = derivs(x)
    r = idfit.fit_breakaway(t, v, f)
    assert r["moved"]
    assert r["Fs_measured"] == pytest.approx(Fs, rel=0.15)


# ── Stability (step 2) ────────────────────────────────────────────────────────

def test_stability_settles():
    t, x, _, _ = simulate(2.0, 8.0, 200.0, x0=0.02, hold_until=0.5, T=4.0)
    _, v, _ = derivs(x)
    r = idfit.stability_metrics(t, x, v)
    assert r["stable"]
    assert r["decay_ratio"] < 0.2


def test_stability_growing_oscillation_is_unstable():
    t, x, _, _ = simulate(2.0, -1.0, 200.0, x0=0.02, hold_until=0.5, T=4.0)
    _, v, _ = derivs(x)
    r = idfit.stability_metrics(t, x, v)
    assert not r["stable"]
    assert r["decay_ratio"] > 1


def test_stability_detects_chatter_in_torques():
    t, x, _, _ = simulate(2.0, 8.0, 200.0, x0=0.02, hold_until=0.5, T=4.0)
    _, v, _ = derivs(x)
    tau = np.column_stack([0.05 * np.sin(2 * np.pi * 30 * t), np.zeros_like(t)])
    r = idfit.stability_metrics(t, x, v, tau=tau)
    assert r["tau_hf_fraction"] > 0.9
    assert not r["stable"]


def test_passivity_k_max():
    assert idfit.passivity_k_max(5.0, 0.0, 0.01) == pytest.approx(1000.0)
    assert idfit.passivity_k_max(5.0, 2.0, 0.01) == pytest.approx(600.0)
    assert np.isnan(idfit.passivity_k_max(2.0, 3.0, 0.01))


# ── Axes ──────────────────────────────────────────────────────────────────────

def test_motion_axis_auto_and_labels():
    s = np.linspace(0, 0.05, 50)
    u_true = np.array([0.0, -1.0, 0.0])
    P = np.outer(s, u_true) + RNG.normal(0, 1e-4, (50, 3))
    u = idfit.motion_axis(P, "auto")
    assert abs(u @ u_true) == pytest.approx(1.0, abs=1e-3)
    assert idfit.axis_label(u) == "y"          # signed so largest component is +
    assert idfit.axis_label(-u) == "-y"
    assert idfit.off_axis_fraction(P, u) < 0.05
    np.testing.assert_allclose(idfit.motion_axis(P, "1,1,0"), np.array([1, 1, 0]) / np.sqrt(2))
    with pytest.raises(ValueError):
        idfit.motion_axis(P, "w")


# ── Kinematics ────────────────────────────────────────────────────────────────

def test_ur3e_dh_zero_pose():
    # All joints at 0: arm straight out along -x. Known UR closed form:
    # (a2 + a3, -(d4 + d6), d1 - d5)
    P, Rm = idbag.make_fk("ur3e")(np.zeros((1, 6)))
    np.testing.assert_allclose(P[0], [-0.24355 - 0.2132, -(0.13105 + 0.0921), 0.15185 - 0.08535],
                               atol=1e-9)
    np.testing.assert_allclose(Rm[0] @ Rm[0].T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("robot", ["gen2", "gen3"])
def test_kinova_fk_loads(robot):
    pytest.importorskip("pinocchio")
    fk = idbag.make_fk(robot)
    q = np.zeros((2, 7))
    q[1, 0] = 0.5                                # turn the base joint
    P, Rm = fk(q)
    assert np.all(np.isfinite(P))
    np.testing.assert_allclose(Rm[0] @ Rm[0].T, np.eye(3), atol=1e-9)
    # turning joint 1 (about base z) keeps the tip's height and radius
    # (to 1e-6: the URDF's joint axes are rounded)
    assert P[1, 2] == pytest.approx(P[0, 2], abs=1e-6)
    assert np.hypot(*P[1, :2]) == pytest.approx(np.hypot(*P[0, :2]), abs=1e-6)


def test_wrench_to_base():
    Rz = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=float)
    np.testing.assert_allclose(idbag.wrench_to_base([[1, 0, 0]], Rz, "tool"), [[0, 1, 0]])
    np.testing.assert_allclose(idbag.wrench_to_base([[1, 0, 0]], Rz, "base"), [[1, 0, 0]])


# ── CLI helpers ───────────────────────────────────────────────────────────────

def test_motion_windows_merge_and_drop():
    t = np.arange(0, 10, 0.1)
    s = np.zeros_like(t)
    s[(t > 1) & (t < 2)] = 1        # burst
    s[(t > 2.5) & (t < 3)] = 1      # merges with the first (gap 0.5 s)
    s[(t > 6) & (t < 6.15)] = 1     # blip, dropped
    w = impedance_id.motion_windows(t, s, 0.5)
    assert len(w) == 1
    assert w[0][0] == pytest.approx(1.1) and w[0][1] == pytest.approx(2.9)


def _row(**kw):
    base = {"robot": "gen2", "pose": "P1", "axis_label": "x", "dt": "0.01", "error": ""}
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_summary_tables():
    rows = [
        _row(test="oscillation", M="2.0", b_robot="5.0"),
        _row(test="oscillation", M="2.2", b_robot="5.4"),
        _row(test="drift", a0="0.5", v_terminal="0.1", terminal_reached="True"),
        _row(test="pull", force_n="3", v_steady="0.1875"),
        _row(test="pull", force_n="5", v_steady="0.4375"),
        _row(test="breakaway", force_n="1.6"),
        _row(test="breakaway", force_n="1.4"),
        _row(test="stability", k_test="100", d_test="0", stable="True"),
        _row(test="stability", k_test="200", d_test="0", stable="True"),
        _row(test="stability", k_test="400", d_test="0", stable="False"),
        _row(test="stability", k_test="800", d_test="0", stable="False"),
        _row(test="oscillation", error="boom"),
    ]
    step1, step2 = impedance_id.summarize(rows)
    assert len(step1) == 1
    s = step1[0]
    assert s["M_kg"] == pytest.approx(2.1)
    assert s["b_Ns_per_m"] == pytest.approx(5.2)
    assert s["F_bias_N"] == pytest.approx(2.1 * 0.5)
    assert s["Fs_N"] == pytest.approx(1.5)
    assert s["b_pull"] == pytest.approx(8.0)
    assert s["Fc_pull"] == pytest.approx(1.5)
    assert len(step2) == 1
    k = step2[0]
    assert k["K_stable_max"] == 200 and k["K_first_unstable"] == 400
    assert k["K_working_max"] == 100
    assert k["K_passivity_pred"] == pytest.approx(2 * 5.2 / 0.01)
    assert k["points"] == "100 200 400* 800*"


# ── End to end: synthetic UR3e bag -> fit -> summary ──────────────────────────

def _write_ur3e_bag(path, segments):
    """
    segments: list of (t_offset, t, x_tangent, f_tangent or None). The tip
    moves along the shoulder-pan tangent: joint 1 = x / r, where r is the
    tip's distance from the base z axis; the wrench is that tangential
    force, expressed in the tool frame.
    """
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from geometry_msgs.msg import WrenchStamped
    from sensor_msgs.msg import JointState

    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    for i, (name, typ) in enumerate([("/joint_states", "sensor_msgs/msg/JointState"),
                                     (idbag.ROBOTS["ur3e"]["wrench"],
                                      "geometry_msgs/msg/WrenchStamped")]):
        try:
            meta = rosbag2_py.TopicMetadata(id=i, name=name, type=typ,
                                            serialization_format="cdr")
        except TypeError:
            meta = rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")
        w.create_topic(meta)

    q_home = np.array([0.3, -1.2, 1.4, -1.8, -1.57, 0.2])
    P0, _ = idbag.make_fk("ur3e")(q_home[None])
    r = np.hypot(P0[0, 0], P0[0, 1])
    t_base = 1_700_000_000.0
    for t_off, t, x, f in segments:
        for ti, xi, fi in zip(t, x, f if f is not None else [None] * len(t)):
            q = q_home.copy()
            q[0] += xi / r
            stamp = t_base + t_off + ti
            js = JointState()
            js.header.stamp.sec, js.header.stamp.nanosec = int(stamp), int((stamp % 1) * 1e9)
            js.name = list(idbag.UR3E_JOINTS)
            js.position = q.tolist()
            ns = int(stamp * 1e9)
            w.write("/joint_states", serialize_message(js), ns)
            # wrench: tangential force (direction of increasing joint 1) in tool frame
            P, Rm = idbag.make_fk("ur3e")(q[None])
            tang = np.array([-P[0, 1], P[0, 0], 0.0])
            tang /= np.linalg.norm(tang)
            f_base = (fi or 0.0) * tang
            f_tool = Rm[0].T @ f_base
            ws = WrenchStamped()
            ws.header = js.header
            ws.header.frame_id = "tool0"
            ws.wrench.force.x, ws.wrench.force.y, ws.wrench.force.z = f_tool.tolist()
            w.write(idbag.ROBOTS["ur3e"]["wrench"], serialize_message(ws), ns + 1000)
    del w


def _ramp(t, on, off, rise=0.15):
    """0 -> 1 -> 0 with raised-cosine edges of `rise` seconds."""
    up = np.clip((t - on) / rise, 0, 1)
    down = np.clip((off - t) / rise, 0, 1)
    s = min(up, down)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def test_axis_groups_cluster_nearby_directions():
    rows = [{"robot": "ur3e", "pose": "P1", "axis_label": "-0.62,0.79,0.00",
             "axis_vec": "-0.62,0.79,0.00"},
            {"robot": "ur3e", "pose": "P1", "axis_label": "0.69,-0.72,0.00",
             "axis_vec": "0.69,-0.72,0.00"},          # same line, other sign
            {"robot": "ur3e", "pose": "P1", "axis_label": "0.00,0.60,0.80",
             "axis_vec": "0.00,0.60,0.80"},           # different direction
            {"robot": "ur3e", "pose": "P1", "axis_label": "-x", "axis_vec": "-1,0,0"}]
    impedance_id.assign_axis_groups(rows)
    assert [r["_axis"] for r in rows] == ["dir1", "dir1", "dir2", "x"]


def test_end_to_end_ur3e_bag(tmp_path):
    pytest.importorskip("rosbag2_py")
    # Pulls kept to < 10 cm, like the real tests: the tip moves on an arc
    # around the base here, so long pulls would swing off the fitted axis.
    M, b, K = 1.5, 20.0, 400.0
    forces = (2.0, 3.0, 4.5)
    t1, x1, _, _ = simulate(M, b, K, x0=0.03, hold_until=0.5, T=5.0, noise=5e-5)
    pulls = []
    for Fp in forces:
        # a hand ramps a pull on and off over ~0.15 s, not as a step
        tp, xp, _, fp = simulate(M, b, Fc=1.0, F=lambda tt, Fp=Fp: Fp * _ramp(tt, 0.5, 1.1),
                                 T=4.0, noise=5e-5)
        pulls.append((tp, xp, fp))
    segs = [(0.0, t1, x1, np.zeros_like(t1))]
    off = 6.0
    for tp, xp, fp in pulls:
        # continue from where the arm ended, as a real bag would
        segs.append((off, tp, xp + segs[-1][2][-1], fp))
        off += 5.0
    bag = tmp_path / "ur3e_test_bag"
    _write_ur3e_bag(bag, segs)

    seg_csv = tmp_path / "segments.csv"
    with open(seg_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(impedance_id.SEGMENT_COLUMNS)
        wr.writerow([bag.name, "ur3e", "P1", "oscillation", 0.0, 5.0, "auto", "", K, 0, ""])
        for i in range(3):
            s0 = 6.0 + 5.0 * i
            wr.writerow([bag.name, "ur3e", "P1", "pull", s0, s0 + 3.9, "auto", "", "", "", ""])

    out = tmp_path / "res"
    impedance_id.main(["fit", str(seg_csv), "--out", str(out)])
    with open(out / "results.csv", newline="") as fh:
        res = list(csv.DictReader(fh))
    assert all(not r.get("error") for r in res), [r.get("error") for r in res]
    osc = res[0]
    assert float(osc["M"]) == pytest.approx(M, rel=0.05)
    assert float(osc["b_robot"]) == pytest.approx(b, rel=0.15)
    assert float(osc["off_axis"]) < 0.1
    for r, Fp in zip(res[1:], forces):
        assert r["has_wrench"] == "True"
        assert abs(float(r["F_measured"])) == pytest.approx(Fp, rel=0.05)
    assert len(list((out / "plots").glob("*.png"))) == 4

    impedance_id.main(["summary", str(out / "results.csv")])
    with open(out / "step1_summary.csv", newline="") as fh:
        s1 = list(csv.DictReader(fh))
    assert len(s1) == 1
    s = s1[0]
    assert float(s["b_pull"]) == pytest.approx(b, rel=0.10)
    assert float(s["Fc_pull"]) == pytest.approx(1.0, abs=0.3)
    assert float(s["ft_M"]) == pytest.approx(M, rel=0.15)
    assert float(s["ft_b"]) == pytest.approx(b, rel=0.15)
    assert s["osc_vs_ft_M_ok"] == "yes"


def test_overview_drafts_segments(tmp_path):
    pytest.importorskip("rosbag2_py")
    t, x, _, _ = simulate(1.5, 5.0, 80.0, x0=0.03, hold_until=0.5, T=5.0, noise=5e-5)
    bag = tmp_path / "ur3e_overview"
    _write_ur3e_bag(bag, [(0.0, t, x, None)])
    out = tmp_path / "ov"
    impedance_id.main(["overview", str(bag), "--robot", "ur3e", "--out", str(out)])
    with open(out / "ur3e_overview_segments.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert float(rows[0]["start"]) < 0.6 < float(rows[0]["end"])
    assert (out / "ur3e_overview_overview.png").exists()
