#!/usr/bin/env python3
"""
impedance_id.py -- measure a robot's own impedance at the tip (step 1 of
the CQimpedance plan) and find its stable spring range (step 2), from
rosbags. Gen2, Gen3 and UR3e.

Only reads bags: starts no ROS nodes and publishes nothing.

Three commands, in the order you use them:

  overview BAG --robot R [--out DIR]
      Plots tip motion over time and drafts a segments CSV with one row
      per burst of motion it finds. Fill in test/pose/axis/force/k/d and
      fix the start/end times by reading the plot.

  fit SEGMENTS.csv [--out DIR]
      Fits every labeled segment. Writes DIR/results.csv and one check
      plot per segment in DIR/plots/ (look at these: a bad fit gives
      confident-looking numbers).

  summary DIR/results.csv [--out DIR]
      Step-1 table per robot x pose x axis (M, b, Fc, Fs, F_bias, plus
      the UR3e F/T cross-check) and the step-2 table (K_max per D vs.
      the passivity prediction).

Segments CSV columns (one row per test; extra columns are kept):
  bag      bag folder (absolute, ~, or relative to the CSV's folder)
  robot    gen2 | gen3 | ur3e
  pose     your label for the test pose (e.g. P1)
  test     drift | breakaway | pull | oscillation | stability
  start, end   seconds since the bag's first joint state (overview's time axis)
  axis     x | y | z | auto | ux,uy,uz   (base frame; auto = main direction moved)
  force_n  force-gauge reading (N): breakaway peak, or the steady pull force
  k_test   test spring stiffness (N/m): oscillation and stability tests
  d_test   test spring damping (N*s/m), default 0
  notes    free text

The tests (see PLAN_steps1-3.md):
  drift        springs off, let go, hands off. Gives initial acceleration
               a0 (F_bias = M*a0) and the terminal drift speed.
  breakaway    pull with slowly rising force until it moves. Fs = gauge
               peak (force_n), or the wrist sensor on the UR3e.
  pull         steady pull at one force; several at different forces give
               b (slope) and Fc (intercept) of force vs. speed.
  oscillation  one soft test spring (k_test), displace ~3 cm, release.
               M = k/wn^2, b = 2*zeta*sqrt(k*M) - d_test. No force sensor needed.
  stability    step-2 sweep point (k_test, d_test): displace ~2 cm, release.
               Settled / chatter / unstable.

Every segment should END with ~1 s hands-off: on the UR3e that stretch
is used to zero the wrist sensor (--tare-s).

UR3e wrist sensor: when the wrench topic is in the bag, every segment also
gets a direct fit of M, b, Fc, F_bias from measured force (columns ft_*).
That's the reference the force-free oscillation method is checked
against. The UR3e launch file doesn't record a bag; record at least
  /joint_states  /force_torque_sensor_broadcaster/wrench
  /virtual_spring_node/joint_torques  /rosout
Check the wrench's frame_id once (--wrench-frame tool|base, default tool)
and its sign with a known pull (--wrench-sign).

Usage (springcontroller venv python, workspace sourced):
  python3 analysis/impedance_id/impedance_id.py overview ~/bags/ur3e_p1 --robot ur3e
  python3 analysis/impedance_id/impedance_id.py fit segments.csv --out impedance_results
  python3 analysis/impedance_id/impedance_id.py summary impedance_results/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idbag  # noqa: E402
import idfit  # noqa: E402

TESTS = ("drift", "breakaway", "pull", "oscillation", "stability")
SEGMENT_COLUMNS = ["bag", "robot", "pose", "test", "start", "end", "axis",
                   "force_n", "k_test", "d_test", "notes"]


# ── Small helpers ─────────────────────────────────────────────────────────────

def _float(s, default=np.nan) -> float:
    if s is None:
        return default
    try:
        return float(s) if str(s).strip() != "" else default
    except (TypeError, ValueError):
        return default


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6g}"
    return v


def _write_csv(path: Path, rows: list[dict], first_cols: list[str] | None = None):
    cols = list(first_cols or [])
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def resolve_bag(bag: str, csv_dir: Path) -> str:
    p = Path(os.path.expanduser(bag))
    return str(p if p.is_absolute() else (csv_dir / p).resolve())


# ── overview ──────────────────────────────────────────────────────────────────

def motion_windows(t, speed, threshold, merge_gap_s=1.0, min_len_s=0.2):
    """[(t_start, t_end)] spans where tip speed > threshold, merging gaps
    shorter than merge_gap_s and dropping blips shorter than min_len_s."""
    on = np.asarray(speed) > threshold
    spans, start = [], None
    for i, m in enumerate(on):
        if m and start is None:
            start = t[i]
        elif not m and start is not None:
            spans.append([start, t[i - 1]])
            start = None
    if start is not None:
        spans.append([start, t[-1]])
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < merge_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_len_s]


def cmd_overview(a):
    data = idbag.read_bag(a.bag, a.robot, a.joint_states, a.wrench_topic)
    fk = idbag.make_fk(a.robot, a.urdf)
    tu, qu, dt = idfit.resample_uniform(data["t"], data["q"])
    P, _ = fk(qu)
    P = P - P[0]
    v = np.column_stack([idfit.sg_derivs(P[:, k], dt, a.sg_window_ms / 1000)[1]
                         for k in range(3)])
    speed = np.linalg.norm(v, axis=1)
    wins = motion_windows(tu, speed, a.threshold)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.bag).name
    rows = []
    for w0, w1 in wins:
        rows.append({"bag": str(Path(a.bag).resolve()), "robot": a.robot, "pose": "",
                     "test": "", "start": round(max(0.0, w0 - 1.0), 2),
                     "end": round(min(tu[-1], w1 + 1.5), 2), "axis": "auto",
                     "force_n": "", "k_test": "", "d_test": "", "notes": ""})
    seg_path = out / f"{name}_segments.csv"
    _write_csv(seg_path, rows, SEGMENT_COLUMNS)

    plt = _plt()
    fig, ax = plt.subplots(5 if "wrench_f" in data else 4, 1, sharex=True,
                           figsize=(14, 10), constrained_layout=True)
    for k, lab in enumerate("xyz"):
        ax[k].plot(tu, 100 * P[:, k], lw=0.8)
        ax[k].set_ylabel(f"tip {lab} (cm)")
    ax[3].plot(tu, 100 * speed, lw=0.8, color="k")
    ax[3].axhline(100 * a.threshold, color="gray", ls=":", lw=0.8)
    ax[3].set_ylabel("speed (cm/s)")
    if "wrench_f" in data:
        for k, lab in enumerate("xyz"):
            ax[4].plot(data["wrench_t"], data["wrench_f"][:, k], lw=0.6, label=f"F{lab}")
        ax[4].set_ylabel(f"wrench ({data.get('wrench_frame') or '?'}) N")
        ax[4].legend(loc="upper right", fontsize=7)
    for i, r in enumerate(rows):
        for axx in ax:
            axx.axvspan(r["start"], r["end"], color="C1", alpha=0.12)
        ax[0].text(r["start"], ax[0].get_ylim()[1], f" {i + 1}", va="top", fontsize=8)
    ax[-1].set_xlabel("seconds since first joint state")
    for axx in ax:
        axx.grid(alpha=0.3)
        axx.minorticks_on()
    fig.suptitle(f"{name} ({a.robot}) -- {len(rows)} motion bursts; numbers = rows in "
                 f"{seg_path.name}")
    png = out / f"{name}_overview.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    print(f"joint states: {data['joint_states_topic']}, {len(data['t'])} msgs, "
          f"{1 / dt:.0f} Hz, {tu[-1]:.1f} s")
    print(f"wrench: {'yes, frame ' + repr(data.get('wrench_frame')) if 'wrench_f' in data else 'no'}"
          f";  spring torques: {'yes' if 'tau' in data else 'no'}")
    print(f"{len(rows)} motion bursts -> {seg_path}")
    print(f"plot -> {png}")
    print("Next: fill in pose/test/axis/force_n/k_test, fix start/end from the plot, "
          "delete rows that aren't tests, then run `fit`.")


# ── fit ───────────────────────────────────────────────────────────────────────

class BagCache:
    def __init__(self, a):
        self.a = a
        self.data = {}
        self.fk = {}

    def get(self, bag, robot):
        key = (bag, robot)
        if key not in self.data:
            self.data[key] = idbag.read_bag(bag, robot, self.a.joint_states, self.a.wrench_topic)
        if robot not in self.fk:
            self.fk[robot] = idbag.make_fk(robot, self.a.urdf)
        return self.data[key], self.fk[robot]


def segment_signals(data, fk, t0, t1, axis, a):
    """Uniformly resampled tip signals for one segment, projected on its axis."""
    sel = (data["t"] >= t0) & (data["t"] <= t1)
    if sel.sum() < 10:
        raise ValueError(f"only {int(sel.sum())} joint-state samples in [{t0}, {t1}]")
    tu, qu, dt = idfit.resample_uniform(data["t"][sel], data["q"][sel])
    P, Rm = fk(qu)
    u = idfit.motion_axis(P, axis)
    x_raw = idfit.project(P, u)
    x, v, acc = idfit.sg_derivs(x_raw, dt, a.sg_window_ms / 1000)
    s = {"t": tu, "dt": dt, "P": P, "u": u, "x": x, "x_raw": x_raw, "v": v, "a": acc,
         "f": None, "tau": None}

    if "wrench_f" in data:
        wt, wf = data["wrench_t"], data["wrench_f"]
        wsel = (wt >= t0 - 0.1) & (wt <= t1 + 0.1)
        if wsel.sum() >= 4:
            fu = np.column_stack([np.interp(tu, wt[wsel], wf[wsel, k]) for k in range(3)])
            fb = idbag.wrench_to_base(fu, Rm, a.wrench_frame) * a.wrench_sign
            f_axis = fb @ u
            tare = tu >= tu[-1] - a.tare_s
            f_axis = f_axis - np.median(f_axis[tare])
            # smooth with the same SG window so force and acceleration line up
            s["f"] = idfit.sg_derivs(f_axis, dt, a.sg_window_ms / 1000)[0]
    if "tau" in data:
        tt, tau = data["tau_t"], data["tau"]
        tsel = (tt >= t0 - 0.1) & (tt <= t1 + 0.1)
        if tsel.sum() >= 4:
            s["tau"] = np.column_stack([np.interp(tu, tt[tsel], tau[tsel, k])
                                        for k in range(tau.shape[1])])
    return s


def fit_segment(row, s, a):
    test = row["test"]
    k = _float(row.get("k_test"), 0.0)
    d = _float(row.get("d_test"), 0.0)
    out = {"axis_vec": ",".join(f"{c:.3f}" for c in s["u"]),
           "axis_label": idfit.axis_label(s["u"]),
           "off_axis": idfit.off_axis_fraction(s["P"], s["u"]),
           "dt": s["dt"], "excursion_cm": 100 * float(np.ptp(s["x"])),
           "has_wrench": s["f"] is not None}
    curve = None
    if test == "drift":
        out.update(idfit.fit_drift(s["t"], s["x"], s["v"], a.threshold))
    elif test == "breakaway":
        out.update(idfit.fit_breakaway(s["t"], s["v"], s["f"], a.threshold))
    elif test == "pull":
        out.update(idfit.fit_pull(s["t"], s["v"], s["f"], a.threshold))
    elif test == "oscillation":
        r = idfit.fit_oscillation(s["t"], s["x"])
        curve = (r.pop("t_fit"), r.pop("x_fit"))
        out.update(r)
        if k > 0:
            out.update(idfit.oscillation_to_mass_damping(r["wn"], r["zeta"], k, d))
        else:
            out["warning"] = "oscillation test without k_test: no M or b"
    elif test == "stability":
        r = idfit.stability_metrics(s["t"], s["x"], s["v"], tau=s["tau"],
                                    hf_hz=a.hf_hz)
        out.update(r)
        o = idfit.fit_oscillation(s["t"], s["x"])
        curve = (o["t_fit"], o["x_fit"])
        out.update(osc_wn=o["wn"], osc_zeta=o["zeta"], osc_r2=o["r2"])
    else:
        raise ValueError(f"unknown test {test!r} (one of {', '.join(TESTS)})")

    if s["f"] is not None:
        # Smoothing smears the release (a step in acceleration) over one SG
        # window; leave that stretch out of the force regression.
        mask = None
        rel = _float(out.get("release_s"))
        if np.isfinite(rel):
            mask = s["t"] >= rel + a.sg_window_ms / 1000
        ft = idfit.regress_dynamics(s["f"], s["v"], s["a"], x=s["x"], k_test=k, d_test=d,
                                    mask=mask)
        out.update({f"ft_{kk}": vv for kk, vv in ft.items()})
    if out["off_axis"] > 0.3:
        out["warning"] = (out.get("warning", "") + "; " if out.get("warning") else "") + \
            f"motion {out['off_axis']:.0%} off-axis"
    return out, curve


def plot_segment(path: Path, row, s, res, curve):
    plt = _plt()
    n = 3 if s["f"] is not None else 2
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(9, 2.4 * n), constrained_layout=True)
    ax[0].plot(s["t"], 100 * s["x_raw"], ".", ms=2, color="0.6", label="tip (raw)")
    ax[0].plot(s["t"], 100 * s["x"], lw=1, label="smoothed")
    if curve is not None and np.isfinite(curve[1]).any():
        ax[0].plot(curve[0], 100 * curve[1], "r--", lw=1, label="fit")
    for key in ("onset_s", "release_s"):
        if np.isfinite(_float(res.get(key))):
            for axx in ax:
                axx.axvline(res[key], color="g", ls=":", lw=1)
    ax[0].set_ylabel(f"along {res['axis_label']} (cm)")
    ax[0].legend(fontsize=7, loc="best")
    ax[1].plot(s["t"], 100 * s["v"], lw=1)
    ax[1].set_ylabel("velocity (cm/s)")
    if s["f"] is not None:
        ax[2].plot(s["t"], s["f"], lw=1)
        ax[2].set_ylabel("F_ext along axis (N)")
    ax[-1].set_xlabel("s")
    for axx in ax:
        axx.grid(alpha=0.3)
    keys = {"drift": ["a0", "v_terminal", "terminal_reached"],
            "breakaway": ["onset_s", "Fs_measured"],
            "pull": ["v_steady", "steady_s", "F_measured"],
            "oscillation": ["M", "b_robot", "zeta", "wn", "r2", "model", "decay"],
            "stability": ["stable", "decay_ratio", "f_dominant", "hf_fraction",
                          "tau_hf_fraction"]}[row["test"]]
    if s["f"] is not None:
        keys = keys + ["ft_M", "ft_b", "ft_Fc", "ft_r2"]
    txt = textwrap.fill("  ".join(f"{kk}={_fmt(res.get(kk))}" for kk in keys), 110)
    if res.get("warning"):
        txt += f"\n[!] {res['warning']}"
    fig.suptitle(f"{row.get('_n')}: {row['robot']} {row.get('pose', '')} {row['test']} "
                 f"[{row['start']}-{row['end']}s]\n{txt}", fontsize=8)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def load_segments(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows, start=1):
        r["_n"] = i
        r["test"] = (r.get("test") or "").strip().lower()
        r["robot"] = (r.get("robot") or "").strip().lower()
        r["axis"] = (r.get("axis") or "auto").strip() or "auto"
    return rows


def cmd_fit(a):
    seg_path = Path(a.segments)
    rows = load_segments(seg_path)
    out = Path(a.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    cache = BagCache(a)
    results = []
    for row in rows:
        n = row["_n"]
        if not row["test"]:
            print(f"row {n}: no test, skipped")
            continue
        if row["robot"] not in idbag.ROBOTS:
            print(f"row {n}: robot {row['robot']!r} not one of {list(idbag.ROBOTS)}, skipped")
            continue
        base = {k: v for k, v in row.items() if not k.startswith("_")}
        base["row"] = n
        try:
            bag = resolve_bag(row["bag"], seg_path.parent)
            data, fk = cache.get(bag, row["robot"])
            s = segment_signals(data, fk, _float(row["start"]), _float(row["end"]),
                                row["axis"], a)
            res, curve = fit_segment(row, s, a)
            png = out / "plots" / f"{n:03d}_{row['robot']}_{row.get('pose', '')}_{row['test']}.png"
            plot_segment(png, row, s, res, curve)
            base.update(res)
            base["plot"] = str(png.relative_to(out))
            print(f"row {n}: {row['robot']} {row.get('pose', '')} {row['test']} "
                  f"{res['axis_label']}: " + _one_line(row["test"], res))
        except Exception as exc:  # keep going; one bad row shouldn't lose the rest
            base["error"] = str(exc)
            print(f"row {n}: ERROR {exc}")
        results.append(base)
    res_path = out / "results.csv"
    _write_csv(res_path, results, ["row"] + SEGMENT_COLUMNS)
    print(f"\n{len(results)} rows -> {res_path}\ncheck plots -> {out / 'plots'}")
    print(f"Next: python3 {Path(__file__).name} summary {res_path}")


def _one_line(test, r):
    pick = {"drift": ["a0", "v_terminal"], "breakaway": ["Fs_measured"],
            "pull": ["v_steady", "F_measured"],
            "oscillation": ["M", "b_robot", "zeta", "r2"],
            "stability": ["stable", "decay_ratio", "f_dominant"]}[test]
    if r.get("has_wrench"):
        pick += ["ft_M", "ft_b"]
    s = ", ".join(f"{k}={_fmt(r.get(k))}" for k in pick)
    return s + (f"  [!] {r['warning']}" if r.get("warning") else "")


# ── summary ───────────────────────────────────────────────────────────────────

def _med(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    return float(np.median(v)) if len(v) else np.nan


def _ratio_ok(a_, b_, tol):
    if not (np.isfinite(a_) and np.isfinite(b_) and b_ != 0):
        return ""
    return "yes" if abs(a_ / b_ - 1) <= tol else "NO"


def assign_axis_groups(rows: list[dict], cluster_deg: float = 20.0):
    """
    Name each row's test axis for grouping (sets r["_axis"]):
      * within 10 deg of a base axis -> "x" / "y" / "z" (sign dropped:
        pushing and pulling along x both measure x)
      * otherwise rows of the same robot + pose whose directions are within
        cluster_deg of each other (either sign) share a name: "dir1", ...
        "auto" axes come out a little different every segment, and a tip
        moving on an arc turns its direction as it goes.
    """
    clusters: dict[tuple, list] = {}
    for r in rows:
        lab = (r.get("axis_label") or "").lstrip("-")
        if lab in ("x", "y", "z") or not r.get("axis_vec"):
            r["_axis"] = lab or (r.get("axis") or "").lstrip("-")
            continue
        u = np.array([float(c) for c in r["axis_vec"].split(",")])
        key = (r.get("robot"), r.get("pose", ""))
        for name, ref in clusters.setdefault(key, []):
            if abs(u @ ref) >= np.cos(np.radians(cluster_deg)):
                r["_axis"] = name
                break
        else:
            name = f"dir{len(clusters[key]) + 1}"
            clusters[key].append((name, u))
            r["_axis"] = name


def summarize(rows: list[dict], tol: float = 0.2, T_override: float | None = None):
    assign_axis_groups(rows)
    good = [r for r in rows if not r.get("error")]

    # Step 1: robot x pose x axis
    step1 = []
    keys = sorted({(r["robot"], r.get("pose", ""), r["_axis"]) for r in good
                   if r["test"] != "stability"})
    for robot, pose, ax in keys:
        g = [r for r in good if (r["robot"], r.get("pose", ""), r["_axis"]) == (robot, pose, ax)]
        by = {t: [r for r in g if r["test"] == t] for t in TESTS}
        osc = by["oscillation"]
        M_osc = _med([_float(r.get("M")) for r in osc])
        b_osc = _med([_float(r.get("b_robot")) for r in osc])
        pulls = by["pull"]
        F_pull = [(_float(r.get("force_n")) if np.isfinite(_float(r.get("force_n")))
                   else _float(r.get("F_measured"))) for r in pulls]
        line = idfit.fit_force_speed_line(F_pull, [_float(r.get("v_steady")) for r in pulls])
        Fs = _med([(_float(r.get("force_n")) if np.isfinite(_float(r.get("force_n")))
                    else _float(r.get("Fs_measured"))) for r in by["breakaway"]])
        a0 = _med([_float(r.get("a0")) for r in by["drift"]])
        v_term = _med([_float(r.get("v_terminal")) for r in by["drift"]
                       if str(r.get("terminal_reached")) == "True"])
        # F/T fit: only segments where a person or spring actually drove the motion
        ft_rows = [r for r in g if r["test"] in ("pull", "oscillation")
                   and np.isfinite(_float(r.get("ft_M")))]
        ft_M = _med([_float(r.get("ft_M")) for r in ft_rows])
        ft_b = _med([_float(r.get("ft_b")) for r in ft_rows])
        ft_Fc = _med([_float(r.get("ft_Fc")) for r in ft_rows])
        # F_bias only from tests without a test spring: with one, the
        # spring's anchor position lands in the same constant term.
        ft_Fb = _med([_float(r.get("ft_F_bias")) for r in g
                      if r["test"] in ("drift", "pull", "breakaway")
                      and np.isfinite(_float(r.get("ft_F_bias")))])
        M_best = M_osc if np.isfinite(M_osc) else ft_M
        b_best = b_osc if np.isfinite(b_osc) else (line["b"] if np.isfinite(line["b"]) else ft_b)
        step1.append({
            "robot": robot, "pose": pose, "axis": ax,
            "M_kg": M_best, "b_Ns_per_m": b_best,
            "Fc_N": line["Fc"] if np.isfinite(line["Fc"]) else ft_Fc,
            "Fs_N": Fs,
            "F_bias_N": M_best * a0 if np.isfinite(M_best) and np.isfinite(a0) else ft_Fb,
            "M_osc": M_osc, "b_osc": b_osc, "n_osc": len(osc),
            "b_pull": line["b"], "Fc_pull": line["Fc"], "pull_r2": line["r2"], "n_pull": line["n"],
            "a0_drift": a0, "v_terminal": v_term, "n_drift": len(by["drift"]),
            "F_bias_check_b_v": (b_best * v_term + (line["Fc"] if np.isfinite(line["Fc"]) else 0)
                                 if np.isfinite(b_best) and np.isfinite(v_term) else np.nan),
            "ft_M": ft_M, "ft_b": ft_b, "ft_Fc": ft_Fc, "ft_F_bias": ft_Fb, "n_ft": len(ft_rows),
            "osc_vs_ft_M_ok": _ratio_ok(M_osc, ft_M, tol),
            "osc_vs_ft_b_ok": _ratio_ok(b_osc, ft_b, tol),
        })

    # Step 2: robot x axis x D
    step2 = []
    stab = [r for r in good if r["test"] == "stability"]
    for robot, ax, d in sorted({(r["robot"], r["_axis"], _float(r.get("d_test"), 0.0)) for r in stab}):
        g = sorted([r for r in stab if (r["robot"], r["_axis"], _float(r.get("d_test"), 0.0))
                    == (robot, ax, d)], key=lambda r: _float(r.get("k_test")))
        ks = [_float(r.get("k_test")) for r in g]
        st = [str(r.get("stable")) == "True" for r in g]
        unstable = [k for k, s in zip(ks, st) if not s]
        k_unst = min(unstable) if unstable else np.nan
        stable_below = [k for k, s in zip(ks, st) if s and (not unstable or k < k_unst)]
        k_stab = max(stable_below) if stable_below else np.nan
        T = T_override or _med([_float(r.get("dt")) for r in g])
        b_rob = _med([s1["b_Ns_per_m"] for s1 in step1
                      if s1["robot"] == robot and s1["axis"] == ax])
        step2.append({
            "robot": robot, "axis": ax, "D": d,
            "K_stable_max": k_stab, "K_first_unstable": k_unst,
            "K_working_max": 0.5 * k_stab if np.isfinite(k_stab) else np.nan,
            "K_passivity_pred": idfit.passivity_k_max(b_rob, d, T),
            "b_robot_used": b_rob, "T_s": T,
            "points": " ".join(f"{k:g}{'' if s else '*'}" for k, s in zip(ks, st)),
        })
    return step1, step2


def _print_table(rows, cols, title):
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    w = {c: max(len(c), *(len(str(_fmt(r.get(c, "")))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(str(_fmt(r.get(c, ""))).ljust(w[c]) for c in cols))


def cmd_summary(a):
    with open(a.results, newline="") as fh:
        rows = list(csv.DictReader(fh))
    step1, step2 = summarize(rows, a.tol, a.T)
    out = Path(a.out) if a.out else Path(a.results).parent
    _write_csv(out / "step1_summary.csv", step1)
    _write_csv(out / "step2_summary.csv", step2)
    _print_table(step1, ["robot", "pose", "axis", "M_kg", "b_Ns_per_m", "Fc_N", "Fs_N",
                         "F_bias_N", "n_osc", "n_pull", "n_drift"],
                 "STEP 1 -- the robot's own impedance at the tip")
    ft = [r for r in step1 if r["n_ft"]]
    if ft:
        _print_table(ft, ["robot", "pose", "axis", "M_osc", "ft_M", "osc_vs_ft_M_ok",
                          "b_osc", "ft_b", "osc_vs_ft_b_ok"],
                     f"Cross-check: oscillation (no sensor) vs wrist F/T, within {a.tol:.0%}?")
    _print_table(step2, ["robot", "axis", "D", "K_stable_max", "K_first_unstable",
                         "K_working_max", "K_passivity_pred", "points"],
                 "STEP 2 -- stable K per D  (points: k tested, * = unstable)")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n{len(errs)} rows had errors and were left out: rows "
              + ", ".join(str(r.get("row")) for r in errs))
    print(f"\n-> {out / 'step1_summary.csv'}\n-> {out / 'step2_summary.csv'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the module docstring (top of this file) for details.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--joint-states", help="joint state topic (default: per robot)")
        p.add_argument("--wrench-topic", help="UR3e wrench topic "
                       "(default /force_torque_sensor_broadcaster/wrench)")
        p.add_argument("--urdf", help="override the robot's URDF (UR3e: use it instead of DH)")
        p.add_argument("--sg-window-ms", type=float, default=150.0,
                       help="Savitzky-Golay smoothing window (default 150)")
        p.add_argument("--threshold", type=float, default=idfit.DEFAULT_MOTION_THRESHOLD,
                       help="tip speed counted as moving, m/s (default 0.005)")
        p.add_argument("--out", default="impedance_results", help="output folder")

    p = sub.add_parser("overview", help="plot a bag and draft its segments CSV")
    p.add_argument("bag")
    p.add_argument("--robot", required=True, choices=list(idbag.ROBOTS))
    common(p)
    p.set_defaults(func=cmd_overview)

    p = sub.add_parser("fit", help="fit every labeled segment")
    p.add_argument("segments")
    common(p)
    p.add_argument("--wrench-frame", choices=["tool", "base"], default="tool",
                   help="frame the UR3e wrench is expressed in (default tool)")
    p.add_argument("--wrench-sign", type=float, default=1.0, choices=[1.0, -1.0],
                   help="-1 if a pull along +axis reads negative")
    p.add_argument("--tare-s", type=float, default=0.5,
                   help="hands-off seconds at each segment's end used to zero the wrench")
    p.add_argument("--hf-hz", type=float, default=10.0,
                   help="stability: power above this counts as chatter (default 10)")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("summary", help="step-1 and step-2 tables from results.csv")
    p.add_argument("results")
    p.add_argument("--out", help="output folder (default: next to results.csv)")
    p.add_argument("--tol", type=float, default=0.2,
                   help="oscillation-vs-F/T agreement counted as OK (default 0.2 = 20%%)")
    p.add_argument("--T", type=float, default=None,
                   help="spring update period for the passivity prediction "
                        "(default: median joint-state period)")
    p.set_defaults(func=cmd_summary)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
