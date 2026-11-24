"""
idfit.py -- the fitting half of impedance_id: pure numpy/scipy, no ROS.

Each robot is treated, along one tip axis u at a time, as a 1-DOF system:

    M*a + b*v + Fc*sign(v) + K*x = F_ext + F_bias

  M       apparent mass at the tip along u (kg)
  b       viscous damping (N*s/m)
  Fc      Coulomb (sliding) friction (N)
  K       a spring the test itself commands (0 except in oscillation tests)
  F_ext   force a person applies (force gauge, or the UR3e's wrist sensor)
  F_bias  leftover force with no springs and no hands: gravity-comp error,
          the "no-spring drift"

The functions here take a tip displacement x(t) already projected onto u
and return those parameters. impedance_id.py does the bag reading and
calls these.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

DEFAULT_SG_WINDOW_S = 0.15
DEFAULT_SG_ORDER = 3
DEFAULT_MOTION_THRESHOLD = 0.005   # m/s; tip speed counted as "moving"


# ── Signal basics ─────────────────────────────────────────────────────────────

def resample_uniform(t, x, dt: float | None = None):
    """Sort, drop repeated stamps, interpolate onto a uniform grid at the
    median sample interval (or dt). x may be 1-D or (N, k)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]
    if len(t) < 2:
        raise ValueError("need at least 2 samples")
    if dt is None:
        dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + dt / 2, dt)
    if x.ndim == 1:
        xu = np.interp(tu, t, x)
    else:
        xu = np.column_stack([np.interp(tu, t, x[:, k]) for k in range(x.shape[1])])
    return tu, xu, dt


def sg_window_samples(window_s: float, dt: float, order: int = DEFAULT_SG_ORDER) -> int:
    """Odd window length (samples) covering window_s, at least order+2."""
    n = max(int(round(window_s / dt)), order + 2)
    return n if n % 2 == 1 else n + 1


def sg_derivs(x, dt: float, window_s: float = DEFAULT_SG_WINDOW_S,
              order: int = DEFAULT_SG_ORDER):
    """Smoothed x, velocity and acceleration of a uniformly sampled 1-D
    signal (Savitzky-Golay: finite differences blow up sensor noise)."""
    x = np.asarray(x, dtype=float)
    w = sg_window_samples(window_s, dt, order)
    if w > len(x):
        w = len(x) if len(x) % 2 == 1 else len(x) - 1
    if w <= order:
        raise ValueError(f"segment too short ({len(x)} samples) for SG order {order}")
    xs = savgol_filter(x, w, order, deriv=0)
    v = savgol_filter(x, w, order, deriv=1, delta=dt)
    a = savgol_filter(x, w, order, deriv=2, delta=dt)
    return xs, v, a


# ── Choosing and projecting onto the test axis ───────────────────────────────

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def motion_axis(p, axis: str = "auto"):
    """
    Unit vector of the test axis, in the robot base frame.

    axis: "x" | "y" | "z" | "auto" | "ux,uy,uz" (any direction, normalized).
    "auto" = the principal direction of the tip's motion in the segment
    (largest-variance direction), signed so its largest component is
    positive -- so an "auto" x-ish axis still reads as +x.
    """
    axis = str(axis).strip().lower()
    if axis in AXES:
        return np.array(AXES[axis])
    if axis == "auto":
        p = np.asarray(p, dtype=float)
        d = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        u = vt[0]
        return u if u[np.argmax(np.abs(u))] > 0 else -u
    parts = [float(s) for s in axis.split(",")]
    if len(parts) != 3:
        raise ValueError(f"axis must be x, y, z, auto or 'ux,uy,uz', not {axis!r}")
    u = np.array(parts)
    n = np.linalg.norm(u)
    if n == 0:
        raise ValueError("axis vector is zero")
    return u / n


def axis_label(u) -> str:
    """'x' / '-y' / ... when u is within ~10 deg of a base axis, else the vector."""
    u = np.asarray(u, dtype=float)
    k = int(np.argmax(np.abs(u)))
    if abs(u[k]) > np.cos(np.radians(10)):
        return ("-" if u[k] < 0 else "") + "xyz"[k]
    return ",".join(f"{c:.2f}" for c in u)


def project(p, u):
    """Displacement along u relative to the first sample (m)."""
    p = np.asarray(p, dtype=float)
    return (p - p[0]) @ np.asarray(u, dtype=float)


def off_axis_fraction(p, u) -> float:
    """RMS motion perpendicular to u over RMS motion along it. Large values
    (> ~0.3) mean the test didn't stay on one axis and the 1-DOF fit is
    suspect."""
    p = np.asarray(p, dtype=float)
    d = p - p.mean(axis=0)
    along = d @ u
    perp = d - np.outer(along, u)
    a = np.sqrt(np.mean(along ** 2))
    return float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))) / a) if a > 0 else float("nan")


# ── Onset / steady-state helpers ──────────────────────────────────────────────

def detect_onset(t, v, threshold: float = DEFAULT_MOTION_THRESHOLD,
                 sustain_s: float = 0.05) -> int | None:
    """Index of the first sample where |v| > threshold and stays above it
    for sustain_s. None if the arm never moves."""
    t = np.asarray(t)
    moving = np.abs(np.asarray(v)) > threshold
    if not moving.any():
        return None
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    n = max(1, int(round(sustain_s / dt)))
    for i in np.flatnonzero(moving):
        if i + n <= len(moving) and moving[i:i + n].all():
            return int(i)
    return None


def steady_mask(v, threshold: float = DEFAULT_MOTION_THRESHOLD, tol: float = 0.2):
    """Samples of a steady pull: moving in the dominant direction, with
    speed within tol (fraction) of the median moving speed."""
    v = np.asarray(v, dtype=float)
    moving = np.abs(v) > threshold
    if not moving.any():
        return np.zeros_like(v, dtype=bool)
    sgn = np.sign(np.median(v[moving]))
    fwd = moving & (np.sign(v) == sgn)
    if not fwd.any():
        return fwd
    med = np.median(v[fwd])
    return fwd & (np.abs(v - med) <= tol * abs(med))


# ── Test 1: drift (let go, no springs) ────────────────────────────────────────

def _drift_model(t, c, A, tau, tr):
    """Step response from rest of M*a + b*v = const, released at tr:
    v = A*(1 - e^(-(t-tr)/tau)), so x = c + A*((t-tr) - tau*(1 - e^(-(t-tr)/tau)))."""
    s = np.clip(t - tr, 0.0, None)
    return c + A * (s - tau * (1 - np.exp(-s / tau)))


def fit_drift(t, x, v, threshold: float = DEFAULT_MOTION_THRESHOLD,
              window_s: float = 0.3, tail_frac: float = 0.3):
    """
    What the arm does when let go with no springs and no hands, as a
    first-order step response (constant bias force against damping):
        v(t) = v_terminal * (1 - e^(-t/tau)),  tau = M/b
      a0          initial acceleration = v_terminal/tau; F_bias = M*a0
                  (minus any static friction it had to overcome)
      v_terminal  final drift speed; F_bias = b*v_terminal + Fc (cross-check)
      tau         M/b
    Falls back to a quadratic over the first window_s when the fit fails.
    Only meaningful over a short drift: gravity-comp error changes with
    pose, so the force isn't constant over a long one.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    out = {"onset_s": np.nan, "a0": np.nan, "v_terminal": np.nan, "tau": np.nan,
           "terminal_reached": False, "moved": False, "r2": np.nan}
    i0 = detect_onset(t, v, threshold)
    if i0 is None:
        return out
    out["moved"] = True
    # back up to where the speed was last near zero (the release moment)
    j = i0
    while j > 0 and abs(v[j - 1]) > 0.2 * threshold:
        j -= 1
    out["onset_s"] = float(t[j])
    sel = t >= t[j] - 0.5
    tt, xx = t[sel], x[sel]
    v_end = float(np.median(v[-max(3, len(v) // 10):]))
    try:
        p, _ = curve_fit(_drift_model, tt, xx,
                         p0=[xx[0], v_end if v_end else v[i0], 0.3, t[j]],
                         bounds=([-np.inf, -np.inf, 1e-3, tt[0]],
                                 [np.inf, np.inf, 100.0, t[i0] + 0.2]),
                         maxfev=20000)
        c, A, tau, tr = p
        out.update(v_terminal=float(A), tau=float(tau), a0=float(A / tau),
                   onset_s=float(tr), r2=_r2(xx, _drift_model(tt, *p)),
                   terminal_reached=bool(t[-1] - tr > 3 * tau and abs(A) > threshold))
    except (RuntimeError, ValueError):
        w = (t >= t[j]) & (t <= t[j] + window_s)
        if w.sum() >= 4:
            out["a0"] = float(2 * np.polyfit(t[w] - t[j], x[w], 2)[0])
        tail = t >= t[-1] - tail_frac * (t[-1] - t[j])
        if tail.sum() >= 4:
            out["v_terminal"] = float(np.median(v[tail]))
    return out


# ── Test 2: breakaway ─────────────────────────────────────────────────────────

def fit_breakaway(t, v, f=None, threshold: float = DEFAULT_MOTION_THRESHOLD,
                  pre_s: float = 0.1):
    """
    Moment the arm starts moving under a slowly increasing pull. With a
    measured force f(t) (UR3e), Fs = the largest |f| in the pre_s before
    that moment. Without one, Fs comes from the gauge reading in the
    segments CSV, and this only confirms the arm did break away.
    """
    t = np.asarray(t, dtype=float)
    i0 = detect_onset(t, v, threshold)
    out = {"onset_s": np.nan, "Fs_measured": np.nan, "moved": i0 is not None}
    if i0 is None:
        return out
    out["onset_s"] = float(t[i0])
    if f is not None:
        f = np.asarray(f, dtype=float)
        sel = (t >= t[i0] - pre_s) & (t <= t[i0])
        if sel.any():
            out["Fs_measured"] = float(np.max(np.abs(f[sel])))
    return out


# ── Test 3: constant pull ─────────────────────────────────────────────────────

def fit_pull(t, v, f=None, threshold: float = DEFAULT_MOTION_THRESHOLD, tol: float = 0.2):
    """Steady speed of one constant pull (and the measured force during it,
    when there's a sensor). b and Fc come from several pulls at different
    forces: see fit_force_speed_line()."""
    v = np.asarray(v, dtype=float)
    m = steady_mask(v, threshold, tol)
    out = {"v_steady": np.nan, "steady_s": 0.0, "F_measured": np.nan}
    if m.sum() < 3:
        return out
    dt = float(np.median(np.diff(np.asarray(t, dtype=float))))
    out["v_steady"] = float(np.median(v[m]))
    out["steady_s"] = float(m.sum() * dt)
    if f is not None:
        out["F_measured"] = float(np.median(np.asarray(f, dtype=float)[m]))
    return out


def fit_force_speed_line(F, v):
    """
    |F| = Fc + b*|v| through several steady pulls (least squares).
    Returns b (slope), Fc (intercept), r2, n. Needs >= 2 pulls at
    different speeds; 3+ recommended.
    """
    F = np.abs(np.asarray(F, dtype=float))
    v = np.abs(np.asarray(v, dtype=float))
    ok = np.isfinite(F) & np.isfinite(v)
    F, v = F[ok], v[ok]
    out = {"b": np.nan, "Fc": np.nan, "r2": np.nan, "n": int(len(F))}
    if len(F) < 2 or np.ptp(v) == 0:
        return out
    A = np.column_stack([v, np.ones_like(v)])
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    out["b"], out["Fc"] = float(coef[0]), float(coef[1])
    out["r2"] = _r2(F, A @ coef)
    return out


# ── Any test with a force sensor: direct dynamic regression ───────────────────

def regress_dynamics(f, v, a, x=None, k_test: float = 0.0, d_test: float = 0.0,
                     v_eps: float = 0.01, min_speed: float = DEFAULT_MOTION_THRESHOLD,
                     two_sided_frac: float = 0.1, mask=None):
    """
    Least-squares fit of  M*a + b*v + Fc*tanh(v/v_eps) - F_bias
                          = f - k_test*x - d_test*v
    f is the measured external force along the axis (UR3e wrist sensor).
    k_test / d_test remove a spring the test itself commanded. tanh is a
    smooth sign(v).

    Samples slower than min_speed are left out: while the arm is stuck,
    friction is static (anything up to Fs), not the Coulomb term.

    Fc and F_bias can only be told apart when the arm moved BOTH ways
    (each direction >= two_sided_frac of the moving samples): one-way,
    Fc*sign(v) is just another constant. Then they're fitted as one
    constant, reported as `const` (= Fc*sign(v) - F_bias), with Fc and
    F_bias NaN.

    Returns M, b, Fc, F_bias, const, two_sided, r2, n. This is the
    "F/T-based fit" the force-free methods are checked against.
    """
    f = np.asarray(f, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    y = f.copy()
    if x is not None and k_test:
        y = y - k_test * np.asarray(x, dtype=float)
    if d_test:
        y = y - d_test * v
    m = np.isfinite(y) & np.isfinite(v) & np.isfinite(a) & (np.abs(v) >= min_speed)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    out = {"M": np.nan, "b": np.nan, "Fc": np.nan, "F_bias": np.nan, "const": np.nan,
           "two_sided": False, "r2": np.nan, "n": int(m.sum())}
    if m.sum() < 8:
        return out
    pos = np.mean(v[m] > 0)
    two = bool(min(pos, 1 - pos) >= two_sided_frac)
    out["two_sided"] = two
    if two:
        A = np.column_stack([a, v, np.tanh(v / v_eps), -np.ones_like(v)])
    else:
        A = np.column_stack([a, v, np.ones_like(v)])
    coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
    out.update(M=float(coef[0]), b=float(coef[1]), r2=_r2(y[m], A[m] @ coef))
    if two:
        out.update(Fc=float(coef[2]), F_bias=float(coef[3]))
    else:
        out["const"] = float(coef[2])
    return out


# ── Test 4 (and step 2): free oscillation against a known test spring ─────────

def _light_smooth(x):
    """Short SG smoothing (7 samples) so sample noise doesn't look like
    peaks or motion. Works on raw or already-smoothed signals."""
    x = np.asarray(x, dtype=float)
    return savgol_filter(x, 7, 2) if len(x) >= 7 else x


def release_index(x, search_frac: float = 0.5) -> int:
    """
    The release moment: where the arm starts moving away from where it
    was held. Found from the velocity (first sample above 10% of peak
    speed, then back to where speed was last under 2% of peak), searched
    in the first search_frac of the segment. Falls back to the largest
    displacement from the final rest position.
    """
    x = np.asarray(x, dtype=float)
    n = max(2, int(len(x) * search_frac))
    rest = np.median(x[-max(3, len(x) // 10):])
    fallback = int(np.argmax(np.abs(x[:n] - rest)))
    if len(x) < 7:
        return fallback
    v = np.abs(np.gradient(_light_smooth(x)))
    vmax = v.max()
    if vmax <= 0:
        return fallback
    # noise floor from the quietest stretch (rest at the end, or the hold)
    noise = min(np.median(v[: max(3, n // 4)]), np.median(v[-max(3, len(v) // 10):]))
    thr = max(0.1 * vmax, 6 * noise)
    above = v[:n] > thr
    sustain = 5
    j = None
    for i in np.flatnonzero(above):
        if above[i:i + sustain].all():
            j = int(i)
            break
    if j is None:
        return fallback
    floor = max(0.02 * vmax, 2 * noise)
    while j > 0 and v[j - 1] > floor:
        j -= 1
    return j


def _underdamped(t, A, sigma, wd, phi, c):
    return A * np.exp(-sigma * t) * np.cos(wd * t + phi) + c


def _overdamped(t, A, s1, ds, c, tr):
    """Released from rest (zero speed) at tr, poles s1 and s2 = s1 + ds:
    x = c + A*(s2 e^(-s1 tau) - s1 e^(-s2 tau))/(s2 - s1), tau = t - tr >= 0."""
    s2 = s1 + ds
    tau = np.clip(t - tr, 0.0, None)
    return c + A * (s2 * np.exp(-s1 * tau) - s1 * np.exp(-s2 * tau)) / ds


def _r2(y, yhat) -> float:
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else float("nan")


def peak_amplitudes(x, c: float, min_frac: float = 0.05):
    """
    One extremum per swing: the largest |x - c| inside each lobe between
    zero crossings of (lightly smoothed) x - c. Lobes smaller than
    min_frac of the first swing are noise around rest and end the count.
    """
    d = _light_smooth(x) - c
    sgn = np.sign(d)
    sgn[sgn == 0] = 1
    cross = np.flatnonzero(sgn[1:] != sgn[:-1]) + 1
    bounds = np.concatenate([[0], cross, [len(d)]])
    idx, amps = [], []
    ref = abs(d[0]) if len(d) else 0.0
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        if s1 <= s0:
            continue
        k = s0 + int(np.argmax(np.abs(d[s0:s1])))
        amp = abs(d[k])
        ref = max(ref, amp) if not amps else ref
        if amp < min_frac * ref:
            break
        idx.append(k)
        amps.append(amp)
    return np.array(idx, dtype=int), np.array(amps)


def decay_type(amps) -> str:
    """
    Viscous damping shrinks each swing by a constant RATIO; Coulomb
    friction by a constant DIFFERENCE. Whichever is more constant (lower
    coefficient of variation) is the hint. Needs 4+ swings.
    """
    amps = np.asarray(amps, dtype=float)
    if len(amps) < 4:
        return "unclear"
    ratios = amps[1:] / amps[:-1]
    diffs = amps[:-1] - amps[1:]
    cv_r = np.std(ratios) / abs(np.mean(ratios)) if np.mean(ratios) else np.inf
    cv_d = np.std(diffs) / abs(np.mean(diffs)) if np.mean(diffs) else np.inf
    if min(cv_r, cv_d) > 0.5:
        return "unclear"
    if cv_r < 0.5 * cv_d:
        return "viscous"
    if cv_d < 0.5 * cv_r:
        return "coulomb"
    return "unclear"


def fit_oscillation(t, x):
    """
    Fit the motion after release with a damped second-order response:
      underdamped  x = A e^(-sigma t) cos(wd t + phi) + c
      overdamped   x = A1 e^(-s1 t) + A2 e^(-s2 t) + c
    (whichever fits better). Returns wn (rad/s), zeta, r2, the model,
    swings counted, a viscous/coulomb hint, the release time, and the
    fitted curve for plotting.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    i0 = release_index(x)
    tt = t[i0:] - t[i0]
    xx = x[i0:]
    out = {"release_s": float(t[i0]), "wn": np.nan, "zeta": np.nan, "r2": np.nan,
           "model": "none", "n_swings": 0, "decay": "unclear",
           "t_fit": tt + t[i0], "x_fit": np.full_like(xx, np.nan)}
    if len(tt) < 10:
        return out
    c0 = float(np.median(xx[-max(3, len(xx) // 10):]))
    A0 = float(xx[0] - c0)
    idx, amps = peak_amplitudes(xx, c0)
    out["n_swings"] = int(len(amps))
    out["decay"] = decay_type(amps)

    fits = []
    # underdamped: guess wd from the spacing of extrema (half periods)
    if len(idx) >= 2:
        half = np.median(np.diff(tt[idx]))
        wd0 = np.pi / half if half > 0 else 2 * np.pi
        ratio = amps[1] / amps[0] if amps[0] > 0 else 0.5
        sigma0 = max(-np.log(max(min(ratio, 0.99), 1e-3)) / half, 1e-3)
    else:
        wd0 = 2 * np.pi / max(tt[-1], 1e-3)
        sigma0 = 1.0
    try:
        p, _ = curve_fit(_underdamped, tt, xx, p0=[A0, sigma0, wd0, 0.0, c0],
                         bounds=([-np.inf, -50, 0.05, -np.pi, -np.inf],
                                 [np.inf, 500, 2000, np.pi, np.inf]),
                         maxfev=20000)
        fits.append(("underdamped", p, _r2(xx, _underdamped(tt, *p))))
    except (RuntimeError, ValueError):
        pass
    # overdamped: two real poles, released from rest. Several starting
    # guesses: a single start often collapses both poles onto one rate.
    best_od = None
    span = max(tt[-1], 1e-3)
    for s1_0 in (1.0 / span, 3.0 / span, 10.0 / span):
        for ratio in (3.0, 10.0, 30.0):
            try:
                p, _ = curve_fit(_overdamped, tt, xx,
                                 p0=[A0, s1_0, s1_0 * (ratio - 1), c0, 0.0],
                                 bounds=([-np.inf, 1e-4, 1e-4, -np.inf, -0.1],
                                         [np.inf, 1e4, 1e4, np.inf, 0.1]),
                                 maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            r2 = _r2(xx, _overdamped(tt, *p))
            if best_od is None or r2 > best_od[2]:
                best_od = ("overdamped", p, r2)
    if best_od is not None:
        fits.append(best_od)
    if not fits:
        return out
    # When both fit about as well, let the data decide: swings back and
    # forth -> underdamped; no overshoot -> overdamped.
    fits.sort(key=lambda f: f[2], reverse=True)
    best = fits[0]
    prefer = "underdamped" if out["n_swings"] >= 3 else (
        "overdamped" if out["n_swings"] <= 1 else None)
    for f in fits:
        if f[0] == prefer and f[2] >= best[2] - 0.005:
            best = f
            break
    name, p, r2 = best
    if name == "underdamped":
        _, sigma, wd, _, _ = p
        wn = float(np.hypot(sigma, wd))
        zeta = float(sigma / wn)
        out["x_fit"] = _underdamped(tt, *p)
    else:
        _, s1, ds, _, _ = p
        s2 = s1 + ds
        wn = float(np.sqrt(s1 * s2))
        zeta = float((s1 + s2) / (2 * wn))
        out["x_fit"] = _overdamped(tt, *p)
    out.update(wn=wn, zeta=zeta, r2=r2, model=name)
    return out


def oscillation_to_mass_damping(wn: float, zeta: float, k_test: float, d_test: float = 0.0):
    """
    From a free oscillation against a test spring k_test (N/m) with
    commanded damping d_test (N*s/m):
      M       = k_test / wn^2
      b_total = 2*zeta*sqrt(k_test*M)     (everything damping the swing)
      b_robot = b_total - d_test          (the robot's own share)
    """
    if not (np.isfinite(wn) and wn > 0 and k_test > 0):
        return {"M": np.nan, "b_total": np.nan, "b_robot": np.nan}
    M = k_test / wn ** 2
    b_total = 2 * zeta * np.sqrt(k_test * M)
    return {"M": float(M), "b_total": float(b_total), "b_robot": float(b_total - d_test)}


# ── Step 2: stability of one sweep point ──────────────────────────────────────

def stability_metrics(t, x, v, early_s: float = 0.5, late_start_s: float = 2.0,
                      late_s: float = 0.5, hf_hz: float = 10.0, tau=None,
                      decay_limit: float = 0.2, hf_limit: float = 0.3):
    """
    One point of the K x D sweep: displace, release, watch.

      decay_ratio   peak-to-peak amplitude in [late_start, late_start+late_s]
                    after release over that in the first early_s.
                    < decay_limit means it settled.
      f_dominant    strongest frequency in the tip velocity after release (Hz).
      hf_fraction   share of tip-velocity power above hf_hz -- chatter.
      tau_hf_fraction  the same for the commanded spring torques, if given
                    (an (N, n_joints) array on the same time base). Chatter
                    shows up there first.
      stable        decay_ratio < decay_limit and neither hf_fraction is
                    above hf_limit.

    At 100 Hz joint states anything above 50 Hz aliases, so very fast
    buzzing can show up at a lower, wrong frequency -- still caught by
    the decay test, since it doesn't die out.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    i0 = release_index(x)
    tr = t - t[i0]
    out = {"release_s": float(t[i0]), "decay_ratio": np.nan, "f_dominant": np.nan,
           "hf_fraction": np.nan, "tau_hf_fraction": np.nan, "stable": False,
           "late_window_ok": False}
    early = (tr >= 0) & (tr <= early_s)
    late = (tr >= late_start_s) & (tr <= late_start_s + late_s)
    if early.sum() >= 3 and late.sum() >= 3:
        e = np.ptp(x[early])
        out["decay_ratio"] = float(np.ptp(x[late]) / e) if e > 0 else np.nan
        out["late_window_ok"] = True
    after = tr >= 0
    dt = float(np.median(np.diff(t)))
    out["f_dominant"], out["hf_fraction"] = _spectrum_stats(v[after], dt, hf_hz)
    if tau is not None:
        tau = np.asarray(tau, dtype=float)
        if tau.ndim == 1:
            tau = tau[:, None]
        fr = [_spectrum_stats(tau[after, k], dt, hf_hz)[1] for k in range(tau.shape[1])]
        fr = [f for f in fr if np.isfinite(f)]
        out["tau_hf_fraction"] = float(max(fr)) if fr else np.nan
    hf_bad = any(np.isfinite(h) and h > hf_limit
                 for h in (out["hf_fraction"], out["tau_hf_fraction"]))
    out["stable"] = bool(np.isfinite(out["decay_ratio"])
                         and out["decay_ratio"] < decay_limit and not hf_bad)
    return out


def _spectrum_stats(sig, dt: float, hf_hz: float):
    sig = np.asarray(sig, dtype=float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < 8:
        return np.nan, np.nan
    sig = sig - np.mean(sig)
    P = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    f = np.fft.rfftfreq(len(sig), dt)
    P[0] = 0.0
    tot = P.sum()
    if tot <= 0:
        return np.nan, np.nan
    return float(f[np.argmax(P)]), float(P[f > hf_hz].sum() / tot)


def passivity_k_max(b_robot: float, d: float, T: float) -> float:
    """Colgate & Schenkel: b > K*T/2 + |D|  ->  K_max = 2*(b - |D|)/T.
    Optimistic (1-DOF, no delay); the sweep decides. NaN if b <= |D|."""
    if not (np.isfinite(b_robot) and T > 0):
        return np.nan
    k = 2 * (b_robot - abs(d)) / T
    return float(k) if k > 0 else np.nan
