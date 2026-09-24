# Sign of the Jacobian point-shift in `urdf_arm_configuration.py`

**Status: reported, not fixed.** No code was changed. Fixing this alters the
dynamics of an already-run study, so it needs to be a deliberate decision.

Found while deriving eq. (4) of `docs/spring_math.tex` for the paper.

## The claim

`UrdfArmConfiguration.get_jacobian` shifts the Jacobian from the link frame
origin to the attachment point with the wrong sign
(`springcontroller/springcontroller/urdf_arm_configuration.py:332`):

```python
skew_p = np.array([
    [ 0,  -pz,  py],
    [ pz,   0, -px],
    [-py,  px,   0],
])
J[:3, :] += skew_p @ J[3:, :]     # <-- should be -=
```

`skew_p` is the standard `[r]_x`, so this computes `Jv_o + [r]_x Jw`. The
correct shift is `Jv_o - [r]_x Jw`.

## Why

`pin.ReferenceFrame.LOCAL_WORLD_ALIGNED` returns a Jacobian whose linear block
is the velocity of the frame **origin**, expressed in world-aligned axes. The
attachment point is rigidly fixed to the link at world-aligned offset
`r = R @ local_point`, so

```
p_dot = o_dot + omega x r
      = o_dot - [r]_x omega        (since omega x r = -(r x omega) = -[r]_x omega)
```

giving `Jv = Jv_o - [r]_x Jw`. The code adds where it should subtract, which
negates the entire rotation-induced (tangential) component of the attachment
point's velocity.

## Reproduction

`jac_check.py` (below) — pure numpy, no pinocchio, so it runs off the lab
machine. It builds a 3-DOF spatial chain with hand-rolled FK, computes both the
LWA frame Jacobian and the true point Jacobian by finite differences, and
compares the two candidate shifts at `r = [0, 0, 0.1]`:

```
true Jv_p (finite diff):
 [[-0.196462  0.106212  0.068254]
  [ 0.259907  0.032855 -0.026367]
  [ 0.        0.093643 -0.068163]]
err with '+= skew@Jw' (code): 0.1875516492995083
err with '-= skew@Jw'      : 4.955904371953257e-08
```

```python
import numpy as np

axes = [np.array([0, 0, 1.]), np.array([0, 1., 0]), np.array([1., 0, 0])]
offs = [np.array([0, 0, .3]), np.array([.4, 0, 0]), np.array([0, .2, .1])]


def rot(a, t):
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def fk(q):
    R, p = np.eye(3), np.zeros(3)
    for i in range(3):
        p = p + R @ offs[i]
        R = R @ rot(axes[i], q[i])
    return R, p


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


q = np.array([0.3, -0.7, 1.1])
r_local = np.array([0.0, 0.0, 0.1])   # same shape as the study's local_point
h = 1e-6

Jv_f = np.zeros((3, 3))
Jw = np.zeros((3, 3))
Jv_p = np.zeros((3, 3))
R0, p0 = fk(q)
pt0 = p0 + R0 @ r_local
for i in range(3):
    dq = q.copy()
    dq[i] += h
    R1, p1 = fk(dq)
    Jv_f[:, i] = (p1 - p0) / h
    Jv_p[:, i] = ((p1 + R1 @ r_local) - pt0) / h
    W = ((R1 - R0) / h) @ R0.T            # world-frame angular velocity skew
    Jw[:, i] = np.array([W[2, 1], W[0, 2], W[1, 0]])

r_world = R0 @ r_local
plus = Jv_f + skew(r_world) @ Jw       # what the code does
minus = Jv_f - skew(r_world) @ Jw      # v_p = v_f + omega x r

print("true Jv_p (finite diff):\n", np.round(Jv_p, 6))
print("err with '+= skew@Jw' (code):", np.abs(plus - Jv_p).max())
print("err with '-= skew@Jw'      :", np.abs(minus - Jv_p).max())
```

The check verifies the kinematic identity independently. The one assumption it
cannot test without pinocchio installed is the `LOCAL_WORLD_ALIGNED` convention
itself — that the linear block is the frame origin's velocity in world axes.
Worth confirming on the lab machine before acting, by comparing
`get_jacobian(link, [0,0,0.1])` against a finite difference of
`get_link_transform` on the real model.

## What it affects

Nothing when `local_point` is `[0, 0, 0]` — the shift term vanishes entirely
and `get_jacobian` returns pinocchio's output unmodified. That is the default
in `_load_one_spring` and `_load_one_pose_spring`
(`virtual_spring_node.py:1167`, `:1288`), and covers every spring that attaches
at a frame origin.

It is nonzero for the study's springs, which attach off-origin:

- `reset_spring_local_point: [0.0, 0.0, 0.1]`
  (`springcontroller_ui/config/study_control_panel.yaml:43`)
- the `block` link preset's `[0, 0, 0.15]` (`web/index.html` `LINK_PRESETS`,
  referenced in the same YAML around line 60)

For those, everything downstream of `Jv` in `virtual_spring.py` is affected:

- the damping force `f_damp = -b * (Jv @ q_dot)` — the tangential part of the
  estimated point velocity has the wrong sign, so damping *injects* energy
  along that component instead of removing it
- `Jv.T @ f_total`, the direction the spring force is projected into joint
  torques
- the pose spring's null-space split — `N = I - pinv(Jv) @ Jv` is built from
  the incorrect `Jv`, so `tau_safe` is the null space of the wrong matrix and
  the guarantee `Jv @ tau_safe == 0` holds only against that wrong `Jv`, not
  against true point motion
- `PoseSpring`'s `Jv.T @ f_position` restoring pull

The magnitude of the error scales with `||r||` and with how much of the point's
motion is rotation-induced. At `||r|| = 0.1`–`0.15` m it is not negligible.

Note this does **not** affect where the spring *thinks* the attachment point
is: `get_link_transform` and eq. (1) are untouched, so extension, force
magnitude and all logged positions are correct. The error is confined to the
velocity/torque mapping.

## Tests

`test/test_virtual_spring.py` and `test/test_pose_spring.py` will not catch
this: both use stub arms that return a hand-written Jacobian directly and never
exercise `UrdfArmConfiguration.get_jacobian`. A regression test would need
either a real URDF model or the finite-difference check above.
