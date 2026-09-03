"""
virtual_spring.py

A platform-agnostic virtual spring class for robot arm impedance control.

A VirtualSpring models a spring attached between:
  - an attachment point on a robot arm link (defined in the link's local frame)
  - a target point in world space (where the spring wants to pull that point to)

The class computes:
  - the spring force vector at the attachment point
  - the generalized joint torques via the Jacobian transpose method

Usage:
    from virtual_spring import VirtualSpring

    spring = VirtualSpring(
        link_name="forearm",
        local_attachment_point=np.array([0.0, 0.0, 0.1]),  # tip of forearm link
        target_world_point=np.array([0.5, 0.0, 0.8]),      # desired world position
        stiffness=100.0,                                    # N/m
        damping=5.0,                                        # N·s/m (optional)
    )

    # Update with current arm state and compute torques
    torques = spring.compute_torques(arm_config)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, Optional
import warnings


# ---------------------------------------------------------------------------
# Protocol: ArmConfiguration
# ---------------------------------------------------------------------------
# Any robot arm backend must satisfy this interface. This keeps VirtualSpring
# fully platform-agnostic — it works with MuJoCo, ROS/URDF, PyBullet, a
# custom simulator, or even a unit-test stub.

class ArmConfiguration(Protocol):
    """
    Minimal interface that a robot arm configuration object must implement.

    Attributes
    ----------
    joint_positions : np.ndarray, shape (n_dof,)
        Current joint angles / positions.
    joint_velocities : np.ndarray, shape (n_dof,)
        Current joint velocities (used only when damping > 0).
    n_dof : int
        Number of degrees of freedom.
    """

    @property
    def joint_positions(self) -> np.ndarray: ...

    @property
    def joint_velocities(self) -> np.ndarray: ...

    @property
    def n_dof(self) -> int: ...

    @property
    def joint_names(self) -> list[str]:
        """Joint names in n_dof/joint_velocities order (used by JointSpring
        to resolve a configured joint_name to a DOF index)."""
        ...

    def get_joint_angle(self, joint_index: int) -> float:
        """
        Return the current angle (rad) of the joint at DOF index
        joint_index. Backends decode their own position representation
        here (e.g. pinocchio's cos/sin encoding for continuous joints) so
        callers like JointSpring don't need to know backend-specific
        details.
        """
        ...

    def get_link_transform(self, link_name: str) -> np.ndarray:
        """
        Return the 4×4 homogeneous transform T_world_link for the named link,
        given the current joint positions.

        Parameters
        ----------
        link_name : str
            Name / identifier of the link.

        Returns
        -------
        np.ndarray, shape (4, 4)
            Homogeneous transformation matrix from link frame to world frame.
        """
        ...

    def get_jacobian(
        self,
        link_name: str,
        local_point: np.ndarray,
    ) -> np.ndarray:
        """
        Return the 6×n_dof geometric Jacobian evaluated at *local_point*
        (expressed in the link frame) for the named link.

        The first 3 rows must be the translational Jacobian (Jv) and the
        last 3 rows the rotational Jacobian (Jw).

        Parameters
        ----------
        link_name : str
            Name / identifier of the link.
        local_point : np.ndarray, shape (3,)
            The point in the link's local frame at which the Jacobian is
            evaluated.

        Returns
        -------
        np.ndarray, shape (6, n_dof)
            Full geometric Jacobian.
        """
        ...
    def get_gravity_torques(self) -> np.ndarray:
        """
        Return the gravity compensation torques τ_grav ∈ R^n_dof.

        Implementations that delegate gravity comp to hardware (e.g. UR3)
        should return np.zeros(self.n_dof).  Implementations that need
        software gravity comp (e.g. Kinova Gen3) should compute it here,
        e.g. via pinocchio.computeGeneralizedGravity().
        urdf_arm_configuration.py will do this automatically.
        """
        ...


# ---------------------------------------------------------------------------
# SpringState — snapshot returned after each update
# ---------------------------------------------------------------------------

@dataclass
class SpringState:
    """
    Snapshot of the spring's state after the most recent call to
    :meth:`VirtualSpring.compute_torques`.

    Attributes
    ----------
    world_attachment_point : np.ndarray, shape (3,)
        Current world-space position of the attachment point.
    displacement : np.ndarray, shape (3,)
        Vector from attachment point to target (positive ⇒ spring is stretched).
    extension : float
        Scalar length of the displacement (≥ 0).
    force_world : np.ndarray, shape (3,)
        Spring + damping force vector in world coordinates, applied at the
        attachment point.
    torques : np.ndarray, shape (n_dof,)
        Generalized joint torques produced by this spring.
    """
    world_attachment_point: np.ndarray
    displacement: np.ndarray
    extension: float
    force_world: np.ndarray
    torques: np.ndarray


@dataclass
class JointSpringState:
    """
    Snapshot of a :class:`JointSpring`'s state after the most recent call to
    :meth:`JointSpring.compute_torques`.

    Attributes
    ----------
    current_angle : float
        Current angle (rad) of the joint.
    extension : float
        Signed angular error (rad) between current_angle and target_angle
        (shortest-path, wrap-aware). Named `extension` -- not `angle_error`
        -- so a JointSpring duck-types against SpringState for the shared
        plotting/bookkeeping code in virtual_spring_node.py that reads
        spring._last_state.extension regardless of spring type. It's an
        angle here, not a distance.
    torque : float
        Scalar torque (N·m) commanded on this joint.
    torques : np.ndarray, shape (n_dof,)
        Full generalized torque vector (zero everywhere except this joint).
    """
    current_angle: float
    extension: float
    torque: float
    torques: np.ndarray


@dataclass
class PoseSpringState:
    """
    Snapshot of a :class:`PoseSpring`'s state after the most recent call to
    :meth:`PoseSpring.compute_torques`.

    Attributes
    ----------
    world_face_normal : np.ndarray, shape (3,)
        Current world-space direction of the link's face normal.
    desired_direction : np.ndarray, shape (3,)
        Unit vector from the attachment point toward target_world_point --
        the direction the face normal is being pulled toward.
    extension : float
        Angular error (rad) between world_face_normal and desired_direction.
        Named `extension` -- not `angle_error` -- for the same duck-typing
        reason as JointSpringState above: shared plotting/bookkeeping code
        in virtual_spring_node.py reads spring._last_state.extension
        regardless of spring type. It's an angle here, not a distance.
    moment_world : np.ndarray, shape (3,)
        Restoring moment (N·m) in world coordinates, before the position-
        radius gate described in compute_torques -- the moment this spring
        *wants* to apply, not scaled by risky_scale.
    position_distance : float
        Current distance (m) from the attachment point to position_center.
        EXPERIMENTAL (2026-09-02): backs the position-radius gate on
        torques -- see compute_torques.
    torques : np.ndarray, shape (n_dof,)
        Generalized joint torques actually produced by this spring (after
        the position-radius gate).
    """
    world_face_normal: np.ndarray
    desired_direction: np.ndarray
    extension: float
    moment_world: np.ndarray
    position_distance: float
    torques: np.ndarray


# ---------------------------------------------------------------------------
# VirtualSpring
# ---------------------------------------------------------------------------

class VirtualSpring:
    """
    A virtual spring between a point on a robot arm link and a fixed world point.

    The spring exerts a force proportional to the displacement of the
    attachment point from the target (Hooke's law), plus an optional
    viscous damping force that opposes the velocity of the attachment point.
    Joint torques are computed via the Jacobian-transpose method:

        τ = Jv(q)ᵀ · F_world

    where Jv is the 3×n_dof translational Jacobian evaluated at the
    attachment point.

    Parameters
    ----------
    link_name : str
        Name of the robot link the spring is attached to.
    local_attachment_point : array-like, shape (3,)
        Position of the spring's attachment point expressed in the link's
        local frame.
    target_world_point : array-like, shape (3,)
        Fixed target position in world coordinates. The spring pulls the
        attachment point toward this location.
    stiffness : float
        Spring stiffness k (N/m). Must be ≥ 0.
    damping : float, optional
        Viscous damping coefficient b (N·s/m). Default 0. Must be ≥ 0.
        Requires the arm configuration to expose joint velocities.
    rest_length : float, optional
        Natural (rest) length of the spring (m). Default 0 (zero-length spring).
        A non-zero rest length means the spring only pulls when the distance
        exceeds rest_length, and pushes when the distance is less than it.
        Ignored whenever a deadband is configured (outer_radius >
        inner_radius): the plain-spring region beyond outer_radius is then
        anchored at inner_radius instead, so the deadband ramp and the
        plain-spring region agree exactly at extension == outer_radius. See
        compute_torques.
    enabled : bool, optional
        If False the spring produces zero force. Useful for toggling without
        removing the object. Default True.
    name : str
        Human-readable label for logging / debugging.
    inner_radius: float, optional
        inside of the deadband (inside this radius the spring will be active, outside it will not exert force until it reaches outer_radius
    outer_radius: float, optional
        outside of the deadband (outside this radius the spring will be active)
    """

    def __init__(
        self,
        link_name: str,
        local_attachment_point: np.ndarray,
        target_world_point: np.ndarray,
        stiffness: float,
        damping: float = 0.0,
        rest_length: float = 0.0,
        inner_radius: float = 0.0,
        outer_radius: float = 0.0,    
        enabled: bool = True,
        name: str = "",
    ):
        self.link_name = link_name
        self.local_attachment_point = np.asarray(local_attachment_point, dtype=float)
        self.target_world_point = np.asarray(target_world_point, dtype=float)
        self.stiffness = stiffness
        self.damping = damping
        self.rest_length = rest_length
        self.enabled = enabled
        self.name = name or f"spring_{link_name}"

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius


        # Validate shapes
        if self.local_attachment_point.shape != (3,):
            raise ValueError(
                f"local_attachment_point must have shape (3,), "
                f"got {self.local_attachment_point.shape}"
            )
        if self.target_world_point.shape != (3,):
            raise ValueError(
                f"target_world_point must have shape (3,), "
                f"got {self.target_world_point.shape}"
            )
        if stiffness < 0:
            raise ValueError(f"stiffness must be ≥ 0, got {stiffness}")
        if damping < 0:
            raise ValueError(f"damping must be ≥ 0, got {damping}")
        if rest_length < 0:
            raise ValueError(f"rest_length must be ≥ 0, got {rest_length}")

        # Cached state from the last compute call
        self._last_state: Optional[SpringState] = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_torques(self, arm: ArmConfiguration) -> np.ndarray:

        """
        Compute the generalized joint torques produced by this spring given
        the current arm configuration.

        Parameters
        ----------
        arm : ArmConfiguration
            Current state of the robot arm (must satisfy the ArmConfiguration
            protocol).

        Returns
        -------
        np.ndarray, shape (n_dof,)
            Joint torques. Returns a zero vector if ``self.enabled`` is False.
        """
        n_dof = arm.n_dof
        zero_torques = np.zeros(n_dof)
        if not self.enabled:
            self._last_state = None
            warnings.warn(
                f"{self.name}: spring disabled, contributing zero torque "
                f"(gravity comp, if enabled, is unaffected).",
                stacklevel=2,
            )
            return zero_torques

        # 1. World-space position of the attachment point
        T = arm.get_link_transform(self.link_name)          # (4, 4)
        p_local_h = np.append(self.local_attachment_point, 1.0)  # homogeneous
        p_world = (T @ p_local_h)[:3]                       # (3,)

        #print(f"{self.name} attachment point in world frame: {p_world}")  # debug

        # 2. Displacement and extension
        displacement = self.target_world_point - p_world    # points toward target
        extension = np.linalg.norm(displacement)
        #print(f"{self.name} displacement: {displacement}, extension: {extension}")  # debug

        # 3. Spring force (Hooke's law with optional rest length)
        # Zero-extension (target == current position, e.g. a freshly
        # resolved auto-target) divides 0/0 -> NaN; only happens to be
        # harmless today because inner_radius defaults to 0.0, so the
        # extension <= inner_radius branch below hard-codes f_spring to
        # zero without ever using `direction`. Guard explicitly instead of
        # relying on that -- any nonzero inner_radius, or floating-point
        # noise landing extension just above zero, would let a NaN
        # direction reach f_spring and then the published torque.
        direction = displacement / extension if extension > 1e-9 else np.zeros(3)
        deadband_active = self.outer_radius > self.inner_radius
        if extension <= self.inner_radius:
            # Attachment point is inside allowable target range
            f_spring = np.zeros(3)
        elif deadband_active and extension <= self.outer_radius:
            t = (extension - self.inner_radius) / (self.outer_radius - self.inner_radius)
            f_spring = self.stiffness * t * (extension - self.inner_radius) * direction
        elif deadband_active:
            # Beyond outer_radius, with a deadband configured: anchor the
            # plain-spring region at inner_radius (not rest_length) so it
            # matches the deadband branch's value at extension ==
            # outer_radius exactly -- continuous by construction,
            # regardless of whatever rest_length happens to be configured.
            # Anchoring at rest_length instead let a mismatched rest_length
            # (e.g. 0 instead of inner_radius) produce a hard force
            # discontinuity right at outer_radius: with extension noise
            # straddling that exact boundary, the commanded force flickered
            # between stiffness*(outer_radius-inner_radius) and
            # stiffness*extension every cycle (tip_spring alternating
            # ~1.27N/~6.35N, live 2026-09-02).
            stretch = extension - self.inner_radius
            f_spring = self.stiffness * stretch * direction
        else:
            stretch = extension - self.rest_length          # can be negative
            f_spring = self.stiffness * stretch * direction
        #print(f"{self.name} spring force: {f_spring}")  # debug

        # 4. Damping force (opposes velocity of the attachment point)
        f_damp = np.zeros(3)
        J = arm.get_jacobian(self.link_name, self.local_attachment_point)
        Jv = J[:3, :]                                    # translational part
        #print("jacobian", Jv)

        if self.damping > 0.0:
            p_dot = Jv @ arm.joint_velocities # world-space velocity
            f_damp = -self.damping * p_dot
        # 5. Total force
        f_total = f_spring + f_damp

        # 6. Joint torques via Jacobian transpose
        if self.damping == 0.0:
            pass
        torques = Jv.T @ f_total                            # (n_dof,)

        #print(f"{self.name} joint torques: {torques}")  # debug
        # 7. Cache state
        self._last_state = SpringState(
            world_attachment_point=p_world,
            displacement=displacement,
            extension=float(extension),
            force_world=f_total,
            torques=torques,
        )
        #print(f"{self.name} state cached: {self._last_state}")  # debug
        return torques

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_spring_force(self, arm: ArmConfiguration) -> np.ndarray:
        """
        Return only the world-space spring+damping force vector (N).
        Calls compute_torques internally and caches the state.
        """
        self.compute_torques(arm)
        if self._last_state is None:
            return np.zeros(3)
        return self._last_state.force_world.copy()

    def get_potential_energy(self, arm: ArmConfiguration) -> float:
        """
        Return the elastic potential energy stored in the spring (J).
        U = ½ k (extension - rest_length)²
        """
        T = arm.get_link_transform(self.link_name)
        p_local_h = np.append(self.local_attachment_point, 1.0)
        p_world = (T @ p_local_h)[:3]
        extension = float(np.linalg.norm(self.target_world_point - p_world))
        stretch = extension - self.rest_length
        return 0.5 * self.stiffness * stretch ** 2

    @property
    def last_state(self) -> Optional[SpringState]:
        """
        The :class:`SpringState` snapshot from the most recent
        :meth:`compute_torques` call, or ``None`` if it has never been called
        or the spring is disabled.
        """
        return self._last_state

    def move_target(self, new_target: np.ndarray) -> None:
        """Update the world-space target point."""
        new_target = np.asarray(new_target, dtype=float)
        if new_target.shape != (3,):
            raise ValueError(f"new_target must have shape (3,), got {new_target.shape}")
        self.target_world_point = new_target

    def __repr__(self) -> str:
        return (
            f"VirtualSpring(name={self.name!r}, "
            f"link={self.link_name!r}, "
            f"k={self.stiffness} N/m, "
            f"b={self.damping} N·s/m, "
            f"rest_length={self.rest_length} m, "
            f"enabled={self.enabled})"
        )


# ---------------------------------------------------------------------------
# JointSpring
# ---------------------------------------------------------------------------

class JointSpring:
    """
    A virtual torsional spring that pulls a single joint toward a target
    angle, independent of any link's Cartesian pose.

    Unlike :class:`VirtualSpring`, this acts directly on one joint's own
    DOF -- no Jacobian involved -- so it can supply restoring torque on
    rotational axes a Cartesian spring is structurally blind to (e.g. a
    wrist joint whose axis passes through the spring's attachment point,
    where rotating that joint doesn't move the attachment point in world
    space at all, so a VirtualSpring there produces exactly zero torque
    regardless of stiffness).

        τ_j = -k · Δθ - b · θ̇

    where Δθ is the shortest-path signed angular error (current - target,
    wrapped into (-π, π]) and θ̇ is the joint's own velocity.

    Parameters
    ----------
    joint_name : str
        Name of the joint (must appear in arm.joint_names).
    target_angle : float
        Target angle (rad) the spring pulls the joint toward.
    stiffness : float
        Spring stiffness k (N·m/rad). Must be ≥ 0.
    damping : float, optional
        Viscous damping coefficient b (N·m·s/rad). Default 0. Must be ≥ 0.
    enabled : bool, optional
        If False the spring produces zero torque. Default True.
    name : str
        Human-readable label for logging / debugging.
    """

    def __init__(
        self,
        joint_name: str,
        target_angle: float,
        stiffness: float,
        damping: float = 0.0,
        enabled: bool = True,
        name: str = "",
    ):
        self.joint_name = joint_name
        self.target_angle = float(target_angle)
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = enabled
        self.name = name or f"joint_spring_{joint_name}"

        if stiffness < 0:
            raise ValueError(f"stiffness must be ≥ 0, got {stiffness}")
        if damping < 0:
            raise ValueError(f"damping must be ≥ 0, got {damping}")

        self._last_state: Optional[JointSpringState] = None

    def compute_torques(self, arm: ArmConfiguration) -> np.ndarray:
        """
        Compute the generalized joint torques produced by this spring given
        the current arm configuration.

        Returns
        -------
        np.ndarray, shape (n_dof,)
            Zero everywhere except this spring's joint. Returns an all-zero
            vector if ``self.enabled`` is False.
        """
        n_dof = arm.n_dof
        if not self.enabled:
            self._last_state = None
            warnings.warn(
                f"{self.name}: joint spring disabled, contributing zero "
                f"torque (gravity comp, if enabled, is unaffected).",
                stacklevel=2,
            )
            return np.zeros(n_dof)

        joint_index = arm.joint_names.index(self.joint_name)
        current_angle = arm.get_joint_angle(joint_index)

        # Shortest signed path from target to current -- avoids yanking a
        # continuous joint the long way around near the wrap point.
        angle_error = math.remainder(current_angle - self.target_angle, 2 * math.pi)
        velocity = arm.joint_velocities[joint_index]

        torque = -self.stiffness * angle_error - self.damping * velocity

        torques = np.zeros(n_dof)
        torques[joint_index] = torque

        self._last_state = JointSpringState(
            current_angle=current_angle,
            extension=float(angle_error),
            torque=float(torque),
            torques=torques,
        )
        return torques

    @property
    def last_state(self) -> Optional[JointSpringState]:
        """
        The :class:`JointSpringState` snapshot from the most recent
        :meth:`compute_torques` call, or ``None`` if it has never been
        called or the spring is disabled.
        """
        return self._last_state

    def move_target(self, new_target_angle: float) -> None:
        """Update the target angle (rad)."""
        self.target_angle = float(new_target_angle)

    def __repr__(self) -> str:
        return (
            f"JointSpring(name={self.name!r}, "
            f"joint={self.joint_name!r}, "
            f"k={self.stiffness} N·m/rad, "
            f"b={self.damping} N·m·s/rad, "
            f"enabled={self.enabled})"
        )


# ---------------------------------------------------------------------------
# PoseSpring
# ---------------------------------------------------------------------------

class PoseSpring:
    """
    A virtual spring that aligns a direction fixed in a link's local frame
    (a "face normal") with the direction from that link's attachment point
    to a static target point in world space -- a "look at" constraint --
    while also bounding how far the attachment point itself is allowed to
    drift while doing so. Position AND orientation, not orientation alone
    (renamed from OrientationSpring, 2026-09-02 -- see below).

    Where VirtualSpring pulls a point toward a target (and is structurally
    blind to any rotation about an axis through that point), this spring's
    moment is still built purely from the rotational Jacobian (Jw) -- the
    same "look at" math as before, unchanged. But unlike the old
    OrientationSpring, Jv (the point's translational Jacobian) IS now used,
    to split the resulting joint torque into a position-safe part (can
    never move the attachment point, applied at full strength always) and
    the remainder (which can move it, applied at full strength within
    position_radius of position_center and ramped down beyond it -- see
    compute_torques). EXPERIMENTAL as of 2026-09-02: a live incident showed
    that with no position bound at all, a modest orientation moment could
    still pull the attachment point well outside a paired position spring's
    own workspace (e.g. ~0.17m outside a 0.12m outer_radius), since on a
    serial chain most joints affect both position and orientation
    simultaneously -- "zero direct Cartesian force" (still true of the
    moment itself) doesn't mean "the point can't move" once real torque is
    applied. An earlier, stricter design (project ALL of the risky
    component out unconditionally, everywhere) was tried and rejected as
    too weak in practice -- see compute_torques's own comment for why.

    The desired heading is recomputed every cycle from the *current*
    attachment position, so as the arm/block moves the spring keeps
    pointing the face at target_world_point, not a rotation frozen at
    spring-creation time.

        M = k * angle * axis_hat - b * ω

    where axis_hat/angle is the axis-angle rotation taking the current face
    normal to the desired direction (via their cross product), and ω is the
    link's world-frame angular velocity.

    Parameters
    ----------
    link_name : str
        Name of the robot link the spring is attached to.
    local_attachment_point : array-like, shape (3,)
        Point (link-local frame) the face normal and moment are defined
        about -- e.g. the gripper's held-block center.
    local_face_normal : array-like, shape (3,)
        Direction (link-local frame) of the face to orient, e.g. [0,0,1]
        for a face pointing along the link's local +z. Normalized
        internally.
    target_world_point : array-like, shape (3,)
        Static world point the face should point at (e.g. the participant's
        face). Only ever used to derive a *direction* -- this spring exerts
        no force toward it, only torque.
    stiffness : float
        Rotational stiffness k (N·m/rad). Must be ≥ 0.
    position_center : array-like, shape (3,)
        World point defining the center of this spring's allowed operating
        region -- typically the SAME target a paired VirtualSpring (e.g. a
        position/reach-center spring) is already using, so both springs
        agree on where the arm is supposed to be. Required -- there's no
        safe default that reproduces the old unconstrained-drift behavior.
    position_radius : float
        Radius (m) around position_center within which this spring has
        full authority to move the attachment point while correcting
        orientation, AND within which position_stiffness applies zero
        restoring force (a true dead zone -- the attachment point is free
        to move anywhere in here). Beyond it, the orientation-correcting
        torque components that would move the point further are ramped
        toward zero -- see compute_torques. Required, must be > 0.
    position_stiffness : float, optional
        Translational stiffness (N/m) of a restoring pull back toward
        position_center, active only beyond position_radius (zero inside
        it -- see position_radius). Default 0: no restoring pull at all,
        which reproduces this spring's original behavior (position_radius/
        position_center only throttle orientation-authority, exactly as
        before this parameter existed) -- that default assumed a paired
        VirtualSpring was already anchoring the arm; a pose spring used
        with no such paired spring needs a nonzero value here or the
        attachment point has nothing at all pulling it back once it drifts
        past position_radius (confirmed live 2026-09-03: it can wander off
        and settle in an arbitrary orientation-satisfying pose, e.g.
        looking down at the target from above instead of up at it from
        below). Must be ≥ 0.
    damping : float, optional
        Rotational viscous damping b (N·m·s/rad). Default 0. Must be ≥ 0.
    enabled : bool, optional
        If False the spring produces zero torque. Default True.
    name : str
        Human-readable label for logging / debugging.
    """

    def __init__(
        self,
        link_name: str,
        local_attachment_point: np.ndarray,
        local_face_normal: np.ndarray,
        target_world_point: np.ndarray,
        stiffness: float,
        position_center: np.ndarray,
        position_radius: float,
        position_stiffness: float = 0.0,
        damping: float = 0.0,
        enabled: bool = True,
        name: str = "",
    ):
        self.link_name = link_name
        self.local_attachment_point = np.asarray(local_attachment_point, dtype=float)
        self.local_face_normal = np.asarray(local_face_normal, dtype=float)
        self.target_world_point = np.asarray(target_world_point, dtype=float)
        self.stiffness = stiffness
        self.position_center = np.asarray(position_center, dtype=float)
        self.position_radius = float(position_radius)
        self.position_stiffness = float(position_stiffness)
        self.damping = damping
        self.enabled = enabled
        self.name = name or f"pose_spring_{link_name}"

        if self.local_attachment_point.shape != (3,):
            raise ValueError(
                f"local_attachment_point must have shape (3,), "
                f"got {self.local_attachment_point.shape}"
            )
        if self.local_face_normal.shape != (3,):
            raise ValueError(
                f"local_face_normal must have shape (3,), "
                f"got {self.local_face_normal.shape}"
            )
        face_normal_norm = np.linalg.norm(self.local_face_normal)
        if face_normal_norm < 1e-9:
            raise ValueError("local_face_normal must be nonzero")
        self.local_face_normal = self.local_face_normal / face_normal_norm
        if self.target_world_point.shape != (3,):
            raise ValueError(
                f"target_world_point must have shape (3,), "
                f"got {self.target_world_point.shape}"
            )
        if self.position_center.shape != (3,):
            raise ValueError(
                f"position_center must have shape (3,), "
                f"got {self.position_center.shape}"
            )
        if self.position_radius <= 0:
            raise ValueError(f"position_radius must be > 0, got {self.position_radius}")
        if self.position_stiffness < 0:
            raise ValueError(f"position_stiffness must be ≥ 0, got {self.position_stiffness}")
        if stiffness < 0:
            raise ValueError(f"stiffness must be ≥ 0, got {stiffness}")
        if damping < 0:
            raise ValueError(f"damping must be ≥ 0, got {damping}")

        self._last_state: Optional[PoseSpringState] = None

    def compute_torques(self, arm: ArmConfiguration) -> np.ndarray:
        """
        Compute the generalized joint torques produced by this spring given
        the current arm configuration.

        Returns
        -------
        np.ndarray, shape (n_dof,)
            Joint torques. Returns a zero vector if ``self.enabled`` is False.
        """
        n_dof = arm.n_dof
        zero_torques = np.zeros(n_dof)
        if not self.enabled:
            self._last_state = None
            warnings.warn(
                f"{self.name}: pose spring disabled, contributing "
                f"zero torque (gravity comp, if enabled, is unaffected).",
                stacklevel=2,
            )
            return zero_torques

        # 1. World-space pose of the link
        T = arm.get_link_transform(self.link_name)          # (4, 4)
        R = T[:3, :3]
        p_local_h = np.append(self.local_attachment_point, 1.0)
        p_world = (T @ p_local_h)[:3]

        # 2. Current face normal, in world frame
        n_current = R @ self.local_face_normal
        n_current = n_current / np.linalg.norm(n_current)

        # 3. Desired ("look at") direction, recomputed from the *current*
        # attachment position every cycle -- this is what makes it a look-at
        # constraint rather than a rotation frozen at construction time.
        to_target = self.target_world_point - p_world
        dist = np.linalg.norm(to_target)
        if dist < 1e-9:
            # Attachment point coincides with the target -- "point at" is
            # undefined here; hold last heading (zero moment) rather than
            # dividing 0/0 into a NaN direction.
            n_desired = n_current.copy()
        else:
            n_desired = to_target / dist

        # 4. Axis-angle rotation error taking n_current -> n_desired.
        # atan2(|cross|, dot) stays well-conditioned near angle=0 and
        # angle=pi, unlike acos(dot) alone.
        cross = np.cross(n_current, n_desired)
        sin_angle = np.linalg.norm(cross)
        cos_angle = np.clip(np.dot(n_current, n_desired), -1.0, 1.0)
        angle = math.atan2(sin_angle, cos_angle)
        axis = cross / sin_angle if sin_angle > 1e-9 else np.zeros(3)

        # 5. Restoring moment (Hooke's law in angle, about `axis`) -- same
        # pure "look at" moment as before this spring gained a position
        # bound; the gating happens below, in how it's turned into torque.
        m_spring = self.stiffness * angle * axis

        # 6. Damping (opposes the link's angular velocity)
        J = arm.get_jacobian(self.link_name, self.local_attachment_point)
        Jv = J[:3, :]                                        # translational part
        Jw = J[3:, :]                                        # rotational part
        m_damp = np.zeros(3)
        if self.damping > 0.0:
            omega = Jw @ arm.joint_velocities                # world angular velocity
            m_damp = -self.damping * omega
        m_total = m_spring + m_damp

        # 7. Split Jw.T @ m_total into a position-safe part (never moves
        # the attachment point, via the null space of Jv -- Jv @ tau_safe
        # == 0 exactly, by the Moore-Penrose identity Jv @ Jv^+ @ Jv ==
        # Jv) and the remainder (tau_risky, the part that WOULD move it).
        # tau_safe is always applied at full strength -- it's free, it
        # can't cause the drift this whole design exists to bound.
        # tau_risky is scaled by how close the attachment point currently
        # is to position_center: full strength inside position_radius,
        # smoothly decreasing beyond it (min(1, radius/dist), continuous
        # at the boundary -- no separate margin parameter, no
        # discontinuity to chatter against as the point crosses back and
        # forth). EXPERIMENTAL (2026-09-02): this replaces an earlier,
        # rejected design that projected out tau_risky unconditionally
        # everywhere -- confirmed live that this made the spring far too
        # weak in general, since on a serial chain most joint-torque
        # directions that correct orientation also move the point, and a
        # real arm's null space can be small or near-empty in many
        # configurations (including the one that caused the original
        # incident). This version gives full authority near
        # position_center (where a paired position spring is presumably
        # already anchoring the arm) and only backs off once it would
        # pull the point out of that zone.
        tau_orient = Jw.T @ m_total
        null_space_projector = np.eye(n_dof) - np.linalg.pinv(Jv) @ Jv
        tau_safe = null_space_projector @ tau_orient
        tau_risky = tau_orient - tau_safe

        dist_from_center = np.linalg.norm(p_world - self.position_center)
        risky_scale = (
            min(1.0, self.position_radius / dist_from_center)
            if dist_from_center > 1e-9 else 1.0
        )
        torques = tau_safe + risky_scale * tau_risky

        # 7b. position_stiffness: an explicit restoring pull back toward
        # position_center, zero inside position_radius (true dead zone)
        # and Hooke's-law beyond it -- same "excess distance past a
        # radius, along the unit vector back to center" shape
        # VirtualSpring.compute_torques uses beyond its own inner_radius
        # (deadband_active=False branch). Needed because tau_risky above
        # only ever *throttles* orientation torque -- it never pulls the
        # point back on its own -- so with no paired VirtualSpring
        # anchoring the arm (confirmed live 2026-09-03), nothing else in
        # this spring restores position at all once past position_radius.
        if self.position_stiffness > 0.0 and dist_from_center > self.position_radius:
            direction_to_center = (self.position_center - p_world) / dist_from_center
            f_position = self.position_stiffness * (dist_from_center - self.position_radius) * direction_to_center
            torques = torques + Jv.T @ f_position

        # 8. Cache state
        self._last_state = PoseSpringState(
            world_face_normal=n_current,
            desired_direction=n_desired,
            extension=float(angle),
            moment_world=m_total,
            position_distance=float(dist_from_center),
            torques=torques,
        )
        return torques

    @property
    def last_state(self) -> Optional[PoseSpringState]:
        """
        The :class:`PoseSpringState` snapshot from the most recent
        :meth:`compute_torques` call, or ``None`` if it has never been
        called or the spring is disabled.
        """
        return self._last_state

    def move_target(self, new_target: np.ndarray) -> None:
        """Update the world-space point the face should point at."""
        new_target = np.asarray(new_target, dtype=float)
        if new_target.shape != (3,):
            raise ValueError(f"new_target must have shape (3,), got {new_target.shape}")
        self.target_world_point = new_target

    def __repr__(self) -> str:
        return (
            f"PoseSpring(name={self.name!r}, "
            f"link={self.link_name!r}, "
            f"k={self.stiffness} N·m/rad, "
            f"b={self.damping} N·m·s/rad, "
            f"position_center={self.position_center.tolist()}, "
            f"position_radius={self.position_radius} m, "
            f"position_stiffness={self.position_stiffness} N/m, "
            f"enabled={self.enabled})"
        )


# ---------------------------------------------------------------------------
# SpringCollection — manage multiple springs at once
# ---------------------------------------------------------------------------

class SpringCollection:
    """
    A container that holds multiple :class:`VirtualSpring` objects and
    aggregates their torques.

    Example
    -------
    >>> springs = SpringCollection()
    >>> springs.add(VirtualSpring("link2", [0,0,0.1], [0.5,0,0.8], k=100))
    >>> springs.add(VirtualSpring("wrist", [0,0,0],  [0.3,0.2,1.0], k=50))
    >>> total_torques = springs.compute_total_torques(arm_config)
    """

    def __init__(self) -> None:
        self._springs: list[VirtualSpring] = []

    def add(self, spring: VirtualSpring) -> None:
        """Add a spring to the collection."""
        self._springs.append(spring)

    def remove(self, name: str) -> None:
        """Remove a spring by name. Raises KeyError if not found."""
        for i, s in enumerate(self._springs):
            if s.name == name:
                self._springs.pop(i)
                return
        raise KeyError(f"No spring named {name!r}")

    def compute_total_torques(
        self, arm: ArmConfiguration, add_gravity_compensation, spring_scale: float = 1.0
    ) -> np.ndarray:
        """
        Compute and sum torques from all enabled springs.

        spring_scale scales only the summed spring torque, not gravity
        compensation -- gravity comp must stay at full magnitude on every
        cycle it's enabled, since it's what's holding the arm up against
        gravity; scaling it down (e.g. as part of a torque-mode-enable
        ramp) lets the arm sag/drop. Only ever ramp spring_scale, never
        gravity comp itself. See torque_ramp_gravity_comp_lesson.

        Returns
        -------
        np.ndarray, shape (n_dof,)
        """
        total = np.zeros(arm.n_dof)
        for spring in self._springs:
            total += spring.compute_torques(arm)
        total *= spring_scale
        if add_gravity_compensation:
            total += arm.get_gravity_torques()
        return total

    def get_spring(self, name: str) -> VirtualSpring:
        """Retrieve a spring by name."""
        for s in self._springs:
            if s.name == name:
                return s
        raise KeyError(f"No spring named {name!r}")

    def __len__(self) -> int:
        return len(self._springs)

    def __iter__(self):
        return iter(self._springs)

    def __repr__(self) -> str:
        return f"SpringCollection([{', '.join(s.name for s in self._springs)}])"
