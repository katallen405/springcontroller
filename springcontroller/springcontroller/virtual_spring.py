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
class OrientationSpringState:
    """
    Snapshot of an :class:`OrientationSpring`'s state after the most recent
    call to :meth:`OrientationSpring.compute_torques`.

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
        Restoring moment (N·m) in world coordinates.
    torques : np.ndarray, shape (n_dof,)
        Generalized joint torques produced by this spring.
    """
    world_face_normal: np.ndarray
    desired_direction: np.ndarray
    extension: float
    moment_world: np.ndarray
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
# OrientationSpring
# ---------------------------------------------------------------------------

class OrientationSpring:
    """
    A virtual rotational spring that aligns a direction fixed in a link's
    local frame (a "face normal") with the direction from that link's
    attachment point to a static target point in world space -- without
    exerting any translational pull.

    Where VirtualSpring pulls a point toward a target (and is structurally
    blind to any rotation about an axis through that point), OrientationSpring
    does the opposite: it produces a pure restoring moment about the
    attachment point and zero translational force, by using only the
    rotational Jacobian (Jw) -- Jv is never referenced. The desired heading
    is recomputed every cycle from the *current* attachment position, so as
    the arm/block moves the spring keeps pointing the face at
    target_world_point (a "look at" constraint), not a rotation frozen at
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
        damping: float = 0.0,
        enabled: bool = True,
        name: str = "",
    ):
        self.link_name = link_name
        self.local_attachment_point = np.asarray(local_attachment_point, dtype=float)
        self.local_face_normal = np.asarray(local_face_normal, dtype=float)
        self.target_world_point = np.asarray(target_world_point, dtype=float)
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = enabled
        self.name = name or f"orientation_spring_{link_name}"

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
        if stiffness < 0:
            raise ValueError(f"stiffness must be ≥ 0, got {stiffness}")
        if damping < 0:
            raise ValueError(f"damping must be ≥ 0, got {damping}")

        self._last_state: Optional[OrientationSpringState] = None

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
                f"{self.name}: orientation spring disabled, contributing "
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

        # 5. Restoring moment (Hooke's law in angle, about `axis`)
        m_spring = self.stiffness * angle * axis

        # 6. Damping (opposes the link's angular velocity)
        J = arm.get_jacobian(self.link_name, self.local_attachment_point)
        Jw = J[3:, :]                                        # rotational part
        m_damp = np.zeros(3)
        if self.damping > 0.0:
            omega = Jw @ arm.joint_velocities                # world angular velocity
            m_damp = -self.damping * omega
        m_total = m_spring + m_damp

        # 7. Joint torques via Jacobian transpose -- rotational rows ONLY.
        # Jv is deliberately never referenced: that's what keeps this a pure
        # orienting torque with no translational pull on the block.
        torques = Jw.T @ m_total

        # 8. Cache state
        self._last_state = OrientationSpringState(
            world_face_normal=n_current,
            desired_direction=n_desired,
            extension=float(angle),
            moment_world=m_total,
            torques=torques,
        )
        return torques

    @property
    def last_state(self) -> Optional[OrientationSpringState]:
        """
        The :class:`OrientationSpringState` snapshot from the most recent
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
            f"OrientationSpring(name={self.name!r}, "
            f"link={self.link_name!r}, "
            f"k={self.stiffness} N·m/rad, "
            f"b={self.damping} N·m·s/rad, "
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
