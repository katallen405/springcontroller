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
    enabled : bool, optional
        If False the spring produces zero force. Useful for toggling without
        removing the object. Default True.
    name : str, optional
        Human-readable label for logging / debugging.
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
        print("computing torques")
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
        #print(f"Computing torques for {self.name} (enabled={self.enabled})")  # debug
        if not self.enabled:
            self._last_state = None
            print(f"{self.name} is disabled; returning zero torques.")
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
        if extension <= self.inner_radius:
            # Attachment point is inside allowable target range
            f_spring = np.zeros(3)
        elif self.outer_radius > self.inner_radius and extension <=self.outer_radius:
            t = (extension - self.inner_radius) / (self.outer_radius - self.inner_radius)
            f_spring = self.stiffness * t * (extension - self.inner_radius) * direction
        else:
            direction = displacement / extension
            stretch = extension - self.rest_length          # can be negative
            f_spring = self.stiffness * stretch * direction
        #print(f"{self.name} spring force: {f_spring}")  # debug

        # 4. Damping force (opposes velocity of the attachment point)
        f_damp = np.zeros(3)
        if self.damping > 0.0:
            J = arm.get_jacobian(self.link_name, self.local_attachment_point)
            Jv = J[:3, :]                                    # translational part
            print("jacobian", Jv)
            p_dot = Jv @ arm.joint_velocities                # world-space velocity
            f_damp = -self.damping * p_dot
           # print(f"{self.name} damping force: {f_damp}")  # debug
        # 5. Total force
        f_total = f_spring + f_damp
        print(f"{self.name} total force: {f_total}")  # debug

        # 6. Joint torques via Jacobian transpose
        if self.damping == 0.0:
            # Jacobian may not have been fetched yet
            J = arm.get_jacobian(self.link_name, self.local_attachment_point)
            Jv = J[:3, :]

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

    def compute_total_torques(self, arm: ArmConfiguration) -> np.ndarray:
        """
        Compute and sum torques from all enabled springs.

        Returns
        -------
        np.ndarray, shape (n_dof,)
        """
        total = np.zeros(arm.n_dof)
        for spring in self._springs:
            total += spring.compute_torques(arm)
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
