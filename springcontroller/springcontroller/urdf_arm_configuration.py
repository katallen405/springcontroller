"""
urdf_arm_configuration.py
ArmConfiguration implementation backed by a URDF + pinocchio.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from typing import Optional
import os

# ---------------------------------------------------------------------------
# Locked-joint model reduction
# ---------------------------------------------------------------------------

def _build_pinocchio_model(
    urdf_path: str, locked_joint_names: list[str] | None
) -> pin.Model:
    """
    Build a pinocchio model from a URDF, optionally locking (fixing) named
    joints via model reduction.

    A locked joint's link mass/inertia stays folded into its parent link, so
    it still contributes to gravity/dynamics, but it's removed from the
    active DOF and from `q`. Use this for joints present in the URDF but not
    reported on /joint_states (e.g. an unmeasured gripper) — dropping them
    from the URDF entirely would silently exclude their weight from gravity
    compensation.
    """
    full_model = pin.buildModelFromUrdf(urdf_path)
    if not locked_joint_names:
        return full_model
    joint_ids = [full_model.getJointId(name) for name in locked_joint_names]
    reference_configuration = pin.neutral(full_model)
    return pin.buildReducedModel(full_model, joint_ids, reference_configuration)


# ---------------------------------------------------------------------------
# CollisionStatus — returned by get_collision_status()
# ---------------------------------------------------------------------------

@dataclass
class CollisionStatus:
    """
    Snapshot of the arm's self-collision state.

    Attributes
    ----------
    min_distance : float
        Minimum distance (m) between any non-adjacent link pair.
        Negative values mean interpenetration.
    closest_pair : tuple[str, str]
        Names of the two links forming the closest pair.
    scale_factor : float
        1.0 when clear, linearly ramps to 0.0 as min_distance approaches
        danger_threshold. Use this to scale spring torques down gracefully.
    in_danger : bool
        True when min_distance < danger_threshold.
    in_collision : bool
        True when min_distance <= 0.
    """
    min_distance: float
    closest_pair: tuple[str, str]
    scale_factor: float
    in_danger: bool
    in_collision: bool


class URDFArmConfiguration:
    """
    ArmConfiguration backed by a URDF file and pinocchio kinematics.

    Parameters
    ----------
    urdf_path : str
        Absolute path to the robot's URDF file.
    srdf_path : str
        Absolute path to the robot's SRDF for collision checking
    q : np.ndarray, shape (n_dof,)
        Current joint positions.
    qdot : np.ndarray, shape (n_dof,), optional
        Current joint velocities. Defaults to zero.
    danger_threshold : float, optional
        Distance in metres at which scale_factor begins ramping toward 0.
        Default 0.05 (5 cm). Only used when collision model is loaded.
    locked_joint_names : list[str], optional
        Joint names to lock (see `_build_pinocchio_model`). Their mass still
        counts toward gravity compensation; they're just removed from the
        active DOF because they aren't reported on /joint_states.
    """

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str,
        q: np.ndarray,
        qdot: np.ndarray | None = None,
        danger_threshold: float = 0.05,
        locked_joint_names: list[str] | None = None,
    ) -> None:
        self._model = _build_pinocchio_model(urdf_path, locked_joint_names)
        self._data = self._model.createData()
        self._danger_threshold = danger_threshold

        # Collision model — loaded separately so kinematic-only use still works
        # if the URDF has no collision geometry or hpp-fcl is unavailable.
        self._collision_model: Optional[pin.GeometryModel] = None
        self._collision_data: Optional[pin.GeometryData] = None
        self._load_collision_model(urdf_path, srdf_path, locked_joint_names)

        # Geometry objects added by add_environment_object() (scene obstacles)
        # live in the same GeometryModel as the robot's own link geometry, so
        # they're indexed >= _n_robot_geoms and share the same distance-check
        # machinery in get_collision_status(). Recorded here, right after the
        # robot's own collision geometry is loaded and before anything else
        # can be added.
        self._n_robot_geoms = (
            len(self._collision_model.geometryObjects)
            if self._collision_model is not None else 0
        )
        self._environment_object_ids: set[str] = set()

        self.update(q, qdot)


    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_collision_model(
        self,
        urdf_path: str,
        srdf_path: str,
        locked_joint_names: list[str] | None = None,
    ) -> None:
        """
        Attempt to load the collision geometry from the URDF.
        Silently skips if the URDF has no collision meshes or if hpp-fcl
        support is not compiled into this pinocchio build.
        """
        try:
            full_model, collision_model, _ = pin.buildModelsFromUrdf(urdf_path)

            if locked_joint_names:
                joint_ids = [full_model.getJointId(name) for name in locked_joint_names]
                reference_configuration = pin.neutral(full_model)
                _, collision_model = pin.buildReducedModel(
                    full_model, collision_model, joint_ids, reference_configuration
                )

            if len(collision_model.geometryObjects) == 0:
                return  # URDF has no collision geometry — nothing to do

            # addAllCollisionPairs() only skips pairs on the *same* joint —
            # it does NOT skip adjacent parent/child links, which are always
            # touching at the joint and will otherwise register as permanent
            # false-positive self-collisions. An SRDF with <disable_collisions>
            # entries (e.g. MoveIt's Setup Assistant output) is required to
            # exclude those; without srdf_path set, expect false positives.
            collision_model.addAllCollisionPairs()
            if srdf_path and os.path.isfile(srdf_path):
                pin.removeCollisionPairs(self._model, collision_model, srdf_path)

            self._collision_model = collision_model
            self._collision_data = pin.GeometryData(collision_model)

        except Exception as e:
            # Degrade gracefully — kinematic control still works without this
            import warnings
            warnings.warn(
                f"Could not load collision model: {e}. "
                "Self-collision checking will be unavailable.",
                stacklevel=2,
            )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        srdf_path: str="",
        danger_threshold: float = 0.05,
        locked_joint_names: list[str] | None = None,
    ) -> "URDFArmConfiguration":
        """Construct with a zero joint configuration inferred from the model."""
        model = _build_pinocchio_model(urdf_path, locked_joint_names)
        return cls(
            urdf_path, srdf_path, np.zeros(model.nq),
            danger_threshold=danger_threshold, locked_joint_names=locked_joint_names,
        )
    @classmethod
    def from_xml_string(
        cls,
        urdf_xml: str,
        srdf_path: str = "",
        danger_threshold: float = 0.05,
        locked_joint_names: list[str] | None = None,
    ) -> "URDFArmConfiguration":
        """Construct from a URDF XML string (e.g. from /robot_description topic)."""
        import tempfile
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".urdf", delete=False
        ) as f:
            f.write(urdf_xml)
            tmp_path = f.name
        try:
            model = _build_pinocchio_model(tmp_path, locked_joint_names)
            return cls(
                tmp_path, srdf_path, np.zeros(model.nq),
                danger_threshold=danger_threshold, locked_joint_names=locked_joint_names,
            )
        finally:
            os.unlink(tmp_path)
    # ------------------------------------------------------------------
    # ArmConfiguration protocol
    # ------------------------------------------------------------------

    @property
    def joint_positions(self) -> np.ndarray:
        return self._q.copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self._qdot.copy()

    @property
    def n_dof(self) -> int:
        return self._model.nv

    @property
    def n_q(self) -> int:
        return self._model.nq

    @property
    def joint_names(self) -> list[str]:
        """Joint names in pinocchio's ordering (index 0 is 'universe', skip it)."""
        return [self._model.names[i] for i in range(1, self._model.njoints)]

    def get_link_transform(self, link_name: str) -> np.ndarray:
        frame_id = self._model.getFrameId(link_name)
        placement = self._data.oMf[frame_id]
        T = np.eye(4)
        T[:3, :3] = placement.rotation
        T[:3, 3] = placement.translation
        return T

    def get_jacobian(self, link_name: str, local_point: np.ndarray) -> np.ndarray:
        frame_id = self._model.getFrameId(link_name)
        J = pin.getFrameJacobian(
            self._model,
            self._data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).copy()

        T = self._data.oMf[frame_id]
        p_world = T.rotation @ local_point
        px, py, pz = p_world
        skew_p = np.array([
            [ 0,  -pz,  py],
            [ pz,   0, -px],
            [-py,  px,   0],
        ])
        J[:3, :] += skew_p @ J[3:, :]
        return J

    def get_gravity_torques(self) -> np.ndarray:
        pin.computeGeneralizedGravity(self._model, self._data, self._q)
        return self._data.g.copy()

    def get_joint_angle(self, joint_index: int) -> float:
        """
        Return the current angle (rad) of the joint at DOF index joint_index
        (0-based, matching joint_names/n_dof ordering). Inverse of the
        per-joint encoding in update_from_angles: unbounded/continuous
        joints are stored as (cos, sin) and decoded via atan2; others are
        stored directly.
        """
        joint = self._model.joints[joint_index + 1]
        if joint.nq == 2:
            return float(np.arctan2(self._q[joint.idx_q + 1], self._q[joint.idx_q]))
        return float(self._q[joint.idx_q])

    # ------------------------------------------------------------------
    # Collision API
    # ------------------------------------------------------------------

    @property
    def has_collision_model(self) -> bool:
        """True if collision geometry was successfully loaded."""
        return self._collision_model is not None

    def get_collision_status(self) -> Optional[CollisionStatus]:
        """
        Compute minimum self-collision distance across all non-adjacent link pairs.

        Returns None if no collision model is loaded (so callers can decide
        whether to treat that as safe or as an error).

        This is cheap enough to call every control cycle — all geometry
        placements are already up to date from the most recent update() call.
        """
        if self._collision_model is None or self._collision_data is None:
            return None

        n_pairs = len(self._collision_model.collisionPairs)
        if n_pairs == 0:
            return None

        pin.computeDistances(
            self._model,
            self._data,
            self._collision_model,
            self._collision_data,
            self._q,
        )

        min_dist = float("inf")
        closest_idx = 0
        for i in range(n_pairs):
            d = self._collision_data.distanceResults[i].min_distance
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Resolve link names for the closest pair
        pair = self._collision_model.collisionPairs[closest_idx]
        name_a = self._collision_model.geometryObjects[pair.first].name
        name_b = self._collision_model.geometryObjects[pair.second].name

        in_collision = min_dist <= 0.0
        in_danger = min_dist < self._danger_threshold

        # Linear ramp: 1.0 when clear, 0.0 at zero distance
        if in_collision:
            scale_factor = 0.0
        elif in_danger:
            scale_factor = min_dist / self._danger_threshold
        else:
            scale_factor = 1.0

        return CollisionStatus(
            min_distance=min_dist,
            closest_pair=(name_a, name_b),
            scale_factor=scale_factor,
            in_danger=in_danger,
            in_collision=in_collision,
        )

    # ------------------------------------------------------------------
    # Environment (scene) collision objects
    # ------------------------------------------------------------------
    #
    # Obstacles are added as extra, world-fixed GeometryObjects in the same
    # GeometryModel used for self-collision, with a collision pair against
    # every robot-link geometry (never against each other). This means
    # get_collision_status() above needs no changes at all -- it already
    # scans every pair and reports whichever is closest, so scene objects
    # get the same in_collision/in_danger/scale_factor treatment as
    # self-collision for free.

    def is_environment_object(self, name: str) -> bool:
        """True if `name` is a scene object added via add_environment_object
        (as opposed to one of the robot's own link geometries)."""
        return name in self._environment_object_ids

    def add_environment_object(
        self,
        object_id: str,
        shape: str,
        dimensions: list[float],
        pose: pin.SE3,
    ) -> None:
        """
        Add (or replace) a static collision object in the scene.

        Parameters
        ----------
        object_id : str
            Unique name for this object. Adding again with the same id
            replaces it (matches moveit_msgs/CollisionObject's ADD
            semantics).
        shape : str
            "box" or "cylinder".
        dimensions : list[float]
            "box": [x, y, z] full side lengths (m).
            "cylinder": [radius, height] -- both full-size, matching
            hppfcl's own constructor convention.
        pose : pin.SE3
            World-frame placement of the object.
        """
        if self._collision_model is None or self._collision_data is None:
            raise RuntimeError(
                "No collision model loaded -- environment collision "
                "checking requires the robot's own collision geometry to "
                "already be available."
            )

        # This pinocchio build's GeometryObject expects coal::CollisionGeometry
        # (the current name for what used to be hpp-fcl) -- the separately
        # installed `hppfcl` package looks API-compatible but is a different,
        # incompatible boost::python type registration and fails at the
        # GeometryObject constructor. Confirmed empirically: `coal` is what's
        # actually wired into this pinocchio build.
        import coal

        if shape == "box":
            geom = coal.Box(*dimensions)
        elif shape == "cylinder":
            radius, height = dimensions
            geom = coal.Cylinder(radius, height)
        else:
            raise ValueError(f"Unknown environment object shape: {shape!r}")

        # Idempotent replace, mirroring CollisionObject.ADD's "if the object
        # previously existed, it is replaced".
        self.remove_environment_object(object_id)

        geometry_object = pin.GeometryObject(object_id, 0, pose, geom)
        gid = self._collision_model.addGeometryObject(geometry_object)
        for i in range(self._n_robot_geoms):
            self._collision_model.addCollisionPair(pin.CollisionPair(i, gid))

        # Pair topology changed -- GeometryData must be rebuilt to match.
        self._collision_data = pin.GeometryData(self._collision_model)
        self._environment_object_ids.add(object_id)

    def remove_environment_object(self, object_id: str) -> bool:
        """Remove a previously-added environment object. Returns False if
        no object with that id exists."""
        if object_id not in self._environment_object_ids:
            return False
        # Also drops this object's collision pairs.
        self._collision_model.removeGeometryObject(object_id)
        self._environment_object_ids.discard(object_id)
        self._collision_data = pin.GeometryData(self._collision_model)
        return True

    def move_environment_object(self, object_id: str, pose: pin.SE3) -> bool:
        """
        Update the world-frame placement of an existing environment object
        without touching collision-pair topology. Cheap enough to call every
        control cycle for a continuously-tracked object (e.g. a person).
        Returns False if no object with that id exists.
        """
        if object_id not in self._environment_object_ids:
            return False
        gid = self._collision_model.getGeometryId(object_id)
        self._collision_model.geometryObjects[gid].placement = pose
        return True

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_from_angles(
        self, angles: np.ndarray, qdot: np.ndarray | None = None
    ) -> None:
        """Update from plain joint angles, handling pinocchio's nq encoding."""
        q = pin.neutral(self._model)
        for i, angle in enumerate(angles):
            joint = self._model.joints[i + 1]
            if joint.nq == 2:  # unbounded revolute: stored as (cos, sin)
                q[joint.idx_q]     = np.cos(angle)
                q[joint.idx_q + 1] = np.sin(angle)
            else:
                q[joint.idx_q] = angle
        self.update(q, qdot)

    def update(self, q: np.ndarray, qdot: np.ndarray | None = None) -> None:
        """Update joint state and recompute forward kinematics."""
        self._q = np.asarray(q, dtype=float)
        self._qdot = (
            np.asarray(qdot, dtype=float)
            if qdot is not None
            else np.zeros(self._model.nv)
        )
        pin.forwardKinematics(self._model, self._data, self._q)
        pin.updateFramePlacements(self._model, self._data)
        pin.computeJointJacobians(self._model, self._data, self._q)

        # Keep collision geometry in sync — this is the only extra cost vs before.
        # updateGeometryPlacements just propagates the already-computed FK
        # transforms into the geometry model; it does not redo FK.
        if self._collision_model is not None:
            pin.updateGeometryPlacements(
                self._model,
                self._data,
                self._collision_model,
                self._collision_data,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def link_names(self) -> list[str]:
        return [self._model.frames[i].name for i in range(len(self._model.frames))]

    def validate_link_name(self, link_name: str) -> None:
        available = [self._model.frames[i].name for i in range(len(self._model.frames))]
        if link_name not in available:
            raise ValueError(
                f"Link '{link_name}' not found in URDF. "
                f"Available frames: {sorted(available)}"
            )

    def validate_joint_name(self, joint_name: str) -> None:
        if joint_name not in self.joint_names:
            raise ValueError(
                f"Joint '{joint_name}' not found in URDF. "
                f"Available joints: {sorted(self.joint_names)}"
            )
