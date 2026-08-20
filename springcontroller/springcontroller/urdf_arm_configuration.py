"""
urdf_arm_configuration.py
ArmConfiguration implementation backed by a URDF + pinocchio.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import warnings
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
    # getJointId() returns the same sentinel index for any name it doesn't
    # recognize instead of raising, so two-or-more missing names silently
    # collide into "duplicate index" -- filter to names the model actually
    # has first, so a URDF source without some locked joint (e.g. a
    # gripper-less /robot_description) degrades to "nothing to lock" rather
    # than crashing with a confusing pinocchio error.
    present = [name for name in locked_joint_names if full_model.existJointName(name)]
    missing = [name for name in locked_joint_names if name not in present]
    if missing:
        warnings.warn(
            f"locked_joint_names not found in URDF, skipping: {missing} "
            "(their mass/inertia is only counted if present in this model)"
        )
    if not present:
        return full_model
    joint_ids = [full_model.getJointId(name) for name in present]
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

            present_locked_names = [
                name for name in (locked_joint_names or [])
                if full_model.existJointName(name)
            ]
            if present_locked_names:
                joint_ids = [full_model.getJointId(name) for name in present_locked_names]
                reference_configuration = pin.neutral(full_model)
                _, collision_model = pin.buildReducedModel(
                    full_model, collision_model, joint_ids, reference_configuration
                )

            if len(collision_model.geometryObjects) == 0:
                return  # URDF has no collision geometry — nothing to do

            self._simplify_collision_geometry(collision_model)

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

    def _simplify_collision_geometry(self, collision_model: "pin.GeometryModel") -> None:
        """
        Replace each loaded triangle mesh (BVHModelOBBRSS -- often several
        thousand triangles per link) with a conservative axis-aligned
        bounding box in the mesh's own local frame, in place.

        Offline-profiled 2026-08-07/08-12: mesh-mesh distance queries via
        pin.computeDistances cost ~126ms/call for this arm's ~91 collision
        pairs -- capping get_collision_status() (called every control
        cycle) to ~8Hz instead of the intended ~100Hz, later confirmed to
        be the dominant cause of a full day's worth of live command-
        staleness/fault incidents. Bounding boxes over the same pairs cost
        ~0.01ms/call, a ~10,000x speedup -- negligible against a 10ms
        cycle budget.

        The box always encloses the true mesh (it's exactly the mesh's own
        AABB), so reported distances are conservative: equal to or smaller
        than the true mesh-to-mesh distance, never larger. That's the safe
        direction for a collision clamp -- it can trigger early on an
        elongated/diagonal link's loose bounding box, but it can never
        report "clear" when the real mesh isn't.

        Skips anything that isn't a BVH mesh (e.g. environment objects
        added later via add_environment_object, already simple
        primitives) -- nothing to simplify there.
        """
        import coal

        for go in collision_model.geometryObjects:
            mesh = go.geometry
            if mesh.getObjectType() != coal.OT_BVH:
                continue
            mesh.computeLocalAABB()
            aabb = mesh.aabb_local
            box = coal.Box(aabb.width(), aabb.height(), aabb.depth())
            center = np.asarray(aabb.center(), dtype=float).reshape(3)
            go.geometry = box
            go.placement = go.placement * pin.SE3(np.eye(3), center)

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

    def _angles_to_q(self, angles: np.ndarray) -> np.ndarray:
        """Plain per-joint angles -> pinocchio's nq encoding (see update_from_angles)."""
        q = pin.neutral(self._model)
        for i, angle in enumerate(angles):
            joint = self._model.joints[i + 1]
            if joint.nq == 2:  # unbounded revolute: stored as (cos, sin)
                q[joint.idx_q]     = np.cos(angle)
                q[joint.idx_q + 1] = np.sin(angle)
            else:
                q[joint.idx_q] = angle
        return q

    def _collision_status_from(
        self, q: np.ndarray, data: pin.Data, collision_data: pin.GeometryData,
    ) -> Optional[CollisionStatus]:
        """
        Shared distance/scale-factor computation behind get_collision_status()
        and get_collision_status_for_angles() -- takes explicit q/data/
        collision_data so the latter can compute into scratch buffers without
        disturbing the live tracked pose the real-time control loop reads.
        """
        if self._collision_model is None:
            return None

        n_pairs = len(self._collision_model.collisionPairs)
        if n_pairs == 0:
            return None

        pin.computeDistances(
            self._model, data, self._collision_model, collision_data, q,
        )

        min_dist = float("inf")
        closest_idx = 0
        for i in range(n_pairs):
            d = collision_data.distanceResults[i].min_distance
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

    def get_collision_status(self) -> Optional[CollisionStatus]:
        """
        Compute minimum self-collision distance across all non-adjacent link pairs,
        at the current tracked pose (last update()/update_from_angles() call).

        Returns None if no collision model is loaded (so callers can decide
        whether to treat that as safe or as an error).

        This is cheap enough to call every control cycle — all geometry
        placements are already up to date from the most recent update() call.
        """
        if self._collision_data is None:
            return None
        return self._collision_status_from(self._q, self._data, self._collision_data)

    def get_collision_status_for_angles(self, angles: np.ndarray) -> Optional[CollisionStatus]:
        """
        Compute collision status at a hypothetical joint configuration,
        without touching the tracked pose get_collision_status() (or the
        real-time control loop) reads. For pre-flight checks -- e.g. "would
        moving to this target put the arm in collision" -- against the
        currently-loaded model, scene objects included.

        Builds scratch pin.Data/GeometryData rather than reusing self._data/
        self._collision_data, so this is safe to call at any time regardless
        of what the tracked pose currently is.
        """
        if self._collision_model is None:
            return None
        q = self._angles_to_q(np.asarray(angles, dtype=float))
        data = pin.Data(self._model)
        collision_data = pin.GeometryData(self._collision_model)
        pin.forwardKinematics(self._model, data, q)
        pin.updateGeometryPlacements(self._model, data, self._collision_model, collision_data)
        return self._collision_status_from(q, data, collision_data)

    def get_repulsion_torques(
        self, caution_threshold: float, max_force_n: float,
    ) -> np.ndarray:
        """
        Joint torques from a low-strength repulsion field around scene
        (environment) collision objects.

        Distinct from get_collision_status()'s danger_threshold clamp: that
        one only ever *removes* spring torque as the arm nears an obstacle;
        this one *adds* an outward push, ramping from 0 at caution_threshold
        to max_force_n at self._danger_threshold, then holding at
        max_force_n for anything closer (including interpenetration, where
        min_distance goes negative) -- unlike the clamp, this is meant to
        keep helping exactly when the arm is already deep in the danger
        zone, not cut out there.

        Only pairs where one side is an environment object contribute
        (is_environment_object()) -- self-collision pairs are out of scope
        for this field. Reuses self._collision_data.distanceResults from
        whatever pin.computeDistances call most recently populated it (i.e.
        the same query get_collision_status() just ran) rather than
        recomputing distances itself, so call get_collision_status() (or
        update()) first in the same cycle.

        Returns a zero vector if no collision model is loaded, either
        threshold is non-positive, or nothing is within caution_threshold.
        """
        torques = np.zeros(self._model.nv)
        if self._collision_model is None or self._collision_data is None:
            return torques
        if caution_threshold <= 0.0 or max_force_n <= 0.0:
            return torques

        # Guards against a misconfigured caution_threshold <= danger_threshold
        # (no room for a ramp) rather than dividing by ~0 in that case.
        span = max(caution_threshold - self._danger_threshold, 1e-9)

        n_pairs = len(self._collision_model.collisionPairs)
        for i in range(n_pairs):
            pair = self._collision_model.collisionPairs[i]
            geom_a = self._collision_model.geometryObjects[pair.first]
            geom_b = self._collision_model.geometryObjects[pair.second]
            a_is_env = geom_a.name in self._environment_object_ids
            b_is_env = geom_b.name in self._environment_object_ids
            if a_is_env == b_is_env:
                # Neither is a scene object (self-collision, out of scope
                # here) -- add_environment_object() never pairs two
                # environment objects together, so both-True can't happen.
                continue

            result = self._collision_data.distanceResults[i]
            d = result.min_distance
            if d >= caution_threshold:
                continue

            if b_is_env:
                robot_geom, robot_point = geom_a, result.getNearestPoint1()
                obstacle_point = result.getNearestPoint2()
            else:
                robot_geom, robot_point = geom_b, result.getNearestPoint2()
                obstacle_point = result.getNearestPoint1()

            # A geometry object's own .name is "{link}_{index}" (a link can
            # have multiple <collision> elements), not the URDF/frame name
            # get_jacobian()/get_link_transform() expect -- resolve via
            # parentFrame instead, same as add_environment_object's
            # exclude_links matching above.
            robot_name = self._model.frames[robot_geom.parentFrame].name

            robot_point = np.asarray(robot_point, dtype=float)
            obstacle_point = np.asarray(obstacle_point, dtype=float)
            direction = robot_point - obstacle_point
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                # Deep penetration with coincident witness points -- no
                # reliable push direction. Skip rather than divide by ~0;
                # get_collision_status()'s hard clamp is already holding
                # spring torque at zero in this regime.
                continue
            direction = direction / norm

            if d < self._danger_threshold:
                magnitude = max_force_n
            else:
                t = (caution_threshold - d) / span
                magnitude = max_force_n * (t * t)

            force_world = magnitude * direction

            T = self.get_link_transform(robot_name)
            local_point = T[:3, :3].T @ (robot_point - T[:3, 3])
            J = self.get_jacobian(robot_name, local_point)
            torques += J[:3, :].T @ force_world

        return torques

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
        exclude_links: list[str] | None = None,
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
        exclude_links : list[str], optional
            URDF link names to skip when creating collision pairs against
            this object -- the environment-object equivalent of an SRDF
            <disable_collisions> entry. Use this for permanent, expected
            "contact" like the base link sitting on the table it's mounted
            to: without it, that link would register a small permanent
            near-miss (or worse, an overlap) against every object placed at
            the mounting surface's height, which is a modeling artifact, not
            a real hazard, and would blanket-scale torques for no reason.
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
        excluded = set(exclude_links or [])
        for i in range(self._n_robot_geoms):
            if excluded:
                link_name = self._model.frames[
                    self._collision_model.geometryObjects[i].parentFrame
                ].name
                if link_name in excluded:
                    continue
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
        q = self._angles_to_q(np.asarray(angles, dtype=float))
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
