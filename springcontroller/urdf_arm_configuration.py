"""
urdf_arm_configuration.py
ArmConfiguration implementation backed by a URDF + pinocchio.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


class URDFArmConfiguration:
    """
    ArmConfiguration backed by a URDF file and pinocchio kinematics.

    Parameters
    ----------
    urdf_path : str
        Absolute path to the robot's URDF file.
    q : np.ndarray, shape (n_dof,)
        Current joint positions.
    qdot : np.ndarray, shape (n_dof,), optional
        Current joint velocities. Defaults to zero.
    """

    def __init__(
        self,
        urdf_path: str,
        q: np.ndarray,
        qdot: np.ndarray | None = None,
    ) -> None:
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self.update(q, qdot)

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
    
    @classmethod
    def from_urdf(cls, urdf_path: str) -> "URDFArmConfiguration":
        """Construct with a zero joint configuration inferred from the model."""
        model = pin.buildModelFromUrdf(urdf_path)
        return cls(urdf_path, np.zeros(model.nq))

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
        ).copy()   # (6, n_dof)

        # Shift Jacobian from frame origin to local_point
        R = self._data.oMf[frame_id].rotation
        p_offset = R @ np.asarray(local_point, dtype=float)
        for i in range(self._model.nv):
            J[:3, i] += np.cross(J[3:, i], -p_offset)

#        print(f"[get_jacobian] {link_name} Jv=\n{J[:3, :]}")
        return J
    

    def update_from_angles(self, angles: np.ndarray, qdot: np.ndarray | None = None) -> None:
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def model(self):
        return self._model

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


    @property
    def link_names(self) -> list[str]:
        """All frame names available in the URDF."""
        return [self._model.frames[i].name for i in range(len(self._model.frames))]

    def validate_link_name(self, link_name: str) -> None:
        """Raise ValueError if link_name is not a known frame in the model."""
        available = [self._model.frames[i].name for i in range(len(self._model.frames))]
        if link_name not in available:
            raise ValueError(
                f"Link '{link_name}' not found in URDF. "
                f"Available frames: {sorted(available)}"
            )
