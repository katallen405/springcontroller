"""
Generalized-momentum observer for external joint torque, plus the
amplification law used to lower the arm's perceived impedance in
gravity-comp-only mode.

Dynamics convention (matches pinocchio and the torque command sent to the
Gen3):  M(q) q̈ + C(q, q̇) q̇ + g(q) = τ + τ_ext

With p = M q̇ and Ṁ = C + Cᵀ:   ṗ = τ + τ_ext + Cᵀ q̇ − g

Observer (De Luca et al.):
    r = K_o ( p − p₀ − ∫ (τ + Cᵀ q̇ − g + r) dt )
r is a first-order low-pass (bandwidth K_o) estimate of τ_ext that needs no
joint acceleration.

Model error (wrong masses, torque-sensor offsets) shows up in r as a
constant "external" torque. While the arm is at rest that constant is
absorbed into a slowly-adapting, clamped bias b̂, and the reported estimate
is r − b̂. Without this, amplification would multiply gravity-model drift.
"""

from __future__ import annotations

import numpy as np


class ExternalTorqueObserver:
    def __init__(
        self,
        n_dof: int,
        gain: float = 20.0,
        bias_max: np.ndarray | float = 1.0,
        rest_velocity: float = 0.02,
        rest_time: float = 1.0,
        bias_time_constant: float = 2.0,
    ) -> None:
        if gain <= 0:
            raise ValueError(f"observer gain must be > 0, got {gain}")
        self._n = n_dof
        self.gain = float(gain)
        self.bias_max = np.broadcast_to(np.asarray(bias_max, dtype=float), (n_dof,)).copy()
        self.rest_velocity = float(rest_velocity)
        self.rest_time = float(rest_time)
        self.bias_time_constant = float(bias_time_constant)

        self._bias = np.zeros(n_dof)
        self._initialized = False
        self._p0 = np.zeros(n_dof)
        self._integral = np.zeros(n_dof)
        self._r = np.zeros(n_dof)
        self._rest_elapsed = 0.0

    @property
    def residual(self) -> np.ndarray:
        """Raw observer output r (includes model bias)."""
        return self._r.copy()

    @property
    def bias(self) -> np.ndarray:
        return self._bias.copy()

    @property
    def estimate(self) -> np.ndarray:
        """Bias-corrected external torque estimate r − b̂."""
        return self._r - self._bias

    def reset(self) -> None:
        """Restart integration from the next step's momentum. Keeps the learned bias."""
        self._initialized = False
        self._integral[:] = 0.0
        self._r[:] = 0.0
        self._rest_elapsed = 0.0

    def step(
        self,
        M: np.ndarray,
        C: np.ndarray,
        g: np.ndarray,
        qdot: np.ndarray,
        tau_measured: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Advance one sample and return the bias-corrected estimate."""
        qdot = np.asarray(qdot, dtype=float)
        p = M @ qdot
        if not self._initialized:
            self._p0 = p
            self._integral[:] = 0.0
            self._r[:] = 0.0
            self._initialized = True
            return self.estimate

        beta = g - C.T @ qdot
        self._integral += (np.asarray(tau_measured, dtype=float) - beta + self._r) * dt
        self._r = self.gain * (p - self._p0 - self._integral)

        at_rest = bool(np.all(np.abs(qdot) < self.rest_velocity))
        self._rest_elapsed = self._rest_elapsed + dt if at_rest else 0.0
        if at_rest and self._rest_elapsed >= self.rest_time and self.bias_time_constant > 0:
            alpha = min(1.0, dt / self.bias_time_constant)
            self._bias += alpha * (self._r - self._bias)
            np.clip(self._bias, -self.bias_max, self.bias_max, out=self._bias)

        return self.estimate


def amplification_torque(
    tau_ext: np.ndarray,
    alpha: float,
    deadband: np.ndarray | float,
    max_torque: np.ndarray | float,
) -> np.ndarray:
    """α · deadband(τ_ext), clipped per joint. Deadband is subtracted, not gated, so it's continuous."""
    tau_ext = np.asarray(tau_ext, dtype=float)
    d = np.asarray(deadband, dtype=float)
    shrunk = np.sign(tau_ext) * np.maximum(np.abs(tau_ext) - d, 0.0)
    lim = np.asarray(max_torque, dtype=float)
    return np.clip(alpha * shrunk, -lim, lim)
