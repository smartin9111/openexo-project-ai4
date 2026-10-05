from typing import Optional

import numpy as np

from src.common.types import PerceptionState
from src.perception.base import BasePerception


class AnkleEKFPerception(BasePerception):
    """
    Extended Kalman Filter for ankle state estimation (one instance per leg).

    State:
        x = [ankle_angle, ankle_angular_velocity]

    Measurement:
        noisy ankle angle (None or NaN = sensor dropout, the filter only predicts)

    Input:
        ankle actuator torque

    Process noise:
        accel_noise set  -> white-noise-acceleration Q (recommended)
        accel_noise None -> legacy diagonal Q = eye * process_noise
    """

    def __init__(
        self,
        dt: float,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
        accel_noise: Optional[float] = None,
        gravity_gain: float = 5.0,
        damping: float = 0.5,
        torque_gain: float = 1.0,
    ):
        self.dt = dt

        # Effective model parameters.
        # These are estimator parameters, not claimed physical constants.
        self.gravity_gain = gravity_gain
        self.damping = damping
        self.torque_gain = torque_gain

        self.x = np.zeros(2)

        self.P = np.eye(2) * 0.1
        if accel_noise is not None:
            # White-noise-acceleration process noise: the velocity is driven by an
            # unknown acceleration, so angle and velocity noise are correlated.
            # A diagonal Q (the legacy path below) leaves the velocity almost
            # unobservable from angle-only measurements.
            self.Q = accel_noise * np.array([
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt],
            ])
        else:
            self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

        self.initialized = False

    def reset(self, angle: float = 0.0, velocity: float = 0.0):
        self.x = np.array([angle, velocity], dtype=float)
        self.P = np.eye(2) * 0.1
        self.initialized = True

    def _process_model(self, x, torque):
        theta, omega = x

        theta_next = theta + self.dt * omega

        omega_next = omega + self.dt * (
            -self.gravity_gain * np.sin(theta)
            - self.damping * omega
            + self.torque_gain * torque
        )

        return np.array([
            theta_next,
            omega_next,
        ])

    def _process_jacobian(self, x):
        theta, _ = x

        return np.array([
            [1.0, self.dt],
            [
                -self.dt * self.gravity_gain * np.cos(theta),
                1.0 - self.dt * self.damping,
            ],
        ])

    def predict(self, torque: float):
        F = self._process_jacobian(self.x)

        self.x = self._process_model(
            self.x,
            torque,
        )

        self.P = F @ self.P @ F.T + self.Q

    def update(self, angle_measurement: float):
        # Measurement model:
        # z = theta
        H = np.array([[1.0, 0.0]])

        z = np.array([angle_measurement])

        innovation = z - H @ self.x

        S = H @ self.P @ H.T + self.R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ innovation).ravel()

        # Joseph-form covariance update
        I = np.eye(2)
        A = I - K @ H

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

    def process(self, observation) -> PerceptionState:
        # A missing measurement (None or NaN) is a sensor dropout: the filter
        # then only predicts and keeps its uncertainty growing.
        raw = observation.get("ankle_angle_measurement")
        valid = raw is not None and bool(np.isfinite(raw))

        torque = float(observation.get("ankle_torque", 0.0))

        if not self.initialized:
            self.reset(
                angle=float(raw) if valid else 0.0,
                velocity=0.0,
            )
        else:
            self.predict(torque)
            if valid:
                self.update(float(raw))

        return PerceptionState(
            joint_positions=np.array([self.x[0]]),
            joint_velocities=np.array([self.x[1]]),
            muscle_activations=observation.get(
                "muscle_activations"
            ),
            foot_contacts=observation.get(
                "foot_contacts"
            ),
            gait_phase=observation.get(
                "gait_phase"
            ),
        )