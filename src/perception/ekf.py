import numpy as np

from src.common.types import PerceptionState
from src.perception.base import BasePerception


class AnkleEKFPerception(BasePerception):
    """
    Extended Kalman Filter for right ankle state estimation.

    State:
        x = [ankle_angle, ankle_angular_velocity]

    Measurement:
        noisy ankle angle

    Input:
        ankle actuator torque
    """

    def __init__(
        self,
        dt: float,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
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
        angle_measurement = float(
            observation["ankle_angle_measurement"]
        )

        torque = float(
            observation.get("ankle_torque", 0.0)
        )

        if not self.initialized:
            self.reset(
                angle=angle_measurement,
                velocity=0.0,
            )
        else:
            self.predict(torque)
            self.update(angle_measurement)

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