from typing import Optional

import numpy as np

from src.common.types import PerceptionState
from src.perception.base import BasePerception


class GaitPhaseEKFPerception(BasePerception):
    """
    Gait-phase EKF for one ankle (one instance per leg).

    Instead of a generic angle/velocity model, the state is the position in the
    gait cycle and the stride frequency, and the measured ankle angle is predicted
    from an average ankle trajectory over one gait cycle (the template).

    State:
        x = [phase, stride_frequency]     phase in [0, 1) cycles, frequency in Hz

    Measurement:
        noisy ankle angle, z = template(phase)
        (None or NaN = sensor dropout, the filter only predicts)

    Output:
        joint_positions  = template(phase)               [rad]
        joint_velocities = template'(phase) * frequency  [rad/s]
        gait_phase       = phase in [0, 1)  (None until the filter is initialised)

    Initialisation and recovery:
        The phase is found by fitting the template to the last `init_window`
        measurements. If the squared innovation stays large (the filter lost the
        gait), the same fit is repeated.

    Notes:
        * `dt` is the time between process() calls. The defaults assume a fast
          loop (500 Hz). For a slow control loop (e.g. 30 Hz) use a much smaller
          `init_window` (about 1 s worth of samples).
        * The template should be the mean ankle angle over one gait cycle, with
          phase 0 at the start of the stride, e.g. from MyoAssist's
          rl_train/reference_data/segmented.npz (q_ankle_angle_r / _l).
    """

    def __init__(
        self,
        dt: float,
        template,
        measurement_noise: float = 0.03**2,
        phase_rate_noise: float = 1e-5,
        n_harmonics: int = 12,
        init_window: int = 100,
        nis_threshold: float = 0.01,
        freq_min: float = 0.5,
        freq_max: float = 1.5,
        init_freqs=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
    ):
        self.dt = dt
        self.R = measurement_noise
        self.qw = phase_rate_noise
        self.init_window = int(init_window)
        self.nis_threshold = nis_threshold
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.init_freqs = tuple(init_freqs)

        # Periodic Fourier model of the template (smooth, with analytic derivative)
        template = np.asarray(template, dtype=float).ravel()
        n = template.size
        phases = np.arange(n) / n
        self.n_harmonics = int(n_harmonics)
        cols = [np.ones(n)]
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.cos(2.0 * np.pi * k * phases))
            cols.append(np.sin(2.0 * np.pi * k * phases))
        self.coef = np.linalg.lstsq(np.stack(cols, axis=1), template, rcond=None)[0]

        self.reset()

    # ------------------------------------------------------------- template
    def template_angle(self, phase):
        p = np.asarray(phase, dtype=float)
        out = self.coef[0] * np.ones_like(p)
        for k in range(1, self.n_harmonics + 1):
            out = (
                out
                + self.coef[2 * k - 1] * np.cos(2.0 * np.pi * k * p)
                + self.coef[2 * k] * np.sin(2.0 * np.pi * k * p)
            )
        return out

    def template_slope(self, phase):
        """Derivative of the template in rad per cycle."""
        p = np.asarray(phase, dtype=float)
        out = np.zeros_like(p)
        for k in range(1, self.n_harmonics + 1):
            out = out + 2.0 * np.pi * k * (
                -self.coef[2 * k - 1] * np.sin(2.0 * np.pi * k * p)
                + self.coef[2 * k] * np.cos(2.0 * np.pi * k * p)
            )
        return out

    # ---------------------------------------------------------------- state
    def reset(self):
        self.x = np.array([0.0, 0.9])
        self.P = np.diag([0.05**2, 0.1**2])
        self.initialized = False
        self._buffer = []
        self._nis = 0.0
        self._since_init = 0

    @property
    def phase(self) -> Optional[float]:
        return float(self.x[0]) if self.initialized else None

    def _initialize_from_buffer(self) -> bool:
        z = np.array(self._buffer[-self.init_window:], dtype=float)
        ok = np.isfinite(z)
        if ok.sum() < max(10, self.init_window // 4):
            return False

        t = np.arange(z.size) * self.dt
        best = None
        for f in self.init_freqs:
            for p0 in np.arange(0.0, 1.0, 0.01):
                err = np.mean((self.template_angle(p0 + f * t)[ok] - z[ok]) ** 2)
                if best is None or err < best[0]:
                    best = (err, p0, f)

        _, p0, f = best
        self.x = np.array([(p0 + f * z.size * self.dt) % 1.0, f])
        self.P = np.diag([0.05**2, 0.1**2])
        self.initialized = True
        self._nis = 0.0
        self._since_init = 0
        return True

    # ------------------------------------------------------------------ EKF
    def predict(self):
        dt = self.dt
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.x[0] %= 1.0

        Q = self.qw * np.array([
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ])
        self.P = F @ self.P @ F.T + Q

    def update(self, angle_measurement: float):
        H = np.array([[float(self.template_slope(self.x[0])), 0.0]])
        S = (H @ self.P @ H.T)[0, 0] + self.R
        K = (self.P @ H.T)[:, 0] / S

        innovation = angle_measurement - float(self.template_angle(self.x[0]))

        self.x = self.x + K * innovation
        self.x[0] %= 1.0
        self.x[1] = min(max(self.x[1], self.freq_min), self.freq_max)

        # Joseph-form covariance update
        A = np.eye(2) - np.outer(K, H[0])
        self.P = A @ self.P @ A.T + self.R * np.outer(K, K)

        # Slow average of the squared innovation, used to detect a lost gait
        self._nis = 0.998 * self._nis + 0.002 * innovation**2

    def process(self, observation) -> PerceptionState:
        raw = observation.get("ankle_angle_measurement")
        valid = raw is not None and bool(np.isfinite(raw))

        self._buffer.append(float(raw) if valid else np.nan)
        self._buffer = self._buffer[-self.init_window:]

        if not self.initialized:
            if len(self._buffer) >= self.init_window:
                self._initialize_from_buffer()
        else:
            self.predict()
            if valid:
                self.update(float(raw))
            self._since_init += 1
            if self._since_init > 500 and self._nis > self.nis_threshold:
                self._initialize_from_buffer()

        if self.initialized:
            angle = float(self.template_angle(self.x[0]))
            velocity = float(self.template_slope(self.x[0]) * self.x[1])
            gait_phase = float(self.x[0])
        else:
            # Warm-up: pass the raw measurement through, velocity unknown
            angle = float(raw) if valid else 0.0
            velocity = 0.0
            gait_phase = None

        return PerceptionState(
            joint_positions=np.array([angle]),
            joint_velocities=np.array([velocity]),
            muscle_activations=observation.get("muscle_activations"),
            foot_contacts=observation.get("foot_contacts"),
            gait_phase=gait_phase,
        )