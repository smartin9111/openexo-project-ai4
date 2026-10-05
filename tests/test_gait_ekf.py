"""
Smoke test of the gait-phase EKF on a synthetic walker (the template itself plus
noise and sensor dropout). This only checks that the code works; the realistic
evaluation is tests/eval_gait_ekf_reference.py.
"""
import numpy as np

from src.perception.gait_ekf import GaitPhaseEKFPerception


DT = 0.002
STD = 0.03
FREQ = 0.9  # Hz
SECONDS = 12

template = np.load("data/ankle_template_r.npy")
rng = np.random.default_rng(0)

ekf = GaitPhaseEKFPerception(dt=DT, template=template, measurement_noise=STD**2)

# True signal: the same Fourier template, advancing at FREQ, starting at phase 0.3
t = np.arange(int(SECONDS / DT)) * DT
true_phase = (0.3 + FREQ * t) % 1.0
true_angle = ekf.template_angle(true_phase)
true_vel = ekf.template_slope(true_phase) * FREQ

z = true_angle + rng.normal(0.0, STD, t.size)
z[3000:3025] = np.nan  # 50 ms dropout
z[4500:4600] = np.nan  # 200 ms dropout

est_angle, est_vel, est_phase = [], [], []
for zi in z:
    s = ekf.process({"ankle_angle_measurement": zi})
    est_angle.append(s.joint_positions[0])
    est_vel.append(s.joint_velocities[0])
    est_phase.append(np.nan if s.gait_phase is None else s.gait_phase)

est_angle, est_vel, est_phase = map(np.asarray, (est_angle, est_vel, est_phase))
w = 500  # skip start-up
phase_err = np.abs(((est_phase[w:] - true_phase[w:]) + 0.5) % 1.0 - 0.5)

print("angle RMSE   :", np.sqrt(np.mean((est_angle[w:] - true_angle[w:]) ** 2)))
print("velocity RMSE:", np.sqrt(np.mean((est_vel[w:] - true_vel[w:]) ** 2)))
print("phase error  : mean %.4f cycles, max %.4f cycles" % (phase_err.mean(), phase_err.max()))

assert np.all(np.isfinite(est_angle)) and np.all(np.isfinite(est_vel))
assert phase_err.mean() < 0.02
print("Gait-phase EKF smoke test passed!")