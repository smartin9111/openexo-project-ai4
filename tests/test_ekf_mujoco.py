import numpy as np

from src.environment.openexo_env import OpenExoEnvironment
from src.perception.ekf import AnkleEKFPerception


STEPS = 500
MEASUREMENT_STD = 0.03
EXO_PEAK = -0.01  # exo command peak, allowed range: [-1, 0]
SIDES = ("r", "l")

rng = np.random.default_rng(42)

env = OpenExoEnvironment()
dt = env.model.opt.timestep

# One EKF per leg
ekf = {
    side: AnkleEKFPerception(
        dt=dt,
        accel_noise=1e3,
        measurement_noise=MEASUREMENT_STD**2,
    )
    for side in SIDES
}

log = {
    side: {"true_angle": [], "true_vel": [], "meas": [], "est_angle": [], "est_vel": []}
    for side in SIDES
}

print("MuJoCo timestep:", dt)
print("Exo actuator ids:", env.exo_ids)
print("Ankle qpos:", env.ankle_qpos, "| qvel:", env.ankle_qvel)

for k in range(STEPS):
    t = k * dt

    # Exo command in [-1, 0], same on both legs
    exo_command = EXO_PEAK * 0.5 * (1.0 - np.cos(2.0 * np.pi * 0.5 * t))

    gt = env.step(exo_command)

    for side in SIDES:
        true_angle = gt[f"ankle_angle_{side}"]
        true_velocity = gt[f"ankle_velocity_{side}"]

        # Simulated noisy encoder measurement
        measured_angle = true_angle + rng.normal(0.0, MEASUREMENT_STD)

        observation = {
            "ankle_angle_measurement": measured_angle,
            # Exo torque normalized to [-1, 0] (actuator gain = 100 Nm)
            "ankle_torque": gt[f"ankle_torque_{side}"] / 100.0,
        }

        state = ekf[side].process(observation)

        log[side]["true_angle"].append(true_angle)
        log[side]["true_vel"].append(true_velocity)
        log[side]["meas"].append(measured_angle)
        log[side]["est_angle"].append(float(state.joint_positions[0]))
        log[side]["est_vel"].append(float(state.joint_velocities[0]))

warmup = 50

print("\n=== EKF + MUJOCO RESULTS ===")
for side in SIDES:
    d = {k: np.asarray(v) for k, v in log[side].items()}

    raw_rmse = np.sqrt(np.mean((d["meas"][warmup:] - d["true_angle"][warmup:]) ** 2))
    ekf_rmse = np.sqrt(np.mean((d["est_angle"][warmup:] - d["true_angle"][warmup:]) ** 2))
    vel_rmse = np.sqrt(np.mean((d["est_vel"][warmup:] - d["true_vel"][warmup:]) ** 2))

    print(f"[{side.upper()}] raw angle RMSE: {raw_rmse:.4f} | "
          f"EKF angle RMSE: {ekf_rmse:.4f} | EKF velocity RMSE: {vel_rmse:.4f}")

    for arr in d.values():
        assert np.all(np.isfinite(arr))

print("\nEKF MuJoCo integration test passed!")