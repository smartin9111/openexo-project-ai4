import mujoco
import numpy as np

from src.environment.openexo_env import OpenExoEnvironment
from src.perception.ekf import AnkleEKFPerception


# Model mapping found during inspection
ANKLE_JOINT_ID = 13
ANKLE_ACTUATOR_ID = 18

STEPS = 500
MEASUREMENT_STD = 0.03

rng = np.random.default_rng(42)

env = OpenExoEnvironment()
model = env.model
data = env.data

qpos_index = model.jnt_qposadr[ANKLE_JOINT_ID]
qvel_index = model.jnt_dofadr[ANKLE_JOINT_ID]

dt = model.opt.timestep

ekf = AnkleEKFPerception(
    dt=dt,
    process_noise=1e-4,
    measurement_noise=MEASUREMENT_STD**2,
)

true_angles = []
true_velocities = []
measurements = []
estimated_angles = []
estimated_velocities = []


print("MuJoCo timestep:", dt)
print("Ankle qpos index:", qpos_index)
print("Ankle qvel index:", qvel_index)


for k in range(STEPS):
    t = k * dt

    # Small test torque.
    # We use the real OpenSourceLeg ankle actuator.
    torque_command = 0.01 * np.sin(
        2.0 * np.pi * 0.5 * t
    )

    data.ctrl[ANKLE_ACTUATOR_ID] = torque_command

    # Advance the real MuJoCo simulation
    mujoco.mj_step(model, data)
    
    actual_actuator_force = float(
        data.qfrc_actuator[qvel_index]
    )

    # Ground truth from MuJoCo
    true_angle = float(data.qpos[qpos_index])
    true_velocity = float(data.qvel[qvel_index])

    # Simulated noisy encoder measurement
    measured_angle = (
        true_angle
        + rng.normal(0.0, MEASUREMENT_STD)
    )

    observation = {
        "ankle_angle_measurement": measured_angle,
        "ankle_torque": actual_actuator_force,
    }

    state = ekf.process(observation)

    true_angles.append(true_angle)
    true_velocities.append(true_velocity)
    measurements.append(measured_angle)

    estimated_angles.append(
        float(state.joint_positions[0])
    )

    estimated_velocities.append(
        float(state.joint_velocities[0])
    )


true_angles = np.asarray(true_angles)
true_velocities = np.asarray(true_velocities)
measurements = np.asarray(measurements)
estimated_angles = np.asarray(estimated_angles)
estimated_velocities = np.asarray(estimated_velocities)


# Ignore initialization transient
warmup = 50

raw_angle_rmse = np.sqrt(
    np.mean(
        (
            measurements[warmup:]
            - true_angles[warmup:]
        ) ** 2
    )
)

ekf_angle_rmse = np.sqrt(
    np.mean(
        (
            estimated_angles[warmup:]
            - true_angles[warmup:]
        ) ** 2
    )
)

ekf_velocity_rmse = np.sqrt(
    np.mean(
        (
            estimated_velocities[warmup:]
            - true_velocities[warmup:]
        ) ** 2
    )
)


print("\n=== EKF + MUJOCO RESULTS ===")
print("Raw noisy angle RMSE:", raw_angle_rmse)
print("EKF angle RMSE:", ekf_angle_rmse)
print("EKF velocity RMSE:", ekf_velocity_rmse)

print("\nFinal true angle:", true_angles[-1])
print("Final estimated angle:", estimated_angles[-1])

print("Final true velocity:", true_velocities[-1])
print(
    "Final estimated velocity:",
    estimated_velocities[-1],
)


# Integration safety checks
assert np.all(np.isfinite(true_angles))
assert np.all(np.isfinite(true_velocities))
assert np.all(np.isfinite(estimated_angles))
assert np.all(np.isfinite(estimated_velocities))

print("\nEKF MuJoCo integration test passed!")