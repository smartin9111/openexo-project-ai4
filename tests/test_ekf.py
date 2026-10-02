import numpy as np

from src.perception.ekf import AnkleEKFPerception


DT = 0.002
STEPS = 2500

GRAVITY_GAIN = 5.0
DAMPING = 0.5
TORQUE_GAIN = 1.0

MEASUREMENT_STD = 0.03

rng = np.random.default_rng(42)

ekf = AnkleEKFPerception(
    dt=DT,
    process_noise=1e-6,
    measurement_noise=MEASUREMENT_STD**2,
    gravity_gain=GRAVITY_GAIN,
    damping=DAMPING,
    torque_gain=TORQUE_GAIN,
)

# True initial state
theta = 0.2
omega = 0.0

true_angles = []
true_velocities = []
measurements = []
estimated_angles = []
estimated_velocities = []

for k in range(STEPS):
    t = k * DT

    # Known test torque
    torque = 0.8 * np.sin(2.0 * np.pi * 0.7 * t)

    # Simulated nonlinear ankle dynamics
    theta_next = theta + DT * omega

    omega_next = omega + DT * (
        -GRAVITY_GAIN * np.sin(theta)
        -DAMPING * omega
        +TORQUE_GAIN * torque
    )

    theta = theta_next
    omega = omega_next

    # Simulated noisy encoder measurement
    measured_angle = theta + rng.normal(
        0.0,
        MEASUREMENT_STD,
    )

    observation = {
        "ankle_angle_measurement": measured_angle,
        "ankle_torque": torque,
    }

    state = ekf.process(observation)

    true_angles.append(theta)
    true_velocities.append(omega)
    measurements.append(measured_angle)

    estimated_angles.append(
        state.joint_positions[0]
    )

    estimated_velocities.append(
        state.joint_velocities[0]
    )


true_angles = np.array(true_angles)
true_velocities = np.array(true_velocities)
measurements = np.array(measurements)
estimated_angles = np.array(estimated_angles)
estimated_velocities = np.array(estimated_velocities)

# Ignore initial filter transient
warmup = 100

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

print("Raw noisy angle RMSE:", raw_angle_rmse)
print("EKF angle RMSE:", ekf_angle_rmse)
print("EKF velocity RMSE:", ekf_velocity_rmse)

assert np.all(np.isfinite(estimated_angles))
assert np.all(np.isfinite(estimated_velocities))

assert ekf_angle_rmse < raw_angle_rmse

print("EKF test passed!")