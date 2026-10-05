import numpy as np

from src.environment.openexo_env import OpenExoEnvironment
from src.perception.ekf import AnkleEKFPerception
from src.controllers.dummy import DummyController


STEPS = 1000
MEASUREMENT_STD = 0.03
EXO_SCALE = 0.01  # scales the controller output so the exo stays gentle
SIDES = ("r", "l")

rng = np.random.default_rng(42)

env = OpenExoEnvironment()
dt = env.model.opt.timestep

# One perception module per leg, one controller shared by both legs
perception = {
    side: AnkleEKFPerception(
        dt=dt,
        process_noise=1e-4,
        measurement_noise=MEASUREMENT_STD**2,
    )
    for side in SIDES
}
controller = DummyController()

gt = env.reset()
commands = []
stance_steps = {side: 0 for side in SIDES}

for k in range(STEPS):
    command = {}

    for side in SIDES:
        # sensor layer: noisy ankle angle + foot-contact based gait phase
        in_contact = bool(np.any(gt[f"foot_contacts_{side}"] > 0.0))
        stance_steps[side] += int(in_contact)

        observation = {
            "ankle_angle_measurement": gt[f"ankle_angle_{side}"] + rng.normal(0.0, MEASUREMENT_STD),
            "ankle_torque": gt[f"ankle_torque_{side}"] / 100.0,
            "foot_contacts": gt[f"foot_contacts_{side}"],
            "gait_phase": "stance" if in_contact else "swing",
        }

        state = perception[side].process(observation)
        action = controller.compute_action(state)  # -1.0 (stance) or 0.0 (swing)
        command[side] = action * EXO_SCALE

    exo_command = (command["r"], command["l"])
    gt = env.step(exo_command)
    commands.append(exo_command)

commands = np.asarray(commands)

print("steps:", STEPS, "| sim time:", round(gt["time"], 3), "s")
print("stance steps R/L:", stance_steps["r"], "/", stance_steps["l"])
print("exo command min/max:", commands.min(), commands.max())
assert np.all(np.isfinite(commands))
assert commands.min() >= -1.0 and commands.max() <= 0.0
print("Pipeline test passed!")