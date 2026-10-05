import mujoco

from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()
model = env.model
data = env.data

for k in range(10):
    exo_command = -0.2  # exo command range: [-1, 0], same on both legs

    env.set_exo_command(exo_command)
    mujoco.mj_step(model, data)

    print(
        f"step={k:2d}",
        f"ctrl_R={data.ctrl[env.exo_ids['r']]: .4f}",
        f"ctrl_L={data.ctrl[env.exo_ids['l']]: .4f}",
        f"torque_R={data.qfrc_actuator[env.ankle_qvel['r']]: .4f}",
        f"torque_L={data.qfrc_actuator[env.ankle_qvel['l']]: .4f}",
    )