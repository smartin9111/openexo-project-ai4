import mujoco
import numpy as np

from src.environment.openexo_env import OpenExoEnvironment


ANKLE_JOINT_ID = 13
ANKLE_ACTUATOR_ID = 18

env = OpenExoEnvironment()
model = env.model
data = env.data

qvel_index = model.jnt_dofadr[ANKLE_JOINT_ID]

for k in range(10):
    torque_command = 0.2

    data.ctrl[ANKLE_ACTUATOR_ID] = torque_command

    mujoco.mj_step(model, data)

    print(
        f"step={k:2d}",
        f"ctrl={data.ctrl[ANKLE_ACTUATOR_ID]: .4f}",
        f"qfrc_actuator={data.qfrc_actuator[qvel_index]: .4f}",
        f"qpos={data.qpos[model.jnt_qposadr[ANKLE_JOINT_ID]]: .4f}",
        f"qvel={data.qvel[qvel_index]: .4f}",
    )