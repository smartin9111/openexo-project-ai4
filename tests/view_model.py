import time

import mujoco.viewer

from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()
model = env.model
data = env.data

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)