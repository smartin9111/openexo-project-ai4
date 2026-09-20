import mujoco
import numpy as np

from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()

initial_time = env.data.time

for _ in range(100):
    mujoco.mj_step(env.model, env.data)

assert env.data.time > initial_time
assert np.all(np.isfinite(env.data.qpos))
assert np.all(np.isfinite(env.data.qvel))

print("Simulation smoke test passed!")
print("Simulated time:", env.data.time)
print("qpos finite:", np.all(np.isfinite(env.data.qpos)))
print("qvel finite:", np.all(np.isfinite(env.data.qvel)))
