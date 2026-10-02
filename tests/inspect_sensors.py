import mujoco

from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()
model = env.model

print("\n=== ALL SENSORS ===")

for i in range(model.nsensor):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        i,
    )

    sensor_type = mujoco.mjtSensor(
        model.sensor_type[i]
    )

    print(
        f"{i:2d}",
        f"name={name}",
        f"type={sensor_type.name}",
        f"dim={model.sensor_dim[i]}",
        f"address={model.sensor_adr[i]}",
    )