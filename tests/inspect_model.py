import mujoco

from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()
model = env.model

print("\n=== MODEL INFO ===")
print(env.get_model_info())


print("\n=== JOINTS ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i,
    )
    print(i, name)


print("\n=== ACTUATORS ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i,
    )
    print(i, name)


print("\n=== SENSORS ===")
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
        i,
        name,
        f"type={sensor_type.name}",
        f"dim={model.sensor_dim[i]}",
        f"address={model.sensor_adr[i]}",
        print("\n=== RIGHT ANKLE ===")
		print("qpos index:", env.ankle_qpos)
		print("qvel index:", env.ankle_qvel)
		print("Exo_R actuator:", env.exo_r_id)
		print("Timestep:", model.opt.timestep)
    )