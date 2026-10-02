import mujoco

from src.environment.openexo_env import OpenExoEnvironment


env = OpenExoEnvironment()
model = env.model

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

    print(
        i,
        name,
        "dim=",
        model.sensor_dim[i],
    )
    print("\n=== RIGHT ANKLE DETAILS ===")

joint_id = 13
sensor_id = 10
actuator_id = 18

print("Joint name:",
      mujoco.mj_id2name(
          model,
          mujoco.mjtObj.mjOBJ_JOINT,
          joint_id,
      ))

print("qpos address:", model.jnt_qposadr[joint_id])
print("qvel address:", model.jnt_dofadr[joint_id])

print("Sensor name:",
      mujoco.mj_id2name(
          model,
          mujoco.mjtObj.mjOBJ_SENSOR,
          sensor_id,
      ))

print("Sensor type:", model.sensor_type[sensor_id])
print("Sensor address:", model.sensor_adr[sensor_id])
print("Sensor dimension:", model.sensor_dim[sensor_id])

print("Actuator name:",
      mujoco.mj_id2name(
          model,
          mujoco.mjtObj.mjOBJ_ACTUATOR,
          actuator_id,
      ))

print("Simulation timestep:", model.opt.timestep)