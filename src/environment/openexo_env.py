from pathlib import Path

import mujoco as mj
import numpy as np
from myoassist_utils.compose import compose_env_model

MSK_KEY = "myolegs22"
DEVICE_KEY = "OpenExo_L1"


class OpenExoEnvironment:
    """
    MuJoCo environment: myolegs22 musculoskeletal model + bilateral OpenExo_L1
    ankle exo on a flat ground plane.

    The exo actuators (Exo_R, Exo_L) take a command in [-1, 0]
    (actuator gain = 100, so -1.0 is about -100 Nm at the ankle).
    """

    def __init__(self, keyframe: str = "stand"):
        self.keyframe = keyframe

        # load_combined() returns a model without a ground plane (the body falls
        # through the floor). compose_env_model() adds a flat ground (terrain=None).
        xml_path = Path(__file__).resolve().parent / "openexo_flat.xml"
        compose_env_model(MSK_KEY, DEVICE_KEY, terrain=None, export_path=xml_path)
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

        # Look everything up by name, never by hard-coded index.
        self.exo_ids = {
            "r": self._id(mj.mjtObj.mjOBJ_ACTUATOR, "Exo_R"),
            "l": self._id(mj.mjtObj.mjOBJ_ACTUATOR, "Exo_L"),
        }

        self.ankle_qpos = {}
        self.ankle_qvel = {}
        for side in ("r", "l"):
            jid = self._id(mj.mjtObj.mjOBJ_JOINT, f"ankle_angle_{side}")
            self.ankle_qpos[side] = int(self.model.jnt_qposadr[jid])
            self.ankle_qvel[side] = int(self.model.jnt_dofadr[jid])

        # Foot contact (touch) sensors
        self.foot_sensors = {
            "r": [self._sensor_adr("r_foot"), self._sensor_adr("r_toes")],
            "l": [self._sensor_adr("l_foot"), self._sensor_adr("l_toes")],
        }

        self.exo_ctrl_range = (
            float(self.model.actuator_ctrlrange[self.exo_ids["r"]][0]),
            float(self.model.actuator_ctrlrange[self.exo_ids["r"]][1]),
        )

        self.reset()

    # ------------------------------------------------------------------ utils
    def _id(self, obj_type, name):
        idx = mj.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"'{name}' not found in model ({DEVICE_KEY})")
        return idx

    def _sensor_adr(self, name):
        sid = self._id(mj.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid])

    # ------------------------------------------------------------ simulation
    def reset(self):
        mj.mj_resetData(self.model, self.data)
        key = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, self.keyframe)
        if key >= 0:
            mj.mj_resetDataKeyframe(self.model, self.data, key)
        mj.mj_forward(self.model, self.data)
        return self.get_ground_truth()

    def set_exo_command(self, command):
        """
        Exo command, clipped to the actuator range [-1, 0].

        command: a float (same command on both legs) or (right, left).
        """
        cmd = np.broadcast_to(np.asarray(command, dtype=float), (2,))
        low, high = self.exo_ctrl_range
        self.data.ctrl[self.exo_ids["r"]] = float(np.clip(cmd[0], low, high))
        self.data.ctrl[self.exo_ids["l"]] = float(np.clip(cmd[1], low, high))

    def step(self, exo_command=0.0, muscle_ctrl=None):
        """
        Advance one physics step.

        exo_command: float (both legs) or (right, left), range [-1, 0].
        muscle_ctrl: optional array of muscle activations (length = nu - 2).
        Without a human policy the muscles stay at zero and the model collapses.
        """
        if muscle_ctrl is not None:
            n_muscles = self.model.nu - 2
            self.data.ctrl[:n_muscles] = muscle_ctrl
        self.set_exo_command(exo_command)
        mj.mj_step(self.model, self.data)
        return self.get_ground_truth()

    # ----------------------------------------------------------- observation
    def get_ground_truth(self):
        """True ankle state and foot contacts for both legs (no noise)."""
        gt = {"time": float(self.data.time)}
        for side in ("r", "l"):
            gt[f"ankle_angle_{side}"] = float(self.data.qpos[self.ankle_qpos[side]])
            gt[f"ankle_velocity_{side}"] = float(self.data.qvel[self.ankle_qvel[side]])
            gt[f"ankle_torque_{side}"] = float(self.data.qfrc_actuator[self.ankle_qvel[side]])
            gt[f"foot_contacts_{side}"] = np.array(
                [float(self.data.sensordata[a]) for a in self.foot_sensors[side]]
            )
        return gt

    def get_model_info(self):
        return {
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "nsensordata": self.model.nsensordata,
            "dt": self.model.opt.timestep,
            "exo_ids": self.exo_ids,
            "ankle_qpos": self.ankle_qpos,
            "ankle_qvel": self.ankle_qvel,
        }