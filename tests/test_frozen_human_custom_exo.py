"""
Frozen human walking policy + our own exo command.

The pretrained PPO drives the 22 human muscles (action[0:22]); the exo actions
(action[22:24]) are thrown away and replaced by a command of ours.

Run from the MyoAssist clone (the config paths are relative to it):

    cd myoassist
    PYTHONPATH=.:.. ../.venv/bin/python ../tests/test_frozen_human_custom_exo.py \
        --session ../assets/walking_policy --composed \
        --exo 1.0 0.9 --steps 600

--composed  use the current assist_sim model (myolegs22 + OpenExo_L1) instead of the
            old config's model_path (a hand-made MJCF that is not in this MyoAssist).

--session   training session folder (has session_config.json and trained_models/)
--model     optional checkpoint path; default = the newest one in trained_models/
--exo       exo action value(s), applied to both legs, one rollout per value.
            1.0 = exo off (ctrl 0). Smaller = more assistance.
--steps     control steps per rollout (30 Hz -> 600 steps = 20 s)

It prints the exo ctrl that MuJoCo really got (read AFTER the step), whether the
human fell, and saves the 30 Hz signals of each rollout to --out (npz).
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def newest_model(session_dir):
    best = None
    for path in glob.glob(os.path.join(session_dir, "trained_models", "model_*")):
        m = re.search(r"model_(\d+)", os.path.basename(path))
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), path)
    if best is None:
        raise FileNotFoundError(f"no model_* in {session_dir}/trained_models")
    return best[1][:-4] if best[1].endswith(".zip") else best[1]


def read_signals(sim):
    """True ankle state, foot contact and pelvis height from the MuJoCo data."""
    d = sim.data
    out = {}
    for s in ("r", "l"):
        out[f"ankle_angle_{s}"] = float(d.joint(f"ankle_angle_{s}").qpos[0])
        out[f"ankle_vel_{s}"] = float(d.joint(f"ankle_angle_{s}").qvel[0])
        out[f"foot_force_{s}"] = float(d.sensor(f"{s}_foot").data[0] + d.sensor(f"{s}_toes").data[0])
    out["pelvis_height"] = float(d.body("pelvis").xpos[2])
    out["pelvis_x"] = float(d.body("pelvis").xpos[0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--exo", type=float, nargs="+", default=[1.0, 0.9])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--out", default="../data/rollouts")
    p.add_argument(
        "--composed",
        action="store_true",
        help="ignore the model_path of the old config and build the model with the current "
        "assist_sim pipeline (myolegs22 + OpenExo_L1, flat ground) instead",
    )
    args = p.parse_args()

    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
    from rl_train.utils.data_types import DictionableDataclass

    with open(os.path.join(args.session, "session_config.json"), "r") as f:
        config_dict = json.load(f)
    config_type = EnvironmentHandler.get_config_type_from_session_id(config_dict["env_params"]["env_id"])
    config = DictionableDataclass.create(config_type, config_dict)

    # Same evaluation settings as MyoAssist's GaitEvaluatorBase
    config.env_params.num_envs = 1
    config.env_params.custom_max_episode_steps = 10**9
    config.env_params.out_of_trajectory_threshold = 10**6

    if args.composed:
        # The old session was trained on a hand-made MJCF (env_params.model_path) that does
        # not exist in the current MyoAssist. Build the equivalent model with the new pipeline.
        config.env_params.model_path = None
        config.env_params.msk_key = "myolegs22"
        config.env_params.device_key = "OpenExo_L1"
        config.env_params.terrain = None
    print(f"model_path={config.env_params.model_path!r} msk_key={config.env_params.msk_key!r} "
          f"device_key={config.env_params.device_key!r}")

    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)
    model_path = args.model or newest_model(args.session)
    model = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=model_path)
    print(f"model: {model_path}")
    print(f"control dt: {1.0 / config.env_params.control_framerate:.4f} s")

    base = env.unwrapped
    if config.evaluate_param_list:
        ev = config.evaluate_param_list[0]
        v_min = ev.get("min_target_velocity", 1.25)
        v_max = ev.get("max_target_velocity", 1.25)
        base.set_target_velocity_mode_manually(
            MyoAssistLegBase.VelocityMode[ev.get("velocity_mode", "UNIFORM")],
            0,
            (v_min + v_max) / 2,
            v_min,
            v_max,
            target_velocity_period=ev.get("target_velocity_period", 2),
        )

    os.makedirs(args.out, exist_ok=True)

    for exo_value in args.exo:
        obs, info = env.reset()
        log = {}
        fell_at = None
        for t in range(args.steps):
            full_action, _ = model.predict(obs, deterministic=True)
            action = np.array(full_action, dtype=np.float32).copy()
            raw_exo = action[..., 22:24].copy()
            action[..., 22:24] = exo_value  # our exo command (replaces the PPO's)

            obs, reward, terminated, truncated, info = env.step(action)

            ctrl = base.sim.data.ctrl[22:24].copy()  # read AFTER the step
            if t < 3:
                print(f"  [exo={exo_value}] t={t} raw PPO exo {raw_exo} -> sent {action[..., 22:24]} "
                      f"-> MuJoCo ctrl {ctrl}")

            sig = read_signals(base.sim)
            sig["exo_ctrl_r"], sig["exo_ctrl_l"] = float(ctrl[0]), float(ctrl[1])
            for k, v in sig.items():
                log.setdefault(k, []).append(v)

            if terminated or truncated:
                fell_at = t
                break

        n = len(log["pelvis_height"])
        dt = 1.0 / config.env_params.control_framerate
        dist = log["pelvis_x"][-1] - log["pelvis_x"][0]
        print(
            f"[exo={exo_value}] steps {n}/{args.steps} ({n * dt:.1f} s)"
            f" | ended early: {fell_at is not None}"
            f" | min pelvis height {min(log['pelvis_height']):.2f} m"
            f" | speed {dist / (n * dt):.2f} m/s"
            f" | exo ctrl range [{min(log['exo_ctrl_r']):.3f}, {max(log['exo_ctrl_r']):.3f}]"
        )
        out_path = os.path.join(args.out, f"rollout_exo_{exo_value}.npz")
        np.savez(out_path, dt=dt, **{k: np.asarray(v) for k, v in log.items()})
        print(f"  saved {out_path}")

    env.close()


if __name__ == "__main__":
    main()