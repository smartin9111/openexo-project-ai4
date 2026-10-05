"""
Status plot: env -> noisy sensor -> EKF -> gait phase -> controller -> exo (both legs).

Run from the project root:
    python -m tests.plot_status
It writes plots/status.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from src.environment.openexo_env import OpenExoEnvironment
from src.perception.ekf import AnkleEKFPerception
from src.controllers.dummy import DummyController


STEPS = 1000
MEASUREMENT_STD = 0.03
EXO_SCALE = 0.01  # keeps the exo gentle while there is no walking policy
SIDES = ("r", "l")
NAMES = {"r": "Right ankle", "l": "Left ankle"}

rng = np.random.default_rng(42)

env = OpenExoEnvironment()
dt = env.model.opt.timestep

perception = {
    s: AnkleEKFPerception(dt=dt, accel_noise=1e3, measurement_noise=MEASUREMENT_STD**2)
    for s in SIDES
}
controller = DummyController()

keys = ("t", "true_angle", "meas", "est_angle", "true_vel", "est_vel", "cmd", "stance")
log = {s: {k: [] for k in keys} for s in SIDES}

gt = env.reset()
for k in range(STEPS):
    cmds = {}
    for s in SIDES:
        in_contact = bool(np.any(gt[f"foot_contacts_{s}"] > 0.0))
        meas = gt[f"ankle_angle_{s}"] + rng.normal(0.0, MEASUREMENT_STD)

        state = perception[s].process({
            "ankle_angle_measurement": meas,
            "ankle_torque": gt[f"ankle_torque_{s}"] / 100.0,
            "gait_phase": "stance" if in_contact else "swing",
        })
        cmds[s] = controller.compute_action(state) * EXO_SCALE

        L = log[s]
        L["t"].append(gt["time"])
        L["true_angle"].append(gt[f"ankle_angle_{s}"])
        L["meas"].append(meas)
        L["est_angle"].append(float(state.joint_positions[0]))
        L["true_vel"].append(gt[f"ankle_velocity_{s}"])
        L["est_vel"].append(float(state.joint_velocities[0]))
        L["cmd"].append(cmds[s])
        L["stance"].append(float(in_contact))

    gt = env.step((cmds["r"], cmds["l"]))

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
for col, s in enumerate(SIDES):
    L = {k: np.asarray(v) for k, v in log[s].items()}
    t = L["t"]

    ax = axes[0, col]
    ax.plot(t, L["meas"], color="0.75", lw=0.8, label="noisy measurement")
    ax.plot(t, L["true_angle"], color="tab:blue", lw=1.5, label="true")
    ax.plot(t, L["est_angle"], color="tab:red", lw=1.2, ls="--", label="EKF")
    ax.set_title(NAMES[s])
    ax.set_ylabel("ankle angle [rad]")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, col]
    ax.plot(t, L["true_vel"], color="tab:blue", lw=1.5, label="true")
    ax.plot(t, L["est_vel"], color="tab:red", lw=1.2, ls="--", label="EKF")
    ax.set_ylabel("ankle velocity [rad/s]")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, col]
    ax.fill_between(t, 0, 1, where=L["stance"] > 0, color="tab:green", alpha=0.2,
                    transform=ax.get_xaxis_transform(), label="stance (foot contact)")
    ax.plot(t, L["cmd"], color="tab:purple", lw=1.5, label="exo command")
    ax.set_ylabel("exo command [-1..0]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="best", fontsize=8)

    w = 50  # skip the filter start-up
    print(f"[{s.upper()}] angle RMSE raw {np.sqrt(np.mean((L['meas'][w:] - L['true_angle'][w:])**2)):.4f}"
          f" | EKF {np.sqrt(np.mean((L['est_angle'][w:] - L['true_angle'][w:])**2)):.4f}"
          f" | EKF velocity {np.sqrt(np.mean((L['est_vel'][w:] - L['true_vel'][w:])**2)):.3f} rad/s"
          f" | stance {int(L['stance'].sum())}/{len(t)} steps")

fig.suptitle("Pipeline status (no walking policy yet: the body collapses, signals are not gait)",
             fontsize=11)
fig.tight_layout()

out = Path("plots")
out.mkdir(exist_ok=True)
fig.savefig(out / "status.png", dpi=130)
print("saved", out / "status.png")