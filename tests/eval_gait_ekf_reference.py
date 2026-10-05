"""
Evaluate the perception methods on real walking kinematics (MyoAssist reference
gait), with sensor noise and dropout. Run from the project root:

    python -m tests.eval_gait_ekf_reference /path/to/myoassist/rl_train/reference_data

Notes: the template (data/ankle_template_r.npy) comes from the first strides of
the reference trial, so the evaluation uses a later segment (default 20-30 s).
"""
import sys
from pathlib import Path

import numpy as np

from src.perception.ekf import AnkleEKFPerception
from src.perception.gait_ekf import GaitPhaseEKFPerception

REF_DIR = sys.argv[1] if len(sys.argv) > 1 else "rl_train/reference_data"

if not (Path(REF_DIR) / "short_reference_gait.npz").exists():
    sys.exit(
        f"Reference data not found: {Path(REF_DIR) / 'short_reference_gait.npz'}\n"
        "Pass the path of the MyoAssist clone's rl_train/reference_data folder as the "
        "first argument, e.g.\n"
        "  python -m tests.eval_gait_ekf_reference ~/myoassist/rl_train/reference_data"
    )
START_S, LENGTH_S, SEEDS = 20, 10, 2
DT, STD = 1 / 500, 0.03

series = np.load(f"{REF_DIR}/short_reference_gait.npz", allow_pickle=True)["series_data"].item()
i0, i1 = int(START_S / DT), int((START_S + LENGTH_S) / DT)
A = np.asarray(series["q_ankle_angle_r"])[i0:i1]
V = np.asarray(series["dq_ankle_angle_r"])[i0:i1]
template = np.load("data/ankle_template_r.npy")


def dropout_mask(rng, n, p_start, length=25):
    m = np.zeros(n, bool)
    i = 0
    while i < n:
        if rng.random() < p_start:
            m[i:i + length] = True
            i += length
        else:
            i += 1
    return m


def rmse(x, y, w=500):
    return float(np.sqrt(np.mean((x[w:] - y[w:]) ** 2)))


def run(name, z):
    ea, ev = np.zeros_like(A), np.zeros_like(A)
    if name == "gait":
        f = GaitPhaseEKFPerception(dt=DT, template=template, measurement_noise=STD**2)
    elif name == "kf":
        f = AnkleEKFPerception(dt=DT, measurement_noise=STD**2, accel_noise=100.0)
    if name in ("gait", "kf"):
        for i, zi in enumerate(z):
            s = f.process({"ankle_angle_measurement": zi})
            ea[i], ev[i] = s.joint_positions[0], s.joint_velocities[0]
    else:  # low-pass (hold last value on dropout) + smoothed derivative
        last, lp = (z[0] if np.isfinite(z[0]) else 0.0), np.zeros_like(A)
        for i in range(A.size):
            last = z[i] if np.isfinite(z[i]) else last
            lp[i] = last if i == 0 else 0.3 * last + 0.7 * lp[i - 1]
        fd, ev = np.gradient(lp, DT), np.zeros_like(A)
        for i in range(1, fd.size):
            ev[i] = 0.1 * fd[i] + 0.9 * ev[i - 1]
        ea = lp
    return rmse(ea, A), rmse(ev, V)


print(f"reference walking {START_S}-{START_S + LENGTH_S} s, noise std {STD} rad, "
      f"{SEEDS} seeds | zero-velocity guess RMSE {rmse(np.zeros_like(V), V):.2f} rad/s\n")
print(f"{'missing':>8s} | {'method':>16s} | angle RMSE | velocity RMSE")
for p_start in (0.0, 0.004, 0.015):
    res = {"gait": [], "kf": [], "lp": []}
    frac = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        z = A + rng.normal(0, STD, A.size)
        m = dropout_mask(rng, A.size, p_start)
        frac.append(m.mean())
        z[m] = np.nan
        for k in res:
            res[k].append(run(k, z))
    for k, label in (("gait", "gait-phase EKF"), ("kf", "kinematic KF"), ("lp", "low-pass + diff")):
        a, v = np.mean(res[k], axis=0)
        print(f"{np.mean(frac) * 100:7.1f}% | {label:>16s} | {a:10.4f} | {v:13.3f}")