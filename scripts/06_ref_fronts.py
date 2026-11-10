"""06 v1.1：ε-约束法生成高精度参考前沿（压力容器 / 减速器 / 焊接梁）。

- PV/WB：连续变量直接 ε-约束扫描（min f1 s.t. f2≤ε, g≤0，多起点 SLSQP+热启动）。
- SR：z∈{17..28} 整数枚举 × 每 z 一条连续子前沿 → 并集非支配（混合整数参考）。
- WB 与 pymoo 官方参考前沿交叉验证（GD/IGD 小 → 方法学自证）。
- 端点与 01 单目标最优对照（应 ≤0.05%）。
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.problems import get_problem

from common import RESULTS, save_fig, save_json
from src.nsga2.local_search import eref_front, nondominated_mask
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer
from src.nsga2.metrics import gd, igd

NEPS = {"pv": 31, "sr": 17, "wb": 41}


def merge_fronts(F_list, X_list):
    F = np.vstack(F_list)
    X = np.vstack(X_list)
    keep = nondominated_mask(F)
    F, X = F[keep], X[keep]
    _, idx = np.unique(np.round(F, 10), axis=0, return_index=True)
    idx = np.sort(idx)
    F, X = F[idx], X[idx]
    order = np.argsort(F[:, 0])
    return F[order], X[order]


def main() -> None:
    import matplotlib.pyplot as plt

    out: dict = {"method": "epsilon-constraint + multi-start SLSQP + hot start",
                 "n_starts": 3}
    t0 = time.time()

    # ---------- 压力容器 ----------
    pv = PressureVessel()
    Fpv, Xpv = eref_front(pv, n_eps=NEPS["pv"], n_starts=3, seed=0)
    out["pressure_vessel"] = {
        "n": int(Fpv.shape[0]),
        "f1_range": [float(Fpv[:, 0].min()), float(Fpv[:, 0].max())],
        "f2_range": [float(Fpv[:, 1].min()), float(Fpv[:, 1].max())],
        "monotone_f2_decreasing": bool(np.all(np.diff(Fpv[:, 1]) <= 1e-3)),
    }
    print(f"PV eref: n={Fpv.shape[0]} f1∈[{Fpv[0,0]:.2f},{Fpv[-1,0]:.2f}] "
          f"f2∈[{Fpv[-1,1]:.2f},{Fpv[0,1]:.2f}]")

    # ---------- 减速器（z 整数枚举） ----------
    sr = SpeedReducer()
    Fs_list, Xs_list, z_used = [], [], []
    for z in range(17, 29):
        Fz, Xz = eref_front(sr, n_eps=NEPS["sr"], n_starts=2, seed=z,
                           fixed={2: float(z)})
        if Fz.shape[0] == 0:
            continue
        Xz[:, 2] = float(z)
        Fs_list.append(Fz)
        Xs_list.append(Xz)
        z_used.append(z)
    Fsr, Xsr = merge_fronts(Fs_list, Xs_list)
    z_on_front = sorted(set(int(v) for v in Xsr[:, 2]))
    out["speed_reducer"] = {
        "n": int(Fsr.shape[0]),
        "f1_range": [float(Fsr[:, 0].min()), float(Fsr[:, 0].max())],
        "f2_range": [float(Fsr[:, 1].min()), float(Fsr[:, 1].max())],
        "z_solved": z_used,
        "z_on_front": z_on_front,
        "monotone_f2_decreasing": bool(np.all(np.diff(Fsr[:, 1]) <= 1e-3)),
    }
    print(f"SR eref: n={Fsr.shape[0]} z_on_front={z_on_front} "
          f"f1 min {Fsr[0,0]:.3f} f2 min {Fsr[-1,1]:.3f}")

    # ---------- 焊接梁（vs pymoo 官方前沿交叉验证） ----------
    wb = get_problem("welded_beam")
    Fwb, Xwb = eref_front(wb, n_eps=NEPS["wb"], n_starts=3, seed=0)
    ref_official = np.asarray(wb.pareto_front(), dtype=float)
    gd_own_vs_off = gd(Fwb, ref_official)
    igd_own_vs_off = igd(Fwb, ref_official)
    out["welded_beam"] = {
        "n": int(Fwb.shape[0]),
        "f1_range": [float(Fwb[:, 0].min()), float(Fwb[:, 0].max())],
        "f2_range": [float(Fwb[:, 1].min()), float(Fwb[:, 1].max())],
        "crosscheck_vs_pymoo_official": {
            "gd_own_to_official": gd_own_vs_off,
            "igd_own_to_official": igd_own_vs_off,
            "n_official": int(ref_official.shape[0]),
        },
        "monotone_f2_decreasing": bool(np.all(np.diff(Fwb[:, 1]) <= 1e-6)),
    }
    print(f"WB eref: n={Fwb.shape[0]} vs official GD={gd_own_vs_off:.2e} "
          f"IGD={igd_own_vs_off:.2e}")

    # ---------- 端点 vs 01 单目标 ----------
    import json
    with open(RESULTS / "01_problem_validation.json", encoding="utf-8") as f:
        v1 = json.load(f)
    checks = {
        "pv_f1_vs_01_pct": abs(Fpv[0, 0] / v1["pressure_vessel"]["min_cost"]["f_min"] - 1) * 100,
        "pv_f2_vs_01_pct": abs(Fpv[-1, 1] / v1["pressure_vessel"]["min_weight"]["f_min"] - 1) * 100,
        "sr_f1_vs_01_pct": abs(Fsr[0, 0] / v1["speed_reducer"]["min_weight"]["f_min"] - 1) * 100,
        "sr_f2_vs_01_pct": abs(Fsr[-1, 1] / v1["speed_reducer"]["min_stress"]["f_min"] - 1) * 100,
    }
    out["endpoint_checks_pct"] = checks
    print("endpoint vs 01:", {k: round(v, 4) for k, v in checks.items()})

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (title, F, X) in zip(axes, [
            ("pressure vessel (cost vs weight)", Fpv, Xpv),
            ("speed reducer (weight vs stress)", Fsr, Xsr),
            ("welded beam (cost vs deflection)", Fwb, Xwb)]):
        ax.plot(F[:, 0], F[:, 1], "-o", ms=3, lw=1.2, color="#d62728",
                label="e-constraint ref")
        if title.startswith("welded"):
            ax.plot(ref_official[:, 0], ref_official[:, 1], "k.", ms=3,
                    alpha=0.5, label="pymoo official")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend(fontsize=7)
    save_fig("06_ref_fronts.png")

    np.savez(RESULTS / "06_ref_fronts.npz",
             pv_F=Fpv, pv_X=Xpv, sr_F=Fsr, sr_X=Xsr, wb_F=Fwb, wb_X=Xwb)
    out["runtime_s"] = round(time.time() - t0, 1)
    save_json("06_ref_fronts.json", out)
    print("saved: results/06_ref_fronts.npz + .json + figures/06_ref_fronts.png")


if __name__ == "__main__":
    main()
