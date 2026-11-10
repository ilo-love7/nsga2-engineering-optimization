"""02 NSGA-II Pareto 前沿复现：ZDT1–3 + 焊接梁/压力容器/减速器/离合器。

- ZDT 对解析前沿、焊接梁对 pymoo 官方参考前沿 → GD/IGD。
- 压力容器/减速器/离合器：3 种子池化非支配前沿 + 端点 vs 01 单目标最优。
"""
from __future__ import annotations

import json
import time

import numpy as np
from pymoo.problems import get_problem

from common import RESULTS, save_fig, save_json
from src.nsga2.metrics import gd, hv_ref_point, hypervolume, igd
from src.nsga2.runner import feasible_mask, nondominated_front, run_nsga2
from src.problems.clutch_wrap import Clutch
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer

POP, GEN, SEEDS = 100, 150, [1, 2, 3]


def pool_front(problem, seeds=SEEDS):
    """多种子池化 → 可行非支配前沿（F, X 配对去重）。"""
    Fs, Xs = [], []
    for s in seeds:
        res = run_nsga2(problem, pop_size=POP, n_gen=GEN, seed=s)
        fm = feasible_mask(res.G)
        Fs.append(res.F[fm])
        Xs.append(res.X[fm])
    F = np.vstack(Fs)
    X = np.vstack(Xs)
    # 非支配过滤（保留对应 X）：被任何点支配的 i 删除
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dom[i] = False
        if dom.any():
            keep[i] = False
    F, X = F[keep], X[keep]
    # 去重（按 F 行）
    _, idx = np.unique(np.round(F, 10), axis=0, return_index=True)
    idx = np.sort(idx)
    F, X = F[idx], X[idx]
    order = np.argsort(F[:, 0])
    return F[order], X[order]


def main() -> None:
    import matplotlib.pyplot as plt

    out: dict = {"config": {"pop": POP, "gen": GEN, "seeds": SEEDS}}
    with open(RESULTS / "01_problem_validation.json", encoding="utf-8") as f:
        v1 = json.load(f)

    panels = []

    # ---------- ZDT1–3：解析前沿对照 ----------
    for name in ["zdt1", "zdt2", "zdt3"]:
        prob = get_problem(name)
        F, X = pool_front(prob, seeds=[1, 2])
        ref = prob.pareto_front()
        out[name] = {
            "n_front": int(F.shape[0]),
            "gd": gd(F, ref), "igd": igd(F, ref),
            "hv": hypervolume(F, np.array([1.1, 1.1])),
        }
        print(f"{name}: n={F.shape[0]} GD={out[name]['gd']:.2e} "
              f"IGD={out[name]['igd']:.2e} HV={out[name]['hv']:.4f}")
        panels.append((f"{name}: IGD={out[name]['igd']:.2e}", F, ref, None, None))

    # ---------- 焊接梁：官方参考前沿 ----------
    wb = get_problem("welded_beam")
    Fwb, Xwb = pool_front(wb, seeds=[1, 2, 3])
    ref_wb = wb.pareto_front()
    out["welded_beam"] = {
        "n_front": int(Fwb.shape[0]),
        "gd": gd(Fwb, ref_wb), "igd": igd(Fwb, ref_wb),
        "hv": hypervolume(Fwb, hv_ref_point(ref_wb, 1.1)),
        "hv_ref": hv_ref_point(ref_wb, 1.1).tolist(),
        "F_min_f1": Fwb[np.argmin(Fwb[:, 0])].tolist(),
        "F_min_f2": Fwb[np.argmin(Fwb[:, 1])].tolist(),
        "X_min_f1": Xwb[np.argmin(Fwb[:, 0])].tolist(),
        "X_min_f2": Xwb[np.argmin(Fwb[:, 1])].tolist(),
    }
    print(f"welded_beam: n={Fwb.shape[0]} GD={out['welded_beam']['gd']:.2e} "
          f"IGD={out['welded_beam']['igd']:.2e}")
    panels.append(("welded beam (cost vs deflection)", Fwb, ref_wb, Xwb, "wb"))

    # ---------- 压力容器 ----------
    pv = PressureVessel()
    Fpv, Xpv = pool_front(pv, seeds=[1, 2, 3])
    # 端点 vs 单目标参考（01）
    f1_min_row = Fpv[np.argmin(Fpv[:, 0])]
    f2_min_row = Fpv[np.argmin(Fpv[:, 1])]
    ref_cost = v1["pressure_vessel"]["min_cost"]["f_min"]
    ref_weight = v1["pressure_vessel"]["min_weight"]["f_min"]
    out["pressure_vessel"] = {
        "n_front": int(Fpv.shape[0]),
        "hv": hypervolume(Fpv, hv_ref_point(Fpv, 1.15)),
        "hv_ref": hv_ref_point(Fpv, 1.15).tolist(),
        "front_min_f1": float(f1_min_row[0]), "front_min_f2": float(f2_min_row[1]),
        "single_obj_min_f1": ref_cost, "single_obj_min_f2": ref_weight,
        "f1_gap_pct": abs(f1_min_row[0] / ref_cost - 1) * 100,
        "f2_gap_pct": abs(f2_min_row[1] / ref_weight - 1) * 100,
        "X_min_f1": Xpv[np.argmin(Fpv[:, 0])].tolist(),
        "X_min_f2": Xpv[np.argmin(Fpv[:, 1])].tolist(),
    }
    print(f"pressure_vessel: n={Fpv.shape[0]} "
          f"endpoint gap {out['pressure_vessel']['f1_gap_pct']:.2f}% / "
          f"{out['pressure_vessel']['f2_gap_pct']:.2f}%")
    panels.append(("pressure vessel (cost vs weight)", Fpv, None, Xpv, "pv"))

    # ---------- 减速器 ----------
    sr = SpeedReducer()
    Fsr, Xsr = pool_front(sr, seeds=[1, 2, 3])
    ref_w = v1["speed_reducer"]["min_weight"]["f_min"]
    ref_s = v1["speed_reducer"]["min_stress"]["f_min"]
    f1_min_row = Fsr[np.argmin(Fsr[:, 0])]
    f2_min_row = Fsr[np.argmin(Fsr[:, 1])]
    out["speed_reducer"] = {
        "n_front": int(Fsr.shape[0]),
        "hv": hypervolume(Fsr, hv_ref_point(Fsr, 1.15)),
        "hv_ref": hv_ref_point(Fsr, 1.15).tolist(),
        "front_min_f1": float(f1_min_row[0]), "front_min_f2": float(f2_min_row[1]),
        "single_obj_min_f1": ref_w, "single_obj_min_f2": ref_s,
        "f1_gap_pct": abs(f1_min_row[0] / ref_w - 1) * 100,
        "f2_gap_pct": abs(f2_min_row[1] / ref_s - 1) * 100,
        "X_min_f1": Xsr[np.argmin(Fsr[:, 0])].tolist(),
        "X_min_f2": Xsr[np.argmin(Fsr[:, 1])].tolist(),
    }
    print(f"speed_reducer: n={Fsr.shape[0]} "
          f"endpoint gap {out['speed_reducer']['f1_gap_pct']:.2f}% / "
          f"{out['speed_reducer']['f2_gap_pct']:.2f}%")
    panels.append(("speed reducer (weight vs stress)", Fsr, None, Xsr, "sr"))

    # ---------- 离合器（整数） ----------
    cl = Clutch()
    Fcl, Xcl = pool_front(cl, seeds=[1, 2, 3])
    out["clutch"] = {
        "n_front": int(Fcl.shape[0]),
        "hv": hypervolume(Fcl, hv_ref_point(Fcl, 1.15)),
        "hv_ref": hv_ref_point(Fcl, 1.15).tolist(),
        "F_min_f1": Fcl[np.argmin(Fcl[:, 0])].tolist(),
        "F_min_f2": Fcl[np.argmin(Fcl[:, 1])].tolist(),
        "X_min_f1": Xcl[np.argmin(Fcl[:, 0])].tolist(),
        "X_min_f2": Xcl[np.argmin(Fcl[:, 1])].tolist(),
    }
    print(f"clutch: n={Fcl.shape[0]}")
    panels.append(("clutch/brake (mass vs stopping time)", Fcl, None, Xcl, "cl"))

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, (title, F, ref, X, tag) in zip(axes.ravel(), panels):
        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 1], "k.", ms=3, alpha=0.5, label="reference PF")
        ax.plot(F[:, 0], F[:, 1], "o", ms=3.5, color="#1f77b4", label="NSGA-II")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    save_fig("02_pareto_fronts.png")

    # 保存前沿数据（TOPSIS 用）
    np.savez(RESULTS / "02_fronts.npz",
             wb_F=Fwb, wb_X=Xwb, pv_F=Fpv, pv_X=Xpv,
             sr_F=Fsr, sr_X=Xsr, cl_F=Fcl, cl_X=Xcl)
    save_json("02_pareto_fronts.json", out)
    print("saved: results/02_pareto_fronts.json + 02_fronts.npz + figures/02_pareto_fronts.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time()-t0:.1f}s")
