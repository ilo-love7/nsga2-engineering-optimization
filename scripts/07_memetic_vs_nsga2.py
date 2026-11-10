"""07 v1.1：memetic 深化——NSGA-II 前沿 vs ASF 精修 + 端点局部搜索。

对 WB/PV/SR 三问题：
  raw      = 3 种子 NSGA-II 池化非支配前沿（v1.0 口径，pop100/gen150）
  polished = raw + 逐点 ASF（Tchebycheff）局部精修 + 两端点单目标精修
评价（对 06 的 ε-约束真参考前沿）：GD / IGD / HV（统一 HV 参考点）+ 端点 gap。
"""
from __future__ import annotations

import json
import time

import numpy as np
from pymoo.problems import get_problem

from common import RESULTS, save_fig, save_json
from src.nsga2.local_search import (
    endpoint_polish, nondominated_mask, polish_front,
)
from src.nsga2.metrics import gd, hypervolume, igd
from src.nsga2.runner import feasible_mask, run_nsga2
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer

POP, GEN, SEEDS = 100, 150, [1, 2, 3]


def pool_front(problem, seeds=SEEDS):
    Fs, Xs = [], []
    for s in seeds:
        res = run_nsga2(problem, pop_size=POP, n_gen=GEN, seed=s)
        fm = feasible_mask(res.G)
        if fm is not None:
            Fs.append(res.F[fm])
            Xs.append(res.X[fm])
        else:
            Fs.append(res.F)
            Xs.append(res.X)
    F, X = np.vstack(Fs), np.vstack(Xs)
    keep = nondominated_mask(F)
    F, X = F[keep], X[keep]
    _, idx = np.unique(np.round(F, 10), axis=0, return_index=True)
    idx = np.sort(idx)
    F, X = F[idx], X[idx]
    order = np.argsort(F[:, 0])
    return F[order], X[order]


def memetic_refine(problem, F, X):
    """ASF 逐点精修 + 两端点单目标精修 → 合并非支配。"""
    Fp, Xp = polish_front(problem, X, F)
    pts_f, pts_x = [Fp], [Xp]
    for idx in (0, 1):
        k = int(np.argmin(Fp[:, idx]))
        x, fv, ok = endpoint_polish(problem, idx, Xp[k])
        if ok:
            from src.nsga2.local_search import eval_fg
            Fx, _ = eval_fg(problem, x)
            pts_f.append(Fx[None, :])
            pts_x.append(x[None, :])
    F2 = np.vstack(pts_f)
    X2 = np.vstack(pts_x)
    keep = nondominated_mask(F2)
    F2, X2 = F2[keep], X2[keep]
    _, idx = np.unique(np.round(F2, 10), axis=0, return_index=True)
    idx = np.sort(idx)
    F2, X2 = F2[idx], X2[idx]
    order = np.argsort(F2[:, 0])
    return F2[order], X2[order]


def evaluate_set(F, ref, hv_point):
    return {"n": int(F.shape[0]),
            "gd": gd(F, ref), "igd": igd(F, ref),
            "hv": hypervolume(F, hv_point)}


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    d = np.load(RESULTS / "06_ref_fronts.npz")
    with open(RESULTS / "01_problem_validation.json", encoding="utf-8") as f:
        v1 = json.load(f)

    problems = {
        "wb": (get_problem("welded_beam"), d["wb_F"], None),
        "pv": (PressureVessel(), d["pv_F"], v1["pressure_vessel"]),
        "sr": (SpeedReducer(), d["sr_F"], v1["speed_reducer"]),
    }
    out: dict = {"config": {"pop": POP, "gen": GEN, "seeds": SEEDS,
                            "polish": "ASF(Tchebycheff)+endpoint SLSQP"}}

    panels_front, panels_bar = [], []
    for tag, (prob, ref_F, v1p) in problems.items():
        t1 = time.time()
        Fraw, Xraw = pool_front(prob)
        Fpol, Xpol = memetic_refine(prob, Fraw, Xraw)
        assert nondominated_mask(Fpol).all(), f"{tag}: polished front not ND"

        # HV 参考点：真前沿 nadir 与两侧解集 max 的更大者 ×1.15（raw/polished 同口径）
        hv_point = np.maximum(np.maximum(ref_F.max(axis=0), Fraw.max(axis=0)),
                              Fpol.max(axis=0)) * 1.15
        m_raw = evaluate_set(Fraw, ref_F, hv_point)
        m_pol = evaluate_set(Fpol, ref_F, hv_point)

        row = {"raw": m_raw, "polished": m_pol, "hv_ref": hv_point.tolist()}
        if v1p is not None:
            k = "min_cost" if tag == "pv" else "min_weight"
            k2 = "min_weight" if tag == "pv" else "min_stress"
            ref1 = v1p[k]["f_min"]
            ref2 = v1p[k2]["f_min"]
            row["endpoint_gap_pct"] = {
                "raw_f1": abs(Fraw[0, 0] / ref1 - 1) * 100,
                "raw_f2": abs(Fraw[-1, 1] / ref2 - 1) * 100,
                "pol_f1": abs(Fpol[0, 0] / ref1 - 1) * 100,
                "pol_f2": abs(Fpol[-1, 1] / ref2 - 1) * 100,
            }
        out[tag] = row
        eg = row.get("endpoint_gap_pct", {})
        print(f"[{tag}] IGD raw {m_raw['igd']:.4e} → pol {m_pol['igd']:.4e} | "
              f"HV {m_raw['hv']:.6g} → {m_pol['hv']:.6g} | "
              f"endpoints {eg.get('raw_f1', float('nan')):.2f}→"
              f"{eg.get('pol_f1', float('nan')):.2f}% / "
              f"{eg.get('raw_f2', float('nan')):.2f}→"
              f"{eg.get('pol_f2', float('nan')):.2f}%  ({time.time()-t1:.0f}s)")
        panels_front.append((tag, ref_F, Fraw, Fpol, Xpol))
        panels_bar.append((tag, m_raw, m_pol))
        np.savez(RESULTS / f"07_front_{tag}.npz",
                 raw_F=Fraw, raw_X=Xraw, pol_F=Fpol, pol_X=Xpol)

    # 图：上排前沿对比，下排指标条形
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    names = {"wb": "welded beam", "pv": "pressure vessel", "sr": "speed reducer"}
    for j, (tag, ref_F, Fraw, Fpol, Xpol) in enumerate(panels_front):
        ax = axes[0, j]
        ax.plot(ref_F[:, 0], ref_F[:, 1], "k-", lw=1.2, label="e-constraint ref")
        ax.plot(Fraw[:, 0], Fraw[:, 1], "o", ms=4, mfc="none", mec="#1f77b4",
                label="NSGA-II raw")
        ax.plot(Fpol[:, 0], Fpol[:, 1], "s", ms=3.5, color="#d62728",
                label="+ ASF polish")
        ax.set_title(names[tag], fontsize=9)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend(fontsize=7)
    metrics = [("IGD", "igd", True), ("HV", "hv", False), ("GD", "gd", True)]
    for j, (mname, key, logy) in enumerate(metrics):
        ax = axes[1, j]
        tags = [t for t, _, _ in panels_bar]
        raw_v = [m_raw[key] for _, m_raw, _ in panels_bar]
        pol_v = [m_pol[key] for _, _, m_pol in panels_bar]
        xs = np.arange(len(tags))
        ax.bar(xs - 0.18, raw_v, 0.36, label="raw", color="#1f77b4")
        ax.bar(xs + 0.18, pol_v, 0.36, label="polished", color="#d62728")
        ax.set_xticks(xs)
        ax.set_xticklabels(tags, fontsize=8)
        ax.set_title(mname + (" (log)" if logy else " (larger better)"),
                     fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    save_fig("07_memetic_vs_nsga2.png")

    out["runtime_s"] = round(time.time() - t0, 1)
    save_json("07_memetic.json", out)
    print("saved: results/07_memetic.json + 07_front_*.npz + "
          "figures/07_memetic_vs_nsga2.png")


if __name__ == "__main__":
    main()
