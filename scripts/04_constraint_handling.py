"""04 约束处理策略对比（压力容器，4 约束）：

A 可行性规则（pymoo NSGA-II 默认，Deb 约束支配）
B1 静态罚函数-仅 f1（1e5·viol 加到 f1）——双目标结构性失效（见报告）
B2 静态罚函数-全目标（1e5·viol 加到每个目标）
C 死刑法（不可行目标置 1e12）
D 修复算子（Ts/Th 取 0.0625 倍数 + 夹界）+ 可行性规则

评价：最终代可行率（**修复后口径**：D 先 repair 再查原约束）、HV / IGD
（vs 02 池化参考前沿）。空可行集的种子记 NaN 不参与均值。
"""
from __future__ import annotations

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.constraint_strategies import (
    DeathPenaltyWrapper, PenaltyWrapper, RepairWrapper,
)
from src.nsga2.metrics import gd, hypervolume, igd
from src.nsga2.runner import feasible_mask, run_nsga2
from src.problems.pressure_vessel import PressureVessel, repair_pv

SEEDS = [1, 2, 3]
POP, GEN = 100, 150


def eval_strategy(problem, wrapped, name, ref_F, hv_ref, repair_fn=None):
    rows = []
    for s in SEEDS:
        res = run_nsga2(wrapped, pop_size=POP, n_gen=GEN, seed=s)
        X = res.X if repair_fn is None else repair_fn(res.X)
        # 原约束口径复核（修复策略在修复后的 X 上评估）
        _, G = problem.evaluate(X, return_values_of=["F", "G"])
        F, _ = problem.evaluate(X, return_values_of=["F", "G"])
        fm = feasible_mask(G)
        feas_ratio = float(fm.mean())
        F_feas = np.asarray(F)[fm]
        if F_feas.shape[0] == 0:
            rows.append({"feas_ratio": feas_ratio, "hv": float("nan"),
                         "igd": float("nan"), "gd": float("nan"), "n_front": 0})
            continue
        rows.append({
            "feas_ratio": feas_ratio,
            "hv": hypervolume(F_feas, hv_ref),
            "igd": igd(F_feas, ref_F),
            "gd": gd(F_feas, ref_F),
            "n_front": int(F_feas.shape[0]),
        })

    def mean_finite(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {"feas_ratio": mean_finite("feas_ratio"), "hv": mean_finite("hv"),
           "igd": mean_finite("igd"), "gd": mean_finite("gd"),
           "n_front": mean_finite("n_front")}
    print(f"{name:>26s}  feas={agg['feas_ratio']:.3f}  HV={agg['hv']:.4g}  "
          f"IGD={agg['igd']:.4g}  GD={agg['gd']:.4g}")
    return {"per_seed": rows, "mean": agg}


def main() -> None:
    import matplotlib.pyplot as plt

    d = np.load(RESULTS / "02_fronts.npz")
    ref_F = d["pv_F"]
    hv_ref = np.array(json_hv_ref())

    pv = PressureVessel()
    out = {"config": {"pop": POP, "gen": GEN, "seeds": SEEDS, "penalty_w": 1e5},
           "hv_ref": hv_ref.tolist()}

    print(f"{'strategy':>26s}  results (mean over 3 seeds)")
    out["A_feasibility_rules"] = eval_strategy(pv, pv, "A feasibility rules",
                                               ref_F, hv_ref)
    out["B1_penalty_f1_1e5"] = eval_strategy(pv, PenaltyWrapper(pv, 1e5, "f1"),
                                             "B1 penalty on f1 only", ref_F, hv_ref)
    out["B2_penalty_all_1e5"] = eval_strategy(pv, PenaltyWrapper(pv, 1e5, "all"),
                                              "B2 penalty on all objs", ref_F, hv_ref)
    out["C_death_penalty"] = eval_strategy(pv, DeathPenaltyWrapper(pv),
                                           "C death penalty", ref_F, hv_ref)
    out["D_repair_plus_rules"] = eval_strategy(
        pv, RepairWrapper(pv, repair_pv), "D repair + rules", ref_F, hv_ref,
        repair_fn=repair_pv)

    # 图：HV/IGD/feas 对比条形图
    names = ["A rules", "B1 pen f1", "B2 pen all", "C death", "D repair"]
    keys = ["A_feasibility_rules", "B1_penalty_f1_1e5", "B2_penalty_all_1e5",
            "C_death_penalty", "D_repair_plus_rules"]
    hvv = [out[k]["mean"]["hv"] for k in keys]
    ig = [out[k]["mean"]["igd"] for k in keys]
    fr = [out[k]["mean"]["feas_ratio"] for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, vals, ttl in [(axes[0], hvv, "HV (larger better)"),
                          (axes[1], ig, "IGD (smaller better)"),
                          (axes[2], fr, "feasible ratio")]:
        ax.bar(names, vals,
               color=["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"])
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=7)
        if ttl.startswith("IGD"):
            ax.set_yscale("log")
    save_fig("04_constraint_handling.png")

    save_json("04_constraint_handling.json", out)
    print("saved: results/04_constraint_handling.json + figures/04_constraint_handling.png")


def json_hv_ref():
    import json
    with open(RESULTS / "02_pareto_fronts.json", encoding="utf-8") as f:
        return json.load(f)["pressure_vessel"]["hv_ref"]


if __name__ == "__main__":
    main()
