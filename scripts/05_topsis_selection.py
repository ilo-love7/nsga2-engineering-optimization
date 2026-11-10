"""05 TOPSIS 帕累托选点 + 全项目汇总（00_summary.json）。

对 02 的四条工程前沿做等权 TOPSIS，输出折中解的决策变量与目标值；
汇总 01–04 关键指标。
"""
from __future__ import annotations

import json

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.topsis import topsis

LABELS = {
    "wb": ("welded beam", ["h", "l", "t", "b"], ["cost", "deflection"]),
    "pv": ("pressure vessel", ["Ts", "Th", "R", "L"], ["cost", "weight"]),
    "sr": ("speed reducer", ["b", "m", "z", "l1", "l2", "d1", "d2"],
           ["weight", "max stress"]),
    "cl": ("clutch/brake", ["ri", "ro", "t", "F", "Z"], ["mass", "stop time"]),
}


def main() -> None:
    import matplotlib.pyplot as plt

    d = np.load(RESULTS / "02_fronts.npz")
    out: dict = {}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))

    for ax, tag in zip(axes.ravel(), ["wb", "pv", "sr", "cl"]):
        F, X = d[f"{tag}_F"], d[f"{tag}_X"]
        name, var_names, obj_names = LABELS[tag]
        r = topsis(F)
        k = r["index"]
        out[tag] = {
            "problem": name,
            "selected_F": r["selected_F"].tolist(),
            "selected_X": X[k].tolist(),
            "closeness_max": float(r["closeness"].max()),
            "var_names": var_names, "obj_names": obj_names,
        }
        print(f"{name:>15s}: TOPSIS x={np.round(X[k],4)}  "
              f"F={np.round(r['selected_F'],5)}  C={r['closeness'].max():.3f}")
        ax.plot(F[:, 0], F[:, 1], "o", ms=3.5, color="#1f77b4", label="front")
        ax.plot(F[k, 0], F[k, 1], "*", ms=14, color="#d62728", label="TOPSIS")
        ax.set_title(f"{name}: TOPSIS compromise", fontsize=9)
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.legend(fontsize=7)
    save_fig("05_topsis_selection.png")
    save_json("05_topsis_selection.json", out)

    # ---------- 汇总 ----------
    def load(n):
        with open(RESULTS / n, encoding="utf-8") as f:
            return json.load(f)

    v1, v2, v3, v4 = (load("01_problem_validation.json"),
                      load("02_pareto_fronts.json"),
                      load("03_convergence.json"),
                      load("04_constraint_handling.json"))
    summary = {
        "project": "N9 NSGA-II 工程多目标优化",
        "solver": "pymoo 0.6.2 NSGA-II (Deb et al. 2002)",
        "metrics": {
            "zdt1_igd": v2["zdt1"]["igd"], "zdt1_hv": v2["zdt1"]["hv"],
            "zdt2_igd": v2["zdt2"]["igd"], "zdt3_igd": v2["zdt3"]["igd"],
            "welded_beam_gd": v2["welded_beam"]["gd"],
            "welded_beam_igd": v2["welded_beam"]["igd"],
            "pv_endpoint_gap_pct": [v2["pressure_vessel"]["f1_gap_pct"],
                                    v2["pressure_vessel"]["f2_gap_pct"]],
            "sr_endpoint_gap_pct": [v2["speed_reducer"]["f1_gap_pct"],
                                    v2["speed_reducer"]["f2_gap_pct"]],
            "constraint_best_strategy": max(
                ["A_feasibility_rules", "B1_penalty_f1_1e5", "B2_penalty_all_1e5",
                 "C_death_penalty", "D_repair_plus_rules"],
                key=lambda k: (v4[k]["mean"]["hv"] if np.isfinite(v4[k]["mean"]["hv"])
                               else -np.inf)),
            "constraint_hv": {k: v4[k]["mean"]["hv"] for k in
                              ["A_feasibility_rules", "B1_penalty_f1_1e5",
                               "B2_penalty_all_1e5", "C_death_penalty",
                               "D_repair_plus_rules"]},
            "constraint_feas": {k: v4[k]["mean"]["feas_ratio"] for k in
                                ["A_feasibility_rules", "B1_penalty_f1_1e5",
                                 "B2_penalty_all_1e5", "C_death_penalty",
                                 "D_repair_plus_rules"]},
            "topsis": {k: out[k]["selected_F"] for k in ["wb", "pv", "sr", "cl"]},
            "single_obj_refs": {
                "pv_min_cost": v1["pressure_vessel"]["min_cost"]["f_min"],
                "sr_min_weight": v1["speed_reducer"]["min_weight"]["f_min"],
            },
        },
    }
    save_json("00_summary.json", summary)
    print("saved: results/05_topsis_selection.json + 00_summary.json + "
          "figures/05_topsis_selection.png")


if __name__ == "__main__":
    main()
