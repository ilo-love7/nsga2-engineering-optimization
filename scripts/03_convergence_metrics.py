"""03 NSGA-II 调参与收敛指标：种群规模/代数 → GD/IGD/HV 收敛曲线。

问题：ZDT1（解析前沿，指标绝对可信）+ 焊接梁（官方参考前沿）。
口径：每配置独立 3 种子，指标取均值；HV 参考点固定（ZDT1 (1.1,1.1)，
焊接梁用参考前沿 max×1.1）。
"""
from __future__ import annotations

import numpy as np
from pymoo.problems import get_problem

from common import save_fig, save_json
from src.nsga2.metrics import gd, hv_ref_point, hypervolume, igd
from src.nsga2.runner import run_nsga2

POPS = [40, 100, 200]
GENS = [25, 50, 100, 200]
SEEDS = [1, 2, 3]


def sweep(problem, ref, hv_ref):
    """种群 × 代数 网格扫描，返回记录列表。"""
    rows = []
    for pop in POPS:
        for gen in GENS:
            g, i, h = [], [], []
            for s in SEEDS:
                res = run_nsga2(problem, pop_size=pop, n_gen=gen, seed=s)
                F = res.F
                # 只统计可行解（约束问题）
                if res.G is not None:
                    F = F[(res.G <= 1e-6).all(axis=1)]
                if F.shape[0] == 0:
                    continue
                g.append(gd(F, ref))
                i.append(igd(F, ref))
                h.append(hypervolume(F, hv_ref))
            rows.append({
                "pop": pop, "gen": gen,
                "gd": float(np.mean(g)), "igd": float(np.mean(i)),
                "hv": float(np.mean(h)),
            })
            print(f"  pop={pop:4d} gen={gen:4d}  GD={np.mean(g):.2e}  "
                  f"IGD={np.mean(i):.2e}  HV={np.mean(h):.4f}")
    return rows


def main() -> None:
    import matplotlib.pyplot as plt

    out: dict = {}

    # ---------- ZDT1 ----------
    zdt1 = get_problem("zdt1")
    ref1 = zdt1.pareto_front()
    print("ZDT1:")
    out["zdt1"] = sweep(zdt1, ref1, np.array([1.1, 1.1]))

    # ---------- 焊接梁 ----------
    wb = get_problem("welded_beam")
    ref_wb = wb.pareto_front()
    hv_ref_wb = hv_ref_point(ref_wb, 1.1)
    print("welded_beam:")
    out["welded_beam"] = sweep(wb, ref_wb, hv_ref_wb)

    # ---------- 收敛图：IGD vs gen（各 pop） ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, key, title in [(axes[0], "zdt1", "ZDT1"),
                           (axes[1], "welded_beam", "welded beam")]:
        for pop in POPS:
            rows = [r for r in out[key] if r["pop"] == pop]
            rows.sort(key=lambda r: r["gen"])
            ax.plot([r["gen"] for r in rows], [r["igd"] for r in rows],
                    "o-", ms=4, label=f"pop={pop}")
        ax.set_yscale("log")
        ax.set_xlabel("generations")
        ax.set_ylabel("IGD (log)")
        ax.set_title(f"{title}: IGD convergence")
        ax.legend(fontsize=8)
    save_fig("03_convergence.png")

    save_json("03_convergence.json", out)
    print("saved: results/03_convergence.json + figures/03_convergence.png")


if __name__ == "__main__":
    main()
