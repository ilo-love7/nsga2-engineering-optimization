"""08 v1.1：收敛过程统计——IGD-代数曲线（多种子）+ 等预算 pop/gen 对比。

A. IGD vs generation：ZDT1（解析前沿）与 PV（06 ε-约束参考前沿），
   10 种子，每代记录种群非支配集 IGD → mean ± min/max band。
B. 等预算对比：固定评估预算 NFE≈pop·gen，比较 (pop,gen) 组合的最终 IGD
   （5 种子均值）——验证「种群比代数划算」的 v1.0 结论在等预算下是否成立。
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.problems import get_problem

from common import RESULTS, save_fig, save_json
from src.nsga2.metrics import igd
from src.nsga2.runner import make_algorithm, nondominated_front
from pymoo.optimize import minimize

N_SEEDS = 10
POP, GEN = 100, 150
BUDGET = 15000
BUDGET_CONFIGS = [(75, 200), (100, 150), (150, 100), (200, 75)]
N_SEEDS_BUDGET = 5


def run_with_curve(problem, ref, pop, gen, seed):
    """跑 NSGA-II，逐代记录非支配集 IGD。"""
    curve = []
    alg = make_algorithm(problem, pop_size=pop)

    def cb(algorithm):
        F = algorithm.opt.get("F")
        if F is None:
            return
        Fr = nondominated_front(np.asarray(F, dtype=float))
        curve.append(igd(Fr, ref))

    minimize(problem, alg, ("n_gen", gen), seed=seed, callback=cb, verbose=False)
    return np.asarray(curve)


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    out: dict = {"config": {"n_seeds": N_SEEDS, "pop": POP, "gen": GEN,
                            "budget_nfe": BUDGET,
                            "budget_configs": BUDGET_CONFIGS,
                            "n_seeds_budget": N_SEEDS_BUDGET}}

    zdt1 = get_problem("zdt1")
    ref_zdt1 = np.asarray(zdt1.pareto_front(), dtype=float)
    d = np.load(RESULTS / "06_ref_fronts.npz")
    ref_pv = d["pv_F"]
    from src.problems.pressure_vessel import PressureVessel
    pv = PressureVessel()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for ax, (name, prob, ref) in zip(
            axes[0], [("zdt1 (analytic PF)", zdt1, ref_zdt1),
                      ("pressure vessel (e-ref PF)", pv, ref_pv)]):
        curves = []
        for s in range(N_SEEDS):
            c = run_with_curve(prob, ref, POP, GEN, seed=s + 1)
            curves.append(c)
        L = min(len(c) for c in curves)
        M = np.vstack([c[:L] for c in curves])
        gens = np.arange(1, L + 1)
        ax.plot(gens, M.mean(axis=0), "-", lw=1.8, color="#1f77b4",
                label="mean")
        ax.fill_between(gens, M.min(axis=0), M.max(axis=0),
                        color="#1f77b4", alpha=0.18, label="min/max")
        ax.set_yscale("log")
        ax.set_xlabel("generation")
        ax.set_ylabel("IGD")
        ax.set_title(f"{name}: IGD convergence ({N_SEEDS} seeds)", fontsize=9)
        ax.legend(fontsize=7)
        out[name.split()[0]] = {
            "igd_final_mean": float(M[:, -1].mean()),
            "igd_final_std": float(M[:, -1].std()),
            "igd_final_min": float(M[:, -1].min()),
            "igd_final_max": float(M[:, -1].max()),
            "curve_mean": M.mean(axis=0).tolist(),
            "curve_min": M.min(axis=0).tolist(),
            "curve_max": M.max(axis=0).tolist(),
        }
        print(f"{name}: final IGD {M[:, -1].mean():.3e} ± {M[:, -1].std():.2e}")

    # 等预算：NFE≈BUDGET 下不同 (pop,gen) 的最终 IGD
    for ax, (name, prob, ref) in zip(
            axes[1], [("zdt1", zdt1, ref_zdt1), ("pressure vessel", pv, ref_pv)]):
        labels, means, stds = [], [], []
        for pop, gen in BUDGET_CONFIGS:
            vals = []
            for s in range(N_SEEDS_BUDGET):
                c = run_with_curve(prob, ref, pop, gen, seed=s + 11)
                vals.append(c[-1])
            labels.append(f"p{pop}×g{gen}")
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            print(f"{name} {pop}x{gen} (nfe={pop*gen}): "
                  f"IGD {np.mean(vals):.3e} ± {np.std(vals):.1e}")
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=3, color="#ff7f0e",
               alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{name}: equal-budget ({BUDGET} nfe) final IGD",
                     fontsize=9)
        out.setdefault("equal_budget", {})[name] = {
            "configs": [list(c) for c in BUDGET_CONFIGS],
            "igd_mean": means, "igd_std": stds,
        }
    save_fig("08_convergence_stats.png")
    out["runtime_s"] = round(time.time() - t0, 1)
    save_json("08_convergence_stats.json", out)
    print("saved: results/08_convergence_stats.json + figures/08_convergence_stats.png")


if __name__ == "__main__":
    main()
