"""10 v1.2 E20：多算法等预算对比——NSGA-II vs NSGA-III vs MOEA/D。

口径（与 manyobj 模块一致）：
- 等 NFE 预算（("n_evals", budget)）+ 等种群（pop = len(ref_dirs)）。
- MOEA/D 原生不支持约束（pymoo 断言）→ 约束问题经 DeathPenaltyWrapper；
  NSGA-II/III 原生可行性规则。res.F/G 一律回原问题重评。
- 指标在可行非支配子集上算；多量纲问题用 igd_norm/gd_norm。

问题：ZDT1 / ZDT3（解析前沿）+ 压力容器（06 真参考）+ 三目标减速器
（标量化参考）。假设：2 目标三算法打平；3 目标 NSGA-III/MOEA-D 的
分布性（IGD）应优于 NSGA-II——非支配排序在 3 目标下选择压力崩塌。
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.problems import get_problem

from common import RESULTS, save_fig, save_json
from src.nsga2.local_search import nondominated_mask
from src.nsga2.manyobj import (
    das_dennis, pooled_front, run_moea, scalarized_ref_front,
)
from src.nsga2.metrics import (
    gd_norm, hv_ref_point, hypervolume, igd_norm, normalize_fronts,
)
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer3 import SpeedReducer3

BUDGET = 20000
SEEDS = [1, 2, 3]
ALGOS = ["nsga2", "nsga3", "moead"]


def load_pv_ref():
    d = np.load(RESULTS / "06_ref_fronts.npz")
    return d["pv_F"], d["pv_X"]


def build_sr3_ref():
    """三目标减速器标量化参考前沿（66 权重 + 二遍精修）。"""
    sr3 = SpeedReducer3()
    F, X = scalarized_ref_front(sr3, step=0.1, n_starts=2, seed=0, polish=True)
    np.savez(RESULTS / "10_sr3_ref.npz", F=F, X=X)
    return F, X


def feasible_subset(res):
    if res.G is None:
        return res.F, res.X
    fm = np.all(res.G <= 1e-6, axis=1)
    return res.F[fm], res.X[fm]


def metrics_vs_ref(F, ref, hv_point):
    Fn, Rn = normalize_fronts(F, ref)
    return {
        "n": int(F.shape[0]),
        "gd_norm": gd_norm(F, ref),
        "igd_norm": igd_norm(F, ref),
        "hv": hypervolume(F, hv_point),
    }


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    zdt1 = get_problem("zdt1")
    zdt3 = get_problem("zdt3")
    pv = PressureVessel()
    sr3 = SpeedReducer3()

    pv_F, _ = load_pv_ref()
    print("[10] building SR3 scalarized reference front ...")
    sr3_F, sr3_X = build_sr3_ref()
    print(f"     SR3 ref: {sr3_F.shape[0]} points")

    problems = {
        "zdt1": (zdt1, zdt1.pareto_front(), 2),
        "zdt3": (zdt3, zdt3.pareto_front(), 2),
        "pv": (pv, pv_F, 2),
        "sr3": (sr3, sr3_F, 3),
    }

    out: dict = {"config": {"budget_nfe": BUDGET, "seeds": SEEDS,
                            "algos": ALGOS,
                            "moead_constrained": "DeathPenaltyWrapper",
                            "equal_pop": "pop = len(Das-Dennis ref_dirs)"}}
    panels_front, panels_conv = [], []

    for tag, (prob, ref, m) in problems.items():
        out[tag] = {}
        n_part = 99 if m == 2 else 12
        ref_dirs = das_dennis(m, n_part)
        # HV 参考点：真前沿 nadir×1.15（统一口径）
        hv_point = ref.max(axis=0) * 1.15

        pooled = {}
        for algo in ALGOS:
            t1 = time.time()
            rows, Fs, Xs = [], [], []
            for s in SEEDS:
                res = run_moea(prob, algo, budget=BUDGET, seed=s,
                               ref_dirs=ref_dirs, n_partitions=n_part)
                Ff, Xf = feasible_subset(res)
                rows.append(metrics_vs_ref(Ff, ref, hv_point))
                rows[-1]["feas_rate"] = (
                    float(np.all(res.G <= 1e-6, axis=1).mean())
                    if res.G is not None else 1.0)
                rows[-1]["nd_frac"] = float(
                    nondominated_mask(Ff).mean()) if Ff.shape[0] else 0.0
                Fs.append(Ff)
                Xs.append(Xf)
            Fpool, Xpool = pooled_front(Fs, Xs)
            pooled[algo] = (Fpool, Xpool)
            agg = {}
            for key in ("gd_norm", "igd_norm", "hv", "feas_rate", "nd_frac"):
                vals = [r[key] for r in rows]
                agg[key + "_mean"] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))
            agg["pool"] = metrics_vs_ref(Fpool, ref, hv_point)
            out[tag][algo] = {"per_seed": rows, "agg": agg}
            print(f"[{tag}] {algo:6s} IGD_norm {agg['igd_norm_mean']:.4e}"
                  f"±{agg['igd_norm_std']:.1e}  HV {agg['hv_mean']:.6g}"
                  f"  nd_frac {agg['nd_frac_mean']:.2f}"
                  f"  feas {agg['feas_rate_mean']:.2f}  ({time.time()-t1:.0f}s)")
        panels_front.append((tag, ref, pooled, m))

    # 收敛曲线（seed=1，ZDT1 + SR3，20 检查点）
    for tag in ("zdt1", "sr3"):
        prob, ref, m = problems[tag]
        n_part = 99 if m == 2 else 12
        ref_dirs = das_dennis(m, n_part)
        for algo in ALGOS:
            res = run_moea(prob, algo, budget=BUDGET, seed=1, ref_dirs=ref_dirs,
                           n_partitions=n_part, save_history=True)
            npts, igds = [], []
            hist = res.history
            idxs = np.unique(np.linspace(0, len(hist) - 1, 20).astype(int))
            for i in idxs:
                h = hist[i]
                X = np.atleast_2d(h.pop.get("X"))
                r = prob.evaluate(X, return_values_of=["F", "G"])
                F = np.asarray(r[0], dtype=float)
                if prob.n_ieq_constr:
                    G = np.asarray(r[1], dtype=float)
                    F = F[np.all(G <= 1e-6, axis=1)]
                if F.shape[0] == 0:
                    continue
                keep = nondominated_mask(F)
                npts.append(int(h.evaluator.n_eval))
                igds.append(igd_norm(F[keep], ref))
            panels_conv.append((tag, algo, npts, igds))

    # ---------------- 图 1：前沿对比（2×2） ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"nsga2": "#1f77b4", "nsga3": "#d62728", "moead": "#2ca02c"}
    for j, (tag, ref, pooled, m) in enumerate(panels_front):
        ax = axes[j // 2, j % 2]
        if m == 2:
            ax.plot(ref[:, 0], ref[:, 1], "k-", lw=1.2, label="reference")
            for algo, (Fp, _) in pooled.items():
                ax.plot(Fp[:, 0], Fp[:, 1], "o", ms=3.5, mfc="none",
                        mec=colors[algo], label=algo)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
        else:
            ax.plot(ref[:, 0], ref[:, 1], "k-", lw=1.0, alpha=0.5,
                    label="reference (f1-f2)")
            for algo, (Fp, _) in pooled.items():
                ax.plot(Fp[:, 0], Fp[:, 1], "o", ms=3.5, mfc="none",
                        mec=colors[algo], label=algo)
            ax.set_xlabel("f1 weight")
            ax.set_ylabel("f2 stress")
        ax.set_title(f"{tag} (pooled front)", fontsize=9)
        ax.legend(fontsize=7)
    save_fig("10_algorithm_fronts.png")

    # ---------------- 图 2：收敛曲线 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, ax in enumerate(axes):
        tag = ("zdt1", "sr3")[k]
        for t, algo, npts, igds in panels_conv:
            if t == tag:
                ax.loglog(npts, igds, "o-", ms=3.5, lw=1.5,
                          color=colors[algo], label=algo)
        ax.set_xlabel("NFE")
        ax.set_ylabel("IGD (normalized)")
        ax.set_title(f"{tag}: convergence", fontsize=9)
        ax.legend(fontsize=8)
    save_fig("10_algorithm_convergence.png")

    out["runtime_s"] = round(time.time() - t0, 1)
    save_json("10_algorithm_comparison.json", out)
    print("saved: results/10_algorithm_comparison.json + 10_sr3_ref.npz + "
          "figures/10_algorithm_*.png")


if __name__ == "__main__":
    main()
