"""11 v1.2 E21：三目标减速器深化——NSGA-III 主场 + 参考方向密度 + 三目标选点。

内容：
1. NSGA-III / NSGA-II / MOEA-D 在 SR3 上的前沿对比（对 E20 的标量化参考）。
2. 参考方向密度扫描（n_partitions ∈ {6, 9, 12, 18, 24}）→ IGD/HV 分布性；
   假设：方向太少分布差、太多稀释选择压力——存在内点。
3. 三目标 TOPSIS 选点 + 权重敏感性（与 v1.1 §14 同口径：±10% 扰动稳定率）。
4. 第三目标的价值量化：双目标前沿（f1,f2）上的点在 f3 上的分布 vs 三目标
   前沿的 f3 分布——"加了体积轴后能省多少体积"。
"""
from __future__ import annotations

import time

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.local_search import nondominated_mask
from src.nsga2.manyobj import das_dennis, pooled_front, run_moea
from src.nsga2.metrics import hypervolume, igd_norm
from src.nsga2.topsis import topsis
from src.problems.speed_reducer3 import SpeedReducer3

BUDGET = 20000
SEEDS = [1, 2, 3]


def feasible_subset(res):
    if res.G is None:
        return res.F, res.X
    fm = np.all(res.G <= 1e-6, axis=1)
    return res.F[fm], res.X[fm]


def run_algo_front(prob, algo, ref_dirs, seeds=SEEDS, budget=BUDGET):
    Fs, Xs = [], []
    for s in seeds:
        res = run_moea(prob, algo, budget=budget, seed=s, ref_dirs=ref_dirs)
        Ff, Xf = feasible_subset(res)
        Fs.append(Ff)
        Xs.append(Xf)
    return pooled_front(Fs, Xs)


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    sr3 = SpeedReducer3()
    ref = np.load(RESULTS / "10_sr3_ref.npz")
    ref_F, ref_X = ref["F"], ref["X"]
    hv_point = ref_F.max(axis=0) * 1.15
    out: dict = {"config": {"budget_nfe": BUDGET, "seeds": SEEDS,
                            "ref": "10_sr3_ref.npz (66-weight ASF + polish)"}}

    # ---------- 1. 三算法对比 ----------
    print("[11] 1/3 三算法 SR3 对比 ...")
    rd12 = das_dennis(3, 12)
    fronts = {}
    for algo in ("nsga2", "nsga3", "moead"):
        Fp, Xp = run_algo_front(sr3, algo, rd12)
        fronts[algo] = (Fp, Xp)
        out.setdefault("algorithms", {})[algo] = {
            "n": int(Fp.shape[0]),
            "igd_norm": igd_norm(Fp, ref_F),
            "hv": hypervolume(Fp, hv_point),
        }
        print(f"   {algo:6s} n={Fp.shape[0]:3d} "
              f"IGD_norm={out['algorithms'][algo]['igd_norm']:.4e} "
              f"HV={out['algorithms'][algo]['hv']:.6g}")

    # ---------- 2. 参考方向密度扫描（NSGA-III） ----------
    print("[11] 2/3 参考方向密度扫描 ...")
    scan = []
    for p in (6, 9, 12, 18, 24):
        rd = das_dennis(3, p)
        igds, hvs, ns = [], [], []
        for s in SEEDS:
            res = run_moea(sr3, "nsga3", budget=BUDGET, seed=s, ref_dirs=rd)
            Ff, _ = feasible_subset(res)
            keep = nondominated_mask(Ff)
            Fk = Ff[keep]
            igds.append(igd_norm(Fk, ref_F))
            hvs.append(hypervolume(Fk, hv_point))
            ns.append(int(Fk.shape[0]))
        row = {"n_partitions": p, "n_ref_dirs": int(rd.shape[0]),
               "igd_norm_mean": float(np.mean(igds)),
               "igd_norm_std": float(np.std(igds)),
               "hv_mean": float(np.mean(hvs)), "hv_std": float(np.std(hvs)),
               "n_front_mean": float(np.mean(ns))}
        scan.append(row)
        print(f"   p={p:2d} dirs={rd.shape[0]:3d} "
              f"IGD={row['igd_norm_mean']:.4e}±{row['igd_norm_std']:.1e} "
              f"HV={row['hv_mean']:.6g}")
    out["ref_dir_scan"] = scan

    # ---------- 3. 三目标 TOPSIS + 权重敏感性 ----------
    print("[11] 3/3 TOPSIS-3D 与权重敏感性 ...")
    F3, X3 = fronts["nsga3"]
    t = topsis(F3)
    k = int(t["index"])
    out["topsis"] = {
        "selected_F": F3[k].tolist(), "selected_X": X3[k].tolist(),
        "closeness": float(t["closeness"][k]),
    }
    # ±10% 权重扰动稳定率（精确索引口径 + 跳变幅度，同 v1.1 §14）
    base_w = np.array([1.0, 1.0, 1.0])
    idxs, jumps = [], []
    Fn = (F3 - F3.min(axis=0)) / np.maximum(F3.max(axis=0) - F3.min(axis=0), 1e-12)
    for j in range(3):
        for sgn in (-1, 1):
            w = base_w.copy()
            w[j] *= 1.1 if sgn > 0 else 0.9
            tj = topsis(F3, weights=w)
            idxs.append(int(tj["index"]))
            jumps.append(float(np.linalg.norm(Fn[int(tj["index"])] - Fn[k])))
    stable = float(np.mean([i == k for i in idxs]))
    out["topsis"]["perturb_stability"] = stable
    out["topsis"]["perturb_jump_mean"] = float(np.mean(jumps))
    out["topsis"]["perturb_jump_max"] = float(np.max(jumps))
    print(f"   等权选点 F={np.round(F3[k], 3)} C*={t['closeness'][k]:.3f} "
          f"±10% 稳定率 {stable:.2f} 跳变中位 {np.median(jumps):.3f}")

    # ---------- 4. 第三目标的价值 ----------
    # 双目标 NSGA-II（同预算）的 f3 分布 vs 三目标前沿的 f3 分布
    from pymoo.core.problem import Problem

    class SR2obj(Problem):
        def __init__(self):
            super().__init__(n_var=7, n_obj=2, n_ieq_constr=11, vtype=float)
            self.xl, self.xu = sr3.xl.copy(), sr3.xu.copy()

        def _evaluate(self, x, out_, *a, **k):
            F, G = sr3.evaluate(x, return_values_of=["F", "G"])
            out_["F"] = np.asarray(F, dtype=float)[:, :2]
            out_["G"] = np.asarray(G, dtype=float)

    sr2 = SR2obj()
    F2p, X2p = run_algo_front(sr2, "nsga2", das_dennis(2, 99), budget=BUDGET)
    # 在三目标问题上重评双目标解的 f3
    # 坑（pymoo 0.6）：return_values_of=["F"] 单项时直接返回数组而非 tuple
    r = sr3.evaluate(np.atleast_2d(X2p), return_values_of=["F"])
    F2r = r[0] if isinstance(r, tuple) else r
    f3_of_2obj = np.atleast_2d(np.asarray(F2r, dtype=float))[:, 2]
    out["third_objective_value"] = {
        "f3_min_on_3obj_front": float(F3[:, 2].min()),
        "f3_min_on_2obj_front": float(np.min(f3_of_2obj)),
        "f3_median_on_2obj_front": float(np.median(f3_of_2obj)),
        "f3_saving_pct_vs_2obj_median":
            float((np.median(f3_of_2obj) / F3[:, 2].min() - 1) * 100),
    }
    print(f"   f3(2obj 中位)={np.median(f3_of_2obj):.1f} vs "
          f"f3(3obj 最小)={F3[:, 2].min():.1f}")

    # ---------------- 图 ----------------
    fig = plt.figure(figsize=(12, 4.2))
    pairs = [(0, 1, "f1 weight", "f2 stress"),
             (0, 2, "f1 weight", "f3 volume"),
             (1, 2, "f2 stress", "f3 volume")]
    for j, (a, b, la, lb) in enumerate(pairs):
        ax = fig.add_subplot(1, 3, j + 1)
        ax.plot(ref_F[:, a], ref_F[:, b], "k-", lw=1.0, alpha=0.45,
                label="ref")
        for algo, (Fp, _) in fronts.items():
            ax.plot(Fp[:, a], Fp[:, b], "o", ms=3.5, mfc="none",
                    mec={"nsga2": "#1f77b4", "nsga3": "#d62728",
                         "moead": "#2ca02c"}[algo], label=algo)
        ax.plot(F3[k, a], F3[k, b], "r*", ms=12, label="TOPSIS")
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.legend(fontsize=7)
    fig.suptitle("E21: speed reducer 3-objective fronts", fontsize=10)
    save_fig("11_three_objective_fronts.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ps = [r["n_ref_dirs"] for r in scan]
    axes[0].errorbar(ps, [r["igd_norm_mean"] for r in scan],
                     yerr=[r["igd_norm_std"] for r in scan], fmt="o-",
                     capsize=3, color="#d62728")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of reference directions")
    axes[0].set_ylabel("IGD (normalized)")
    axes[0].set_title("NSGA-III: ref-dir density vs accuracy", fontsize=9)
    axes[1].errorbar(ps, [r["hv_mean"] for r in scan],
                     yerr=[r["hv_std"] for r in scan], fmt="s-",
                     capsize=3, color="#1f77b4")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of reference directions")
    axes[1].set_ylabel("HV")
    axes[1].set_title("NSGA-III: ref-dir density vs coverage", fontsize=9)
    save_fig("11_ref_dir_sensitivity.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].hist(f3_of_2obj, bins=20, alpha=0.65, color="#1f77b4",
                 label="f3 of 2-obj front")
    axes[0].axvline(F3[:, 2].min(), color="crimson", lw=1.5,
                    label="f3 min on 3-obj front")
    axes[0].set_xlabel("f3 volume [in^3]")
    axes[0].set_title("third objective: hidden cost on 2-obj designs",
                      fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].bar(["2-obj median", "3-obj min"],
                [np.median(f3_of_2obj), F3[:, 2].min()],
                color=["#1f77b4", "#d62728"])
    axes[1].set_ylabel("f3 volume [in^3]")
    axes[1].set_title("volume saving via 3rd objective", fontsize=9)
    save_fig("11_third_objective.png")

    out["runtime_s"] = round(time.time() - t0, 1)
    save_json("11_three_objective.json", out)
    print("saved: results/11_three_objective.json + figures/11_*.png")


if __name__ == "__main__":
    main()
