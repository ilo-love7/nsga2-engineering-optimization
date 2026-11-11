"""12 v1.2 E22：鲁棒优化——制造公差下名义最优的脆弱性与稳健解的代价。

内容：
1. 公差扫描：名义前沿（06 真参考）逐点 p_feas(tol)，tol ∈ {0.25..5}%。
   假设（冒烟已证实）：名义最优贴约束边界 → ±1% 公差下 p_feas 崩塌。
2. 鲁棒前沿：RobustWrapper[reliability]（P[G≤0] ≥ 95%）+ NSGA-II
   vs 名义前沿；稳健解退到内点，F_mean 代价量化（"稳健溢价"）。
3. mean/worst 口径对照：三种鲁棒定义给出的前沿位置差异。
4. 可靠度-代价曲线：p_fail_max ∈ {1%, 5%, 10%, 20%} → 最优 F_mean。
"""
from __future__ import annotations

import time

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.local_search import nondominated_mask
from src.nsga2.manyobj import pooled_front
from src.nsga2.robust import (
    RobustWrapper, evaluate_scenarios, reliability, robust_stats,
    sample_perturbations,
)
from src.nsga2.runner import run_nsga2
from src.nsga2.topsis import topsis
from src.problems.pressure_vessel import PressureVessel

TOL = 0.01
N_SAMPLES = 400
SEEDS = [1, 2, 3]
P_FAIL_LIST = [0.01, 0.05, 0.10, 0.20]


def eval_F(problem, X):
    """pymoo 0.6 坑：return_values_of=['F'] 单项直接返回数组（不是 tuple）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    r = problem.evaluate(X, return_values_of=["F"])
    F = r[0] if isinstance(r, tuple) else r
    return np.atleast_2d(np.asarray(F, dtype=float))


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    pv = PressureVessel()
    xl, xu = pv.xl, pv.xu
    pert = sample_perturbations(N_SAMPLES, pv.n_var, TOL, xl, xu, seed=42)
    out: dict = {"config": {"tol_frac": TOL, "n_samples": N_SAMPLES,
                            "seeds": SEEDS, "p_fail_list": P_FAIL_LIST}}

    # ---------- 1. 名义前沿的可行概率 ----------
    d = np.load(RESULTS / "06_ref_fronts.npz")
    nom_F, nom_X = d["pv_F"], d["pv_X"]
    print(f"[12] 1/4 名义前沿 {nom_F.shape[0]} 点 × tol={TOL:.1%} ...")
    p_feas_list, f_mean_list = [], []
    for x in nom_X:
        st = robust_stats(pv, x, pert)
        p_feas_list.append(st["p_feas"])
        f_mean_list.append(st["F_mean"])
    p_feas_arr = np.array(p_feas_list)
    out["nominal_front_reliability"] = {
        "p_feas_min": float(p_feas_arr.min()),
        "p_feas_median": float(np.median(p_feas_arr)),
        "p_feas_max": float(p_feas_arr.max()),
        "frac_below_95pct": float(np.mean(p_feas_arr < 0.95)),
        "frac_below_50pct": float(np.mean(p_feas_arr < 0.50)),
    }
    print(f"   p_feas: min {p_feas_arr.min():.3f} / "
          f"median {np.median(p_feas_arr):.3f} / max {p_feas_arr.max():.3f}；"
          f"<95% 占 {np.mean(p_feas_arr < 0.95):.0%}")

    # 公差扫描：端点 x_nom（名义成本最优）+ 随机内点对照
    x_nom = np.array([0.7782, 0.3846, 40.32, 200.0])
    tol_sweep = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
    sweep = []
    for tol in tol_sweep:
        p = sample_perturbations(N_SAMPLES, pv.n_var, tol, xl, xu, seed=42)
        st = robust_stats(pv, x_nom, p)
        sweep.append({"tol_frac": tol, "p_feas": float(st["p_feas"]),
                      "F_worst": st["F_worst"].tolist(),
                      "F_mean": st["F_mean"].tolist()})
        print(f"   tol={tol:.2%}  p_feas={st['p_feas']:.3f}")
    out["tol_sweep_nominal_optimum"] = sweep

    # ---------- 2. 可靠度-代价曲线 ----------
    print("[12] 2/4 可靠度-代价（reliability 模式，NSGA-II）...")
    rel_fronts = {}
    for pf in P_FAIL_LIST:
        rw = RobustWrapper(pv, pert, mode="reliability", p_fail_max=pf)
        Fs, Xs = [], []
        for s in SEEDS:
            res = run_nsga2(rw, pop_size=100, n_gen=100, seed=s)
            fm = np.all(np.asarray(res.G) <= 1e-6, axis=1) \
                if res.G is not None else np.ones(res.F.shape[0], bool)
            Fs.append(res.F[fm])
            Xs.append(res.X[fm])
        Fp, Xp = pooled_front(Fs, Xs)
        rel_fronts[pf] = (Fp, Xp)
        out.setdefault("reliability_fronts", {})[str(pf)] = {
            "n": int(Fp.shape[0]),
            "F_mean_min_f1": float(Fp[:, 0].min()) if Fp.shape[0] else None,
            "F_mean_min_f2": float(Fp[:, 1].min()) if Fp.shape[0] else None,
        }
        print(f"   p_fail≤{pf:.0%}: n={Fp.shape[0]} "
              f"F1min={Fp[:, 0].min():.1f} F2min={Fp[:, 1].min():.0f}")

    # ---------- 3. mean / worst 口径对照 ----------
    print("[12] 3/4 mean / worst 口径对照 ...")
    mode_fronts = {}
    for mode in ("mean", "worst"):
        rw = RobustWrapper(pv, pert, mode=mode)
        Fs, Xs = [], []
        for s in SEEDS:
            res = run_nsga2(rw, pop_size=100, n_gen=100, seed=s)
            Fs.append(res.F)
            Xs.append(res.X)
        Fp, Xp = pooled_front(Fs, Xs)
        mode_fronts[mode] = (Fp, Xp)
        # 稳健解在名义口径下的表现 + 可行概率
        r = pv.evaluate(Xp, return_values_of=["F", "G"])
        F_nom = np.asarray(r[0], dtype=float)
        G_nom = np.asarray(r[1], dtype=float)
        feas_nom = float(np.mean(np.all(G_nom <= 1e-6, axis=1)))
        pfs = [robust_stats(pv, x, pert)["p_feas"] for x in Xp]
        out.setdefault("mode_fronts", {})[mode] = {
            "n": int(Fp.shape[0]),
            "nominal_feas_rate": feas_nom,
            "p_feas_mean": float(np.mean(pfs)),
            "p_feas_min": float(np.min(pfs)),
        }
        print(f"   {mode:5s}: n={Fp.shape[0]} 名义可行率 {feas_nom:.2f} "
              f"p_feas 均值 {np.mean(pfs):.3f}")

    # 稳健溢价：reliability(5%) 前沿 TOPSIS 点 vs 名义 TOPSIS 点
    t_nom = topsis(nom_F)
    F5, X5 = rel_fronts[0.05]
    t_rel = topsis(F5)
    k_nom, k_rel = int(t_nom["index"]), int(t_rel["index"])
    # 名义 TOPSIS 点的 F_mean 与稳健 TOPSIS 点的名义 F
    st_nom = robust_stats(pv, nom_X[k_nom], pert)
    F5_nominal = eval_F(pv, X5[k_rel][None, :])[0]
    out["robustness_premium"] = {
        "nominal_topsis_F": nom_F[k_nom].tolist(),
        "nominal_topsis_p_feas": float(st_nom["p_feas"]),
        "nominal_topsis_F_mean": st_nom["F_mean"].tolist(),
        "robust_topsis_F_mean": F5[k_rel].tolist(),
        "robust_topsis_nominal_F": F5_nominal.tolist(),
        "premium_f1_pct": float(
            (F5[k_rel][0] / max(st_nom["F_mean"][0], 1e-12) - 1) * 100),
        "premium_f2_pct": float(
            (F5[k_rel][1] / max(st_nom["F_mean"][1], 1e-12) - 1) * 100),
        "robust_topsis_p_feas": float(
            robust_stats(pv, X5[k_rel], pert)["p_feas"]),
    }
    print(f"   稳健溢价: f1 {out['robustness_premium']['premium_f1_pct']:+.1f}% "
          f"f2 {out['robustness_premium']['premium_f2_pct']:+.1f}%；"
          f"稳健点 p_feas="
          f"{out['robustness_premium']['robust_topsis_p_feas']:.3f}")

    # ---------- 4. 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    sc = ax.scatter(nom_F[:, 0], nom_F[:, 1], c=p_feas_arr, cmap="RdYlGn",
                    vmin=0, vmax=1, s=28, edgecolors="k", lw=0.4)
    plt.colorbar(sc, ax=ax, label="P(feasible) @ ±1%")
    ax.set_xlabel("f1 cost")
    ax.set_ylabel("f2 weight")
    ax.set_title("nominal front: reliability under tolerance", fontsize=9)

    ax = axes[0, 1]
    ax.plot([s["tol_frac"] * 100 for s in sweep],
            [s["p_feas"] for s in sweep], "o-", color="#d62728", lw=1.8)
    ax.axhline(0.95, color="gray", ls="--", lw=1.0, label="95% target")
    ax.set_xlabel("tolerance (% of range)")
    ax.set_ylabel("P(feasible)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("nominal optimum: fragility vs tolerance", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(nom_F[:, 0], nom_F[:, 1], "k-", lw=1.2, label="nominal front")
    for pf, (Fp, _) in rel_fronts.items():
        ax.plot(Fp[:, 0], Fp[:, 1], "o-", ms=3.5,
                label=f"reliability p_fail≤{pf:.0%}")
    ax.set_xlabel("f1 (robust F_mean)")
    ax.set_ylabel("f2 (robust F_mean)")
    ax.set_title("reliability-constrained robust fronts", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    names = ["nominal TOPSIS", "robust TOPSIS (5%)"]
    pfe = [out["robustness_premium"]["nominal_topsis_p_feas"],
           out["robustness_premium"]["robust_topsis_p_feas"]]
    ax.bar(names, pfe, color=["#1f77b4", "#d62728"])
    ax.axhline(0.95, color="gray", ls="--", lw=1.0)
    ax.set_ylabel("P(feasible) @ ±1%")
    ax.set_ylim(0, 1.1)
    ax.set_title("TOPSIS points: reliability comparison", fontsize=9)
    save_fig("12_robust_optimization.png")

    # 图 2：三口径前沿（在名义 F 空间）
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(nom_F[:, 0], nom_F[:, 1], "k-", lw=1.3, label="nominal front")
    for mode, (Fp, Xp) in mode_fronts.items():
        Fn = eval_F(pv, Xp)
        keep = nondominated_mask(Fn)
        Fn = Fn[keep]
        ax.plot(Fn[:, 0], Fn[:, 1], "o", ms=3.5, mfc="none",
                mec={"mean": "#1f77b4", "worst": "#2ca02c"}[mode],
                label=f"{mode} robust (nominal F)")
    F5p, X5p = rel_fronts[0.05]
    Fn5 = eval_F(pv, X5p)
    keep5 = nondominated_mask(Fn5)
    ax.plot(Fn5[keep5][:, 0], Fn5[keep5][:, 1], "s", ms=3.5, mfc="none",
            mec="#d62728", label="reliability 5% (nominal F)")
    ax.set_xlabel("f1 cost (nominal)")
    ax.set_ylabel("f2 weight (nominal)")
    ax.set_title("robust designs in nominal objective space", fontsize=9)
    ax.legend(fontsize=7.5)
    save_fig("12_robust_modes_nominal_view.png")

    out["runtime_s"] = round(time.time() - t0, 1)
    save_json("12_robust_optimization.json", out)
    print("saved: results/12_robust_optimization.json + figures/12_*.png")


if __name__ == "__main__":
    main()
