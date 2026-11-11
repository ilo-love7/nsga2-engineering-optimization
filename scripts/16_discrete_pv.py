"""16 v1.3 E25：PV 离散版整数 NSGA-II——三管线对比 + 网格分辨率扫描。

口径（与 discrete 模块一致）：
- 管线：eval_repair（v1.0 D 算子观）/ op_repair（算子级修复）/
  grid_ops（网格感知算子）；grid=0.0625、pop=100 × n_gen=200、种子 1/2/3。
- 网格扫描：grid ∈ {0.0625, 0.125, 0.25, 0.5} × grid_ops × 3 种子。
- 指标回原问题重评（X 先 quantize）：HV / igd_norm / gd_norm（连续参考
  06_ref_fronts.npz pv_F）/ feas_rate / 网格占据（Ts/Th 唯一水平数）/
  修复位移 mean|X_raw − X|。
- 量化 Δviol：对连续参考前沿 pv_X 逐点量化后的约束违反增量。

假设（报告 §25）：grid_ops 在离散空间分布更均（占据水平多）、修复位移
为零，IGD/HV 不劣于 op_repair；但 v1.0 已证离散化 HV 损失是物理性的，
任何管线都消不掉。网格粗化 → HV 损失单调增。
"""
from __future__ import annotations

import time

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.discrete import (
    PIPELINES,
    quantize_to_grid,
    quantize_violation_delta,
    run_grid_nsga2,
)
from src.nsga2.local_search import nondominated_mask
from src.nsga2.metrics import gd_norm, hv_ref_point, hypervolume, igd_norm
from src.problems.pressure_vessel import PressureVessel

SEEDS = [1, 2, 3]
POP, NGEN = 100, 200
GRID = 0.0625
GRID_SCAN = [0.0625, 0.125, 0.25, 0.5]
COLS = (0, 1)
TOL = 1e-6


def metrics(F, G, ref, hv_point, X=None, X_raw=None, grid=GRID):
    feas = (G <= TOL).all(axis=1)
    res = {"feas_rate": float(feas.mean())}
    if X_raw is not None and X is not None:
        res["repair_disp"] = float(np.mean(np.abs(X_raw - X)))
    if X is not None:
        res["levels_Ts"] = int(len(np.unique(np.round(X[:, 0] / grid))))
        res["levels_Th"] = int(len(np.unique(np.round(X[:, 1] / grid))))
    if not feas.any():
        res.update({"igd_norm": float("nan"), "gd_norm": float("nan"), "hv": 0.0})
        return res
    Ff = F[feas]
    Fnd = Ff[nondominated_mask(Ff)]
    res.update({"igd_norm": igd_norm(Fnd, ref), "gd_norm": gd_norm(Fnd, ref),
                "hv": hypervolume(Fnd, hv_point)})
    return res


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    d6 = np.load(RESULTS / "06_ref_fronts.npz")
    ref = d6["pv_F"]
    ref_X = d6["pv_X"]
    hv_point = hv_ref_point(ref, scale=1.1)
    pv = PressureVessel()

    out = {"config": {"seeds": SEEDS, "pop": POP, "n_gen": NGEN,
                      "grid": GRID, "grid_scan": GRID_SCAN,
                      "pipelines": list(PIPELINES), "tol": TOL},
           "pipelines": {}, "grid_scan": {}}

    # ---- 三管线对比（grid=0.0625）----
    for pipe in PIPELINES:
        runs = []
        for seed in SEEDS:
            print(f"[16] {pipe} grid={GRID} seed={seed}", flush=True)
            r = run_grid_nsga2(pv, pipe, grid=GRID, discrete_cols=COLS,
                               pop_size=POP, n_gen=NGEN, seed=seed)
            m = metrics(r["F"], r["G"], ref, hv_point,
                        X=r["X"], X_raw=r["X_raw"], grid=GRID)
            m["seed"] = seed
            runs.append(m)
        agg = {}
        for key in ("feas_rate", "igd_norm", "gd_norm", "hv",
                    "levels_Ts", "levels_Th", "repair_disp"):
            if key in runs[0]:
                vals = np.array([x[key] for x in runs], dtype=float)
                agg[key + "_mean"] = float(np.nanmean(vals))
                agg[key + "_std"] = float(np.nanstd(vals))
        out["pipelines"][pipe] = {"runs": runs, "agg": agg}

    # ---- 网格分辨率扫描（grid_ops）----
    for g in GRID_SCAN:
        runs = []
        for seed in SEEDS:
            print(f"[16] grid_scan grid={g} seed={seed}", flush=True)
            r = run_grid_nsga2(pv, "grid_ops", grid=g, discrete_cols=COLS,
                               pop_size=POP, n_gen=NGEN, seed=seed)
            m = metrics(r["F"], r["G"], ref, hv_point,
                        X=r["X"], X_raw=r["X_raw"], grid=g)
            m["seed"] = seed
            runs.append(m)
        agg = {}
        for key in ("feas_rate", "igd_norm", "gd_norm", "hv", "levels_Ts"):
            vals = np.array([x[key] for x in runs], dtype=float)
            agg[key + "_mean"] = float(np.nanmean(vals))
            agg[key + "_std"] = float(np.nanstd(vals))
        out["grid_scan"][str(g)] = {"runs": runs, "agg": agg}

    # ---- 量化 Δviol（连续参考前沿 pv_X）----
    out["quantize_delta"] = quantize_violation_delta(pv, ref_X, GRID, COLS)

    # ---- 连续参考的口径锚点（HV=1 归一）----
    ref_only = metrics(ref, np.zeros((ref.shape[0], 1)), ref, hv_point)
    out["continuous_ref_anchor"] = {
        "igd_norm": 0.0, "hv": ref_only["hv"], "note": "HV 归一锚点",
    }

    out["runtime_s"] = time.time() - t0
    save_json("16_discrete_pv.json", out)

    # ---- 图 1：三管线 HV/IGD 对比 ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    xs = np.arange(len(PIPELINES))
    for ax, key, title in ((axes[0], "hv", "HV (ref 1.1x)"),
                           (axes[1], "igd_norm", "IGD_norm"),
                           (axes[2], "feas_rate", "feasibility rate")):
        means = [out["pipelines"][p]["agg"][key + "_mean"] for p in PIPELINES]
        stds = [out["pipelines"][p]["agg"][key + "_std"] for p in PIPELINES]
        ax.bar(xs, means, yerr=stds, capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(PIPELINES, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
    save_fig("16_discrete_pipelines.png")

    # ---- 图 2：网格分辨率 → HV 损失 ----
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    gs = [float(g) for g in out["grid_scan"]]
    hv_means = [out["grid_scan"][str(g)]["agg"]["hv_mean"] for g in out["grid_scan"]]
    hv_stds = [out["grid_scan"][str(g)]["agg"]["hv_std"] for g in out["grid_scan"]]
    ax.errorbar(gs, hv_means, yerr=hv_stds, marker="o", capsize=3)
    ax.axhline(out["continuous_ref_anchor"]["hv"], ls="--", color="gray",
               label="continuous ref HV")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("grid size (in)")
    ax.set_ylabel("HV")
    ax.set_title("PV discrete: grid resolution vs HV")
    ax.legend(fontsize=8)
    save_fig("16_grid_resolution.png")

    print(f"[16] done in {out['runtime_s']:.1f}s -> 16_discrete_pv.json")


if __name__ == "__main__":
    main()
