"""14 v1.3 E23：约束处理等预算对比——rules/death/static/dynamic/adaptive/eps。

口径（与 constraints 模块一致）：
- 问题：压力容器 PV（06 真参考）+ 双目标减速器 SR（06 真参考）+ 三目标
  减速器 SR3（10 标量化参考）；等预算 pop=100 × n_gen=200（20k NFE）、
  种子 1/2/3、同 SBX/PM 算子。
- 指标回原问题重评（原 F/G、原约束判可行 1e-6）：feas_rate / mean_viol /
  igd_norm / gd_norm / HV（hv_ref_point(ref, 1.1)）。
- 轨迹：逐代 mean_viol 与 feas_rate（六方法对比搜索引导差异）。

假设（报告 §23）：动态/自适应罚的可行率应优于固定权重（早期容忍不可行、
后期收紧）；但 Deb 规则仍应整体最优（v1.0 结论外推）。负结果候选：罚调度
不敏感（约束已归一化后权重不敏感）。
"""
from __future__ import annotations

import time

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.constraints import METHODS, run_handled
from src.nsga2.local_search import nondominated_mask
from src.nsga2.metrics import gd_norm, hv_ref_point, hypervolume, igd_norm
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer
from src.problems.speed_reducer3 import SpeedReducer3

SEEDS = [1, 2, 3]
POP, NGEN = 100, 200
TOL = 1e-6
# 七种配置：METHODS 六种 + dynamic_matched（终值权重配平 static 1e5 的对照，
# 回答"动态罚差是调度形状还是标度不足"）
CONFIGS = [(m, {}) for m in METHODS] + [("dynamic_matched", {"method": "dynamic", "c": 1.5})]
LABELS = [c[0] for c in CONFIGS]


def load_refs():
    d6 = np.load(RESULTS / "06_ref_fronts.npz")
    d10 = np.load(RESULTS / "10_sr3_ref.npz")
    return {
        "pv": (PressureVessel(), d6["pv_F"]),
        "sr": (SpeedReducer(), d6["sr_F"]),
        "sr3": (SpeedReducer3(), d10["F"]),
    }


def feas_metrics(F, G, ref, hv_point):
    feas = (G <= TOL).all(axis=1)
    n_feas = int(feas.sum())
    mean_viol = float(np.maximum(0.0, G).sum(axis=1).mean())
    if n_feas == 0:
        return {"feas_rate": 0.0, "mean_viol": mean_viol, "n_feas": 0,
                "igd_norm": float("nan"), "gd_norm": float("nan"),
                "hv": 0.0}
    Ff = F[feas]
    nd = nondominated_mask(Ff)
    Fnd = Ff[nd]
    return {
        "feas_rate": float(feas.mean()),
        "mean_viol": mean_viol,
        "n_feas": n_feas,
        "igd_norm": igd_norm(Fnd, ref),
        "gd_norm": gd_norm(Fnd, ref),
        "hv": hypervolume(Fnd, hv_point),
    }


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    refs = load_refs()
    out = {"config": {"seeds": SEEDS, "pop": POP, "n_gen": NGEN,
                      "nfe": POP * NGEN, "tol": TOL,
                      "methods": list(METHODS)},
           "problems": {}}

    traces = {}   # (prob, method) -> list of trace lists
    for pname, (prob, ref) in refs.items():
        hv_point = hv_ref_point(ref, scale=1.1)
        out["problems"][pname] = {}
        for label, params in CONFIGS:
            method = params.get("method", label)
            kw = {k: v for k, v in params.items() if k != "method"}
            runs = []
            for seed in SEEDS:
                print(f"[14] {pname} {label} seed={seed}", flush=True)
                r = run_handled(prob, method, pop_size=POP, n_gen=NGEN, seed=seed, **kw)
                m = feas_metrics(r["F"], r["G"], ref, hv_point)
                m["seed"] = seed
                runs.append(m)
                traces.setdefault((pname, label), []).append(r["trace"])
            agg = {}
            for key in ("feas_rate", "mean_viol", "igd_norm", "gd_norm", "hv"):
                vals = np.array([x[key] for x in runs], dtype=float)
                agg[key + "_mean"] = float(np.nanmean(vals))
                agg[key + "_std"] = float(np.nanstd(vals))
            out["problems"][pname][label] = {"runs": runs, "agg": agg}
        # 逐配置横向汇总（按 IGD 排序）
        ranking = sorted(LABELS,
                         key=lambda m: (out["problems"][pname][m]["agg"]["igd_norm_mean"]
                                        != out["problems"][pname][m]["agg"]["igd_norm_mean"],
                                        out["problems"][pname][m]["agg"]["igd_norm_mean"]))
        out["problems"][pname]["ranking_by_igd"] = ranking

    # ---- 初始采样可行域覆盖诊断（解释 death 悬崖）----
    rng = np.random.default_rng(0)
    samp_diag = {}
    for pname, (prob, _) in refs.items():
        counts = []
        for _ in range(20):
            X = rng.random((100, prob.n_var)) * (prob.xu - prob.xl) + prob.xl
            _, G = prob.evaluate(X, return_values_of=["F", "G"])
            counts.append(int((G <= TOL).all(axis=1).sum()))
        samp_diag[pname] = {"feasible_per_100_mean": float(np.mean(counts)),
                            "min": int(min(counts)), "max": int(max(counts))}
        print(f"[14] sampling {pname}: {samp_diag[pname]}", flush=True)
    out["sampling_feasibility"] = samp_diag

    out["runtime_s"] = time.time() - t0
    save_json("14_constraint_handling.json", out)

    # ---- 图 1：mean_viol 轨迹（3 问题 × 6 方法，多种子均值）----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (pname, _) in zip(axes, refs.items()):
        for method in METHODS:
            ts = traces.get((pname, method), [])
            if not ts:
                continue
            L = min(len(t) for t in ts)
            mat = np.array([[rec["mean_viol"] for rec in t[:L]] for t in ts])
            ax.plot(np.arange(1, L + 1), mat.mean(axis=0), label=method)
        ax.set_title(f"{pname}: mean violation")
        ax.set_xlabel("generation")
        ax.set_yscale("symlog", linthresh=1e-6)
    axes[-1].legend(fontsize=7, ncol=2)
    save_fig("14_constraint_viol_trace.png")

    # ---- 图 2：feas_rate 轨迹 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (pname, _) in zip(axes, refs.items()):
        for method in METHODS:
            ts = traces.get((pname, method), [])
            if not ts:
                continue
            L = min(len(t) for t in ts)
            mat = np.array([[rec["feas_rate"] for rec in t[:L]] for t in ts])
            ax.plot(np.arange(1, L + 1), mat.mean(axis=0), label=method)
        ax.set_title(f"{pname}: feasibility rate")
        ax.set_xlabel("generation")
        ax.set_ylim(-0.02, 1.02)
    axes[-1].legend(fontsize=7, ncol=2)
    save_fig("14_constraint_feas_trace.png")

    # ---- 图 3：终态 IGD/HV 条形 ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (pname, _) in zip(axes, refs.items()):
        aggs = [out["problems"][pname][m]["agg"] for m in LABELS]
        xs = np.arange(len(LABELS))
        ax.bar(xs, [a["igd_norm_mean"] for a in aggs],
               yerr=[a["igd_norm_std"] for a in aggs], capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{pname}: IGD_norm (feasible ND)")
    save_fig("14_constraint_igd_bars.png")

    print(f"[14] done in {out['runtime_s']:.1f}s -> 14_constraint_handling.json")


if __name__ == "__main__":
    main()
