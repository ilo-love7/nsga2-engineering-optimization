"""15 v1.3 E24：ELECTRE III / PROMETHEE II vs TOPSIS 偏好选点对比。

口径（与 outranking 模块一致）：
- 前沿：06 真参考（pv_F/sr_F/wb_F）+ SR3 标量化参考（10_sr3_ref）；
  全部视为连续 Pareto 近似。
- 方法：topsis / promethee（线性偏好，p=10% 极差）/ electre（q/p/v =
  2%/8%/25% 极差，净可信度流排序——非完整蒸馏，报告如实标注）。
- 稳定性协议（沿用 v1.1 §14 的双口径，防"稠密前沿索引稳定率"陷阱）：
  等权基准选点 + 100 次 ±10% 权重扰动 → ① index 稳定率（同索引占比）
  ② 跳变中位（选点 F 的归一化距离中位数）。
- 效率：uto_distance（选点到理想点的归一化距离）。
- 附加：三方法一致率（同索引/邻点）+ ELECTRE 阈值敏感性（q/p/v 网格 27 组合）。

假设（报告 §24）：三方法选点相近（同为妥协解）；ELECTRE 否决阈值可能
更拒极端点 → 更稳。负结果候选：方法间高度一致（换方法不改结论），或
ELECTRE 阈值反而引入新超参敏感性。
"""
from __future__ import annotations

import time

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.outranking import (
    criterion_ranges,
    electre3,
    promethee2,
    select_compromise,
    uto_distance,
)
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer
from src.problems.speed_reducer3 import SpeedReducer3

N_DRAWS = 100
PERTURB = 0.10
RNG_SEED = 20261111
METHODS = ("topsis", "promethee", "electre")


def load_fronts():
    d6 = np.load(RESULTS / "06_ref_fronts.npz")
    d10 = np.load(RESULTS / "10_sr3_ref.npz")
    return {
        "pv": d6["pv_F"],
        "sr": d6["sr_F"],
        "wb": d6["wb_F"],
        "sr3": d10["F"],
    }


def stability(F, method, rng, n_draws=N_DRAWS, perturb=PERTURB, **kw):
    """等权基准 + 权重扰动双口径稳定性。"""
    m = F.shape[1]
    base = select_compromise(method, F, None, **kw)
    base_idx = base["selected"]
    base_F = np.asarray(base["selected_F"], dtype=float)
    rng_scale = criterion_ranges(F)
    same = 0
    jumps = []
    idxs = []
    for _ in range(n_draws):
        w = 1.0 + rng.uniform(-perturb, perturb, size=m)
        r = select_compromise(method, F, w, **kw)
        idxs.append(r["selected"])
        if r["selected"] == base_idx:
            same += 1
        d = np.linalg.norm((np.asarray(r["selected_F"]) - base_F) / rng_scale)
        jumps.append(float(d))
    return {
        "base_index": base_idx,
        "base_F": base_F.tolist(),
        "uto": uto_distance(F, base_idx),
        "index_stability": same / n_draws,
        "jump_median": float(np.median(jumps)),
        "jump_max": float(np.max(jumps)),
        "n_unique_selected": int(len(set(idxs))),
    }


def main() -> None:
    import matplotlib.pyplot as plt

    t0 = time.time()
    fronts = load_fronts()
    out = {"config": {"n_draws": N_DRAWS, "perturb": PERTURB,
                      "methods": list(METHODS),
                      "electre_default": {"q_frac": 0.02, "p_frac": 0.08,
                                          "v_frac": 0.25},
                      "promethee_default": {"q_frac": 0.0, "p_frac": 0.10}},
           "fronts": {}}

    rng = np.random.default_rng(RNG_SEED)
    for fname, F in fronts.items():
        entry = {"n": int(F.shape[0]), "m": int(F.shape[1]), "methods": {}}
        for method in METHODS:
            print(f"[15] {fname} {method}", flush=True)
            entry["methods"][method] = stability(F, method, rng)
        # 三方法一致率（等权基准）
        idxs = {m: entry["methods"][m]["base_index"] for m in METHODS}
        Fs = {m: np.asarray(entry["methods"][m]["base_F"]) for m in METHODS}
        agree_same = sum(1 for a in METHODS for b in METHODS
                         if a < b and idxs[a] == idxs[b])
        pair_dists = []
        for a in METHODS:
            for b in METHODS:
                if a < b:
                    pair_dists.append(float(np.linalg.norm(
                        (Fs[a] - Fs[b]) / criterion_ranges(F))))
        entry["agreement"] = {
            "same_index_pairs": agree_same,
            "n_pairs": 3,
            "pair_dist_median": float(np.median(pair_dists)),
        }
        # ELECTRE 阈值敏感性：q/p/v 3×3×3 网格
        sens = []
        for qf in (0.01, 0.02, 0.04):
            for pf in (0.05, 0.08, 0.12):
                for vf in (0.15, 0.25, 0.35):
                    if not (qf <= pf <= vf):
                        continue
                    r = electre3(F, None, q_frac=qf, p_frac=pf, v_frac=vf)
                    sens.append({"q": qf, "p": pf, "v": vf,
                                 "selected": r["selected"]})
        sel_counts = {}
        for s in sens:
            sel_counts[s["selected"]] = sel_counts.get(s["selected"], 0) + 1
        entry["electre_threshold_sensitivity"] = {
            "n_settings": len(sens),
            "n_unique_selected": len(sel_counts),
            "selected_counts": {str(k): v for k, v in sel_counts.items()},
            "top_share": max(sel_counts.values()) / len(sens),
        }
        out["fronts"][fname] = entry

    out["runtime_s"] = time.time() - t0
    save_json("15_outranking.json", out)

    # ---- 图 1：稳定率双口径（index vs jump-median）----
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(fronts.keys())
    xs = np.arange(len(names))
    width = 0.25
    for k, method in enumerate(METHODS):
        axes[0].bar(xs + (k - 1) * width,
                    [out["fronts"][n]["methods"][method]["index_stability"]
                     for n in names], width, label=method)
        axes[1].bar(xs + (k - 1) * width,
                    [out["fronts"][n]["methods"][method]["jump_median"]
                     for n in names], width, label=method)
    for ax, title in zip(axes, ["index stability (trap-prone)",
                                "jump median (F distance)"]):
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    save_fig("15_outranking_stability.png")

    # ---- 图 2：ELECTRE 阈值敏感性 ----
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    shares = [out["fronts"][n]["electre_threshold_sensitivity"]["top_share"]
              for n in names]
    uniq = [out["fronts"][n]["electre_threshold_sensitivity"]["n_unique_selected"]
            for n in names]
    ax.bar(xs, shares, width=0.5, label="top-share")
    for x, u in zip(xs, uniq):
        ax.text(x, 0.02, f"unique={u}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_title("ELECTRE threshold sensitivity (top-share of selection)")
    save_fig("15_electre_threshold_sensitivity.png")

    print(f"[15] done in {out['runtime_s']:.1f}s -> 15_outranking.json")


if __name__ == "__main__":
    main()
