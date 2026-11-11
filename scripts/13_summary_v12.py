"""13 v1.2 汇总：从 10/11/12 的 JSON 生成 v12_summary_tables.md + 关键指标打印。"""
from __future__ import annotations

import json

from common import RESULTS


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    d10 = load("10_algorithm_comparison.json")
    d11 = load("11_three_objective.json")
    d12 = load("12_robust_optimization.json")

    lines = ["# N9 v1.2 汇总表（自动生成 by 13_summary_v12.py）", ""]

    # ---------------- E20 ----------------
    lines += ["## E20 多算法等预算对比（NFE=%d）" % d10["config"]["budget_nfe"], ""]
    lines += ["| 问题 | 算法 | IGD_norm (mean±std) | HV (mean) | 可行率 | 非支配占比 |",
              "|---|---|---|---|---|---|"]
    for tag in ("zdt1", "zdt3", "pv", "sr3"):
        for algo in ("nsga2", "nsga3", "moead"):
            a = d10[tag][algo]["agg"]
            lines += [
                "| {tag} | {algo} | {ig:.3e} ± {igs:.1e} | {hv:.6g} | {fr:.2f} | {nd:.2f} |".format(
                    tag=tag, algo=algo, ig=a["igd_norm_mean"], igs=a["igd_norm_std"],
                    hv=a["hv_mean"], fr=a["feas_rate_mean"], nd=a["nd_frac_mean"])]
    lines += [""]

    # ---------------- E21 ----------------
    lines += ["## E21 三目标减速器", ""]
    lines += ["| 算法 | n | IGD_norm | HV |", "|---|---|---|---|"]
    for algo, r in d11["algorithms"].items():
        lines.append(f"| {algo} | {r['n']} | {r['igd_norm']:.4e} | {r['hv']:.6g} |")
    lines += ["", "### NSGA-III 参考方向密度扫描", "",
              "| n_dirs | IGD_norm | HV | 前沿点数 |", "|---|---|---|---|"]
    for r in d11["ref_dir_scan"]:
        lines.append(
            "| {nd} | {ig:.4e} ± {igs:.1e} | {hv:.6g} | {n:.0f} |".format(
                nd=r["n_ref_dirs"], ig=r["igd_norm_mean"], igs=r["igd_norm_std"],
                hv=r["hv_mean"], n=r["n_front_mean"]))
    tp = d11["topsis"]
    tov = d11["third_objective_value"]
    lines += [
        "",
        f"- TOPSIS 等权选点 F={ [round(v,2) for v in tp['selected_F']] } "
        f"C*={tp['closeness']:.3f}；±10% 权重扰动稳定率 {tp['perturb_stability']:.2f}，"
        f"跳变中位 {tp['perturb_jump_mean']:.3f} / 最大 {tp['perturb_jump_max']:.3f}",
        f"- 第三目标价值：2-obj 前沿 f3 中位 {tov['f3_median_on_2obj_front']:.1f} "
        f"vs 3-obj 前沿 f3 最小 {tov['f3_min_on_3obj_front']:.1f} "
        f"（省 {tov['f3_saving_pct_vs_2obj_median']:.1f}%）",
        "",
    ]

    # ---------------- E22 ----------------
    lines += [f"## E22 鲁棒优化（PV，±1% 公差，N={d12['config']['n_samples']} 样本）", ""]
    nr = d12["nominal_front_reliability"]
    lines += [
        f"- 名义前沿可行概率：min {nr['p_feas_min']:.3f} / "
        f"median {nr['p_feas_median']:.3f} / max {nr['p_feas_max']:.3f}；"
        f"<95% 占 {nr['frac_below_95pct']:.0%}，<50% 占 {nr['frac_below_50pct']:.0%}",
        "",
        "| 公差 | 名义最优点 p_feas |", "|---|---|",
    ]
    for s in d12["tol_sweep_nominal_optimum"]:
        lines.append(f"| {s['tol_frac']:.2%} | {s['p_feas']:.3f} |")
    lines += ["", "| p_fail_max | 前沿点数 | F1_min | F2_min |", "|---|---|---|---|"]
    for k, r in d12["reliability_fronts"].items():
        lines.append(f"| {k} | {r['n']} | {r['F_mean_min_f1']:.1f} | {r['F_mean_min_f2']:.0f} |")
    rp = d12["robustness_premium"]
    lines += [
        "",
        f"- 稳健溢价（TOPSIS 点，F_mean 口径）：f1 {rp['premium_f1_pct']:+.1f}% / "
        f"f2 {rp['premium_f2_pct']:+.1f}%；名义 TOPSIS p_feas={rp['nominal_topsis_p_feas']:.3f} → "
        f"稳健 TOPSIS p_feas={rp['robust_topsis_p_feas']:.3f}",
        "",
    ]

    text = "\n".join(lines) + "\n"
    p = RESULTS / "v12_summary_tables.md"
    p.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved: {p}")


if __name__ == "__main__":
    main()
