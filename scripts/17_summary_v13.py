"""17 v1.3 汇总表：从 14/15/16 JSON 生成 markdown 对照表（报告素材）。"""
from __future__ import annotations

import json

from common import RESULTS

FILES = {
    "d14": RESULTS / "14_constraint_handling.json",
    "d15": RESULTS / "15_outranking.json",
    "d16": RESULTS / "16_discrete_pv.json",
}


def fmt(x, spec=".4g"):
    if isinstance(x, float):
        if x != x:
            return "NaN"
        return format(x, spec)
    return str(x)


def main() -> None:
    d14 = json.loads(FILES["d14"].read_text(encoding="utf-8"))
    d15 = json.loads(FILES["d15"].read_text(encoding="utf-8"))
    d16 = json.loads(FILES["d16"].read_text(encoding="utf-8"))

    lines = ["# N9 v1.3 汇总表（E23/E24/E25）", ""]

    # ---- E23 ----
    lines += ["## E23 约束处理等预算对比（pop=100×gen=200，3 种子）", ""]
    for pname, pdata in d14["problems"].items():
        lines += [f"### 问题 {pname}", "",
                  "| 方法 | feas_rate | mean_viol | IGD_norm | GD_norm | HV |",
                  "|---|---|---|---|---|---|"]
        labels = [k for k in pdata.keys() if k != "ranking_by_igd"]
        for m in labels:
            a = pdata[m]["agg"]
            lines.append(
                f"| {m} | {fmt(a['feas_rate_mean'])}±{fmt(a['feas_rate_std'])}"
                f" | {fmt(a['mean_viol_mean'])}"
                f" | {fmt(a['igd_norm_mean'])}±{fmt(a['igd_norm_std'])}"
                f" | {fmt(a['gd_norm_mean'])}"
                f" | {fmt(a['hv_mean'])} |")
        lines += ["", f"IGD 排序（好→差）：{' < '.join(pdata['ranking_by_igd'])}", ""]

    # ---- E24 ----
    lines += ["## E24 ELECTRE / PROMETHEE vs TOPSIS（100 次 ±10% 权重扰动）", "",
              "| 前沿 | 方法 | base_idx | uto | index 稳定率 | 跳变中位 | 唯一选点数 |",
              "|---|---|---|---|---|---|---|"]
    for fname, e in d15["fronts"].items():
        for m, s in e["methods"].items():
            lines.append(
                f"| {fname} | {m} | {s['base_index']} | {fmt(s['uto'])}"
                f" | {fmt(s['index_stability'], '.2f')} | {fmt(s['jump_median'])}"
                f" | {s['n_unique_selected']} |")
    lines += ["", "| 前沿 | 三方法同索引对 | pair_dist 中位 | ELECTRE 唯一选点/阈值组 | top_share |",
              "|---|---|---|---|---|"]
    for fname, e in d15["fronts"].items():
        ag, es = e["agreement"], e["electre_threshold_sensitivity"]
        lines.append(
            f"| {fname} | {ag['same_index_pairs']}/{ag['n_pairs']}"
            f" | {fmt(ag['pair_dist_median'])}"
            f" | {es['n_unique_selected']}/{es['n_settings']} | {fmt(es['top_share'], '.2f')} |")
    lines.append("")

    # ---- E25 ----
    lines += [f"## E25 PV 离散整数 NSGA-II（grid={d16['config']['grid']}，3 种子）", "",
              "| 管线 | feas_rate | IGD_norm | HV | Ts 水平数 | Th 水平数 | 修复位移 |",
              "|---|---|---|---|---|---|---|"]
    for pipe, pdata in d16["pipelines"].items():
        a = pdata["agg"]
        lines.append(
            f"| {pipe} | {fmt(a['feas_rate_mean'])}"
            f" | {fmt(a['igd_norm_mean'])}±{fmt(a['igd_norm_std'])}"
            f" | {fmt(a['hv_mean'])}±{fmt(a['hv_std'])}"
            f" | {fmt(a['levels_Ts_mean'], '.3g')} | {fmt(a['levels_Th_mean'], '.3g')}"
            f" | {fmt(a['repair_disp_mean'])} |")
    lines += ["", f"连续参考 HV 锚点：{fmt(d16['continuous_ref_anchor']['hv'])}；"
              f"量化 Δviol（参考前沿）：mean {fmt(d16['quantize_delta']['mean_delta'])}, "
              f"max {fmt(d16['quantize_delta']['max_delta'])}, "
              f"frac_worse {fmt(d16['quantize_delta']['frac_worse'])}", "",
              "| grid | feas_rate | IGD_norm | HV | Ts 水平数 |", "|---|---|---|---|---|"]
    for g, pdata in d16["grid_scan"].items():
        a = pdata["agg"]
        lines.append(
            f"| {g} | {fmt(a['feas_rate_mean'])}"
            f" | {fmt(a['igd_norm_mean'])}±{fmt(a['igd_norm_std'])}"
            f" | {fmt(a['hv_mean'])}±{fmt(a['hv_std'])}"
            f" | {fmt(a['levels_Ts_mean'], '.3g')} |")
    lines += ["", f"runtime：E23 {fmt(d14['runtime_s'], '.0f')}s / "
              f"E24 {fmt(d15['runtime_s'], '.2f')}s / E25 {fmt(d16['runtime_s'], '.0f')}s", ""]

    out = RESULTS / "v13_summary_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
