#!/usr/bin/env bash
# N9 NSGA-II 一键复现：01–05（v1.0）+ 06–09（v1.1）+ 10–13（v1.2）+ 14–17（v1.3）→ 单测
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

echo "=== 01 问题建模校验（单目标端点） ==="
"$PY" scripts/01_problem_validation.py
echo "=== 02 Pareto 前沿复现（ZDT + 4 工程问题） ==="
"$PY" scripts/02_pareto_fronts.py
echo "=== 03 收敛指标扫描（GD/IGD/HV，较长） ==="
"$PY" scripts/03_convergence_metrics.py
echo "=== 04 约束处理策略对比 ==="
"$PY" scripts/04_constraint_handling.py
echo "=== 05 TOPSIS 选点 + 汇总 ==="
"$PY" scripts/05_topsis_selection.py
echo "=== 06 e-约束参考前沿（v1.1） ==="
"$PY" scripts/06_ref_fronts.py
echo "=== 07 memetic 精修对比（v1.1） ==="
"$PY" scripts/07_memetic_vs_nsga2.py
echo "=== 08 收敛统计 + 等预算（v1.1，较长） ==="
"$PY" scripts/08_convergence_stats.py
echo "=== 09 TOPSIS 敏感性（v1.1） ==="
"$PY" scripts/09_topsis_sensitivity.py
echo "=== 10 多算法等预算对比（v1.2，长） ==="
"$PY" scripts/10_algorithm_comparison.py
echo "=== 11 三目标减速器深化（v1.2，长） ==="
"$PY" scripts/11_three_objective.py
echo "=== 12 鲁棒优化（v1.2） ==="
"$PY" scripts/12_robust_optimization.py
echo "=== 13 v1.2 汇总表 ==="
"$PY" scripts/13_summary_v12.py
echo "=== 14 约束处理对比（v1.3，长） ==="
"$PY" scripts/14_constraint_handling.py
echo "=== 15 ELECTRE/PROMETHEE 选点（v1.3） ==="
"$PY" scripts/15_outranking.py
echo "=== 16 PV 离散整数 NSGA-II（v1.3，长） ==="
"$PY" scripts/16_discrete_pv.py
echo "=== 17 v1.3 汇总表 ==="
"$PY" scripts/17_summary_v13.py
echo "=== pytest ==="
"$PY" -m pytest tests/ -q
echo "ALL DONE"
