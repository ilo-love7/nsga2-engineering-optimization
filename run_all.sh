#!/usr/bin/env bash
# N9 NSGA-II 一键复现：01–05 实验 → 单测 → 汇总
set -euo pipefail
cd "$(dirname "$0")/.."
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
echo "=== pytest ==="
"$PY" -m pytest tests/ -q
echo "ALL DONE"
