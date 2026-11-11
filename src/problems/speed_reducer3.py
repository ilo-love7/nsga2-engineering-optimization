"""三目标减速器设计（v1.2 扩展）：重量 / 轴应力水平 / 体积。

变量、边界、11 条约束与双目标 SpeedReducer 完全一致（公式口径同源），
仅追加第三目标：
  f3 体积 = π/4·(b·(m·z)²·0.25 + l1·d1² + l2·d2²)
      （齿轮坯 + 两轴的特征体积 [in³]，工程 trade-off 的"占地/材料"轴）

动机（报告 §17）：NSGA-III/MOEA-D 的主场是 3+ 目标——非支配排序在 3 目标
下选择压力骤降（大量互不支配），参考方向/分解类算法应显出优势。双目标
问题上三算法应近似打平，作为对照。
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from .speed_reducer import XL, XU, repair_sr


class SpeedReducer3(Problem):
    """三目标：f1 重量 / f2 最大轴应力水平 / f3 特征体积（全为最小化）。"""

    def __init__(self):
        super().__init__(n_var=7, n_obj=3, n_ieq_constr=11, vtype=float)
        self.xl = XL.copy()
        self.xu = XU.copy()

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        b, m, z, l1, l2, d1, d2 = (x[:, i] for i in range(7))
        f1 = (0.7854 * b * m**2 * (3.3333 * z**2 + 14.933 * z - 43.0934)
              - 1.508 * b * (d1**2 + d2**2)
              + 7.4777 * (d1**3 + d2**3)
              + 0.7854 * (l1 * d1**2 + l2 * d2**2))
        s1 = np.sqrt((745.0 * l1 / (m * z)) ** 2 + 16.9e6) / d1**3
        s2 = np.sqrt((745.0 * l2 / (m * z)) ** 2 + 157.5e6) / d2**3
        f2 = np.maximum(s1, s2)
        f3 = (np.pi / 4.0) * (0.25 * b * (m * z) ** 2 + l1 * d1**2 + l2 * d2**2)

        g1 = 27.0 / (b * m**2 * z) - 1.0
        g2 = 397.5 / (b * m**2 * z**2) - 1.0
        g3 = 1.93 * l1**3 / (m * z * d1**4) - 1.0
        g4 = 1.93 * l2**3 / (m * z * d2**4) - 1.0
        g5 = np.sqrt((745.0 * l1 / (m * z)) ** 2 + 16.9e6) / (110.0 * d1**3) - 1.0
        g6 = np.sqrt((745.0 * l2 / (m * z)) ** 2 + 157.5e6) / (85.0 * d2**3) - 1.0
        g7 = m * z / 40.0 - 1.0
        g8 = 5.0 * m / b - 1.0
        g9 = b / (12.0 * m) - 1.0
        g10 = (1.5 * d1 + 1.9) / l1 - 1.0
        g11 = (1.1 * d2 + 1.9) / l2 - 1.0

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


def repair_sr3(x: np.ndarray) -> np.ndarray:
    """整数修复：齿数 z 取整，全部夹到边界（与双目标版一致）。"""
    return repair_sr(x)
