"""压力容器设计（经典 benchmark，Sandgren 一系）双目标建模。

变量 x = [Ts, Th, R, L]：
  Ts 圆柱壳厚 [in]，Th 半球封头厚 [in]，R 内半径 [in]，L 圆柱段长 [in]。
边界：Ts, Th ∈ [0.0625, 6.1875]（离散版为 0.0625 的整数倍，repair 处理），
      R ∈ [10, 200]，L ∈ [10, 200]。

目标（双目标 trade-off：制造成本 vs 结构重量）：
  f1 成本 = 0.6224·Ts·R·L + 1.7781·Th·R² + 3.1661·Ts²·L + 19.84·Ts²·R
      （材料 + 焊接/成形项，经典成本式）
  f2 重量 = ρ·(V_shell + V_head)，ρ = 0.2836 lb/in³（钢）
      V_shell = π·((R+Ts)² − R²)·L，V_head = (2/3)π·((R+Th)³ − R³)

约束（≤0，**归一化口径**：各约束除以量级常数，使判据容差 1e-6 对全约束统一有效；
      g3 量级 ~1.3e6，不归一化时活跃约束的浮点残差会误判不可行）：
  g1 = (−Ts + 0.0193·R) / 6.1875        壳厚满足内压
  g2 = (−Th + 0.00954·R) / 6.1875       封头厚满足内压
  g3 = (−π·R²·L − (4/3)π·R³ + 1296000) / 1296000   容积 ≥ 需求
  g4 = (L − 240) / 240                  长度上限
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

RHO_STEEL = 0.2836  # lb/in^3


class PressureVessel(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=4, vtype=float)
        self.xl = np.array([0.0625, 0.0625, 10.0, 10.0])
        self.xu = np.array([6.1875, 6.1875, 200.0, 200.0])

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        Ts, Th, R, L = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        f1 = (0.6224 * Ts * R * L + 1.7781 * Th * R**2
              + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R)
        v_shell = np.pi * ((R + Ts) ** 2 - R**2) * L
        v_head = (2.0 / 3.0) * np.pi * ((R + Th) ** 3 - R**3)
        f2 = RHO_STEEL * (v_shell + v_head)

        g1 = (-Ts + 0.0193 * R) / 6.1875
        g2 = (-Th + 0.00954 * R) / 6.1875
        g3 = (-np.pi * R**2 * L - (4.0 / 3.0) * np.pi * R**3 + 1_296_000.0) / 1_296_000.0
        g4 = (L - 240.0) / 240.0

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4])


def repair_pv(x: np.ndarray) -> np.ndarray:
    """离散修复：Ts/Th 取整到 0.0625 倍数，全部夹到边界。"""
    x = np.array(x, dtype=float, copy=True)
    x[..., 0] = np.round(x[..., 0] / 0.0625) * 0.0625
    x[..., 1] = np.round(x[..., 1] / 0.0625) * 0.0625
    lo = np.array([0.0625, 0.0625, 10.0, 10.0])
    hi = np.array([6.1875, 6.1875, 200.0, 200.0])
    return np.clip(x, lo, hi)


def single_objective_cost(x: np.ndarray) -> float:
    """经典单目标成本（仅 f1），用于端点文献对照。"""
    Ts, Th, R, L = [float(v) for v in np.asarray(x, dtype=float).ravel()[:4]]
    return (0.6224 * Ts * R * L + 1.7781 * Th * R**2
            + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R)
