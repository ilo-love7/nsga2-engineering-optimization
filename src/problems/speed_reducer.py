"""减速器齿轮箱设计（经典 7 变量 benchmark，Kurpati/Garg 一系）双目标建模。

变量 x = [b, m, z, l1, l2, d1, d2]：
  b  齿宽 [in]，m  模数 [in]，z  小齿轮齿数，l1/l2 两轴轴承跨距 [in]，
  d1/d2 两轴直径 [in]。
边界（与 g10/g11 相容性已验证）：b∈[2.6,3.6]，m∈[0.7,0.8]，z∈[17,28]
      （整数，repair 处理），l1,l2∈[7.3,8.3]，d1∈[2.9,3.9]，d2∈[5.0,5.5]。
      （口径说明：benchmark 原始边界表离线不可核；本实现取与经典解区域
       b≈3.5, m=0.7, z=17, l≈7.3–7.8, d1≈3.35, d2≈5.29 相容的自洽边界。）

目标：
  f1 重量（经典齿轮箱重量式）：
     0.7854·b·m²·(3.3333·z² + 14.933·z − 43.0934)
     − 1.508·b·(d1² + d2²) + 7.4777·(d1³ + d2³)
     + 0.7854·(l1·d1² + l2·d2²)
  f2 最大轴应力水平（工程 trade-off：轻量 vs 应力）：
     max( σ1, σ2 )，σi = sqrt((745·li/(m·z))² + ci) / di³
     （g5/g6 强度约束的分子部分，值越小越安全）

约束（≤0，经典 11 条）：
  g1 = 27/(b·m²·z) − 1               齿面接触强度
  g2 = 397.5/(b·m²·z²) − 1           齿根弯曲强度
  g3 = 1.93·l1³/(m·z·d1⁴) − 1        轴 1 挠度
  g4 = 1.93·l2³/(m·z·d2⁴) − 1        轴 2 挠度
  g5 = sqrt((745·l1/(m·z))² + 16.9e6)/(110·d1³) − 1
  g6 = sqrt((745·l2/(m·z))² + 157.5e6)/(85·d2³) − 1
  g7 = m·z/40 − 1                    大齿轮分度圆
  g8 = 5·m/b − 1                     齿宽比
  g9 = b/(12·m) − 1                  齿宽比
  g10 = (1.5·d1 + 1.9)/l1 − 1        轴 1 轴承跨距
  g11 = (1.1·d2 + 1.9)/l2 − 1        轴 2 轴承跨距
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

XL = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0])
XU = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])


class SpeedReducer(Problem):
    def __init__(self):
        super().__init__(n_var=7, n_obj=2, n_ieq_constr=11, vtype=float)
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

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


def repair_sr(x: np.ndarray) -> np.ndarray:
    """整数修复：齿数 z 取整，全部夹到边界。"""
    x = np.array(x, dtype=float, copy=True)
    x[..., 2] = np.round(x[..., 2])
    return np.clip(x, XL, XU)
