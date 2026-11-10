"""pymoo Clutch（离合器/盘式制动器族）修复封装。

pymoo 0.6.2 的 Clutch 声明 n_ieq_constr=19，但 _evaluate 只输出 8 列 G
（g9–g18 是变量界，已由 xl/xu 索引界处理）→ 形状 (n,8) vs (n,19) 直接抛异常。
此处子类化把 n_ieq_constr 修正为 8。整数索引变量 [ri, ro, t, F, Z]：
f1 = 离合器质量（π(ro²−ri²)·t·(Z+1)·ρ），f2 = 制动时间 T。
"""
from __future__ import annotations

from pymoo.problems.multi.clutch import Clutch as _Clutch


class Clutch(_Clutch):
    def __init__(self) -> None:
        super().__init__()
        self.n_ieq_constr = 8
