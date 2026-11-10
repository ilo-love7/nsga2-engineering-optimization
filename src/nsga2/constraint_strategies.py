"""约束处理策略对比：可行性规则（默认）/ 罚函数 / 死刑 / 修复算子。"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem


class PenaltyWrapper(Problem):
    """静态罚函数：把约束违反量加权加进目标，问题转为无约束双目标。

    mode="f1"：只加到 f1（经典单目标罚的形式）——**双目标下结构性失效**：
    不可行但 f2 优于一切可行解的点永远不被可行解支配，罚多大都留在前沿。
    mode="all"：加到所有目标 → 足够大权重下可行解支配一切不可行点。
    """

    def __init__(self, problem: Problem, weight: float = 1e5, mode: str = "f1"):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=0, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem
        self.weight = float(weight)
        if mode not in ("f1", "all"):
            raise ValueError("mode 须为 f1 或 all")
        self.mode = mode

    def _evaluate(self, x, out, *args, **kwargs):
        F, G = self.inner.evaluate(x, return_values_of=["F", "G"])
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        viol = np.maximum(0.0, G).sum(axis=1)
        F2 = F + self.weight * viol[:, None]
        if self.mode == "f1":
            F2 = F.copy()
            F2[:, 0] = F[:, 0] + self.weight * viol
        out["F"] = F2


class DeathPenaltyWrapper(Problem):
    """死刑法：不可行个体目标置极大值（支配一切可行解）。"""

    BIG = 1e12

    def __init__(self, problem: Problem):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=0, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem

    def _evaluate(self, x, out, *args, **kwargs):
        F, G = self.inner.evaluate(x, return_values_of=["F", "G"])
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        infeas = (G > 1e-6).any(axis=1)
        F2 = F.copy()
        F2[infeas] = self.BIG
        out["F"] = F2


class RepairWrapper(Problem):
    """修复算子：eval 前投影到可行域（几何修复）+ 保留原约束给可行性规则。

    repair_fn: x (n,d) -> x (n,d) 逐个体投影。
    """

    def __init__(self, problem: Problem, repair_fn):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=problem.n_ieq_constr, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem
        self.repair_fn = repair_fn

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        F, G = self.inner.evaluate(self.repair_fn(x), return_values_of=["F", "G"])
        out["F"] = np.asarray(F, dtype=float)
        out["G"] = np.asarray(G, dtype=float)


def violation(G: np.ndarray) -> np.ndarray:
    """总约束违反量 Σ max(0, gi)。"""
    G = np.asarray(G, dtype=float)
    return np.maximum(0.0, G).sum(axis=1)
