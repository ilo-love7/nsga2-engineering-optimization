"""NSGA-II 运行封装：按问题类型选算子（连续 SBX/PM，整数走 repair）。"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize


def make_algorithm(problem: Problem, pop_size: int = 100,
                   crossover_prob: float = 0.9, eta_c: float = 15.0,
                   eta_m: float = 20.0, seed_sampling=None):
    """构造 NSGA-II。vtype=int 的问题（Clutch）用整数采样 + 舍入修复。"""
    if str(problem.vtype) == "int" or problem.vtype is int:
        return NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=eta_c, vtype=float,
                          repair=RoundingRepair()),
            mutation=PM(eta=eta_m, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )
    return NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=crossover_prob, eta=eta_c),
        mutation=PM(eta=eta_m),
        eliminate_duplicates=True,
    )


def run_nsga2(problem: Problem, pop_size: int = 100, n_gen: int = 200,
              seed: int = 1, **kw):
    """跑 NSGA-II，返回 res（res.X / res.F / res.G）。"""
    algorithm = make_algorithm(problem, pop_size=pop_size, **kw)
    return minimize(problem, algorithm, ("n_gen", n_gen), seed=seed,
                    verbose=False)


def feasible_mask(G: np.ndarray | None, tol: float = 1e-6) -> np.ndarray:
    """约束可行掩码（G ≤ tol 为可行；G 为 None 表示无约束）。"""
    if G is None:
        return None
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G[:, None]
    return (G <= tol).all(axis=1)


def nondominated_front(F: np.ndarray) -> np.ndarray:
    """返回 F 的非支配子集（行去重、按 f1 排序）。"""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break
    front = F[keep]
    front = np.unique(np.round(front, 12), axis=0)
    return front[np.argsort(front[:, 0])]
