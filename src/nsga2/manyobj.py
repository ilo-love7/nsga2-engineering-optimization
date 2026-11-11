"""v1.2 多算法 / 多目标工具：NSGA-III、MOEA-D、Das-Dennis 参考方向、
单纯形格点权重、加权 ASF 标量化参考前沿。

口径（报告须如实披露）：
1. pymoo 0.6.2 的 MOEAD 在 `_setup` 断言 `not problem.has_constraints()`
   ——原生不支持约束。约束问题走 `DeathPenaltyWrapper` 转无约束再跑
   （v1.0 §5：死刑法可行率 100%、HV 与可行性规则差 ~1.2%）；
   NSGA-II/NSGA-III 保留原生可行性规则。算法对比时该口径差异必须披露。
2. 等预算协议：统一 `("n_evals", budget)` 终止。MOEA/D 的种群规模被源码
   写死为 `len(ref_dirs)`，若按 n_gen 比较三算法预算不等，故必须按评估数。
3. 等种群协议：`pop_size = len(ref_dirs)`，三算法同种群规模、同预算。
4. `run_moea` 返回的 `res.F` / `res.G` 一律在**原问题**上重评——死刑法
   包装下 MOEA/D 的原生 F 是 1e12 惩罚值，不可直接进 GD/IGD/HV。
5. 【v1.2 修复】pymoo 0.6.2 的 `comp_by_cv_then_random`（NSGA-III 锦标赛）
   调 `compare(..., return_random_if_equal=True)` 时**漏传 random_state**，
   `@default_random_state` 装饰器回落 `np.random.default_rng(None)`（OS 熵）
   ——CV 平局的锦标赛即真随机，同 seed 跨调用结果漂移（实测 ~165 次/
   run 熵回落）。本模块用 `comp_by_cv_then_random_fixed` 显式透传
   random_state 修复；NSGA-II 的 `binary_tournament` / MOEA-D 无此问题。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.optimize import minimize

from .constraint_strategies import DeathPenaltyWrapper
from .local_search import _asf_polish, _slsqp, eval_fg, nondominated_mask
from .runner import make_algorithm


def comp_by_cv_then_random_fixed(pop, P, random_state=None, **kwargs):
    """NSGA-III 锦标赛比较（pymoo 0.6.2 漏传 random_state 的修复版）。

    与原版唯一差别：CV 平局掷币显式走 `random_state`（缺省时用
    `default_rng(0)` 兜底，**禁止**熵回落，保证 run_moea(seed=...) 跨调用
    完全可复现）。原版 bug 见模块 docstring 口径 5。
    """
    rng = random_state if random_state is not None else np.random.default_rng(0)
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(
                a,
                pop[a].CV,
                b,
                pop[b].CV,
                method="smaller_is_better",
                return_random_if_equal=True,
                random_state=rng,
            )
        else:
            S[i] = rng.choice([a, b])
    return S[:, None].astype(int)


# ---------------------------------------------------------------- 参考方向
def das_dennis(n_dim: int, n_partitions: int) -> np.ndarray:
    """Das-Dennis 单纯形格点参考方向。

    返回 (C(n_partitions+n_dim-1, n_dim-1), n_dim)，每行非负、和为 1。
    """
    if n_dim < 2:
        raise ValueError("n_dim 须 ≥ 2")
    if n_partitions < 1:
        raise ValueError("n_partitions 须 ≥ 1")

    def gen(rest: int, remaining: int, prefix: list):
        if rest == 1:
            yield prefix + [remaining]
        else:
            for i in range(remaining + 1):
                yield from gen(rest - 1, remaining - i, prefix + [i])

    dirs = np.array(list(gen(n_dim, n_partitions, [])), dtype=float)
    return dirs / float(n_partitions)


def simplex_lattice(m: int, step: float) -> np.ndarray:
    """单纯形格点权重（和为 1、增量为 step 的网格），等价 das_dennis(m, 1/step)。"""
    n = int(round(1.0 / step))
    if n < 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError("step 须为 1/n 形式（n 正整数）")
    return das_dennis(m, n)


# ---------------------------------------------------------------- 多算法封装
def make_moea(name: str, problem: Problem, pop_size: int = 100,
              ref_dirs: np.ndarray | None = None, n_neighbors: int = 20,
              n_partitions: int | None = None):
    """构造 NSGA-II / NSGA-III / MOEA-D，返回 (algorithm, inner_problem)。

    - nsga2：原问题 + 可行性规则（runner.make_algorithm，含整数 repair）。
    - nsga3：原问题 + Das-Dennis 参考方向 + 可行性规则。
    - moead：问题**有约束**时包 DeathPenaltyWrapper（pymoo 限制，见模块
      docstring）；种群规模固定为 len(ref_dirs)。
    """
    name = name.lower()
    m = problem.n_obj
    if ref_dirs is None:
        p = n_partitions if n_partitions is not None else (99 if m == 2 else 12)
        ref_dirs = das_dennis(m, p)

    if name == "nsga2":
        return make_algorithm(problem, pop_size=pop_size), problem
    if name == "nsga3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        # 修复版锦标赛：显式透传 random_state（原版漏传→熵回落，口径 5）
        sel = TournamentSelection(func_comp=comp_by_cv_then_random_fixed)
        return (NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, selection=sel),
                problem)
    if name == "moead":
        from pymoo.algorithms.moo.moead import MOEAD
        inner = DeathPenaltyWrapper(problem) if problem.has_constraints() else problem
        alg = MOEAD(ref_dirs=ref_dirs, n_neighbors=n_neighbors)
        return alg, inner
    raise ValueError(f"未知算法 {name!r}（可选 nsga2/nsga3/moead）")


def run_moea(problem: Problem, algo: str = "nsga2", budget: int = 15000,
             pop_size: int | None = None, seed: int = 1,
             ref_dirs: np.ndarray | None = None, n_neighbors: int = 20,
             n_partitions: int | None = None, save_history: bool = False):
    """等 NFE 预算跑一个多目标算法，返回 res（X/F/G 已在原问题上重评）。

    pop_size=None 时取 len(ref_dirs)（等种群口径；nsga2 也用该值）。
    save_history=True 时 res.history 保留各代状态（收敛曲线用）。
    """
    m = problem.n_obj
    if ref_dirs is None:
        p = n_partitions if n_partitions is not None else (99 if m == 2 else 12)
        ref_dirs = das_dennis(m, p)
    if pop_size is None:
        pop_size = int(ref_dirs.shape[0])
    algorithm, inner = make_moea(algo, problem, pop_size=pop_size,
                                 ref_dirs=ref_dirs, n_neighbors=n_neighbors)
    res = minimize(inner, algorithm, ("n_evals", int(budget)),
                   seed=int(seed), verbose=False, save_history=save_history)

    # 死刑法包装下原生 F 是惩罚值 → 一律回原问题重评
    X = res.X
    if X is None:
        # 小预算/难收敛时 pymoo 无可行最优 → 取末代种群中 CV 最小者兜底
        pop = res.algorithm.pop
        cv = np.asarray(pop.get("CV"), dtype=float).reshape(-1)
        X = np.asarray(pop.get("X"), dtype=float)[cv <= cv.min() + 1e-12]
    X = np.atleast_2d(np.asarray(X, dtype=float))
    r = problem.evaluate(X, return_values_of=["F", "G"])
    res.X = X
    res.F = np.asarray(r[0], dtype=float)
    if problem.n_ieq_constr and problem.n_ieq_constr > 0:
        res.G = np.asarray(r[1], dtype=float)
    else:
        res.G = None
    return res


# ---------------------------------------------------------------- 前沿池化
def pooled_front(Fs: Iterable[np.ndarray], Xs: Iterable[np.ndarray]):
    """多组解合并 → 非支配 + 去重 + 按 f1 排序，返回 (F, X)。"""
    F = np.vstack([np.asarray(f, dtype=float) for f in Fs])
    X = np.vstack([np.asarray(x, dtype=float) for x in Xs])
    keep = nondominated_mask(F)
    F, X = F[keep], X[keep]
    _, idx = np.unique(np.round(F, 10), axis=0, return_index=True)
    idx = np.sort(idx)
    F, X = F[idx], X[idx]
    order = np.argsort(F[:, 0])
    return F[order], X[order]


# ---------------------------------------------------------------- 标量化参考前沿
def scalarized_ref_front(problem: Problem, weights: np.ndarray | None = None,
                         step: float = 0.1, n_starts: int = 2, seed: int = 0,
                         polish: bool = True):
    """加权 ASF（Tchebycheff）标量化解并集 → 高质量 m 目标参考前沿。

    流程：m 个单目标端点（SLSQP 多起点）→ z*/span → 单纯形权重逐个解
    `min ASF(x; w)` → （可选）二遍精修（更新 z*/span 后从一遍解再抛光）
    → 非支配去重。
    """
    rng = np.random.default_rng(seed)
    m = problem.n_obj
    if weights is None:
        weights = simplex_lattice(m, step)
    W = np.asarray(weights, dtype=float)

    # 端点
    endpoints, EF = [], []
    for j in range(m):
        best = None
        x0s = [problem.xl + (problem.xu - problem.xl) * f
               for f in (0.25, 0.5, 0.75)]
        x0s.append(rng.uniform(problem.xl, problem.xu))
        for x0 in x0s:
            x, fv, ok = _slsqp(problem, j, x0)
            if ok and (best is None or fv < best[1]):
                best = (x, fv)
        if best is None:
            raise RuntimeError(f"scalarized_ref_front 端点 f{j} 全部起点失败")
        endpoints.append(best[0])
        Fx, _ = eval_fg(problem, best[0])
        EF.append(Fx)
    EF = np.asarray(EF, dtype=float)
    z_star = EF.min(axis=0)
    span = np.maximum(EF.max(axis=0) - z_star, 1e-12)

    def solve_weight(w, x0):
        x, Fx, ok = _asf_polish(problem, x0, w, z_star, span)
        return (x, Fx) if ok else None

    pts_x = list(endpoints)
    pts_f = [Fx for Fx in EF]
    for w in W:
        got = None
        for s in range(max(1, n_starts)):
            x0 = endpoints[s % m] if s < m else rng.uniform(problem.xl, problem.xu)
            r = solve_weight(w, x0)
            if r is not None and (got is None or r[1][0] < got[1][0]):
                got = r
        if got is not None:
            pts_x.append(got[0])
            pts_f.append(got[1])

    F1 = np.asarray(pts_f, dtype=float)
    X1 = np.asarray(pts_x, dtype=float)

    if polish:
        z2 = F1.min(axis=0)
        span2 = np.maximum(F1.max(axis=0) - z2, 1e-12)
        more_x, more_f = [], []
        for i, w in enumerate(W):
            r = _asf_polish(problem, X1[min(i, X1.shape[0] - 1)], w, z2, span2)
            x, Fx, ok = r
            if ok:
                more_x.append(x)
                more_f.append(Fx)
        if more_x:
            F1 = np.vstack([F1, np.asarray(more_f, dtype=float)])
            X1 = np.vstack([X1, np.asarray(more_x, dtype=float)])

    F, X = pooled_front([F1], [X1])
    return F, X
