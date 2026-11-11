"""v1.3 E23 约束处理对比：动态罚 / 自适应罚 / ε-松弛。

与 v1.0 的 constraint_strategies（静态罚 / 死刑 / 修复）互补。动机：
v1.1 §12 与 v1.2 §22 连续两轮把「动态惩罚与自适应约束处理」列为未做；
且 v1.2 发现 SR3 上 CV 平局主导锦标赛选择——罚权重作为**状态量**
（随代数 / 可行率变化）是否改变搜索引导与最终可行性，需等预算横向对比。

方法口径（run_handled 统一装配，六种）：

- rules    原生可行性规则（Deb 约束支配，pymoo 默认 CV 比较）
- death    死刑法（不可行置 1e12，constraint_strategies.DeathPenaltyWrapper）
- static   静态罚 w=1e5·viol 加到全部目标（PenaltyWrapper mode="all"）
- dynamic  动态罚 w(t)=(C·(t+1))^α · Σ max(0,g_i)^β（Joines & Smith 型）
- adaptive 自适应罚：按当代可行率 r 对目标 r* 调权（r<r* 加权、r>r* 降权，有界）
- eps      ε-松弛：g'=g−ε(t)，ε(t)=ε0·(1−t/T)^k，可行性规则作用在松弛约束上

口径纪律：

- 状态推进用 GenerationHook（pymoo Callback），t=algorithm.n_gen；
  自适应罚的当代可行率由 hook 用**原问题重评** pop.X 得到（确定性、无缓存）。
- 指标一律回原问题重评（原 F、原 G、原约束判可行 tol=1e-6）；罚值、
  1e12、松弛 G 都不进指标。
- violation(G) = Σ max(0, g_i)（问题的约束已按量级归一化，见 problems/）。

纯函数（dynamic_weight / adaptive_weight / epsilon_schedule / violation）
与包装器分离，均可单测；本模块不依赖任何全局随机状态。
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from src.nsga2.constraint_strategies import (
    DeathPenaltyWrapper,
    PenaltyWrapper,
    violation,
)

METHODS = ("rules", "death", "static", "dynamic", "adaptive", "eps")


# ---------------------------------------------------------------------------
# 纯函数：罚/松弛调度（全部单调、有界，单测直接钉）
# ---------------------------------------------------------------------------

def dynamic_weight(t: int, c: float = 0.5, alpha: float = 2.0) -> float:
    """动态罚权重 w(t) = (C·(t+1))^α。t=0 起非负、随 t 单调不减。"""
    if t < 0:
        raise ValueError("t 须 ≥ 0")
    if c <= 0 or alpha <= 0:
        raise ValueError("c、alpha 须 > 0")
    return float((c * (t + 1)) ** alpha)


def adaptive_weight(weight: float, feas_rate: float, target: float = 0.2,
                    gamma: float = 0.5, w_lo: float = 1e-3,
                    w_hi: float = 1e12) -> float:
    """自适应罚权重单步更新（纯函数）：

    r < target（不可行偏多）→ weight·(1+γ) 加压；
    r > target（可行偏多）→ weight/(1+γ) 减压；
    r == target → 不变。结果夹在 [w_lo, w_hi]。
    """
    if not (0.0 <= feas_rate <= 1.0):
        raise ValueError("feas_rate 须在 [0,1]")
    if not (0.0 < target < 1.0):
        raise ValueError("target 须在 (0,1)")
    if gamma <= 0 or w_lo <= 0 or w_hi < w_lo:
        raise ValueError("gamma/w_lo/w_hi 参数非法")
    w = float(weight)
    if feas_rate < target:
        w *= (1.0 + gamma)
    elif feas_rate > target:
        w /= (1.0 + gamma)
    return float(min(max(w, w_lo), w_hi))


def epsilon_schedule(t: int, total_gen: int, eps0: float = 0.5,
                     k: float = 2.0) -> float:
    """ε-松弛量 ε(t) = ε0·(1 − t/T)^k，t∈[0,T] 单调降到 0。"""
    if total_gen <= 0:
        raise ValueError("total_gen 须 > 0")
    if t < 0 or eps0 < 0 or k <= 0:
        raise ValueError("t/eps0/k 参数非法")
    frac = max(0.0, 1.0 - float(t) / float(total_gen))
    return float(eps0 * frac ** k)


# ---------------------------------------------------------------------------
# 包装器：状态型罚 / ε-松弛
# ---------------------------------------------------------------------------

class DynamicPenaltyWrapper(Problem):
    """动态罚：F' = F + w(t)·viol^β（加到全部目标；mode="f1" 仅加第一目标）。

    t 由 GenerationHook 每代写入 set_gen()；未接 hook 时 t=0（退化静态）。
    """

    def __init__(self, problem: Problem, c: float = 0.5, alpha: float = 2.0,
                 beta: float = 1.0, mode: str = "all"):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=0, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem
        self.c, self.alpha, self.beta = float(c), float(alpha), float(beta)
        if mode not in ("f1", "all"):
            raise ValueError("mode 须为 f1 或 all")
        self.mode = mode
        self.t = 0

    def set_gen(self, t: int) -> None:
        self.t = int(t)

    def _evaluate(self, x, out, *args, **kwargs):
        F, G = self.inner.evaluate(x, return_values_of=["F", "G"])
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        viol = (np.maximum(0.0, G) ** self.beta).sum(axis=1)
        w = dynamic_weight(self.t, self.c, self.alpha)
        pen = w * viol
        F2 = F.copy()
        if self.mode == "all":
            F2 = F + pen[:, None]
        else:
            F2[:, 0] = F[:, 0] + pen
        out["F"] = F2


class AdaptivePenaltyWrapper(Problem):
    """自适应罚：静态罚形式，但权重按当代可行率每代自适应（set_rate）。

    与 DynamicPenaltyWrapper 同为"无约束化"包装（n_ieq=0）；feas_rate
    由 GenerationHook 用原问题重评后写入。
    """

    def __init__(self, problem: Problem, weight: float = 1e3,
                 target: float = 0.2, gamma: float = 0.5,
                 w_lo: float = 1e-3, w_hi: float = 1e12):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=0, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem
        self.weight = float(weight)
        self.target, self.gamma = float(target), float(gamma)
        self.w_lo, self.w_hi = float(w_lo), float(w_hi)

    def set_rate(self, feas_rate: float) -> None:
        self.weight = adaptive_weight(
            self.weight, feas_rate, self.target, self.gamma,
            self.w_lo, self.w_hi,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        F, G = self.inner.evaluate(x, return_values_of=["F", "G"])
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        viol = violation(G)
        out["F"] = F + self.weight * viol[:, None]


class EpsilonConstraintWrapper(Problem):
    """ε-松弛约束：G' = G − ε(t)（保留约束 → Deb 可行性规则作用在松弛域）。

    t 由 GenerationHook 写入；t=T 时 ε=0，约束收紧到原问题。
    """

    def __init__(self, problem: Problem, total_gen: int, eps0: float = 0.5,
                 k: float = 2.0):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=problem.n_ieq_constr, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem
        self.total_gen, self.eps0, self.k = int(total_gen), float(eps0), float(k)
        self.t = 0

    def set_gen(self, t: int) -> None:
        self.t = int(t)

    @property
    def eps(self) -> float:
        return epsilon_schedule(self.t, self.total_gen, self.eps0, self.k)

    def _evaluate(self, x, out, *args, **kwargs):
        F, G = self.inner.evaluate(x, return_values_of=["F", "G"])
        out["F"] = np.asarray(F, dtype=float)
        out["G"] = np.asarray(G, dtype=float) - self.eps


# ---------------------------------------------------------------------------
# 每代状态推进 + 轨迹记录
# ---------------------------------------------------------------------------

class GenerationHook(Callback):
    """每代记录 mean_viol / 原始可行率 / 状态参数，并推进罚或 ε。

    记录口径：
    - mean_viol：当代 pop 在**原约束**下的 Σ max(0,g) 均值；
    - feas_rate：原约束（tol=1e-6）可行占比；
    - relaxed_feas_rate：松弛约束可行占比（eps 方法才有意义，其余=feas_rate）；
    - state：当前 w 或 ε。
    """

    def __init__(self, inner: Problem, wrapper=None, method: str = "rules",
                 total_gen: int | None = None):
        super().__init__()
        self.inner = inner
        self.wrapper = wrapper
        self.method = method
        self.total_gen = total_gen
        self.trace: list[dict] = []

    def update(self, algorithm):
        X = np.asarray(algorithm.pop.get("X"), dtype=float)
        _, G = self.inner.evaluate(X, return_values_of=["F", "G"])
        G = np.asarray(G, dtype=float)
        viol = violation(G)
        feas = (G <= 1e-6).all(axis=1)
        feas_rate = float(feas.mean()) if len(feas) else 0.0
        mean_viol = float(viol.mean()) if len(viol) else 0.0

        t = int(getattr(algorithm, "n_gen", len(self.trace) + 1))
        state = None
        relaxed = feas_rate
        if isinstance(self.wrapper, DynamicPenaltyWrapper):
            self.wrapper.set_gen(t)
            state = dynamic_weight(t, self.wrapper.c, self.wrapper.alpha)
        elif isinstance(self.wrapper, AdaptivePenaltyWrapper):
            self.wrapper.set_rate(feas_rate)
            state = self.wrapper.weight
        elif isinstance(self.wrapper, EpsilonConstraintWrapper):
            self.wrapper.set_gen(t)
            state = self.wrapper.eps
            relaxed = float((G - self.wrapper.eps <= 1e-6).all(axis=1).mean())

        self.trace.append({
            "t": t, "mean_viol": mean_viol, "feas_rate": feas_rate,
            "relaxed_feas_rate": relaxed, "state": state,
        })


# ---------------------------------------------------------------------------
# 统一装配
# ---------------------------------------------------------------------------

def wrap_problem(problem: Problem, method: str, total_gen: int,
                 c: float = 0.5, alpha: float = 2.0, beta: float = 1.0,
                 w_static: float = 1e5, adapt_target: float = 0.2,
                 adapt_gamma: float = 0.5, adapt_w0: float = 1e3,
                 eps0: float = 0.5, eps_k: float = 2.0):
    """按方法返回 (wrapped_problem, wrapper_or_None)。rules 直接用原问题。"""
    if method not in METHODS:
        raise ValueError(f"method 须为 {METHODS} 之一，得到 {method!r}")
    if method == "rules":
        return problem, None
    if method == "death":
        return DeathPenaltyWrapper(problem), None
    if method == "static":
        return PenaltyWrapper(problem, weight=w_static, mode="all"), None
    if method == "dynamic":
        w = DynamicPenaltyWrapper(problem, c=c, alpha=alpha, beta=beta, mode="all")
        return w, w
    if method == "adaptive":
        w = AdaptivePenaltyWrapper(problem, weight=adapt_w0,
                                   target=adapt_target, gamma=adapt_gamma)
        return w, w
    w = EpsilonConstraintWrapper(problem, total_gen=total_gen,
                                 eps0=eps0, k=eps_k)
    return w, w


def run_handled(problem: Problem, method: str = "rules",
                pop_size: int = 100, n_gen: int = 100, seed: int = 1,
                **params) -> dict:
    """装配并运行一种约束处理方法，返回原问题口径的最终种群与轨迹。

    返回 dict：
      X (n,d) / F (n,m) / G (n,c)：最终种群**回原问题重评**；
      trace：逐代 mean_viol / feas_rate / relaxed_feas_rate / state；
      method / pop_size / n_gen / seed。
    """
    wrapped, wrapper = wrap_problem(problem, method, total_gen=n_gen, **params)
    hook = GenerationHook(inner=problem, wrapper=wrapper, method=method,
                          total_gen=n_gen)
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15.0),
        mutation=PM(prob=1.0, eta=20.0),
        eliminate_duplicates=True,
    )
    res = minimize(wrapped, algorithm, ("n_gen", n_gen), seed=seed,
                   callback=hook, verbose=False)
    pop = res.algorithm.pop
    X = np.atleast_2d(np.asarray(pop.get("X"), dtype=float))
    F, G = problem.evaluate(X, return_values_of=["F", "G"])
    return {
        "X": X,
        "F": np.asarray(F, dtype=float),
        "G": np.asarray(G, dtype=float),
        "trace": hook.trace,
        "method": method,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "seed": seed,
    }
