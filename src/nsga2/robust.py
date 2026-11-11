"""v1.2 鲁棒优化：设计变量制造/装配不确定性下的稳健解。

场景（报告 §20）：真实产品里变量有公差——名义最优解常常紧贴约束边界
（g≈0），公差一进就变成不可行或目标骤变。鲁棒口径：

- 名义（nominal）：F(x)、G(x) 单点评估（v1.0/v1.1 全部口径）。
- 均值（mean）：F̄(x) = E_δ[F(x+δ)]，Ḡ(x) = E_δ[G(x+δ)]。
- 最坏（worst）：F_w(x) = max_δ F(x+δ)，G_w(x) = max_δ G(x+δ)（逐目标/约束）。
- 可行概率（reliability）：p_feas(x) = P_δ[G(x+δ) ≤ 0]，由样本频率估计。

不确定性模型：δ ~ U(-Δ, Δ)，Δ = tol_frac·(xu − xl)（每变量等比公差），
样本由 `sample_perturbations` 生成（独立于优化器、可复现 seed）。
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem


def sample_perturbations(n_samples: int, n_var: int, tol_frac: float,
                         xl: np.ndarray, xu: np.ndarray, seed: int = 0):
    """生成 (n_samples, n_var) 的绝对扰动矩阵 δ（U(-Δ,Δ)，Δ=tol_frac·range）。"""
    rng = np.random.default_rng(seed)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    delta = tol_frac * (xu - xl)
    return rng.uniform(-delta, delta, size=(int(n_samples), int(n_var)))


def evaluate_scenarios(problem: Problem, x: np.ndarray,
                       perturbations: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """把 x 广播到每个扰动样本上批量评估，返回 (F (n,m), G (n,c) 或 None)。"""
    x = np.asarray(x, dtype=float).ravel()
    X = x[None, :] + np.asarray(perturbations, dtype=float)
    X = np.clip(X, problem.xl, problem.xu)
    r = problem.evaluate(X, return_values_of=["F", "G"])
    F = np.asarray(r[0], dtype=float)
    G = None
    if problem.n_ieq_constr and problem.n_ieq_constr > 0:
        G = np.asarray(r[1], dtype=float)
    return F, G


def reliability(G_scenarios: np.ndarray | None, tol: float = 1e-6) -> float:
    """可行概率估计：P[G ≤ tol] 的样本频率。无约束恒为 1。"""
    if G_scenarios is None:
        return 1.0
    G = np.asarray(G_scenarios, dtype=float)
    return float(np.mean(np.all(G <= tol, axis=1)))


def robust_stats(problem: Problem, x: np.ndarray, perturbations: np.ndarray,
                 tol: float = 1e-6) -> dict:
    """单点鲁棒统计：F 均值/最坏、G 最坏、可行概率。"""
    F, G = evaluate_scenarios(problem, x, perturbations)
    out = {
        "F_mean": F.mean(axis=0),
        "F_worst": F.max(axis=0),
        "F_std": F.std(axis=0),
        "p_feas": reliability(G, tol),
    }
    if G is not None:
        out["G_worst"] = G.max(axis=0)
    return out


class RobustWrapper(Problem):
    """把鲁棒口径包成 pymoo Problem，让任意 MOEA 直接优化稳健目标。

    mode="mean"：目标 = E[F]，约束 = E[G]（保守口径：均值约束）。
    mode="worst"：目标 = max F，约束 = max G（完全保守）。
    mode="reliability"：目标 = E[F]，约束 = (1 − p_feas) ≤ p_fail_max，
        即 P[G ≤ 0] ≥ 1 − p_fail_max（机会约束，样本频率）。
    """

    def __init__(self, problem: Problem, perturbations: np.ndarray,
                 mode: str = "mean", p_fail_max: float = 0.05,
                 tol: float = 1e-6):
        n_ieq = problem.n_ieq_constr
        if mode == "reliability":
            n_ieq = 1   # 单列联合机会约束：P[all g≤0] ≥ 1 − p_fail_max
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj,
                         n_ieq_constr=n_ieq, vtype=problem.vtype)
        self.xl, self.xu = problem.xl, problem.xu
        self.inner = problem
        self.perturbations = np.asarray(perturbations, dtype=float)
        if mode not in ("mean", "worst", "reliability"):
            raise ValueError("mode 须为 mean/worst/reliability")
        self.mode = mode
        self.p_fail_max = float(p_fail_max)
        self.tol = float(tol)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        F_mean = np.empty((n, self.n_obj))
        F_worst = np.empty((n, self.n_obj))
        if self.mode == "reliability":
            G_out = np.empty((n, 1))
        elif self.inner.n_ieq_constr:
            G_out = np.empty((n, self.inner.n_ieq_constr))
        else:
            G_out = None

        for i in range(n):
            F, G = evaluate_scenarios(self.inner, x[i], self.perturbations)
            F_mean[i] = F.mean(axis=0)
            F_worst[i] = F.max(axis=0)
            if self.mode == "reliability":
                p_fail = 1.0 - reliability(G, self.tol)
                G_out[i, 0] = p_fail - self.p_fail_max
            elif G is not None:
                if self.mode == "worst":
                    G_out[i] = G.max(axis=0)
                else:
                    G_out[i] = G.mean(axis=0)

        out["F"] = F_worst if self.mode == "worst" else F_mean
        if G_out is not None:
            out["G"] = G_out
