"""v1.1 局部搜索模块：e-constraint 参考前沿 + ASF（Tchebycheff）memetic 精修。

- `eref_front`：ε-约束法生成高精度近似参考前沿（min f0 s.t. f1≤ε, g≤0，
  多起点 SLSQP + ε 升序热启动）。
- `polish_front`：对 NSGA-II 前沿逐点做成就标量化（ASF/Tchebycheff）局部精修。
- `endpoint_polish`：单目标端点精修（min f_i s.t. g≤0）。
- `nondominated_mask`：非支配掩码（保留不被任何点支配的行——注意方向）。
"""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem
from scipy.optimize import minimize as sp_minimize


# ---------------------------------------------------------------- 评估封装
def eval_fg(problem: Problem, x: np.ndarray):
    """评估单点，返回 (F 行, G 行或 None)。兼容 pymoo 0.6 元组返回。"""
    x = np.asarray(x, dtype=float).ravel()[None, :]
    r = problem.evaluate(x, return_values_of=["F", "G"])
    F = np.asarray(r[0], dtype=float)[0]
    G = None
    if problem.n_ieq_constr and problem.n_ieq_constr > 0:
        G = np.asarray(r[1], dtype=float)[0]
    return F, G


def _feasible(problem: Problem, x: np.ndarray, tol: float = 1e-6) -> bool:
    if not problem.n_ieq_constr:
        return bool(np.all(x >= problem.xl - 1e-9) and np.all(x <= problem.xu + 1e-9))
    _, G = eval_fg(problem, x)
    return bool(np.all(np.asarray(G) <= tol))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """非支配掩码：被任何其他点支配的行 → False。

    领域陷阱（v1.0 教训）：`keep[dom]=False` 删的是**支配者**，方向相反。
    正确：第 i 行若存在 j 支配 i，则 keep[i]=False。
    支配判定带量级感知容差（atol ~1e-9·|F|），否则浮点微差会让
    近重复劣点（如成本 54.7 vs 36.4 而 deflection 差 1e-10）漏网。
    """
    F = np.asarray(F, dtype=float)
    atol = 1e-9 * (np.abs(F).max(axis=0) + 1.0)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        le = np.all(F <= F[i] + atol, axis=1)      # j 不劣于 i
        lt = np.any(F < F[i] - atol, axis=1)       # j 严格优于 i（容差内）
        dominated = le & lt
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


# ---------------------------------------------------------------- SLSQP 内核
def _slsqp(problem: Problem, obj_idx: int, x0: np.ndarray,
           f_ub: tuple[int, float] | None = None, maxiter: int = 400):
    """min f_{obj_idx}(x) s.t. G≤0（+ 可选 f_{k}≤eps）。

    返回 (x, fval, success)。失败时返回起点信息不吞错。
    """
    x0 = np.clip(np.asarray(x0, dtype=float), problem.xl, problem.xu)

    def f(x):
        F, _ = eval_fg(problem, x)
        return float(F[obj_idx])

    cons = []
    if problem.n_ieq_constr and problem.n_ieq_constr > 0:
        def c_g(x):
            _, G = eval_fg(problem, x)
            return -np.asarray(G, dtype=float)   # scipy: c(x) >= 0
        cons.append({"type": "ineq", "fun": c_g})
    if f_ub is not None:
        k, eps = f_ub

        def c_f(x):
            F, _ = eval_fg(problem, x)
            return eps - float(F[k])
        cons.append({"type": "ineq", "fun": c_f})

    res = sp_minimize(f, x0, bounds=list(zip(problem.xl, problem.xu)),
                      constraints=cons, method="SLSQP",
                      options={"maxiter": maxiter, "ftol": 1e-10})
    x = np.clip(np.asarray(res.x, dtype=float), problem.xl, problem.xu)

    # 可行性恢复：SLSQP 常停在活跃约束的不可行侧（"Positive directional
    # derivative"，残差 ~1e-5）。两步修复：
    #   1) 线性违反最小化 → 可行锚点 x_feas（常停得过深、f 偏差大）；
    #   2) 在 x 与 x_feas 连线上二分出可行边界 → f 损失最小的可行点。
    if problem.n_ieq_constr and problem.n_ieq_constr > 0:
        _, G = eval_fg(problem, x)
        G = np.asarray(G, dtype=float)
        if np.any(G > 1e-6) and np.max(G) < 1e-2:
            def viol(u):
                _, Gu = eval_fg(problem, u)
                return float(np.sum(np.maximum(0.0, np.asarray(Gu, dtype=float))))

            def feasible_at(u):
                _, Gu = eval_fg(problem, u)
                return bool(np.all(np.asarray(Gu) <= 1e-6))

            r2 = sp_minimize(viol, x,
                             bounds=list(zip(problem.xl, problem.xu)),
                             method="SLSQP",
                             options={"maxiter": 200, "ftol": 1e-12})
            x_feas = np.clip(np.asarray(r2.x, dtype=float),
                             problem.xl, problem.xu)
            if feasible_at(x_feas):
                if not feasible_at(x):
                    lo_a, hi_a = 0.0, 1.0   # x(α)=x+α(x_feas−x)；α=1 可行
                    for _ in range(60):
                        mid = 0.5 * (lo_a + hi_a)
                        if feasible_at(x + mid * (x_feas - x)):
                            hi_a = mid
                        else:
                            lo_a = mid
                    x = x + hi_a * (x_feas - x)

    F, _ = eval_fg(problem, x)
    ok = _feasible(problem, x) \
        and (f_ub is None or float(F[f_ub[0]]) <= f_ub[1] + 1e-6)
    return x, float(F[obj_idx]), ok


# ---------------------------------------------------------------- e-constraint
def eref_front(problem: Problem, n_eps: int = 25, n_starts: int = 3,
               seed: int = 0, fixed: dict[int, float] | None = None,
               f1_idx: int = 0, f2_idx: int = 1):
    """ε-约束参考前沿：min f1 s.t. f2 ≤ ε（ε 升序热启动）。

    fixed: 可选变量钉死（如 SR 的 z=整数）→ 修改边界实现。
    返回 (F (n,2), X (n,d))，仅收可行非支配点。
    """
    rng = np.random.default_rng(seed)
    if fixed:
        lo, hi = problem.xl.copy(), problem.xu.copy()
        for i, v in fixed.items():
            lo[i] = hi[i] = float(v)
    else:
        lo, hi = problem.xl, problem.xu

    def solve(obj_idx, x0, f_ub=None):
        x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
        # 临时边界钉死
        save_lo, save_hi = problem.xl, problem.xu
        problem.xl, problem.xu = lo, hi
        try:
            return _slsqp(problem, obj_idx, x0, f_ub=f_ub)
        finally:
            problem.xl, problem.xu = save_lo, save_hi

    # 端点（确定性分数起点 + 随机起点；活跃约束处 SLSQP 停点双峰，须多起点）
    # 平局（fv 差 <1e-9 相对）时按另一目标择优——否则 f2 平局会留下成本劣点。
    best = [None, None]
    bestkey = [None, None]
    for idx in (f1_idx, f2_idx):
        starts = [lo + (hi - lo) * frac for frac in (0.2, 0.4, 0.6, 0.8)]
        starts += [rng.uniform(lo, hi) for _ in range(max(2, n_starts))]
        for x0 in starts:
            x, fv, ok = solve(idx, x0)
            if not ok:
                continue
            Fo, _ = eval_fg(problem, x)
            key = (round(fv, 9), float(Fo[1 - idx]))
            if bestkey[idx] is None or key < bestkey[idx]:
                bestkey[idx] = key
                best[idx] = x
        if best[idx] is None:
            raise RuntimeError(f"eref endpoint f{idx} all starts failed")
    x_f1, x_f2 = best[f1_idx], best[f2_idx]
    F1, _ = eval_fg(problem, x_f1)
    F2, _ = eval_fg(problem, x_f2)

    eps_lo = float(F1[f2_idx])   # f2 at f1-min
    eps_hi = float(F2[f2_idx])   # f2-min
    # 升序扫描：从 f2-min 到 f2(f1-min)（eps_lo ≥ eps_hi）
    eps_vals = np.linspace(min(eps_lo, eps_hi), max(eps_lo, eps_hi), n_eps)

    pts_x, pts_f = [x_f1, x_f2], [F1[[f1_idx, f2_idx]], F2[[f1_idx, f2_idx]]]
    warm = x_f2.copy()
    for eps in eps_vals:
        cands = [warm] + [rng.uniform(lo, hi) for _ in range(max(0, n_starts - 1))]
        got = None
        for x0 in cands:
            x, fv, ok = solve(f1_idx, x0, f_ub=(f2_idx, float(eps)))
            if ok and (got is None or fv < got[1]):
                got = (x, fv)
        if got is not None:
            warm = got[0]
            Fx, _ = eval_fg(problem, got[0])
            pts_x.append(got[0])
            pts_f.append(Fx[[f1_idx, f2_idx]])

    # 反向扫描：min f2 s.t. f1 ≤ ε′（凸弯段 ε-均匀采样摊不开，f1-均匀补覆盖）
    f1_lo, f1_hi = float(F1[f1_idx]), float(F2[f1_idx])
    eps2_vals = np.linspace(min(f1_lo, f1_hi), max(f1_lo, f1_hi), n_eps)
    warm = x_f1.copy()
    for eps2 in eps2_vals:
        cands = [warm] + [rng.uniform(lo, hi) for _ in range(max(0, n_starts - 1))]
        got = None
        for x0 in cands:
            x, fv, ok = solve(f2_idx, x0, f_ub=(f1_idx, float(eps2)))
            if ok and (got is None or fv < got[1]):
                got = (x, fv)
        if got is not None:
            warm = got[0]
            Fx, _ = eval_fg(problem, got[0])
            pts_x.append(got[0])
            pts_f.append(Fx[[f1_idx, f2_idx]])

    F = np.array(pts_f, dtype=float)
    X = np.array(pts_x, dtype=float)
    keep = nondominated_mask(F)
    F, X = F[keep], X[keep]
    order = np.argsort(F[:, 0])
    return F[order], X[order]


# ---------------------------------------------------------------- ASF 精修
def _asf_polish(problem: Problem, x0: np.ndarray, w: np.ndarray,
                z_star: np.ndarray, span: np.ndarray, rho: float = 1e-4):
    """成就标量化（Tchebycheff + 小增量项）单点精修。

    min_x  t + rho * Σ_j (f_j - z*_j)/span_j
    s.t.   w_j·(f_j(x) - z*_j) ≤ t  ∀j ；g(x) ≤ 0
    实现为 (x, t) 联合变量的 SLSQP。span 用**全前沿量程**（极端点处 F−z*→0，
    用点距做 span 会让增广项梯度爆炸）。
    """
    m = problem.n_obj
    F0, _ = eval_fg(problem, x0)

    def pack(xt):
        return xt[:-1], float(xt[-1])

    def obj(xt):
        x, t = pack(xt)
        F, _ = eval_fg(problem, x)
        return t + rho * float(np.sum((F - z_star) / span))

    cons = []
    for j in range(m):
        def c_j(xt, j=j):
            x, t = pack(xt)
            F, _ = eval_fg(problem, x)
            return t - w[j] * (float(F[j]) - z_star[j])
        cons.append({"type": "ineq", "fun": c_j})
    if problem.n_ieq_constr and problem.n_ieq_constr > 0:
        def c_g(xt):
            x, _ = pack(xt)
            _, G = eval_fg(problem, x)
            return -np.asarray(G, dtype=float)
        cons.append({"type": "ineq", "fun": c_g})

    t0 = float(np.max(w * (F0 - z_star)))
    xt0 = np.concatenate([np.asarray(x0, dtype=float), [t0]])
    bounds = list(zip(problem.xl, problem.xu)) + [(None, None)]
    res = sp_minimize(obj, xt0, bounds=bounds, constraints=cons,
                      method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
    x = np.clip(np.asarray(res.x[:-1], dtype=float), problem.xl, problem.xu)
    F, _ = eval_fg(problem, x)
    ok = _feasible(problem, x)
    return x, F, ok


def polish_front(problem: Problem, X: np.ndarray, F: np.ndarray | None = None):
    """对前沿逐点 ASF 精修，返回精修后的 (F, X)（可行非支配、按 f1 排序）。

    权重取 w ∝ 1/(f_i - z*_i + δ)（解所在位置的局部权衡方向）。
    """
    X = np.asarray(X, dtype=float)
    if F is None:
        F = np.array([eval_fg(problem, x)[0] for x in X])
    F = np.asarray(F, dtype=float)
    z_star = F.min(axis=0)
    span = np.maximum(F.max(axis=0) - z_star, 1e-12)   # 全前沿量程
    delta = 1e-6 * np.maximum(span, 1.0)

    pts_f, pts_x = [F], [X]
    for i in range(X.shape[0]):
        raw = 1.0 / (F[i] - z_star + delta)
        w = raw / raw.sum()
        x, Fi, ok = _asf_polish(problem, X[i], w, z_star, span)
        if ok:
            pts_f.append(Fi[None, :])
            pts_x.append(x[None, :])
    F2 = np.vstack(pts_f)
    X2 = np.vstack(pts_x)
    keep = nondominated_mask(F2)
    F2, X2 = F2[keep], X2[keep]
    _, idx = np.unique(np.round(F2, 10), axis=0, return_index=True)
    idx = np.sort(idx)
    F2, X2 = F2[idx], X2[idx]
    order = np.argsort(F2[:, 0])
    return F2[order], X2[order]


def endpoint_polish(problem: Problem, obj_idx: int, x_seed: np.ndarray):
    """单目标端点精修：min f_{obj_idx} s.t. g≤0。返回 (x, f, ok)。"""
    return _slsqp(problem, obj_idx, x_seed)
