"""v1.3 E24 偏好选点：ELECTRE III 可信度 + PROMETHEE II 净流。

与 topsis.topsis() 同台：对同一 Pareto 前沿 F、同一权重扰动协议，比较
三种多准则决策（MCDM）妥协解的稳定性与效率（报告 §24）。

口径（如实标注）：

- 目标全部按成本型（越小越好）；
- ELECTRE III 的 q/p/v 阈值（无差异/偏好/否决）以各准则**极差比例**给出
  （默认 2%/8%/25%）；完整 ELECTRE III 蒸馏（descending/ascending
  distillation）未实现——用**净可信度流** φ(a)=Σ_b(σ(a,b)−σ(b,a)) 作
  妥协解排序（报告如实标注为简化排序）；
- PROMETHEE II 用线性偏好函数（q=0、p=极差 10%），净流 φ 给出全预序；
- σ(a,b) 用 ELECTRE III 标准式：
  σ = c(a,b) · Π_{j: d_j(a,b) > c(a,b)} (1−d_j(a,b))/(1−c(a,b))，
  否决阈值使 d_j=1 时 σ=0。

全部纯函数：无全局随机、无状态。权重扰动在实验脚本里做。
"""
from __future__ import annotations

import numpy as np


def _as_F(F) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] == 0:
        raise ValueError("F 须为 (n,m) 非空矩阵")
    return F


def _norm_weights(m: int, weights) -> np.ndarray:
    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != m:
            raise ValueError("weights 长度须等于目标数")
    if np.any(w < 0):
        raise ValueError("weights 须非负")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights 之和须 > 0")
    return w / s


def criterion_ranges(F: np.ndarray) -> np.ndarray:
    """各准则极差（max−min）；退化准则（极差 0）取 1 防除零。"""
    r = F.max(axis=0) - F.min(axis=0)
    return np.where(r > 0, r, 1.0)


# ---------------------------------------------------------------------------
# PROMETHEE II
# ---------------------------------------------------------------------------

def linear_pref(d: np.ndarray, q: float, p: float) -> np.ndarray:
    """线性偏好函数 P(d)∈[0,1]：d≤q → 0；q<d<p → (d−q)/(p−q)；d≥p → 1。"""
    if p < q:
        raise ValueError("p 须 ≥ q")
    d = np.asarray(d, dtype=float)
    if p == q:
        return (d >= p).astype(float)
    return np.clip((d - q) / (p - q), 0.0, 1.0)


def promethee2(F, weights=None, q_frac: float = 0.0,
               p_frac: float = 0.10) -> dict:
    """PROMETHEE II：净流 φ（成本型，越小越优的目标）。

    P_j(a,b) = pref_j(max(0, F[b,j] − F[a,j]))；φ(a) = 均值_b(Σ_j w_j(P_j(a,b) − P_j(b,a))。
    返回 {phi, ranking, selected, selected_F, flows_ab}。
    """
    F = _as_F(F)
    n, m = F.shape
    w = _norm_weights(m, weights)
    rng = criterion_ranges(F)
    q, p = float(q_frac) * rng, float(p_frac) * rng
    if np.any(p < q):
        raise ValueError("p_frac 须 ≥ q_frac")

    # D[a,b,j] = max(0, F[b,j] − F[a,j])：b 比 a 差的幅度（成本型）
    D = np.maximum(0.0, F[None, :, :] - F[:, None, :])
    P = np.zeros((n, n, m))
    for j in range(m):
        P[:, :, j] = linear_pref(D[:, :, j], q[j], p[j])
    flows = (P * w[None, None, :]).sum(axis=2)          # flows[a,b] = P(a,b)
    phi = (flows - flows.T).sum(axis=1) / max(n - 1, 1)
    ranking = list(np.argsort(-phi))
    return {
        "phi": phi,
        "flows_ab": flows,
        "ranking": ranking,
        "selected": int(ranking[0]),
        "selected_F": F[ranking[0]].copy(),
    }


# ---------------------------------------------------------------------------
# ELECTRE III（可信度 + 净可信度流；无完整蒸馏）
# ---------------------------------------------------------------------------

def electre3(F, weights=None, q_frac: float = 0.02, p_frac: float = 0.08,
             v_frac: float = 0.25) -> dict:
    """ELECTRE III 可信度矩阵 σ 与净可信度流排序。

    记 Δ_j(a,b) = F[b,j] − F[a,j]（成本型：Δ>0 即 b 劣于 a 的幅度）：
      一致指数（支持 a S b 的证据）c_j = 1（Δ≥p）/ (Δ−q)/(p−q)（q<Δ<p）/
      0（Δ≤q）——注意方向：a 比 b 好得越多一致指数越高；
      不一致指数（反对 a S b，即 a 明显更差）δ_j = 0（−Δ≤p）/
      (−Δ−p)/(v−p)（−v<−Δ<... 即 Δ∈(−v,−p)）/ 1（Δ≤−v，veto）；
      c(a,b) = Σ w_j c_j；
      σ(a,b) = c(a,b) · Π_{j: δ_j > c} (1−δ_j)/(1−c)（任一 δ_j=1 且 c<1 → σ=0）。
    返回 {sigma, phi, ranking, selected, selected_F, concordance}。
    """
    F = _as_F(F)
    n, m = F.shape
    if not (0.0 <= q_frac <= p_frac <= v_frac <= 1.0):
        raise ValueError("阈值须满足 0≤q_frac≤p_frac≤v_frac≤1")
    w = _norm_weights(m, weights)
    rng = criterion_ranges(F)
    q, p, v = q_frac * rng, p_frac * rng, v_frac * rng

    delta = F[None, :, :] - F[:, None, :]              # delta[a,b,j] = F[b,j]−F[a,j]
    worse = -delta                                     # a 劣于 b 的幅度
    # 一致：Δ≥p → 1；Δ≤q → 0；中间线性
    c_denom = np.where(p > q, p - q, 1.0)
    c_j = np.where(delta >= p, 1.0,
                   np.where(delta <= q, 0.0,
                            np.clip((delta - q) / c_denom, 0.0, 1.0)))
    # 不一致：worse≤p → 0；worse≥v → 1；中间线性
    d_denom = np.where(v > p, v - p, 1.0)
    d_j = np.where(worse <= p, 0.0,
                   np.where(worse >= v, 1.0,
                            np.clip((worse - p) / d_denom, 0.0, 1.0)))
    c = (c_j * w[None, None, :]).sum(axis=2)             # (n,n)

    # σ = c · Π_{j: d_j > c} (1−d_j)/(1−c)；c=1 → σ=1；任一 d_j=1 且 c<1 → 0
    sigma = c.copy()
    for j in range(m):
        bad = (d_j[:, :, j] > c) & (c < 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(bad, (1.0 - d_j[:, :, j]) / (1.0 - c + 1e-30), 1.0)
        sigma = sigma * np.clip(factor, 0.0, 1.0)
    sigma = np.clip(sigma, 0.0, 1.0)

    phi = (sigma - sigma.T).sum(axis=1) / max(n - 1, 1)
    ranking = list(np.argsort(-phi))
    return {
        "sigma": sigma,
        "concordance": c,
        "phi": phi,
        "ranking": ranking,
        "selected": int(ranking[0]),
        "selected_F": F[ranking[0]].copy(),
    }


# ---------------------------------------------------------------------------
# 选点分发 + 稳定性/效率口径（实验脚本共用）
# ---------------------------------------------------------------------------

def select_compromise(method: str, F, weights=None, **kw) -> dict:
    """按方法名选点。method ∈ {"topsis","promethee","electre"}。"""
    if method == "topsis":
        from src.nsga2.topsis import topsis
        r = topsis(F, weights)
        return {"selected": int(r["index"]), "selected_F": np.asarray(r["selected_F"]),
                "score": np.asarray(r["closeness"])}
    if method == "promethee":
        return promethee2(F, weights, **kw)
    if method == "electre":
        return electre3(F, weights, **kw)
    raise ValueError(f"未知 method: {method!r}")


def uto_distance(F, idx: int) -> float:
    """选点效率：选中点到理想点（各列最小）的归一化距离 / sqrt(m)。越小越"均衡"。"""
    F = _as_F(F)
    ideal = F.min(axis=0)
    rng = criterion_ranges(F)
    d = (F[idx] - ideal) / rng
    return float(np.sqrt((d ** 2).sum()) / np.sqrt(F.shape[1]))
