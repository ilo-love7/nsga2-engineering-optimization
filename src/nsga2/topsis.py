"""简版 TOPSIS 帕累托选点（等权；归一化 + 理想/负理想 + 贴近度）。"""
from __future__ import annotations

import numpy as np


def topsis(F: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """对 Pareto 目标矩阵 F (n, m) 做 TOPSIS，返回选中行与贴近度。

    步骤：向量归一化 → 加权 → 理想 A+ / 负理想 A− → 贴近度 C_i。
    所有目标按「越小越好」处理（成本型）。
    """
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    if weights is None:
        weights = np.ones(m)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    norm = F / np.sqrt((F**2).sum(axis=0))
    W = norm * w
    ideal = W.min(axis=0)          # 成本型：各列最小
    nadir = W.max(axis=0)

    d_pos = np.sqrt(((W - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((W - nadir) ** 2).sum(axis=1))
    closeness = d_neg / (d_pos + d_neg + 1e-30)
    k = int(np.argmax(closeness))
    return {
        "index": k,
        "closeness": closeness,
        "selected_F": F[k],
        "d_pos": d_pos, "d_neg": d_neg,
    }
