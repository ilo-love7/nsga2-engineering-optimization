"""收敛/分布指标：GD、IGD、HV（pymoo 指标器 + 轻量独立实现对照）。"""
from __future__ import annotations

import numpy as np
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def gd(F: np.ndarray, ref: np.ndarray) -> float:
    """Generational Distance：解集 → 参考前沿的平均最近距离。"""
    return float(GD(np.asarray(ref, dtype=float))(np.asarray(F, dtype=float)))


def igd(F: np.ndarray, ref: np.ndarray) -> float:
    """Inverted GD：参考前沿 → 解集的平均最近距离（兼顾收敛+分布）。"""
    return float(IGD(np.asarray(ref, dtype=float))(np.asarray(F, dtype=float)))


def hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume（越大越好）。ref_point 须被前沿严格支配。"""
    return float(HV(ref_point=np.asarray(ref_point, dtype=float))(
        np.asarray(F, dtype=float)))


def hv_ref_point(F: np.ndarray, scale: float = 1.1) -> np.ndarray:
    """由目标量级生成 HV 参考点（fmax × scale，常规做法）。"""
    F = np.asarray(F, dtype=float)
    return np.max(F, axis=0) * scale


def gd_igd_numpy(F: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """无 pymoo 依赖的 GD/IGD（测试对照用，欧氏距离）。"""
    F = np.asarray(F, dtype=float)
    R = np.asarray(ref, dtype=float)
    d = np.sqrt(((F[:, None, :] - R[None, :, :]) ** 2).sum(axis=2))
    g = float(d.min(axis=1).mean())
    i = float(d.min(axis=0).mean())
    return g, i


def normalize_fronts(F: np.ndarray, ref: np.ndarray):
    """按参考前沿量程逐目标 min-max 归一化（v1.2：多量纲目标用）。

    scale 取 ref 的 (max−min)，退化维度（常数目标）除 1。
    """
    F = np.asarray(F, dtype=float)
    R = np.asarray(ref, dtype=float)
    lo = R.min(axis=0)
    hi = R.max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    return (F - lo) / scale, (R - lo) / scale


def igd_norm(F: np.ndarray, ref: np.ndarray) -> float:
    """归一化 IGD：先按参考前沿量程归一化再算 IGD（跨量纲可比）。"""
    Fn, Rn = normalize_fronts(F, ref)
    return igd(Fn, Rn)


def gd_norm(F: np.ndarray, ref: np.ndarray) -> float:
    """归一化 GD。"""
    Fn, Rn = normalize_fronts(F, ref)
    return gd(Fn, Rn)
