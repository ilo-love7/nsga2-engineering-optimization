"""v1.2 单测：鲁棒优化（公差采样 / 可行概率 / mean·worst·reliability 包装）。

核心口径：
- 同 seed 采样可复现；评估结果对同一 perturbations 矩阵确定。
- 名义最优（贴约束边界）在 ±1% 公差下 p_feas 显著 < 1；内点 = 1。
- worst 目标 ≥ mean 目标（逐元素）；reliability 模式单列联合机会约束。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.robust import (
    RobustWrapper, evaluate_scenarios, reliability, robust_stats,
    sample_perturbations,
)
from src.problems.pressure_vessel import PressureVessel

PV = PressureVessel()
XL, XU = PV.xl, PV.xu
X_NOM = np.array([0.7782, 0.3846, 40.32, 200.0])   # v1.0 名义成本最优（贴边界）
X_IN = np.array([1.5, 0.8, 50.0, 120.0])           # 内点（远离活跃约束）


def P(n=200, tol=0.01, seed=7):
    return sample_perturbations(n, PV.n_var, tol, XL, XU, seed=seed)


class TestSampling:
    def test_shape_and_reproducibility(self):
        a, b = P(), P()
        assert a.shape == (200, 4)
        assert np.allclose(a, b)

    def test_bound_by_delta(self):
        delta = 0.01 * (XU - XL)
        p = P(n=500, seed=1)
        assert np.all(np.abs(p) <= delta + 1e-12)

    def test_zero_tol_gives_nominal(self):
        p = sample_perturbations(10, 4, 0.0, XL, XU, seed=0)
        assert np.allclose(p, 0.0)
        F, G = evaluate_scenarios(PV, X_NOM, p)
        assert np.allclose(F, F[0])
        assert np.allclose(G, G[0])


class TestScenarios:
    def test_shapes(self):
        F, G = evaluate_scenarios(PV, X_NOM, P())
        assert F.shape == (200, 2) and G.shape == (200, 4)

    def test_reliability_interior_vs_boundary(self):
        p = P()
        assert reliability(evaluate_scenarios(PV, X_IN, p)[1]) == 1.0
        r_nom = reliability(evaluate_scenarios(PV, X_NOM, p)[1])
        assert 0.0 < r_nom < 1.0     # 名义最优贴边界 → 公差下部分不可行

    def test_robust_stats_keys(self):
        st = robust_stats(PV, X_NOM, P())
        for k in ("F_mean", "F_worst", "F_std", "p_feas", "G_worst"):
            assert k in st
        assert st["F_mean"].shape == (2,)
        # 最坏 ≥ 均值（逐元素）
        assert np.all(st["F_worst"] >= st["F_mean"] - 1e-12)


class TestRobustWrapper:
    def test_mean_mode(self):
        rw = RobustWrapper(PV, P(), mode="mean")
        r = rw.evaluate(np.array([X_NOM, X_IN]), return_values_of=["F", "G"])
        F, G = r[0], r[1]
        assert F.shape == (2, 2) and G.shape == (2, 4)
        st = robust_stats(PV, X_NOM, P())
        assert np.allclose(F[0], st["F_mean"])

    def test_worst_mode_dominates_mean(self):
        p = P()
        rw_w = RobustWrapper(PV, p, mode="worst")
        rw_m = RobustWrapper(PV, p, mode="mean")
        Fw, _ = rw_w.evaluate(np.array([X_NOM]), return_values_of=["F", "G"])
        Fm, _ = rw_m.evaluate(np.array([X_NOM]), return_values_of=["F", "G"])
        assert np.all(np.asarray(Fw)[0] >= np.asarray(Fm)[0] - 1e-9)

    def test_reliability_mode_single_constraint(self):
        rw = RobustWrapper(PV, P(), mode="reliability", p_fail_max=0.05)
        assert rw.n_ieq_constr == 1
        # pymoo 单项 return_values_of 直接返回数组（不是 tuple）→ 双取防歧义
        r = rw.evaluate(np.array([X_IN, X_NOM]), return_values_of=["F", "G"])
        G = np.asarray(r[1])
        assert G.shape == (2, 1)
        assert G[0, 0] <= 0.0        # 内点 p_fail=0 → 满足
        assert G[1, 0] > 0.0         # 名义最优 p_fail>0.05 → 违约

    def test_deterministic_given_perturbations(self):
        p = P(seed=3)
        rw1 = RobustWrapper(PV, p, mode="mean")
        rw2 = RobustWrapper(PV, p, mode="mean")
        F1, _ = rw1.evaluate(np.array([X_NOM]), return_values_of=["F", "G"])
        F2, _ = rw2.evaluate(np.array([X_NOM]), return_values_of=["F", "G"])
        assert np.allclose(np.asarray(F1), np.asarray(F2))

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            RobustWrapper(PV, P(), mode="median")

    def test_unconstrained_problem(self):
        from pymoo.problems import get_problem
        z1 = get_problem("zdt1")
        p = sample_perturbations(50, 30, 0.01, z1.xl, z1.xu, seed=0)
        rw = RobustWrapper(z1, p, mode="mean")
        assert rw.n_ieq_constr == 0
        r = rw.evaluate(np.zeros((2, 30)), return_values_of=["F"])
        assert np.asarray(r[0]).shape[-1] == 2
        assert reliability(None) == 1.0
