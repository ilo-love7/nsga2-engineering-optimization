"""v1.2 单测：多算法封装（NSGA-III/MOEA-D）、Das-Dennis 参考方向、
标量化参考前沿。锁定口径：
- MOEA/D 约束问题必须经死刑法包装（pymoo 断言不支持约束）；
- run_moea 在原问题上重评 F/G（惩罚值 1e12 不得泄漏进指标）；
- 等预算 = n_evals。
"""
from __future__ import annotations

import numpy as np
import pytest
from pymoo.problems import get_problem

from src.nsga2.constraint_strategies import DeathPenaltyWrapper
from src.nsga2.local_search import eval_fg, nondominated_mask
from src.nsga2.manyobj import (
    das_dennis, make_moea, pooled_front, run_moea, scalarized_ref_front,
    simplex_lattice,
)
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer3 import SpeedReducer3


class TestRefDirs:
    @pytest.mark.parametrize("m,p,n", [(2, 4, 5), (2, 12, 13), (3, 4, 15),
                                       (3, 12, 91), (4, 5, 56)])
    def test_das_dennis_count(self, m, p, n):
        from math import comb
        rd = das_dennis(m, p)
        assert rd.shape == (comb(p + m - 1, m - 1), m) == (n, m)

    def test_simplex_properties(self):
        rd = das_dennis(3, 8)
        assert np.allclose(rd.sum(axis=1), 1.0)
        assert (rd >= 0).all()

    def test_simplex_lattice_step_validation(self):
        w = simplex_lattice(3, 0.25)
        assert w.shape == (15, 3)
        with pytest.raises(ValueError):
            simplex_lattice(3, 0.3)

    def test_bad_args(self):
        with pytest.raises(ValueError):
            das_dennis(1, 4)
        with pytest.raises(ValueError):
            das_dennis(3, 0)


class TestMakeMoea:
    def test_algorithm_types(self):
        z1 = get_problem("zdt1")
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.nsga3 import NSGA3
        alg, inner = make_moea("nsga2", z1, pop_size=20)
        assert isinstance(alg, NSGA2) and inner is z1
        alg, inner = make_moea("nsga3", z1, pop_size=20,
                               ref_dirs=das_dennis(2, 6))
        assert isinstance(alg, NSGA3) and inner is z1
        alg, inner = make_moea("moead", z1,
                               ref_dirs=das_dennis(2, 6))
        assert isinstance(alg, MOEAD) and inner is z1   # 无约束不包装

    def test_moead_wraps_constraint_problem(self):
        # pymoo MOEAD 断言 not has_constraints → 约束问题必须包装
        pv = PressureVessel()
        alg, inner = make_moea("moead", pv, ref_dirs=das_dennis(2, 6))
        assert isinstance(inner, DeathPenaltyWrapper)
        assert not inner.has_constraints()

    def test_unknown_name(self):
        with pytest.raises(ValueError):
            make_moea("nsga12", get_problem("zdt1"), pop_size=10)


class TestRunMoea:
    @pytest.mark.parametrize("name", ["nsga2", "nsga3", "moead"])
    def test_zdt1_reevaluated_on_original(self, name):
        z1 = get_problem("zdt1")
        res = run_moea(z1, name, budget=1200, pop_size=30,
                       ref_dirs=das_dennis(2, 29), seed=1)
        assert res.F.shape[1] == 2 and res.G is None
        assert np.all(np.isfinite(res.F))
        # 原问题重评一致性
        F2, _ = eval_fg(z1, res.X[0])
        assert np.allclose(res.F[0], F2, rtol=1e-9)

    def test_budget_stops_by_n_evals(self):
        z1 = get_problem("zdt1")
        res = run_moea(z1, "nsga2", budget=900, pop_size=30, seed=2)
        # n_evals 预算终止：评估数应 ≥ 预算且 < 预算+pop
        assert 900 <= res.algorithm.evaluator.n_eval < 900 + 30

    def test_moead_constraint_problem_no_penalty_leak(self):
        # 死刑法包装下 MOEA/D 原生 F 为 1e12 → 重评后不得出现惩罚值
        pv = PressureVessel()
        res = run_moea(pv, "moead", budget=1500, pop_size=24,
                       ref_dirs=das_dennis(2, 23), seed=3)
        assert res.G is not None and res.F.shape[1] == 2
        assert np.max(np.abs(res.F)) < 1e8
        feas = (res.G <= 1e-6).all(axis=1)
        assert feas.mean() > 0.3   # 部分可行（小预算下不要求全可行）


class TestPooledFront:
    def test_dominance_and_dedup(self):
        a = np.array([[1.0, 2.0], [3.0, 1.0]])
        b = np.array([[2.0, 1.5], [3.0, 1.0]])   # 与 a 重复 + 被支配点混入
        F, X = pooled_front([a, b], [np.zeros((2, 3)), np.ones((2, 3))])
        assert F.shape[0] == 3
        assert np.allclose(F[0], [1.0, 2.0])
        assert np.all(np.diff(F[:, 0]) >= 0)


class TestNormalizedMetrics:
    def test_normalize_by_ref_range(self):
        from src.nsga2.metrics import normalize_fronts
        ref = np.array([[0.0, 10.0], [1.0, 20.0]])
        F = np.array([[0.5, 15.0]])
        Fn, Rn = normalize_fronts(F, ref)
        assert np.allclose(Fn, [[0.5, 0.5]])
        assert np.allclose(Rn, [[0.0, 0.0], [1.0, 1.0]])

    def test_igd_norm_scale_invariance(self):
        # 同一几何关系、不同量纲 → igd_norm 相同（igd 会差 1000 倍）
        from src.nsga2.metrics import gd_norm, igd, igd_norm
        ref = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        F = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
        n1 = igd_norm(F, ref)
        F2, ref2 = F * 1000.0, ref * 1000.0
        assert igd_norm(F2, ref2) == pytest.approx(n1, rel=1e-9)
        assert igd(F2, ref2) == pytest.approx(igd(F, ref) * 1000.0, rel=1e-9)
        assert gd_norm(F, ref) > 0

    def test_constant_dimension_scale_one(self):
        from src.nsga2.metrics import normalize_fronts
        ref = np.array([[5.0, 0.0], [5.0, 1.0]])
        Fn, _ = normalize_fronts(np.array([[5.0, 0.5]]), ref)
        assert np.allclose(Fn, [[0.0, 0.5]])   # 常数维度不炸


class TestCrossCallDeterminism:
    """回归：pymoo 0.6.2 comp_by_cv_then_random 漏传 random_state →
    compare() 熵回落（default_rng(None)），同 seed 跨调用漂移。
    修复后 run_moea(seed=...) 必须与前置调用无关。"""

    def test_nsga3_same_seed_reproducible_after_prior_calls(self):
        sr3 = SpeedReducer3()
        rd = das_dennis(3, 4)
        r1 = run_moea(sr3, "nsga3", budget=400, seed=1, ref_dirs=rd)
        # 干扰：不同 seed / 不同算法的前置调用
        run_moea(sr3, "nsga3", budget=400, seed=2, ref_dirs=rd)
        run_moea(sr3, "moead", budget=400, seed=3, ref_dirs=rd)
        r2 = run_moea(sr3, "nsga3", budget=400, seed=1, ref_dirs=rd)
        assert np.array_equal(r1.X, r2.X)
        assert np.array_equal(r1.F, r2.F)

    def test_fixed_comp_forwards_random_state(self):
        from pymoo.operators.selection.tournament import compare
        from src.nsga2.manyobj import comp_by_cv_then_random_fixed
        # 两个 CV 相等的不可行个体 → 平局掷币必须走给定 rng（可复现）
        class Ind:
            def __init__(self, cv):
                self.CV = cv
        pop = [Ind(1.0), Ind(1.0)]
        P = np.array([[0, 1]])
        s1 = comp_by_cv_then_random_fixed(pop, P,
                                          random_state=np.random.default_rng(7))
        s2 = comp_by_cv_then_random_fixed(pop, P,
                                          random_state=np.random.default_rng(7))
        assert np.array_equal(s1, s2)
        # 缺省 random_state 时不得依赖熵（default_rng(0) 兜底仍可复现）
        s3 = comp_by_cv_then_random_fixed(pop, P)
        s4 = comp_by_cv_then_random_fixed(pop, P)
        assert np.array_equal(s3, s4)

    def test_make_moea_uses_fixed_selection(self):
        from src.nsga2.manyobj import comp_by_cv_then_random_fixed
        z1 = get_problem("zdt1")
        alg, _ = make_moea("nsga3", z1, pop_size=20, ref_dirs=das_dennis(2, 6))
        # pymoo 0.6 GeneticAlgorithm 把 selection 包进 Mating → alg.mating.selection
        assert alg.mating.selection.func_comp is comp_by_cv_then_random_fixed


class TestScalarizedRefFront:
    def test_pv_two_obj_small(self):
        pv = PressureVessel()
        W = simplex_lattice(2, 0.25)   # 5 权重
        F, X = scalarized_ref_front(pv, weights=W, n_starts=1, seed=0,
                                    polish=False)
        assert F.shape[1] == 2 and F.shape[0] >= 3
        assert nondominated_mask(F).all()
        # 端点应接近单目标最优（多起点 SLSQP 精度 ~0.5%）
        assert abs(F[:, 0].min() / 5885.335 - 1) < 5e-3
        assert abs(F[:, 1].min() / 6239.534 - 1) < 5e-3
        for x in X:
            _, G = eval_fg(pv, x)
            assert np.all(np.asarray(G) <= 1e-6)

    def test_sr3_three_obj(self):
        sr3 = SpeedReducer3()
        W = simplex_lattice(3, 0.5)   # 6 权重（快）
        F, X = scalarized_ref_front(sr3, weights=W, n_starts=1, seed=1,
                                    polish=False)
        assert F.shape[1] == 3 and F.shape[0] >= 3
        assert nondominated_mask(F).all()
        # 全部可行 + z 连续域内
        for x in X:
            _, G = eval_fg(sr3, x)
            assert np.all(np.asarray(G) <= 1e-6)
            assert 17.0 <= x[2] <= 28.0
