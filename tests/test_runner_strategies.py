"""runner 与约束策略单测：小规模端到端。"""
from __future__ import annotations

import numpy as np
import pytest
from pymoo.problems import get_problem

from src.nsga2.constraint_strategies import (
    DeathPenaltyWrapper, PenaltyWrapper, RepairWrapper, violation,
)
from src.nsga2.runner import feasible_mask, nondominated_front, run_nsga2
from src.problems.pressure_vessel import PressureVessel, repair_pv


class TestRunner:
    def test_zdt1_small_run(self):
        p = get_problem("zdt1")
        res = run_nsga2(p, pop_size=60, n_gen=80, seed=1)
        # eliminate_duplicates 可能收缩种群 → 只断言规模上下界
        assert res.F.shape[1] == 2 and 10 <= res.F.shape[0] <= 60
        assert np.all(np.isfinite(res.F))
        # ZDT1 前沿 f2 = 1 - sqrt(f1)，收敛后中位误差应 < 0.15
        err = np.abs(res.F[:, 1] - (1 - np.sqrt(np.clip(res.F[:, 0], 0, 1))))
        assert np.median(err) < 0.15

    def test_clutch_integer_run(self):
        from src.problems.clutch_wrap import Clutch
        p = Clutch()
        res = run_nsga2(p, pop_size=30, n_gen=15, seed=1)
        assert res.F.shape[1] == 2 and 10 <= res.F.shape[0] <= 30

    def test_feasible_mask(self):
        G = np.array([[0.0, -1.0], [0.1, 0.0], [-0.1, -0.1]])
        m = feasible_mask(G)
        assert list(m) == [True, False, True]
        assert feasible_mask(None) is None

    def test_nondominated_front(self):
        F = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [2.5, 2.5]])
        front = nondominated_front(F)
        assert front.shape[0] == 3
        assert np.allclose(front[0], [1.0, 3.0])


class TestStrategies:
    def _small_run(self, prob, seed=1):
        return run_nsga2(prob, pop_size=30, n_gen=15, seed=seed)

    def test_penalty_wrapper_runs(self):
        pv = PressureVessel()
        w = PenaltyWrapper(pv, 1e5)
        res = self._small_run(w)
        assert res.F.shape[1] == 2 and 5 <= res.F.shape[0] <= 30
        assert w.n_ieq_constr == 0

    def test_death_penalty_big(self):
        pv = PressureVessel()
        w = DeathPenaltyWrapper(pv)
        X = np.array([[0.1, 0.1, 10.0, 10.0]])   # 不可行
        res = w.evaluate(X, return_values_of=["F"])
        F = res[0] if isinstance(res, tuple) else res
        assert F[0, 0] == DeathPenaltyWrapper.BIG

    def test_repair_wrapper_rounds(self):
        pv = PressureVessel()
        w = RepairWrapper(pv, repair_pv)
        res = self._small_run(w)
        # 评估出的 F 应全部有限
        assert np.all(np.isfinite(res.F))

    def test_violation(self):
        G = np.array([[0.0, -1.0], [0.2, 0.3]])
        v = violation(G)
        assert v[0] == 0.0 and v[1] == pytest.approx(0.5)
