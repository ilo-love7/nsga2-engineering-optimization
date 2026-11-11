"""v1.3 E23 约束处理模块单测：调度纯函数 / 包装器 / GenerationHook / run_handled。"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.constraints import (
    METHODS,
    AdaptivePenaltyWrapper,
    DynamicPenaltyWrapper,
    EpsilonConstraintWrapper,
    GenerationHook,
    adaptive_weight,
    dynamic_weight,
    epsilon_schedule,
    run_handled,
    wrap_problem,
)
from src.nsga2.constraint_strategies import violation
from src.problems.pressure_vessel import PressureVessel


class TestSchedules:
    def test_dynamic_weight_monotone_and_positive(self):
        ws = [dynamic_weight(t) for t in range(10)]
        assert ws[0] > 0
        assert all(ws[i] <= ws[i + 1] for i in range(9))

    def test_dynamic_weight_params(self):
        with pytest.raises(ValueError):
            dynamic_weight(-1)
        with pytest.raises(ValueError):
            dynamic_weight(0, c=0.0)
        with pytest.raises(ValueError):
            dynamic_weight(0, alpha=0.0)
        assert dynamic_weight(0, c=1.0, alpha=1.0) == 1.0
        assert dynamic_weight(3, c=1.0, alpha=1.0) == 4.0

    def test_adaptive_weight_direction(self):
        assert adaptive_weight(1e3, 0.0, target=0.2, gamma=0.5) > 1e3
        assert adaptive_weight(1e3, 1.0, target=0.2, gamma=0.5) < 1e3
        assert adaptive_weight(1e3, 0.2, target=0.2, gamma=0.5) == 1e3

    def test_adaptive_weight_bounds(self):
        assert adaptive_weight(1.0, 1.0, target=0.9, gamma=0.99, w_lo=1.0) >= 1.0
        assert adaptive_weight(1e12, 0.0, target=0.1, gamma=10.0, w_hi=1e12) <= 1e12
        with pytest.raises(ValueError):
            adaptive_weight(1.0, 1.5)
        with pytest.raises(ValueError):
            adaptive_weight(1.0, 0.5, target=0.0)

    def test_epsilon_schedule(self):
        eps = [epsilon_schedule(t, 10, eps0=1.0, k=1.0) for t in range(11)]
        assert eps[0] == 1.0
        assert eps[-1] == 0.0
        assert all(eps[i] >= eps[i + 1] - 1e-12 for i in range(10))
        with pytest.raises(ValueError):
            epsilon_schedule(0, 0)
        assert epsilon_schedule(5, 10, eps0=2.0, k=2.0) == pytest.approx(0.5)


class TestWrappers:
    def _pair(self):
        """一行可行、一行不可行（Ts 远小于 0.0193R 要求）。"""
        feas = np.array([[1.0, 0.6, 50.0, 100.0]])
        infeas = np.array([[0.0625, 0.0625, 100.0, 100.0]])
        return feas, infeas

    def test_dynamic_penalty_penalizes_only_infeasible(self):
        pv = PressureVessel()
        feas, infeas = self._pair()
        w = DynamicPenaltyWrapper(pv, c=0.5, alpha=2.0, mode="all")
        Ff, _ = pv.evaluate(feas, return_values_of=["F", "G"])
        Fi, _ = pv.evaluate(infeas, return_values_of=["F", "G"])
        Pf = w.evaluate(feas, return_values_of=["F"])
        Pi = w.evaluate(infeas, return_values_of=["F"])
        assert np.allclose(Pf, np.asarray(Ff))          # viol=0 → 不加罚
        assert np.all(Pi > np.asarray(Fi))               # 全目标加罚

    def test_dynamic_penalty_grows_with_gen(self):
        pv = PressureVessel()
        _, infeas = self._pair()
        w = DynamicPenaltyWrapper(pv, c=0.5, alpha=2.0, mode="all")
        w.set_gen(0)
        P0 = np.asarray(w.evaluate(infeas, return_values_of=["F"])).copy()
        w.set_gen(20)
        P20 = np.asarray(w.evaluate(infeas, return_values_of=["F"]))
        assert np.all(P20 > P0)

    def test_dynamic_penalty_mode_f1_only_first(self):
        pv = PressureVessel()
        _, infeas = self._pair()
        w = DynamicPenaltyWrapper(pv, c=0.5, alpha=2.0, mode="f1")
        Fi, _ = pv.evaluate(infeas, return_values_of=["F", "G"])
        Pi = np.asarray(w.evaluate(infeas, return_values_of=["F"]))
        assert Pi[0, 0] > np.asarray(Fi)[0, 0]
        assert Pi[0, 1] == pytest.approx(np.asarray(Fi)[0, 1])

    def test_adaptive_wrapper_responds_to_rate(self):
        pv = PressureVessel()
        _, infeas = self._pair()
        w = AdaptivePenaltyWrapper(pv, weight=1e3, target=0.2, gamma=0.5)
        w.set_rate(1.0)      # 可行偏多 → 降权
        assert w.weight < 1e3
        low = np.asarray(w.evaluate(infeas, return_values_of=["F"])).copy()
        w.set_rate(0.0)      # 不可行偏多 → 加权
        w.set_rate(0.0)
        high = np.asarray(w.evaluate(infeas, return_values_of=["F"]))
        assert np.all(high > low)

    def test_epsilon_wrapper_relaxes_g(self):
        pv = PressureVessel()
        _, infeas = self._pair()
        w = EpsilonConstraintWrapper(pv, total_gen=10, eps0=0.5, k=1.0)
        w.set_gen(0)
        _, G_raw = pv.evaluate(infeas, return_values_of=["F", "G"])
        _, G_eps = w.evaluate(infeas, return_values_of=["F", "G"])
        assert np.allclose(np.asarray(G_eps), np.asarray(G_raw) - 0.5)
        w.set_gen(10)        # ε→0
        _, G_end = w.evaluate(infeas, return_values_of=["F", "G"])
        assert np.allclose(np.asarray(G_end), np.asarray(G_raw))

    def test_wrap_problem_dispatch(self):
        pv = PressureVessel()
        for m in METHODS:
            prob, wrapper = wrap_problem(pv, m, total_gen=10)
            assert prob.n_var == pv.n_var
        rules, w = wrap_problem(pv, "rules", total_gen=10)
        assert rules is pv and w is None
        with pytest.raises(ValueError):
            wrap_problem(pv, "nope", total_gen=10)


class TestRunHandled:
    def test_smoke_original_reeval(self):
        pv = PressureVessel()
        out = run_handled(pv, "rules", pop_size=20, n_gen=5, seed=1)
        assert out["X"].shape == (20, 4)
        assert out["F"].shape == (20, 2)
        assert out["G"].shape == (20, 4)
        F, G = pv.evaluate(out["X"], return_values_of=["F", "G"])
        assert np.allclose(out["F"], np.asarray(F))
        assert np.allclose(out["G"], np.asarray(G))
        assert len(out["trace"]) == 5
        for rec in out["trace"]:
            assert set(rec) == {"t", "mean_viol", "feas_rate",
                                "relaxed_feas_rate", "state"}
            assert 0.0 <= rec["feas_rate"] <= 1.0
            assert rec["mean_viol"] >= 0.0

    @pytest.mark.parametrize("method", METHODS)
    def test_all_methods_run(self, method):
        pv = PressureVessel()
        out = run_handled(pv, method, pop_size=16, n_gen=4, seed=2)
        assert out["X"].shape[0] == 16
        assert out["method"] == method
        if method == "eps":
            for rec in out["trace"]:
                assert rec["relaxed_feas_rate"] >= rec["feas_rate"] - 1e-12
        if method == "dynamic":
            states = [rec["state"] for rec in out["trace"]]
            assert states == sorted(states)
        if method == "eps":
            states = [rec["state"] for rec in out["trace"]]
            assert states == sorted(states, reverse=True)

    def test_same_seed_deterministic(self):
        pv = PressureVessel()
        r1 = run_handled(pv, "static", pop_size=16, n_gen=4, seed=3)
        r2 = run_handled(pv, "static", pop_size=16, n_gen=4, seed=3)
        assert np.array_equal(r1["X"], r2["X"])
        assert np.array_equal(r1["F"], r2["F"])

    def test_hook_violation_matches_helper(self):
        pv = PressureVessel()
        hook = GenerationHook(inner=pv, method="rules")
        X = np.array([[1.0, 0.6, 50.0, 100.0], [0.0625, 0.0625, 100.0, 100.0]])

        class _Pop:
            def get(self, key):
                return X

        class _Alg:
            pop = _Pop()
            n_gen = 1

        hook.update(_Alg())
        assert len(hook.trace) == 1
        _, G = pv.evaluate(X, return_values_of=["F", "G"])
        assert hook.trace[0]["mean_viol"] == pytest.approx(
            float(violation(np.asarray(G)).mean()))
