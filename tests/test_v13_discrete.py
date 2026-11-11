"""v1.3 E25 离散算子模块单测：量化/索引/修复/网格算子/三管线。"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.discrete import (
    PIPELINES,
    GridCrossover,
    GridMutation,
    GridRepair,
    GridSampling,
    from_grid_index,
    grid_levels,
    make_grid_algorithm,
    quantize_to_grid,
    quantize_violation_delta,
    run_grid_nsga2,
    to_grid_index,
)
from src.problems.pressure_vessel import PressureVessel

GRID = 0.0625
COLS = (0, 1)
LO = np.array([0.0625, 0.0625, 10.0, 10.0])
HI = np.array([6.1875, 6.1875, 200.0, 200.0])


class TestQuantize:
    def test_rounds_to_grid_and_clips(self):
        X = np.array([[0.10, 0.20, 50.0, 100.0],
                      [-1.0, 99.0, 5.0, 500.0]])
        Xq = quantize_to_grid(X, GRID, COLS, LO, HI)
        assert Xq[0, 0] == pytest.approx(0.125)   # 0.10 → 2 格
        assert Xq[0, 1] == pytest.approx(0.1875)  # 0.20 → 3 格
        assert Xq[1, 0] == pytest.approx(LO[0])   # 夹下界
        assert Xq[1, 1] == pytest.approx(HI[1])   # 夹上界
        assert Xq[1, 2] == pytest.approx(LO[2])
        assert Xq[1, 3] == pytest.approx(HI[3])

    def test_idempotent(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 4)) * (HI - LO) + LO
        X1 = quantize_to_grid(X, GRID, COLS, LO, HI)
        X2 = quantize_to_grid(X1, GRID, COLS, LO, HI)
        assert np.array_equal(X1, X2)

    def test_grid_levels_and_index_roundtrip(self):
        lv = grid_levels(0.0625, 6.1875, GRID)
        assert lv[0] == pytest.approx(0.0625)
        assert lv[-1] == pytest.approx(6.1875)
        assert np.allclose(np.diff(lv), GRID)
        idx = to_grid_index(lv, GRID, 0.0625)
        assert np.array_equal(idx, np.arange(len(lv)))
        assert np.allclose(from_grid_index(idx, GRID, 0.0625), lv)

    def test_bad_grid(self):
        with pytest.raises(ValueError):
            quantize_to_grid(np.zeros((1, 4)), -1.0, COLS, LO, HI)


class TestOperators:
    def test_grid_repair_idempotent(self):
        pv = PressureVessel()
        rep = GridRepair(GRID, COLS)
        rng = np.random.default_rng(1)
        X = rng.random((20, 4)) * (HI - LO) + LO

        class _Pop:
            def __init__(self, X):
                self.X = X

            def __iter__(self):
                class _I:
                    pass
                out = []
                for row in self.X:
                    ind = _I()
                    ind.X = row
                    out.append(ind)
                return iter(out)

            def set(self, key, val):
                self.X = val

        pop = _Pop(X.copy())
        rep.do(pv, pop)
        X1 = pop.X
        X2 = quantize_to_grid(X1, GRID, COLS, LO, HI)
        assert np.array_equal(X1, X2)

    def test_grid_sampling_on_grid(self):
        pv = PressureVessel()
        samp = GridSampling(GRID, COLS)
        pop = samp(pv, 30, random_state=np.random.default_rng(2))
        X = pop.get("X")
        assert X.shape == (30, 4)
        assert np.array_equal(X, quantize_to_grid(X, GRID, COLS, LO, HI))
        assert np.all(X >= LO - 1e-12) and np.all(X <= HI + 1e-12)

    def test_grid_crossover_shape_and_grid(self):
        pv = PressureVessel()
        cro = GridCrossover(GRID, COLS)
        rng = np.random.default_rng(3)
        X = rng.random((2, 7, 4)) * (HI - LO) + LO
        Q = cro._do(pv, X, random_state=rng)
        assert Q.shape == (2, 7, 4)
        Qq = quantize_to_grid(Q.reshape(-1, 4), GRID, COLS, LO, HI)
        assert np.allclose(Q.reshape(-1, 4), Qq)

    def test_grid_mutation_stays_on_grid_and_deterministic(self):
        pv = PressureVessel()
        mut = GridMutation(GRID, COLS)
        X = np.full((10, 4), 1.0)
        Y1 = mut._do(pv, X, random_state=np.random.default_rng(4))
        Y2 = mut._do(pv, X, random_state=np.random.default_rng(4))
        assert np.array_equal(Y1, Y2)
        assert np.array_equal(Y1, quantize_to_grid(Y1, GRID, COLS, LO, HI))

    def test_grid_mutation_creep_is_grid_multiple(self):
        pv = PressureVessel()
        mut = GridMutation(GRID, COLS, p_creep=1.0, p_jump=0.0)
        X = np.full((20, 4), 1.0)
        Y = mut._do(pv, X, random_state=np.random.default_rng(5))
        delta = (Y[:, 0] - 1.0) / GRID
        assert np.allclose(delta, np.rint(delta))   # 只走整格
        assert np.all(np.abs(delta) <= 3)           # creep ≤3 格


class TestPipelines:
    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_make_algorithm(self, pipeline):
        pv = PressureVessel()
        alg = make_grid_algorithm(pipeline, GRID, COLS, pop_size=10)
        assert alg is not None

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_run_grid_nsga2_smoke(self, pipeline):
        pv = PressureVessel()
        out = run_grid_nsga2(pv, pipeline, grid=GRID, discrete_cols=COLS,
                             pop_size=16, n_gen=4, seed=1)
        assert out["X"].shape == (16, 4)
        assert out["F"].shape == (16, 2)
        assert out["G"].shape == (16, 4)
        X = out["X"]
        assert np.array_equal(X, quantize_to_grid(X, GRID, COLS, LO, HI))
        F, G = pv.evaluate(X, return_values_of=["F", "G"])
        assert np.allclose(out["F"], np.asarray(F))
        assert np.allclose(out["G"], np.asarray(G))

    def test_run_deterministic(self):
        pv = PressureVessel()
        r1 = run_grid_nsga2(pv, "grid_ops", pop_size=12, n_gen=3, seed=7)
        r2 = run_grid_nsga2(pv, "grid_ops", pop_size=12, n_gen=3, seed=7)
        assert np.array_equal(r1["X"], r2["X"])

    def test_quantize_violation_delta(self):
        pv = PressureVessel()
        rng = np.random.default_rng(6)
        X = rng.random((40, 4)) * (HI - LO) + LO
        d = quantize_violation_delta(pv, X, GRID, COLS)
        assert set(d) == {"mean_delta", "max_delta", "frac_worse"}
        assert 0.0 <= d["frac_worse"] <= 1.0
        # 已在网格上的点：增量为 0
        Xg = quantize_to_grid(X, GRID, COLS, LO, HI)
        d0 = quantize_violation_delta(pv, Xg, GRID, COLS)
        assert d0["max_delta"] == pytest.approx(0.0, abs=1e-12)
