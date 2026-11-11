"""v1.3 E24 ELECTRE III / PROMETHEE II 单测：方向、veto、净流性质。"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.outranking import (
    criterion_ranges,
    electre3,
    linear_pref,
    promethee2,
    select_compromise,
    uto_distance,
)


class TestLinearPref:
    def test_boundaries(self):
        d = np.array([0.0, 0.05, 0.1, 0.2])
        p = linear_pref(d, q=0.05, p=0.1)
        assert p[0] == 0.0
        assert p[1] == 0.0
        assert p[2] == pytest.approx(1.0)
        assert p[3] == pytest.approx(1.0)

    def test_midpoint_and_monotone(self):
        p = linear_pref(np.array([0.075]), q=0.05, p=0.1)
        assert p[0] == pytest.approx(0.5)
        d = np.linspace(0, 1, 50)
        vals = linear_pref(d, q=0.1, p=0.6)
        assert np.all(np.diff(vals) >= -1e-12)

    def test_bad_thresholds(self):
        with pytest.raises(ValueError):
            linear_pref(np.array([0.1]), q=0.5, p=0.1)


class TestPromethee:
    def test_identical_points_zero_flow(self):
        F = np.array([[1.0, 2.0], [1.0, 2.0]])
        r = promethee2(F)
        assert np.allclose(r["phi"], 0.0)

    def test_dominated_direction(self):
        F = np.array([[0.0, 0.0], [1.0, 1.0]])
        r = promethee2(F)
        assert r["phi"][0] > 0 > r["phi"][1]
        assert r["selected"] == 0

    def test_symmetric_tradeoff(self):
        F = np.array([[0.0, 1.0], [1.0, 0.0]])
        r = promethee2(F, weights=np.array([1.0, 1.0]))
        assert r["phi"][0] == pytest.approx(r["phi"][1])

    def test_phi_sums_to_zero(self):
        rng = np.random.default_rng(0)
        F = rng.random((12, 3))
        r = promethee2(F)
        assert r["phi"].sum() == pytest.approx(0.0, abs=1e-9)

    def test_weights_shift_selection(self):
        F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        r1 = promethee2(F, weights=np.array([1.0, 0.0]) + 1e-12)
        r2 = promethee2(F, weights=np.array([0.0, 1.0]) + 1e-12)
        assert r1["selected"] == 0
        assert r2["selected"] == 2


class TestElectre3:
    def test_sigma_bounds_and_self(self):
        rng = np.random.default_rng(1)
        F = rng.random((8, 3))
        r = electre3(F)
        s = r["sigma"]
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_dominated_pair_full_outrank(self):
        F = np.array([[0.0, 0.0], [1.0, 1.0]])
        r = electre3(F)
        assert r["sigma"][0, 1] == pytest.approx(1.0)   # a 全面支配 b → a S b
        assert r["sigma"][1, 0] == pytest.approx(0.0)   # 反向被 veto
        assert r["selected"] == 0

    def test_veto_blocks_outranking(self):
        # a 在 j0 略优、j1 大劣 → c(a,b)=0.5 但 veto 把 σ(a,b) 清零；
        # 反向同样 veto → 互不 outrank（净流 0）。
        F = np.array([[0.0, 1.0], [0.5, 0.0]])
        r = electre3(F)
        assert r["concordance"][0, 1] == pytest.approx(0.5)
        assert r["sigma"][0, 1] == pytest.approx(0.0)
        assert r["sigma"][1, 0] == pytest.approx(0.0)
        assert r["phi"][0] == pytest.approx(0.0)
        assert r["phi"][1] == pytest.approx(0.0)

    def test_veto_threshold_params(self):
        with pytest.raises(ValueError):
            electre3(np.ones((3, 2)), q_frac=0.3, p_frac=0.2, v_frac=0.5)

    def test_single_point(self):
        r = electre3(np.array([[1.0, 2.0]]))
        assert r["selected"] == 0
        assert r["phi"][0] == pytest.approx(0.0)


class TestSelectAndUto:
    def test_select_dispatch(self):
        F = np.array([[0.0, 1.0], [0.6, 0.4], [1.0, 0.0]])
        for m in ("topsis", "promethee", "electre"):
            r = select_compromise(m, F)
            assert 0 <= r["selected"] < 3
            assert np.asarray(r["selected_F"]).shape == (2,)
        with pytest.raises(ValueError):
            select_compromise("nope", F)

    def test_uto_distance(self):
        F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        # F[0] 距理想点 (0,0)：d=(0,1) → ||d||/sqrt(m) = 1/sqrt(2)
        assert uto_distance(F, 0) == pytest.approx(np.sqrt(0.5))
        # 理想点自身距离为 0
        G = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert uto_distance(G, 0) == pytest.approx(0.0)

    def test_ranges_fallback(self):
        F = np.array([[1.0, 3.0], [1.0, 5.0]])
        r = criterion_ranges(F)
        assert r[0] == 1.0     # 退化列取 1
        assert r[1] == 2.0
