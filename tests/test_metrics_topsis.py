"""指标与 TOPSIS 单测：解析可算情形 + numpy 对拍。"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.metrics import gd_igd_numpy, hv_ref_point, hypervolume
from src.nsga2.metrics import gd as gd_pm
from src.nsga2.metrics import igd as igd_pm
from src.nsga2.topsis import topsis


class TestGDIGD:
    def test_identical_sets_zero(self):
        R = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        g, i = gd_igd_numpy(R, R)
        assert g == pytest.approx(0.0, abs=1e-12)
        assert i == pytest.approx(0.0, abs=1e-12)

    def test_single_point_ref(self):
        F = np.array([[1.0, 1.0], [3.0, 4.0]])
        R = np.array([[0.0, 0.0]])
        g, i = gd_igd_numpy(F, R)
        assert g == pytest.approx((np.sqrt(2) + 5.0) / 2)
        assert i == pytest.approx(np.sqrt(2))  # ref → 最近解 (1,1)

    def test_pymoo_matches_numpy(self):
        rng = np.random.default_rng(0)
        F = rng.random((20, 2))
        R = rng.random((30, 2))
        g, i = gd_igd_numpy(F, R)
        assert gd_pm(F, R) == pytest.approx(g, rel=1e-9)
        assert igd_pm(F, R) == pytest.approx(i, rel=1e-9)


class TestHV:
    def test_three_points(self):
        F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        hv = hypervolume(F, np.array([1.1, 1.1]))
        # 支配体积（含斥容斥）：0.11 + 0.36 + 0.11 − 0.06 − 0.01 − 0.06 + 0.01 = 0.46
        assert hv == pytest.approx(0.46, rel=1e-6)

    def test_ref_point_helper(self):
        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(hv_ref_point(F, 1.1), [3.3, 4.4])


class TestTopsis:
    def test_ideal_point_selected(self):
        # 一行被支配？不——构造：点2 两目标均最小 → 必被选中
        F = np.array([[5.0, 5.0], [1.0, 1.0], [3.0, 3.0]])
        r = topsis(F)
        assert r["index"] == 1
        assert r["closeness"][1] == pytest.approx(1.0)

    def test_tradeoff_midpoint(self):
        F = np.array([[1.0, 10.0], [4.0, 4.0], [10.0, 1.0]])
        r = topsis(F)
        assert r["index"] == 1  # 等权 → 中点最贴近理想

    def test_weights_tilt(self):
        F = np.array([[1.0, 10.0], [4.0, 4.0], [10.0, 1.0]])
        r = topsis(F, weights=np.array([9.0, 1.0]))  # 重 f1 → 选 f1 最小
        assert r["index"] == 0
