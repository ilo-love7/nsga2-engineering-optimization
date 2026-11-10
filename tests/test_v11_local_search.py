"""v1.1 单测：e-constraint 参考前沿、非支配掩码方向、ASF/端点精修、膝点。"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.local_search import (
    endpoint_polish, eref_front, eval_fg, nondominated_mask, polish_front,
)
from src.nsga2.topsis import topsis
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer


class TestNondominatedMask:
    def test_direction_regression(self):
        # v1.0 陷阱回归：被支配的行必须删掉（删的是被支配者，不是支配者）
        F = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0]])
        m = nondominated_mask(F)
        assert m.tolist() == [True, False, True]

    def test_near_equal_tie(self):
        # 浮点微差的近重复劣点必须删（WB 端点平局案例）
        F = np.array([[36.42, 4.4e-4], [54.66, 4.4e-4 + 1e-11]])
        m = nondominated_mask(F)
        assert m.tolist() == [True, False]

    def test_keeps_true_front(self):
        F = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 2.0], [5.0, 1.0]])
        assert nondominated_mask(F).all()


class TestEref:
    def test_pv_front_quality(self):
        pv = PressureVessel()
        F, X = eref_front(pv, n_eps=8, n_starts=1, seed=0)
        assert F.shape[0] >= 6 and F.shape[1] == 2
        # 单调：f1 升序 → f2 降序
        assert np.all(np.diff(F[:, 1]) <= 1e-3)
        # 端点 vs 01 单目标参考（0.1%）
        assert abs(F[0, 0] / 5885.335 - 1) < 1e-3
        assert abs(F[-1, 1] / 6239.534 - 1) < 1e-3
        # 全部可行
        for x in X:
            _, G = eval_fg(pv, x)
            assert np.all(np.asarray(G) <= 1e-6)

    def test_sr_fixed_integer_z(self):
        sr = SpeedReducer()
        F, X = eref_front(sr, n_eps=5, n_starts=1, seed=3, fixed={2: 17.0})
        assert X.shape[0] >= 2
        assert np.allclose(X[:, 2], 17.0)
        assert abs(F[0, 0] / 2994.464 - 1) < 3e-3   # z=17 子前沿端点


class TestPolish:
    def test_endpoint_polish_pv_cost(self):
        pv = PressureVessel()
        x0 = np.array([1.0, 0.5, 50.0, 150.0])
        x, f, ok = endpoint_polish(pv, 0, x0)
        assert ok
        assert abs(f / 5885.335 - 1) < 5e-3

    def test_polish_front_nondominated(self):
        pv = PressureVessel()
        # 手造一条粗劣初始前沿
        X = np.array([
            [0.9, 0.45, 45.0, 190.0],
            [1.1, 0.55, 55.0, 120.0],
            [1.25, 0.62, 65.0, 12.0],
        ])
        F = np.array([eval_fg(pv, x)[0] for x in X])
        F2, X2 = polish_front(pv, X, F)
        assert nondominated_mask(F2).all()
        # 精修不得劣化前沿：f1 最小值不高于原 f1 最小值
        assert F2[:, 0].min() <= F[:, 0].min() + 1e-6


class TestKnee:
    def test_knee_on_convex_front(self):
        # 膝点逻辑与 09 一致：归一化前沿到弦 x+y=1 的最大距离点
        F = np.array([[1.0, 10.0], [2.0, 4.0], [3.0, 2.2], [4.0, 1.5],
                      [5.0, 1.1], [10.0, 1.0]])
        Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
        dist = np.abs(Fn[:, 0] + Fn[:, 1] - 1.0) / np.sqrt(2.0)
        k = int(np.argmax(dist))
        assert 1 <= k <= 4   # 膝点在中部，不在端点
