"""v1.2 单测：三目标减速器（重量/应力/体积）公式一致性。"""
from __future__ import annotations

import numpy as np
import pytest

from src.nsga2.local_search import eval_fg
from src.nsga2.topsis import topsis
from src.problems.speed_reducer import SpeedReducer, repair_sr
from src.problems.speed_reducer3 import SpeedReducer3, repair_sr3

X0 = np.array([3.5, 0.7, 17.0, 7.3, 7.7153, 3.3502, 5.2867])


class TestSpeedReducer3:
    def test_matches_two_obj_shared_part(self):
        sr2, sr3 = SpeedReducer(), SpeedReducer3()
        F3, G3 = eval_fg(sr3, X0)
        F2, G2 = eval_fg(sr2, X0)
        assert np.allclose(F3[:2], F2, rtol=1e-12)
        assert np.allclose(G3, G2, rtol=1e-12)
        assert F3.shape[0] == 3

    def test_f3_volume_formula(self):
        F3, _ = eval_fg(SpeedReducer3(), X0)
        b, m, z, l1, l2, d1, d2 = X0
        f3_hand = (np.pi / 4.0) * (0.25 * b * (m * z) ** 2
                                   + l1 * d1**2 + l2 * d2**2)
        assert F3[2] == pytest.approx(f3_hand, rel=1e-12)

    def test_batch_shape(self):
        sr3 = SpeedReducer3()
        X = np.vstack([X0, X0 + 0.01])
        r = sr3.evaluate(X, return_values_of=["F", "G"])
        assert r[0].shape == (2, 3) and r[1].shape == (2, 11)

    def test_repair_rounds_z(self):
        x = repair_sr3(np.array([3.49, 0.71, 17.6, 7.2, 7.8, 3.4, 5.3]))
        assert x[2] == 18.0
        assert np.allclose(x, repair_sr(np.array(
            [3.49, 0.71, 17.6, 7.2, 7.8, 3.4, 5.3])))

    def test_known_design_feasible(self):
        # v1.1 经典解：g8/g11 活跃 → 残差 ~1e-5 但按 2e-3 口径可行
        _, G = eval_fg(SpeedReducer3(), X0)
        assert np.max(G) < 2e-3
        assert G[7] == pytest.approx(0.0, abs=2e-3)    # g8 = 5m/b − 1
        assert G[10] == pytest.approx(0.0, abs=2e-3)   # g11 活跃


class TestTopsis3D:
    def test_equal_weight_prefers_majority_corner(self):
        # 3 目标 TOPSIS 语义（v1.2 钉住）：等权下 (10,1,1)「2 优 1 劣」的
        # 贴近度 > 全中庸 (5.5,5.5,5.5) > (1,10,10)——d_pos 用欧氏距离聚合，
        # 两目标在负理想的代价高于单目标。中庸解不是等权 TOPSIS 的解。
        F = np.array([[1.0, 10.0, 10.0], [5.5, 5.5, 5.5], [10.0, 1.0, 1.0]])
        t = topsis(F)
        assert t["index"] == 2
        assert float(t["closeness"][2]) > float(t["closeness"][1]) > float(t["closeness"][0])
        # 镜像对称：A/C 贴近度互补（和为 1）
        assert float(t["closeness"][0]) + float(t["closeness"][2]) == pytest.approx(1.0)

    def test_weight_shifts_selection(self):
        F = np.array([[1.0, 10.0, 10.0], [5.5, 5.5, 5.5], [10.0, 1.0, 1.0]])
        t0 = topsis(F, weights=np.array([1.0, 1.0, 1.0]))
        t1 = topsis(F, weights=np.array([10.0, 1.0, 1.0]))   # 偏向 f1 最小
        assert t0["index"] == 2 and t1["index"] == 0
