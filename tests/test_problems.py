"""问题建模单测：公式自洽、经典解复现、约束口径。"""
from __future__ import annotations

import numpy as np
import pytest

from src.problems.clutch_wrap import Clutch
from src.problems.pressure_vessel import PressureVessel, repair_pv, single_objective_cost
from src.problems.speed_reducer import SpeedReducer, repair_sr


class TestPressureVessel:
    def test_shapes(self):
        p = PressureVessel()
        X = np.array([[0.8, 0.4, 50.0, 150.0]])
        F, G = p.evaluate(X, return_values_of=["F", "G"])
        assert F.shape == (1, 2) and G.shape == (1, 4)

    def test_classic_continuous_optimum(self):
        """经典连续最优 (0.7782, 0.3847, 40.32, 200) 成本≈5885.36 且可行。"""
        p = PressureVessel()
        x = np.array([[0.7782, 0.3847, 40.3196, 200.0]])
        F, G = p.evaluate(x, return_values_of=["F", "G"])
        assert float(F[0, 0]) == pytest.approx(5885.36, rel=2e-4)
        assert (G[0] <= 1e-3).all(), f"经典最优点应可行: G={G[0]}"
        assert float(F[0, 0]) == pytest.approx(single_objective_cost(x[0]), rel=1e-12)

    def test_repair_discrete(self):
        x = np.array([[0.7782, 0.3847, 40.3, 200.0]])
        xr = repair_pv(x)
        assert (xr[0, :2] % 0.0625 == 0).all()
        assert np.all(xr >= PressureVessel().xl - 1e-12)

    def test_volume_constraint(self):
        """容积不足时 g3>0。"""
        p = PressureVessel()
        _, G = p.evaluate(np.array([[1.0, 1.0, 20.0, 20.0]]),
                          return_values_of=["F", "G"])
        assert G[0, 2] > 0


class TestSpeedReducer:
    def test_classic_optimum_feasible_and_nearoptimal(self):
        """经典解 (3.5,0.7,17,7.3,7.7153,3.3502,5.2867)：重量≈2994.4、可行。"""
        p = SpeedReducer()
        x = np.array([[3.5, 0.7, 17.0, 7.3, 7.7153, 3.3502, 5.2867]])
        F, G = p.evaluate(x, return_values_of=["F", "G"])
        assert float(F[0, 0]) == pytest.approx(2994.4, rel=2e-3)
        assert (G[0] <= 1e-3).all(), f"经典最优点应可行: G={G[0]}"

    def test_g8_g11_active_at_classic(self):
        p = SpeedReducer()
        x = np.array([[3.5, 0.7, 17.0, 7.3, 7.7153, 3.3502, 5.2867]])
        _, G = p.evaluate(x, return_values_of=["F", "G"])
        assert abs(G[0, 7]) < 2e-3   # g8 = 5m/b - 1 ≈ 0
        assert abs(G[0, 10]) < 2e-3  # g11 = (1.1 d2 + 1.9)/l2 - 1 ≈ 0

    def test_repair_integer_z(self):
        xr = repair_sr(np.array([[3.0, 0.75, 20.4, 7.8, 7.8, 3.3, 5.2]]))
        assert float(xr[0, 2]).is_integer()

    def test_bounds_consistency(self):
        """边界与 g10/g11 相容：界内至少存在可行点（经典点即证）。"""
        p = SpeedReducer()
        assert np.all(p.xu > p.xl)


class TestClutch:
    def test_wrapper_shapes(self):
        p = Clutch()
        X = np.array([[10, 5, 1, 200, 4]])   # 整数索引（x4 索引 200 → F=800）
        F, G = p.evaluate(X, return_values_of=["F", "G"])
        assert F.shape == (1, 2) and G.shape == (1, 8)

    def test_mass_formula(self):
        p = Clutch()
        F, _ = p.evaluate(np.array([[10, 5, 1, 200, 4]]),
                          return_values_of=["F", "G"])
        # 索引 10/5/1/200/4 → x1=70, x2=95, x3=1.5, x4=800, x5=6
        ri, ro, t, z = 70.0, 95.0, 1.5, 6.0
        rho = 0.0000078
        f1_ref = np.pi * (ro**2 - ri**2) * t * (z + 1) * rho
        assert float(F[0, 0]) == pytest.approx(f1_ref, rel=1e-9)
