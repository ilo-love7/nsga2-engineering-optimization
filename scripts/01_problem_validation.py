"""01 问题建模校验：单目标端点多起点 SLSQP + ZDT 解析前沿核对。

口径：文献最优值不可外部核实（离线），故参考端点一律用
**本项目独立复算**（多起点 SLSQP + 约束最小化）作为可复现基准，
报告中如实标注「自算参考」。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as sp_minimize

from common import save_json
from src.problems.pressure_vessel import PressureVessel
from src.problems.speed_reducer import SpeedReducer


def _wrap(problem, obj_idx: int):
    """目标 obj_idx 单目标化 + 约束 dict 给 SLSQP。"""
    def f(x):
        F, _ = problem.evaluate(x[None, :], return_values_of=["F", "G"])
        return float(np.asarray(F)[0, obj_idx])

    def c_ub(x):  # scipy convention: c(x) >= 0
        _, G = problem.evaluate(x[None, :], return_values_of=["F", "G"])
        return -np.asarray(G)[0]   # G<=0 → -G >= 0

    return f, c_ub


def solve_single(problem, obj_idx: int, n_starts: int = 40, seed: int = 0,
                 fallback_x0: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    f, c_ub = _wrap(problem, obj_idx)
    bounds = list(zip(problem.xl, problem.xu))
    cons = [{"type": "ineq", "fun": c_ub}]
    best_x, best_f = None, np.inf
    fails = 0
    starts = [rng.uniform(problem.xl, problem.xu) for _ in range(n_starts)]
    if fallback_x0 is not None:
        starts.append(np.asarray(fallback_x0, dtype=float))
    for x0 in starts:
        try:
            r = sp_minimize(f, x0, bounds=bounds, constraints=cons,
                            method="SLSQP", options={"maxiter": 400, "ftol": 1e-10})
        except Exception:
            fails += 1
            continue
        if r.success and r.fun < best_f and np.all(c_ub(r.x) >= -1e-6):
            best_f, best_x = float(r.fun), r.x.copy()
    return {"f_min": best_f if best_x is not None else float("nan"),
            "x_min": best_x.tolist() if best_x is not None else None,
            "n_starts": len(starts), "fails": fails}


def zdt1_front(n: int = 200) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1 - np.sqrt(f1)])


def main() -> None:
    out: dict = {}

    # ---------- 压力容器：min 成本 / min 重量 ----------
    pv = PressureVessel()
    fb_pv = np.array([1.2, 0.6, 60.0, 200.0])   # 手选可行起点兜底
    pv_cost = solve_single(pv, 0, n_starts=60, seed=1, fallback_x0=fb_pv)
    pv_weight = solve_single(pv, 1, n_starts=60, seed=2, fallback_x0=fb_pv)
    out["pressure_vessel"] = {
        "min_cost": pv_cost, "min_weight": pv_weight,
    }
    print(f"PV min cost   : f={pv_cost['f_min']:.4f}  x={np.round(pv_cost['x_min'],4)}")
    print(f"PV min weight : f={pv_weight['f_min']:.4f}  x={np.round(pv_weight['x_min'],4)}")

    # ---------- 减速器：min 重量 / min 应力 ----------
    sr = SpeedReducer()
    fb_sr = np.array([3.5, 0.7, 17.0, 7.8, 7.8, 3.35, 5.29])  # 手选可行起点兜底
    sr_w = solve_single(sr, 0, n_starts=60, seed=3, fallback_x0=fb_sr)
    sr_s = solve_single(sr, 1, n_starts=60, seed=4, fallback_x0=fb_sr)
    out["speed_reducer"] = {"min_weight": sr_w, "min_stress": sr_s}
    def _fmt(d):
        return (f"f={d['f_min']:.4f}  x={np.round(d['x_min'],4)}"
                if d["x_min"] is not None else f"f={d['f_min']}  x=None")
    print(f"SR min weight : {_fmt(sr_w)}")
    print(f"SR min stress : {_fmt(sr_s)}")

    # ---------- ZDT1 解析前沿自检 ----------
    from pymoo.problems import get_problem
    zdt1 = get_problem("zdt1")
    pf = zdt1.pareto_front()
    f1 = pf[:, 0]
    err = np.max(np.abs(pf[:, 1] - (1 - np.sqrt(f1))))
    out["zdt1_analytic_check"] = {"n_points": int(pf.shape[0]), "max_abs_err": float(err)}
    print(f"ZDT1 analytic PF max err: {err:.2e}  (n={pf.shape[0]})")

    # ---------- WeldedBeam 参考前沿可加载性 ----------
    wb = get_problem("welded_beam")
    pf_wb = wb.pareto_front()
    out["welded_beam_ref_pf"] = {
        "n_points": int(pf_wb.shape[0]),
        "f1_range": [float(pf_wb[:, 0].min()), float(pf_wb[:, 0].max())],
        "f2_range": [float(pf_wb[:, 1].min()), float(pf_wb[:, 1].max())],
    }
    print(f"WB ref PF: n={pf_wb.shape[0]}  f1∈[{pf_wb[:,0].min():.4f},{pf_wb[:,0].max():.4f}]"
          f"  f2∈[{pf_wb[:,1].min():.5f},{pf_wb[:,1].max():.5f}]")

    save_json("01_problem_validation.json", out)
    print("saved: results/01_problem_validation.json")


if __name__ == "__main__":
    main()
