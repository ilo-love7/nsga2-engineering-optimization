"""v1.3 E25 PV 离散版整数 NSGA-II：网格量化 / 修复 / 网格感知算子。

压力容器的 Ts/Th 工程上只能取 0.0625 in 整数倍（板材规格）。v1.0 的 D
策略是「连续算子 + eval 修复」；v1.3 对比三种算子管线（报告 §25）：

- eval_repair  连续 SBX/PM，eval 前 repair（RepairWrapper，v1.0 D 的算子观）
- op_repair    连续 SBX/PM + 算子级修复（GridRepair 挂 SBX/PM 的 repair=）
- grid_ops     网格感知算子（GridSampling + GridCrossover + GridMutation：
               Ts/Th 在网格索引空间均匀交叉/爬行/跳变，R/L 连续列仍走 SBX）

另含网格分辨率扫描（HV 损失 vs 连续参考前沿）与量化引起的约束违反增量
（对连续参考前沿每点量化后 Δviol 分布）。

口径：
- 所有管线最终 X 一律 quantize 后回原问题重评（对 eval_repair 是"实际
  设计点"；对另两条是恒等）——指标同口径才可比；
- GridCrossover 的连续列走 pymoo cross_sbx（与 op_repair 同源），仅
  离散列处理不同 → A2 vs B 的差异被隔离在离散列上；
- 自定义算子的随机数一律走 random_state（缺省 default_rng(0) 兜底，
  禁熵回落——v1.2 锦标赛熵泄漏的教训）。
"""
from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX, cross_sbx
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from src.nsga2.constraint_strategies import RepairWrapper

PIPELINES = ("eval_repair", "op_repair", "grid_ops")


# ---------------------------------------------------------------------------
# 纯函数：网格量化 / 索引映射
# ---------------------------------------------------------------------------

def quantize_to_grid(X, grid: float, discrete_cols, xl, xu) -> np.ndarray:
    """离散列取整到 grid 倍数并夹边；连续列仅夹边。幂等。"""
    if grid <= 0:
        raise ValueError("grid 须 > 0")
    X = np.array(X, dtype=float, copy=True)
    X = np.atleast_2d(X)
    xl = np.asarray(xl, dtype=float).ravel()
    xu = np.asarray(xu, dtype=float).ravel()
    for j in discrete_cols:
        X[:, j] = np.round(X[:, j] / grid) * grid
    return np.clip(X, xl, xu)


def to_grid_index(x, grid: float, lo: float) -> np.ndarray:
    """网格坐标 → 整数索引（0 基）。"""
    return np.rint(np.asarray(x, dtype=float) / grid - lo / grid).astype(int)


def from_grid_index(idx, grid: float, lo: float) -> np.ndarray:
    """整数索引 → 网格坐标。"""
    return np.asarray(idx, dtype=float) * grid + lo


def grid_levels(lo: float, hi: float, grid: float) -> np.ndarray:
    """[lo,hi] 上全部合法网格坐标（升序）。"""
    if hi < lo:
        raise ValueError("hi 须 ≥ lo")
    n = int(np.floor((hi - lo) / grid + 1e-9)) + 1
    return lo + grid * np.arange(n)


def quantize_violation_delta(problem, X, grid: float, discrete_cols) -> dict:
    """量化引起的约束违反增量 Δviol = viol(quantize(x)) − viol(x)。

    viol = Σ max(0, g_i)（原问题约束）。返回
    {mean_delta, max_delta, frac_worse}：quantize 通常增加违反（厚度取整
    后可能压过内压约束线），frac_worse 为 Δviol>1e-9 的样本占比。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Xq = quantize_to_grid(X, grid, discrete_cols, problem.xl, problem.xu)
    _, G0 = problem.evaluate(X, return_values_of=["F", "G"])
    _, G1 = problem.evaluate(Xq, return_values_of=["F", "G"])
    v0 = np.maximum(0.0, np.asarray(G0, dtype=float)).sum(axis=1)
    v1 = np.maximum(0.0, np.asarray(G1, dtype=float)).sum(axis=1)
    d = v1 - v0
    return {
        "mean_delta": float(d.mean()),
        "max_delta": float(d.max()),
        "frac_worse": float((d > 1e-9).mean()),
    }


# ---------------------------------------------------------------------------
# pymoo 算子
# ---------------------------------------------------------------------------

class GridRepair(Repair):
    """算子级网格修复：离散列拉回 grid 网格 + 夹边（幂等）。"""

    def __init__(self, grid: float, discrete_cols, **kwargs):
        super().__init__(**kwargs)
        self.grid = float(grid)
        self.discrete_cols = tuple(discrete_cols)

    def _do(self, problem, X, **kwargs):
        return quantize_to_grid(X, self.grid, self.discrete_cols,
                                problem.xl, problem.xu)


class GridSampling(Sampling):
    """均匀采样后离散列量化 → 初始种群直接落在网格上。"""

    def __init__(self, grid: float, discrete_cols):
        super().__init__()
        self.grid = float(grid)
        self.discrete_cols = tuple(discrete_cols)

    def _do(self, problem, n, *args, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng(0)
        lo = np.asarray(problem.xl, dtype=float).ravel()
        hi = np.asarray(problem.xu, dtype=float).ravel()
        X = rng.random((int(n), problem.n_var)) * (hi - lo) + lo
        return quantize_to_grid(X, self.grid, self.discrete_cols, lo, hi)


class GridCrossover(Crossover):
    """离散列：索引空间均匀交叉（+25% 概率邻格爬行混合）；连续列：cross_sbx。

    连续列与 op_repair 管线同源（pymoo cross_sbx，默认 η=15/prob_var=0.5），
    使「A2 vs B」的差异被隔离在离散列的处理上。
    """

    def __init__(self, grid: float, discrete_cols, n_offsprings: int = 2,
                 prob: float = 0.9, **kwargs):
        super().__init__(2, n_offsprings, prob=prob, **kwargs)
        self.grid = float(grid)
        self.discrete_cols = tuple(discrete_cols)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng(0)
        n_matings = X.shape[1]
        xl = np.asarray(problem.xl, dtype=float).ravel()
        xu = np.asarray(problem.xu, dtype=float).ravel()

        # 连续列：与 SBX 相同的 cross_sbx（对全部列跑，离散列稍后覆盖）
        eta = np.full((n_matings, 1), 15.0)
        prob_var = np.full((n_matings, 1), 0.5)
        prob_bin = np.full((n_matings, 1), 0.5)
        Q = cross_sbx(X.astype(float), xl, xu, eta, prob_var, prob_bin,
                      random_state=rng)

        # 离散列：索引空间均匀交叉 + 爬行混合
        for j in self.discrete_cols:
            i1 = to_grid_index(X[0, :, j], self.grid, xl[j])
            i2 = to_grid_index(X[1, :, j], self.grid, xl[j])
            swap = rng.random(n_matings) < 0.5
            k1 = np.where(swap, i2, i1).astype(int)
            k2 = np.where(swap, i1, i2).astype(int)
            # 25% 概率对换来的索引做 ±1 邻格爬行（离散搜索的"微调"）
            creep = rng.random(n_matings) < 0.25
            step = rng.choice(np.array([-1, 1]), size=n_matings)
            k1 = np.where(creep, k1 + step, k1)
            k2 = np.where(creep, k2 - step, k2)
            Q[0, :, j] = from_grid_index(k1, self.grid, xl[j])
            Q[1, :, j] = from_grid_index(k2, self.grid, xl[j])

        Q = quantize_to_grid(Q.reshape(-1, problem.n_var), self.grid,
                             self.discrete_cols, xl, xu)
        Q = Q.reshape(2, n_matings, problem.n_var)      # cross_sbx 恒出 2 子代
        if self.n_offsprings == 1:
            Q = Q[[0]]
        return Q


class GridMutation(Mutation):
    """离散列：爬行（±1..3 格）/ 跳变（全域重采样）；连续列：高斯步。

    与 PM 的差异被隔离在离散列；连续列步长 σ=0.1·(hi−lo) 幅度级匹配。
    """

    def __init__(self, grid: float, discrete_cols, p_creep: float = 0.25,
                 p_jump: float = 0.05, prob: float = 1.0, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.grid = float(grid)
        self.discrete_cols = tuple(discrete_cols)
        self.p_creep = float(p_creep)
        self.p_jump = float(p_jump)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng(0)
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        xl = np.asarray(problem.xl, dtype=float).ravel()
        xu = np.asarray(problem.xu, dtype=float).ravel()
        Y = X.copy()

        for j in self.discrete_cols:
            u = rng.random(n)
            creep = u < self.p_creep
            jump = (u >= self.p_creep) & (u < self.p_creep + self.p_jump)
            step = rng.integers(1, 4, size=n) * rng.choice(np.array([-1, 1]), size=n)
            Y[creep, j] = X[creep, j] + step[creep] * self.grid
            if jump.any():
                lv = grid_levels(xl[j], xu[j], self.grid)
                Y[jump, j] = rng.choice(lv, size=int(jump.sum()))

        cont = np.array([j for j in range(d) if j not in self.discrete_cols])
        if cont.size:
            sigma = 0.1 * (xu[cont] - xl[cont])
            event = rng.random((n, cont.size)) < 0.5
            Y[:, cont] = X[:, cont] + event * rng.normal(0.0, 1.0, (n, cont.size)) * sigma

        return quantize_to_grid(Y, self.grid, self.discrete_cols, xl, xu)


# ---------------------------------------------------------------------------
# 三管线运行
# ---------------------------------------------------------------------------

def make_grid_algorithm(pipeline: str, grid: float, discrete_cols,
                        pop_size: int = 100) -> NSGA2:
    """按管线构造 NSGA-II（算子差异只在离散列/修复位点）。"""
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline 须为 {PIPELINES} 之一")
    cols = tuple(discrete_cols)
    sampling = GridSampling(grid, cols)
    if pipeline == "eval_repair":
        return NSGA2(pop_size=pop_size, sampling=sampling,
                     crossover=SBX(prob=0.9, eta=15.0),
                     mutation=PM(prob=1.0, eta=20.0),
                     eliminate_duplicates=True)
    if pipeline == "op_repair":
        rep = GridRepair(grid, cols)
        return NSGA2(pop_size=pop_size, sampling=sampling,
                     crossover=SBX(prob=0.9, eta=15.0, repair=rep),
                     mutation=PM(prob=1.0, eta=20.0, repair=rep),
                     eliminate_duplicates=True)
    return NSGA2(pop_size=pop_size, sampling=sampling,
                 crossover=GridCrossover(grid, cols),
                 mutation=GridMutation(grid, cols),
                 eliminate_duplicates=True)


def run_grid_nsga2(problem, pipeline: str, grid: float = 0.0625,
                   discrete_cols=(0, 1), pop_size: int = 100,
                   n_gen: int = 100, seed: int = 1) -> dict:
    """跑一条管线，返回量化后回原问题重评的最终种群。

    eval_repair：问题包 RepairWrapper（eval 时修复，v1.0 D 语义）；
    另两条：算子自带量化，问题原样。X 一律 quantize（幂等）再重评。
    """
    cols = tuple(discrete_cols)
    algo = make_grid_algorithm(pipeline, grid, cols, pop_size)
    target = problem
    if pipeline == "eval_repair":
        target = RepairWrapper(problem, repair_fn=lambda X: quantize_to_grid(
            X, grid, cols, problem.xl, problem.xu))
    res = minimize(target, algo, ("n_gen", n_gen), seed=seed, verbose=False)
    X_raw = np.atleast_2d(np.asarray(res.algorithm.pop.get("X"), dtype=float))
    X = quantize_to_grid(X_raw, grid, cols, problem.xl, problem.xu)
    F, G = problem.evaluate(X, return_values_of=["F", "G"])
    return {
        "X_raw": X_raw, "X": X,
        "F": np.asarray(F, dtype=float),
        "G": np.asarray(G, dtype=float),
        "pipeline": pipeline, "grid": float(grid), "seed": seed,
    }
