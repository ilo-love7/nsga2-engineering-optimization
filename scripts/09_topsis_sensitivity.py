"""09 v1.1：TOPSIS 选点稳健性——权重扫描轨迹 + 膝点对比 + 非支配断言。

对 07 精修后的 WB/PV/SR 前沿：
  A. w1 从 0.05→0.95（w2=1−w1）扫描，记录 TOPSIS 选点轨迹（选点随权重
     应沿前沿连续迁移；跳变 = 选点对权重不稳健）。
  B. 膝点（归一化前沿到弦 x+y=1 的最大距离点）与等权 TOPSIS 对比。
  C. 权重 ±10% 扰动下的选点稳定率。
  D. 断言：所有选点均为前沿行（非支配由 07 保证，此处复核）。
"""
from __future__ import annotations

import numpy as np

from common import RESULTS, save_fig, save_json
from src.nsga2.local_search import nondominated_mask
from src.nsga2.topsis import topsis

W_GRID = np.linspace(0.05, 0.95, 19)
PERTURB = 200
LABELS = {
    "wb": ("welded beam", "cost", "deflection"),
    "pv": ("pressure vessel", "cost", "weight"),
    "sr": ("speed reducer", "weight", "max stress"),
}


def knee_index(F: np.ndarray) -> int:
    """归一化前沿到弦 (0,1)–(1,0) 的最大距离点（凸前沿的膝点）。"""
    Fn = (F - F.min(axis=0)) / np.maximum(F.max(axis=0) - F.min(axis=0), 1e-30)
    dist = np.abs(Fn[:, 0] + Fn[:, 1] - 1.0) / np.sqrt(2.0)
    return int(np.argmax(dist))


def main() -> None:
    import matplotlib.pyplot as plt

    out: dict = {"w_grid": W_GRID.tolist()}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))

    for j, tag in enumerate(["wb", "pv", "sr"]):
        d = np.load(RESULTS / f"07_front_{tag}.npz")
        F, X = d["pol_F"], d["pol_X"]
        name, n1, n2 = LABELS[tag]
        assert nondominated_mask(F).all(), f"{tag}: front not ND"

        # A. 权重扫描
        traj_idx = []
        for w1 in W_GRID:
            r = topsis(F, weights=np.array([w1, 1.0 - w1]))
            traj_idx.append(r["index"])
        traj_idx = np.asarray(traj_idx)
        r_eq = topsis(F)  # 等权
        k_eq = int(r_eq["index"])
        k_knee = knee_index(F)

        # C. ±10% 扰动稳定率（等权邻域）
        rng = np.random.default_rng(0)
        same = 0
        for _ in range(PERTURB):
            w = np.array([1.0, 1.0]) * rng.uniform(0.9, 1.1, size=2)
            if topsis(F, weights=w)["index"] == k_eq:
                same += 1
        stability = same / PERTURB

        jumps = int(np.sum(np.diff(traj_idx) != 0))
        out[tag] = {
            "n": int(F.shape[0]),
            "traj_index": traj_idx.tolist(),
            "n_switches": jumps,
            "equal_weight_index": k_eq,
            "equal_weight_F": F[k_eq].tolist(),
            "equal_weight_X": X[k_eq].tolist(),
            "knee_index": k_knee,
            "knee_F": F[k_knee].tolist(),
            "knee_X": X[k_knee].tolist(),
            "knee_vs_equal_same": bool(k_eq == k_knee),
            "stability_pm10pct": stability,
        }
        print(f"[{tag}] switches={jumps} eq={F[k_eq]} knee={F[k_knee]} "
              f"same={k_eq == k_knee} stability={stability:.2f}")

        # 图上排：前沿 + 轨迹
        ax = axes[0, j]
        ax.plot(F[:, 0], F[:, 1], "-o", ms=3.5, lw=1.0, color="#1f77b4",
                label="polished front")
        T = F[traj_idx]
        ax.plot(T[:, 0], T[:, 1], "-", lw=1.4, color="#ff7f0e", alpha=0.8,
                label="TOPSIS trajectory")
        sc = ax.scatter(T[:, 0], T[:, 1], c=W_GRID, cmap="viridis", s=28,
                        zorder=5, label="w1")
        plt.colorbar(sc, ax=ax, fraction=0.046, label="w1")
        ax.plot(*F[k_eq], "*", ms=15, color="#d62728", zorder=6,
                label="equal weight")
        ax.plot(*F[k_knee], "D", ms=8, mfc="none", mec="k", mew=1.6, zorder=6,
                label="knee")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(n1)
        ax.set_ylabel(n2)
        ax.legend(fontsize=6.5, loc="best")

        # 图下排：选点 f1 / 贴近度 vs w1
        ax2 = axes[1, j]
        cl = [topsis(F, weights=np.array([w, 1 - w]))["closeness"][traj_idx[i]]
              for i, w in enumerate(W_GRID)]
        ax2.plot(W_GRID, F[traj_idx, 0] / F[:, 0].max(), "-o", ms=3,
                 label=f"{n1} (norm)")
        ax2.plot(W_GRID, F[traj_idx, 1] / F[:, 1].max(), "-s", ms=3,
                 label=f"{n2} (norm)")
        ax2.plot(W_GRID, np.asarray(cl), ":^", ms=3, color="#7f7f7f",
                 label="closeness")
        ax2.set_xlabel("w1")
        ax2.set_title(f"{name}: selection vs weight", fontsize=9)
        ax2.legend(fontsize=6.5)

    save_fig("09_topsis_sensitivity.png")
    save_json("09_topsis_sensitivity.json", out)
    print("saved: results/09_topsis_sensitivity.json + "
          "figures/09_topsis_sensitivity.png")


if __name__ == "__main__":
    main()
