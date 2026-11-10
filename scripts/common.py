"""脚本公共：路径、JSON、绘图风格。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def save_json(name: str, obj: dict) -> Path:
    p = RESULTS / name

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv),
                 encoding="utf-8")
    return p


def save_fig(name: str) -> Path:
    p = FIGURES / name
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p
