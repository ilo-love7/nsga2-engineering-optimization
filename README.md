# N9 · NSGA-II 工程多目标优化

博远线上科研新项目提案 v2 · N9（机械设计/优化 · ★★）。

pymoo 0.6.2 实现 NSGA-II，求解焊接梁/压力容器/减速器/离合器（制动器）四类约束工程双目标问题，复现 Pareto 前沿并做 GD/IGD/HV 收敛评估、约束处理策略对比与 TOPSIS 折中选点。基准文献：Deb et al., IEEE Trans. Evol. Comput. 2002。

## 流水线

```
pymoo Problem 建模（2 内置 + 2 自研）─→ 01 单目标端点校验（多起点 SLSQP）
        │
        ├─→ 02 NSGA-II 前沿复现（ZDT 解析 / WB 官方参考 / 自算端点对照）
        ├─→ 03 调参扫描 pop×gen → GD/IGD/HV 收敛曲线
        ├─→ 04 约束策略：可行性规则 / 罚(f1) / 罚(全) / 死刑 / 修复
        └─→ 05 TOPSIS 折中选点 + 汇总
```

## 核心结果（v1.0.0）

| 指标 | 数值 |
|---|---|
| ZDT1/2/3 IGD | 5.5e-3 / 6.5e-3 / 4.4e-3（vs 解析前沿） |
| 焊接梁 GD / IGD | 5.35e-2 / 7.10e-2（vs pymoo 官方 257 点参考） |
| 减速器端点 gap | 0.04% / 0.00%（g8/g11 活跃约束精确命中） |
| 压力容器端点 gap | 2.70% / 0.06%（f1 角点=EA 欠收敛，SLSQP 5885.34 稳命中） |
| 约束策略 | 可行性规则最优；**罚仅 f1 结构性失效**（可行率 1.7%） |
| TOPSIS | 四题折中解 C*=0.744–0.923，决策变量全录 |

26 项单测全绿；`run_all.sh` 一键 01–05（~3 min）。

## 关键口径

- 文献数字离线不可核 → 参考前沿两类：解析/官方（ZDT、pymoo welded_beam.pf）+ 自算单目标端点（SLSQP 60 起点，报告如实标注）。
- 约束输出按量级归一化（g3 ~1.3e6 不归一化会误判不可行）。
- 非支配过滤方向单测钉住（方向反了会删优留劣）。

## 复现

```bash
pip install -r requirements.txt
bash run_all.sh          # 01–05 实验 + pytest
```

结果 JSON 在 `results/`，前沿数据 `results/02_fronts.npz`（TOPSIS 可复算），图在 `figures/`，报告在 `reports/`。
