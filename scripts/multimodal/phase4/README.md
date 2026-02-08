# Phase 4：评估与诊断（GT 对齐 + 边界问题定位）

> 本 README 的风格对齐 `scripts/multimodal/phase1/README`：以“可直接复制运行”的命令为主，便于审计与复现。

0. 输入文件约定
- Ground Truth：`data/processed/groundtruth/{system}_ground_truth.json`
- Phase3 输出分区：
  - `data/processed/fusion/{system}_baseline_partition.json`
  - `data/processed/fusion/{system}_cac-final_partition.json`
- **Class order（强烈建议）**：`data/processed/fusion/{system}_class_order.json`
  - 用于将 Phase3 的“按索引编号”的分区 JSON（key 为 "0","1",...）映射回类名，以便与 GT 对齐评估。
- （可选）依赖矩阵：`data/processed/dependency/{system}_dependency_matrix.json`
  - 若该文件不存在，可先跳过 `--dep`，仅做 GT 对齐评估。

1. 评估：GT 对齐（Pairwise F1 / MoJoSim）

> 注意：如果你的 `--pred` 分区文件的 key 是类名（例如 `com.xxx.ClassName`），`--class_order` 可省略；
> 但本项目 Phase3 默认输出多为索引 key（"0","1"...），因此这里**统一推荐加上** `--class_order`。

# 评估 CAC-Final（推荐作为主结果）
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/acmeair_ground_truth.json --pred data/processed/fusion/acmeair_cac-final_partition.json --class_order data/processed/fusion/acmeair_class_order.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/daytrader_ground_truth.json --pred data/processed/fusion/daytrader_cac-final_partition.json --class_order data/processed/fusion/daytrader_class_order.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/jpetstore_ground_truth.json --pred data/processed/fusion/jpetstore_cac-final_partition.json --class_order data/processed/fusion/jpetstore_class_order.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/plants_ground_truth.json --pred data/processed/fusion/plants_cac-final_partition.json --class_order data/processed/fusion/plants_class_order.json

# 评估 Baseline（对照）
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/acmeair_ground_truth.json --pred data/processed/fusion/acmeair_baseline_partition.json --class_order data/processed/fusion/acmeair_class_order.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/daytrader_ground_truth.json --pred data/processed/fusion/daytrader_baseline_partition.json --class_order data/processed/fusion/daytrader_class_order.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/jpetstore_ground_truth.json --pred data/processed/fusion/jpetstore_baseline_partition.json --class_order data/processed/fusion/jpetstore_class_order.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/plants_ground_truth.json --pred data/processed/fusion/plants_baseline_partition.json --class_order data/processed/fusion/plants_class_order.json

# （可选）Sanity Check：Random / Monolith 极端基准
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/plants_ground_truth.json --pred data/processed/fusion/plants_cac-final_partition.json --class_order data/processed/fusion/plants_class_order.json --sanity

2. （可选）评估：加入依赖矩阵后的架构指标（IFN / SM / ICP / NED）

# 注意：只有在 `data/processed/dependency/{system}_dependency_matrix.json` 存在时才运行。
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/acmeair_ground_truth.json --pred data/processed/fusion/acmeair_cac-final_partition.json --class_order data/processed/fusion/acmeair_class_order.json --dep data/processed/dependency/acmeair_dependency_matrix.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/daytrader_ground_truth.json --pred data/processed/fusion/daytrader_cac-final_partition.json --class_order data/processed/fusion/daytrader_class_order.json --dep data/processed/dependency/daytrader_dependency_matrix.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/jpetstore_ground_truth.json --pred data/processed/fusion/jpetstore_cac-final_partition.json --class_order data/processed/fusion/jpetstore_class_order.json --dep data/processed/dependency/jpetstore_dependency_matrix.json
python scripts/multimodal/phase4/evaluate_partition_f1.py --gt data/processed/groundtruth/plants_ground_truth.json --pred data/processed/fusion/plants_cac-final_partition.json --class_order data/processed/fusion/plants_class_order.json --dep data/processed/dependency/plants_dependency_matrix.json

2.1 论文级“一键选最优点”：按 BCubed F1 选择最佳 cap（推荐）

> 目的：对 DPEP 的 `cap` 做敏感性 sweep，并以 **BCubed F1 最大**作为每个系统的“最优分区”选择准则。
> 该流程可直接写进论文方法段（parameter sensitivity + objective selection）。

运行（四系统默认）：
```powershell
python scripts/multimodal/phase4/select_best_by_bcubed.py --caps "0.05,0.08,0.10,0.12,0.14,0.16,0.18,0.20"
```

你也可以只跑某个系统（示例：DayTrader）：
```powershell
python scripts/multimodal/phase4/select_best_by_bcubed.py daytrader --caps "0.05,0.08,0.10,0.12,0.14,0.16,0.18,0.20"
```

主要输出：
- Paper-ready 汇总表：
  - `results/ablation/phase4_best_by_bcubed_*.csv`
  - `results/ablation/phase4_best_by_bcubed_*.md`
- 每个系统的“最佳分区快照”（不会被后续实验覆盖）:
  - `data/processed/fusion/{system}_cac-final_best_by_bcubed_partition.json`
- 每个系统的“最佳点元信息”（最优 cap、baseline vs ours、K、备注等）:
  - `data/processed/fusion/{system}_cac_best_by_bcubed.json`

备注（审稿友好）:
- 脚本会自动标注某些系统的 **Baseline granularity collapse**（例如 DayTrader 出现 K 很小但得分高），
  并写入汇总表的 `Note` 列，便于你在论文表格旁加脚注：
  - `Note: Baseline results in granularity collapse (K=2), whereas CAC maintains expert-recommended granularity (K≈GT_K).`

3. 边界冲突诊断（S_final 高但 U 低的“伪耦合”类对）

# 跑所有系统（默认阈值）
python scripts/multimodal/phase4/diagnose_boundaries.py

# 只诊断一个系统 + 自定义阈值（示例：daytrader）
python scripts/multimodal/phase4/diagnose_boundaries.py --system daytrader --s-th 0.5 --u-th 0.3 --topn 20

4. 说明
- Phase4 脚本只负责“评估与诊断”，不生成分区；分区由 Phase3 (`phase3_cac_evaluation.py`) 生成。
- 如果你更新了 Phase1 的融合矩阵或 Phase2 的不确定性推理，请重新跑 Phase1~Phase3，再跑本阶段。
