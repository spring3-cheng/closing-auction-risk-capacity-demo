# 数据字典

## 数据资产

| 逻辑名 | Release 文件 | 主要用途 | 完整行数 |
|---|---|---|---:|
| `panel` | `closing_auction_panel_2024_sample2m_v1.parquet` | 直接风险与 shape-strength 模型 | 2,000,000 |
| `b1` | `b1_zero_info_10bps_rematch_v1.parquet` | 零信息/固定报价基线 | 544,506 |
| `b3` | `b3_oracle_boundary_2024m12_v1.parquet` | 12 月事后 oracle 上限诊断 | 46,690 |
| `fill` | `fill_safe_capacity_min0p950_v1.parquet` | 静态成交深度约束 | 544,506 |

精确文件大小、SHA256、列名、日期范围与 smoke 合同见 `data/data_manifest.json`。

## 主面板关键字段

| 字段 | 含义 |
|---|---|
| `date`, `sym`, `side`, `quote_strategy` | 场景主键 |
| `x_adv` | 我方申报资金占 ADV 比例 |
| `vcp_145959` | `14:59:59` 可见的参考虚拟成交价 |
| `market_move_bps` | 最终收盘后才能确定的无我方订单市场变化，只用于标签拆解与诊断，禁止进入模型特征 |
| `impact_me_raw_bps` | 我方订单加入后的有符号边际冲击 |
| `impact_me_bad_bps` | 沿订单方向截断为非负的不利边际冲击 |
| `total_move_bps` | 最终收盘价相对 `vcp_145959` 的有符号总变化 |
| `total_bad_move_bps` | 沿订单方向截断为非负的不利总风险标签 |
| `sample_mode`, `sample_cap_rows` | 阶段性分层抽样口径 |
| `sample_stratum` | 清理后的抽样层标识，不含源文件路径 |

主面板共有 152 列。Notebook `01` 只通过显式白名单选取 `14:59:59` 可见特征，并用 `LEAKAGE_COLUMNS` 阻断 `market_move_bps`、最终价格、成交和标签字段；其他列存在于数据中不等于会进入模型。

## 辅助表关键字段

- B1：`submitted_x_adv`、`adv_cash`、`total_bad_move_bps`、`fill_ratio` 和基线状态字段。
- B3：`b3_oracle_refined_x_safe`、事后风险/成交信息及 fallback 状态。B3 使用未来信息，不能作为实时特征。
- Fill：`x_safe_fill`、`fill_model_status`、`fill_ratio_min`。`no_feasible_fill_ratio` 对应缺失容量，最终容量保持缺失并按 fail-closed 处理。

## 网格稀疏性

完整主面板包含 44 个正 `x_adv` 档，但每个 anchor 只观察到 1–14 档，中位数为 4，完整 44 档 anchor 数为 0。容量阶段因此把可见特征的 `x_adv` 设为 0，由模型生成 no-order 风险锚点，再在 `0 + 44` 点网格上统一打分；未观察档不能被解释为真实反事实结果。
