# 收盘集合竞价风险计量与容量评估 Clean-room Notebook 项目设计

> 发布状态补充：原设计按“可公开发布”边界构建；用户在实际 GitHub 交付时选择 private repository。`00_get_data.ipynb` 因此同时支持公开直链与 `GH_TOKEN`/`GITHUB_TOKEN` 鉴权的私人 Release API 下载。

## 1. 目标与边界

本项目在与原工作仓库物理隔离的新目录中，从零编写一个可公开发布、可在新设备复现并可继续优化的 Notebook 项目。项目研究 A 股收盘集合竞价，在 `14:59:59` 已知信息下，评估我方订单加入后最终收盘价相对 `vcp_145959` 的方向化不利风险，并在 10bps 风险预算下估计研究性资金容量上限。

完整复现的起点是经过授权公开再分发的 2024 年真实业务衍生面板，而不是 `adata` 原始库、逐笔委托成交原始表或上游数千个 chunk。项目必须完整覆盖数据校验、标签审计、多分位模型、因果校准、容量反解、成交容量约束、operating-point selection 和结果报告。

项目不继承旧工作仓库或其他现有仓库的 Git 历史，不直接复制既有 Notebook 或 Python 实现，不读取公司内部库，也不声称能够从内部原始库重建上游数据。

## 2. 交付形态

公开交付采用“GitHub 代码仓库 + GitHub Release 数据包”结构：

- Git 仓库保存五个从零编写的自包含 Notebook、说明文档、环境锁定、数据 manifest、测试和小型 smoke sample。
- GitHub Release `data-v1.0.0` 保存一个数据包 `closing_auction_demo_data_v1.zip`。
- 大型训练候选表和模型中间结果不进入 Git 历史，由 Notebook 在本地重新生成。
- `outputs/`、`models/`、完整数据和 Notebook 临时文件默认进入 `.gitignore`。

`00_get_data.ipynb` 从 `git remote origin` 推导 GitHub owner/repository。私人仓库通过 `GH_TOKEN`/`GITHUB_TOKEN` 查询 Release API 并按 asset ID 鉴权下载；公开仓库可使用 Release 直链。若没有 Git remote，则可读取用户显式设置的 `DATA_RELEASE_URL`；本地开发阶段还允许读取同名本地数据包。下载先写入 `.part` 文件，SHA256 校验通过后再原子重命名。

## 3. 数据发布合同

### 3.1 Release 数据包

`closing_auction_demo_data_v1.zip` 包含：

1. `closing_auction_panel_2024_sample2m_v1.parquet`
   - 2024-01-02 至 2024-12-31。
   - 2,000,000 行阶段性分层真实样本。
   - buy/sell 双方向和 44 个正 Cash/ADV 档位，范围为 `0.000002` 至 `0.10`。
   - 保留模型特征、风险标签、价格、盘口、ADV、抽样范围和评价范围字段。
   - 删除 `source_file`，将 `_impact_sample_stratum` 重命名为 `sample_stratum`；清理后预期 152 列。
2. `b1_zero_info_10bps_rematch_v1.parquet`
   - 2024 全年 B1 10bps protection rematch 基线。
   - 544,506 行、69 列。
3. `b3_oracle_boundary_2024m12_v1.parquet`
   - 2024-12-02 至 2024-12-31 的 B3 adaptive/refined oracle 边界。
   - 46,690 行。
   - 删除 `source_file` 和 `source_shard_id`，保留边界、实际结果、零容量 fallback 和评价状态字段。
4. `fill_safe_capacity_min0p950_v1.parquet`
   - 544,506 行静态成交容量上限，`fill_ratio_min=0.950`。
5. `data_manifest.json`
   - 记录数据版本、文件名、字节数、SHA256、行列数、日期范围、关键字段、生成日期、抽样口径和授权用途。

仓库内另保存一个确定性真实 smoke sample，只用于 schema、Notebook 执行和输出合同测试，不用于报告正式 coverage、pinball 或容量结论。

### 3.2 授权与来源

`DATA_LICENSE.md` 记录用户提供的公开再分发授权摘要，不虚构授权编号，不发布内部审批材料。`PROVENANCE.csv` 对每个公开文件记录：

```text
file_path
source_description
author
license_or_authorization
created_date
referenced_internal_material
reviewer
allowed_use
sha256
```

公开前必须扫描压缩包、Notebook、Markdown、JSON、CSV 和 Notebook 输出，确认不存在个人主目录、公司服务器目录、旧工作仓库等绝对路径，不存在旧 Git metadata、访问凭据或 `source_file` 字段。

### 3.3 数据口径限制

主面板的 `sample_mode` 为 `stratified_cap`，`sample_cap_rows` 为 2,000,000，`evaluation_scope` 为 `sample_level_evaluation`。所有结果必须称为阶段性样本离线结果，不得表述为全市场全量生产验证。

该面板是按行分层抽样的稀疏 candidate panel，而不是每个 `date × sym × side × quote_strategy` 都有完整 44 档的反事实网格。当前审计基线为 516,874 个唯一 anchor，每个 anchor 有 1 至 14 个已观测正档，完整 44 档 anchor 数为 0，中位数为 4 档。12 月 B3 的 46,690 个唯一 key 均可在主面板找到 anchor。

因此，模型训练使用稀疏已观测 candidate 行；容量阶段从 data manifest 读取完整 44 档 action grid，并对每个待评价 anchor 重新生成模型预测网格。未观测档位只能视为模型评分点，不能伪装成真实反事实 outcome。容量输出必须记录 observed-grid count、scored-grid count 和 grid source。

主面板已包含当前模型所需的 14:59:59 条件特征，因此单独的 `market_move_145959_features.csv`、`money_1s_panel.parquet` 和上游 chunks 不属于 v1 必需输入。它们不得因下载完成而自动加入 Release。

## 4. Notebook 架构

Notebook 按编号顺序运行，且不依赖仓库内额外 `.py` 文件。测试脚本可以用于合同验证，但业务运行逻辑必须存在于 Notebook 内。

### 4.1 `00_get_data.ipynb`

职责：

- 支持 `RUN_MODE="smoke"` 与 `RUN_MODE="full"`。
- 下载或读取本地 Release 数据包。
- 校验压缩包及包内文件 SHA256。
- 解压到相对路径 `data/raw/`。
- 校验文件存在、行列数、日期范围、关键字段和内部路径扫描结果。
- 写出 `outputs/00_data_contract.json` 和 `outputs/00_data_inventory.csv`。

校验失败时 fail closed，不允许继续运行后续 Notebook。

### 4.2 `01_build_panel_and_labels.ipynb`

职责：

- 标准化 `date`、`sym`、`side`、`x_adv`、`quote_strategy` 和数值字段。
- 生成公开、可解释的标签审计：sell 方向符号为 `-1`，其他有效 buy 方向为 `+1`。
- 按以下公式从公开字段重新计算并核验风险标签：

```text
impact_me_adv_bps = side_sign * impact_me_raw_bps
impact_me_bad_bps = max(impact_me_adv_bps, 0)
total_bad_move_bps = max(side_sign * total_move_bps, 0)
```

- 核对标签差异、空值、异常价格、重复 key、日期覆盖、side 和 Cash/ADV 档位。
- 为每个 anchor 生成 `x_adv=0` 的模型 no-order 风险锚点：total risk 来自只使用 `14:59:59` 可见特征的模型预测与历史校准，impact guardrail 为 0；不得用该日最终 `market_move_bps` 覆盖，该锚点也不作为额外已观测反事实行。
- 只保留经审计的 14:59:59 可见特征白名单。
- 写出 `data/processed/model_panel.parquet`、`outputs/01_label_audit.csv`、`outputs/01_feature_contract.csv`。

`total_bad_move_bps` 是主风险目标；`impact_me_bad_bps` 是我方边际冲击 guardrail。

### 4.3 `02_quantile_risk_model.ipynb`

职责：

- 使用可配置 quantile registry；默认训练 `q50/q80/q85/q90/q95/q99`，允许增加合法的 `(0,1)` quantile heads。
- direct LightGBM 区分 operating quantile 与 raw training tau，并在 manifest 中完整记录映射。默认复现映射为：`q50→0.50`、`q80→0.80`、`q85→0.85`、`q90→0.90`、`q95→0.80`、`q99→0.90`；较软的 raw tau 通过后续 calibration/floor 恢复 operating coverage。shape-strength 各 head 使用自己的训练 tau，并单独记录 shape、strength 和 calibration 身份。
- 对每个 quantile 独立训练 direct LightGBM quantile model。
- 训练并评价 shape-strength 模型族，使每个 quantile 拥有独立 shape/strength 风险候选。
- 使用严格按日期排序的 train/calibration/test 切分，禁止随机打散。
- calibration buffer 只能使用 calibration 残差；因果 rolling floor 只能使用更早 fold。
- final-month audit 只评价模型层，不参与模型、calibration buffer 或 floor 学习。
- 输出整体和 `side × ratio_bucket` coverage、pinball loss、预测分位水平、最差分桶和 crossing diagnostics。

跨 quantile 不做强制排序。每个 quantile 是独立 operating-point candidate；输出必须报告 raw/calibrated/final crossing rate、crossing bps 和最差分桶。允许额外生成 `_ordered` 对照列，但这些列默认不进入 policy selection 或正式容量字段。

### 4.4 `03_capacity_inversion.ipynb`

职责：

- 候选身份由以下字段唯一确定：

```text
model_family
quantile_level
prediction_stage
calibration_variant
floor_variant
boundary_solver
```

- 对每个候选分别构造 total-risk 和 impact-guardrail 风险曲线。
- impact guardrail quantile 独立配置，默认使用 0.95，不随 total-risk operating quantile 自动变化，并写入候选 manifest。
- 从 manifest 的 44 个正档加 `x_adv=0` no-order anchor 构造统一 45 点评分网格，不要求训练样本对每个 anchor 已观测全部档位。
- 只要求同一个候选沿 `x_adv` 的反解曲线单调，不要求不同 quantile 间有序。
- 使用累积最大值消除 ratio grid 局部下降，禁止通过非单调 wiggle 获得虚高容量。
- 在 10bps 风险预算下反解 price-risk capacity。
- direct LightGBM 使用单调线性插值，同时保留 grid-floor 诊断。
- shape-strength 输出 exact piecewise-linear H boundary 和网格对照；若 calibration buffer 随 `x_adv` 变化，exact-H 只能在经过一致性验证后进入正式候选。
- 最终可执行容量为 `min(price-risk capacity, x_safe_fill)`。
- 输出所有候选的 price-only、final-with-fill 容量和单调性诊断。

### 4.5 `04_optimization_and_report.ipynb`

职责：

- 在相同 10bps 风险预算和相同 B3 可评价样本上比较 direct LightGBM 与 shape-strength 候选。
- B3 boundary 缺失行不进入风险率或 submitted-cash 分母。
- 每个入选 model family 的 B3 boundary coverage 不低于 95%。
- 同时报告 cash-weighted、unweighted 和 material `side × ratio_bucket` 边界越界率。
- 在约束满足时最大化 price-only submitted cash，并报告叠加 fill 上限后的容量代价。
- 对比 B1、selected B2 和 B3 oracle；B3 使用事后信息，只是研究上限，不能部署。
- 生成整体 coverage、分桶 coverage、最差桶、pinball、crossing、保守程度、容量、B3 coverage、越界率和 oracle regret scorecard。

v1 固定采用当前复现口径：2024 年 12 月同时用于 operating-point selection 和 baseline evaluation。所有相关输出必须写入：

```text
evaluation_scope = same-sample policy-development diagnostic
selection_evaluation_overlap = true
independent_out_of_sample_policy_evaluation = false
```

不得把该结果描述为冻结 policy 的独立样本外证据。

## 5. 文件结构

```text
closing-auction-risk-capacity-demo/
├── .github/workflows/notebook-smoke.yml
├── .gitignore
├── README.md
├── DATA_LICENSE.md
├── PROVENANCE.csv
├── environment.yml
├── requirements-lock.txt
├── data/
│   ├── data_manifest.json
│   └── smoke/closing_auction_smoke_v1.parquet
├── docs/
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── result_interpretation.md
│   └── superpowers/specs/2026-08-07-closing-auction-clean-room-notebook-design.md
├── notebooks/
│   ├── 00_get_data.ipynb
│   ├── 01_build_panel_and_labels.ipynb
│   ├── 02_quantile_risk_model.ipynb
│   ├── 03_capacity_inversion.ipynb
│   └── 04_optimization_and_report.ipynb
├── reference/
│   └── reference_run_manifest.json
├── tests/
│   ├── test_data_contract.py
│   ├── test_notebook_contracts.py
│   └── test_output_contracts.py
└── outputs/
```

`reference/` 只能保存由 clean-room Notebook 全链运行生成的小型参考指标和 manifest，不复制旧仓库的大型结果表。

## 6. 环境与复现

环境锁定至少包含 Python、JupyterLab、pandas、numpy、pyarrow、scikit-learn、LightGBM、matplotlib、seaborn、joblib 和 nbconvert。所有随机过程使用固定种子；正式 run manifest 记录 Python 版本、包版本、CPU 信息、数据 SHA256、配置和运行时间。

新设备复现顺序：

```text
git clone
创建锁定环境
启动 JupyterLab
依次运行 00 → 01 → 02 → 03 → 04
检查 outputs/full_run_manifest.json
```

路径必须基于仓库根目录解析，不得写入个人绝对路径。

## 7. 错误处理

- 下载中断保留 `.part`，不把部分文件当作有效输入。
- SHA256、schema、日期范围或授权 manifest 不匹配时立即停止。
- 上游标准输出缺失时，后续 Notebook 写出明确错误，不使用旧产物或猜测路径。
- 缓存键至少包含数据 SHA256、quantile registry、特征合同、时间切分和模型配置。
- 缓存键变化时不得复用模型、calibration、floor 或容量结果。
- B3 boundary coverage 低于 95% 时 policy selection fail closed。
- 关闭可选阶段时仍写稳定字段的 skipped CSV，避免 stale artifact 被误读。

## 8. 测试与验收

### 8.1 自动测试

自动测试必须覆盖：

- 数据包与包内文件 SHA256。
- 152 列主面板公开合同及必需字段。
- 标签公式和 side 方向。
- 14:59:59 特征白名单与禁止字段。
- train/calibration/test 日期无重叠。
- calibration 与 rolling floor 不使用 test/final-month outcome。
- quantile registry 可扩展、各 head 独立、crossing 诊断存在且不被静默覆盖。
- 每个候选沿 `x_adv` 的风险曲线单调化。
- 10bps 容量边界和 fill-cap min 逻辑。
- B3 missing 行从分母排除，coverage 阈值为 95%。
- same-sample disclosure 字段稳定存在。
- 开关关闭时 skipped CSV contract 稳定。
- Notebook 不导入仓库内业务 `.py` 模块。

### 8.2 两级运行验收

1. Smoke：CI 对小型真实 sample 顺序执行五个 Notebook，验证执行与合同，不发布正式统计结论。
2. Full：在 2,000,000 行 Release 数据上顺序执行五个 Notebook，生成参考 manifest、整体与分桶指标、容量表和优化报告。

正式发布必须满足：

- 新目录不存在旧 `.git` 历史。
- Release 数据可在无公司环境的新设备下载。
- SHA256 和 schema 校验通过。
- 五个 Notebook 从干净环境顺序执行成功。
- 整体及 `side × ratio_bucket` coverage、最差桶、pinball、crossing、容量代价、单调性和 B3 coverage 均有输出。
- 结果明确标记 `sample_level_evaluation` 与 `same-sample policy-development diagnostic`。
- 仓库与 Release 扫描不含内部路径、凭据、旧 Git metadata 或未授权文件。

## 9. 非目标

v1 不包含：

- `adata` 接入或上游逐笔数据重建。
- 原始 impact/carry chunks 的完整归档。
- 生产交易接口、实盘下单或“安全收益”承诺。
- 将 10bps 容量估计表述为生产保证。
- 将 12 月策略结果表述为独立样本外验证。
- Git LFS 或把大型数据直接写入普通 Git 历史。

项目对外统一表述为“10bps 风险约束下的研究性容量上限估计”或“风险计量与交易容量评估框架”。
