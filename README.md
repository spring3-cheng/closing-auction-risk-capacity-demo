# Closing Auction Risk & Capacity Demo

这是一个从零编写的 A 股收盘集合竞价研究演示项目：在 `14:59:59` 已知信息下估计订单加入后的方向化不利风险，并在 10bps 风险约束下形成研究性容量上限。项目不依赖公司内部数据接口，五本 Notebook 可在有权访问该私人仓库的新设备按顺序独立运行。

## 与实习项目的对应关系

本仓库是经授权从零重写的 clean-room 演示，重点复现实习项目中的下游风险建模、因果校准、容量反解和策略评价流程。实习项目还包含基于原始 Level-2 委托/成交数据的虚拟收盘价、剩余订单簿和反事实撮合重建，以及更早阶段的市场微观结构研究；这些上游内部研究管线不包含在本仓库中。本仓库从经授权发布的加工面板开始，因此仓库中的行数、交易日和 anchor 统计不能与简历所述早期研究样本直接等同。

## 项目结构

| Notebook | 作用 |
|---|---|
| `00_get_data.ipynb` | 获取 smoke 或 GitHub Release 数据，校验 SHA256、行数和 schema |
| `01_build_panel_and_labels.ipynb` | 重建风险标签，声明 `14:59:59` 特征白名单与 no-order anchor |
| `02_quantile_risk_model.ipynb` | 训练真实 quantile head 与 `B_q+A_qH_q` 候选；`A_q` 使用 train-only anchor-level curve fit 并进行因果校准 |
| `03_capacity_inversion.ipynb` | 构造 45 点绝对分位风险曲线、单调化并连续反解 10bps 容量 |
| `04_optimization_and_report.ipynb` | 分开评价 B1/B3，按约束 fail-closed 选择 operating point |

方法、字段和结果边界分别见 `docs/methodology.md`、`docs/data_dictionary.md` 和 `docs/result_interpretation.md`。

## 新设备复现

### 1. 创建环境

```bash
git clone https://github.com/spring3-cheng/closing-auction-risk-capacity-demo.git
cd closing-auction-risk-capacity-demo
conda env create -f environment.yml
conda activate closing-auction-capacity-demo
```

也可直接使用 Python 3.11：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

### 2. Smoke 全链验收

仓库自带约 5–8 MiB/表以内的真实 smoke 数据，无需下载 Release。设置 `RUN_MODE=smoke`，在仓库根目录启动 JupyterLab，并按 `00 → 01 → 02 → 03 → 04` 顺序运行。

```bash
jupyter lab
```

也可非交互执行：

```bash
python -m nbconvert --to notebook --execute notebooks/00_get_data.ipynb --output /tmp/00.executed.ipynb --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/01_build_panel_and_labels.ipynb --output /tmp/01.executed.ipynb --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/02_quantile_risk_model.ipynb --output /tmp/02.executed.ipynb --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks/03_capacity_inversion.ipynb --output /tmp/03.executed.ipynb --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks/04_optimization_and_report.ipynb --output /tmp/04.executed.ipynb --ExecutePreprocessor.timeout=1200
python -m unittest discover -s tests -v
```

PowerShell 先执行 `$env:RUN_MODE='smoke'`；bash 使用 `export RUN_MODE=smoke`。默认值也是 smoke。

### 3. Full 数据复现

完整数据发布在该私人仓库的 `data-v1.0.0` Release，ZIP 不进入 Git 历史。为保证网页上传稳定性，Release 以 6 个分卷资产发布；`00` 会逐卷下载与 SHA256 校验、重组为完整 ZIP，再以总 SHA256 复核。manifest 中记录的总包合同为：

```text
bytes: 460981504
sha256: 841418b344ea11ce8834afd563ecd389955ef1ca7bb52af6408bd3095f0fea56
```

设置 `RUN_MODE=full` 后运行同一组 Notebook。私人 Release 下载需要设置 `GH_TOKEN` 或 `GITHUB_TOKEN`，token 只需具备该仓库的只读访问权限，不会被 Notebook 保存或输出。`00` 会从 Git origin 推导仓库，调用 GitHub Release API，将通过校验的分卷缓存到 `release/.parts/`，再原子生成完整 ZIP 并解压；也可手工把已校验 ZIP 放入 `release/`。

```bash
export RUN_MODE=full
export GH_TOKEN=<token-with-private-repository-read-access>
jupyter lab
```

PowerShell 对应命令为 `$env:RUN_MODE='full'` 和 `$env:GH_TOKEN='<token>'`。若使用自托管下载地址，可另外设置 `DATA_RELEASE_URL`；私有 GitHub asset 仍需 token。

完整数据为 2,000,000 行分层样本。本机已按 `00 → 01 → 02 → 03 → 04` 顺序完成一次 `RUN_MODE=full` 全链；全量结果和边界说明见 `reference/reference_run_manifest.json` 与 `docs/result_interpretation.md`。这仍是 same-sample policy-development diagnostic，不是独立样本外策略验证或生产承诺。

## 数据发布与来源

- `data/smoke/` 和 `data/data_manifest.json` 进入 Git 仓库。
- 460,981,504 字节的完整 ZIP 以 6 个可校验分卷作为私人 GitHub Release assets 单独上传。
- 数据经授权可公开再分发，范围说明见 `DATA_LICENSE.md`；来源台账见 `PROVENANCE.csv`。
- 可发布数据已移除 `source_file`、`source_shard_id` 等追踪列；实际 GitHub 仓库按用户选择设为 private，且不包含内部接口、访问凭据或旧项目 Git 历史。

## 研究边界

- 主面板是 2024 年 200 万行阶段性分层样本，不是全市场全量逐笔数据；每个 anchor 的观测 ratio grid 稀疏。
- 容量使用模型生成的 `0 + 44` 点网格；`x=0` 也是基于可见特征的模型预测，不使用该日最终 `market_move_bps`，未观测档不能当作真实反事实结果。
- quantile registry 可扩展，各 quantile 独立，不强制 q99/q95 或其他跨分位排序；单个候选的 `x_adv` 风险曲线必须单调。`A_q` 的训练标签按 `date × sym × side × quote_strategy` anchor 从观测曲线拟合，单点 anchor 仅提供退化的局部信息。
- 发布面板没有真实 `x_adv=0` 结果；`B_q` 统一来自 direct quantile model 的 `x=0` 特征副本，不能解读为真实 no-order 反事实观测。
- 每个 direct head 使用其声明的真实 quantile alpha；shape-strength 的 side-aware `H_q(x)` 只由 train 构造，容量超过训练支撑域时封顶并标记 `support_cap_reached`。
- 容量约束作用于包含 no-order base、条件增量、校准 buffer 与 causal floor 的 `absolute_final_quantile`，不是只约束边际冲击。
- 12 月同时用于策略 operating-point selection 与 evaluation，属于 `same-sample policy-development diagnostic`，不是独立样本外策略验证。
- B3 使用事后信息，只是研究 oracle 上限；容量不是生产交易承诺。
