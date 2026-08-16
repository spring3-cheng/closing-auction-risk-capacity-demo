# 新设备 Codex 导入与复现指南

本文用于指导另一台设备上的 Codex 导入并复现本项目。该流程针对 GitHub clean-room 仓库，不需要访问公司的内部代码、内部数据库或原始内部路径。

## 1. 项目范围

本仓库复现以下下游流程：

- `14:59:59` 可见信息下的 `total_bad_move_bps` 风险建模；
- 独立 quantile head、train-only `H_q` 和 anchor-level `A_q`；
- calibration buffer、历史 causal floor 和 final-month 评价；
- 10bps 绝对风险容量反解、impact guardrail 和静态 fill 约束；
- B1/B2/B3 边界评价及 same-sample policy-development diagnostic 披露。

实习项目中的原始 Level-2 委托/成交重建、订单簿重建和反事实撮合上游管线不在本仓库内。仓库从经授权发布的加工面板开始，不能把本仓库描述成内部原始数据管线的逐字复制。

## 2. 导入仓库

需要使用有该 private repository 访问权限的 GitHub 账号。

```powershell
git clone https://github.com/spring3-cheng/closing-auction-risk-capacity-demo.git
Set-Location .\closing-auction-risk-capacity-demo
git branch --show-current
git log -1 --oneline
```

应位于 `main` 分支。不要把 GitHub token 写入 Notebook、代码、提交记录或日志。

## 3. 安装环境

### 方案 A：Conda

如果设备已经安装并初始化 Conda：

```powershell
conda env create -f environment.yml
conda activate closing-auction-capacity-demo
```

如果环境已经存在：

```powershell
conda env update -f environment.yml --prune
conda activate closing-auction-capacity-demo
```

### 方案 B：没有 Conda 时使用 Python venv

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

如果 PowerShell 禁止激活脚本，只对当前用户执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新运行：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Smoke 全链复现

Smoke 数据已经在 Git 仓库中，不需要下载完整 Release。PowerShell 中执行：

```powershell
$env:RUN_MODE = "smoke"
jupyter lab
```

在 JupyterLab 中按以下顺序运行，不能跳过前置阶段：

```text
00_get_data.ipynb
01_build_panel_and_labels.ipynb
02_quantile_risk_model.ipynb
03_capacity_inversion.ipynb
04_optimization_and_report.ipynb
```

也可以使用 Codex 或终端非交互执行：

```powershell
$runDir = Join-Path $env:TEMP "closing-auction-demo-runs"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

python -m nbconvert --to notebook --execute notebooks/00_get_data.ipynb --output 00.executed.ipynb --output-dir $runDir --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/01_build_panel_and_labels.ipynb --output 01.executed.ipynb --output-dir $runDir --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/02_quantile_risk_model.ipynb --output 02.executed.ipynb --output-dir $runDir --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks/03_capacity_inversion.ipynb --output 03.executed.ipynb --output-dir $runDir --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks/04_optimization_and_report.ipynb --output 04.executed.ipynb --output-dir $runDir --ExecutePreprocessor.timeout=1200
```

## 5. 验收

Notebook 执行后，在仓库根目录运行：

```powershell
python -m unittest
```

重点检查：

- `outputs/01_stage_contract.json`：标签合同和 `14:59:59` 可见特征合同；
- `outputs/02_stage_contract.json`：
  - `strength_label_granularity = anchor_level_curve_fit`；
  - `base_label_source = direct_quantile_model_at_x0_no_observed_outcome`；
  - `cross_quantile_order_enforced = false`；
  - `outputs/02_anchor_strength_labels.csv` 非空；
- `outputs/03_stage_contract.json`：45 点 action grid、`absolute_final_quantile` 和 `cummax` 单调化；
- `outputs/03_capacity_candidates.parquet`：缺失 fill 时 `x_safe_final` 必须保持缺失；
- `outputs/04_report_summary.json`：
  - `evaluation_scope = same-sample policy-development diagnostic`；
  - `independent_out_of_sample_policy_evaluation = false`。

Smoke 结果只证明 Notebook 链路和输出合同可执行，不代表完整数据上的正式模型效果或生产容量承诺。当前参考清单中的 `full_run_status` 仍为 `not_run`。

## 6. Full Release 复现

完整数据以 private GitHub Release `data-v1.0.0` 发布，完整 ZIP 由 6 个 SHA256 校验分卷组成，总大小约 460,981,504 bytes。需要有该 private repository 读取权限的 token。

PowerShell 中设置环境变量，但不要把 token 写入文件：

```powershell
$env:RUN_MODE = "full"
$env:GH_TOKEN = "<read-only-github-token>"
```

然后仍按 `00 → 04` 顺序运行。`00_get_data.ipynb` 会下载分卷、校验 SHA256、重组并解压数据。完整运行时间和内存需求高于 smoke；运行失败时先保留错误信息和阶段合同，不要删除已有输出后猜测原因。

## 7. 当前边界

- 本仓库使用授权的加工面板，不包含原始 Level-2 上游重建代码。
- 发布面板没有真实 `x_adv=0` 结果；`B_q` 是 direct 模型在 `x=0` 特征副本上的预测。
- quantile heads 独立训练，仓库只报告 crossing diagnostics，不强制 `q99 >= q95`。
- 12 月策略选择和评价存在样本重叠，必须标记为 same-sample policy-development diagnostic。
- 不要用 smoke coverage、smoke capacity 或 crossing 结果包装成完整数据正式结论。

## 8. Codex 工作规则

另一台设备上的 Codex 开始工作前，应先阅读：

```text
README.md
docs/methodology.md
docs/data_dictionary.md
docs/result_interpretation.md
reference/reference_run_manifest.json
```

除非用户明确要求，不要：

- 修改目标、特征、时间切分、校准/floor、容量字段或输出 CSV contract；
- 引入内部路径、内部字段、内部 Git 历史或内部数据；
- 把 full run 未完成描述成已验证；
- 将 GitHub clean-room 项目冒充为公司内部原始工程的直接上传；
- 自动推送新的 GitHub 提交。
