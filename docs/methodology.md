# 方法说明

## 研究问题

在 A 股收盘集合竞价 `14:59:59` 已知信息下，估计我方订单加入后，最终收盘价相对 `vcp_145959` 的方向化不利风险，并在 10bps 风险约束下形成研究性资金容量上限。

方向化标签按 buy/sell 符号统一：

```text
impact_me_bad_bps = max(side_sign × impact_me_raw_bps, 0)
total_bad_move_bps = max(side_sign × total_move_bps, 0)
```

`01` 会从原始变化字段重建两项标签，并要求与发布数据中的标签逐行一致。

## 时间与信息边界

- 特征通过显式白名单限定为 `14:59:59` 可见信息；`market_move_bps`、最终价格、成交结果和风险标签被列入 leakage blocklist。
- 模型按交易日严格划分约 70% train、10% calibration、20% test。
- calibration buffer 和 rolling floor 只使用 calibration 当期或此前历史 fold；test/final month 不参与学习。
- 12 月模型指标属于 final-month holdout 评价，但策略 operating point 仍在同一 12 月 B3 上选择和评价，因此策略层明确标记为 `same-sample policy-development diagnostic`。

## 模型候选

默认 quantile registry 为 `0.50/0.80/0.85/0.90/0.95/0.99`。每个 quantile head 都是独立候选，不强制 `q99 ≥ q95` 或其他跨 quantile 顺序；输出 crossing rate 和 crossing 幅度用于诊断。

- `direct_lgbm`：直接拟合 `total_bad_move_bps`。为复现既有训练探索，q95/q99 的 raw tau 分别为 0.80/0.90，其余采用同名 tau。
- `shape_strength`：直接基于总风险标签，在训练集内估计沿 `x_adv` 单调的 shape `H(q,x)`，再用不含 `x_adv` 的可见条件特征预测强度；不把真实最终自然涨跌当作 base 输入。
- `impact guardrail`：独立的 0.95 quantile 模型，不随 total-risk operating quantile 自动变化。

最终风险由 raw prediction、calibration buffer 和历史 causal floor 组成。报告整体 coverage、`side × ratio_bucket` coverage、最差桶、pinball loss、预测分位水平及 crossing，而不是只追求 coverage。

## 容量反解

容量阶段为每个 B3 anchor 构造 45 点网格（模型 no-order `x=0` 加 44 个正档），分别评分 total risk 和 impact guardrail。`x=0` 的 total risk 来自模型预测及历史校准，不由该日真实 `market_move_bps` 覆盖。每个候选内部沿 `x_adv` 使用 `cummax` 单调化，禁止利用局部下降获得虚高容量；不同 quantile 之间不排序。

正式候选统一用单调线性插值反解 10bps 边界。grid floor 和 shape exact-H 标签仅作为诊断，后者因 ratio bucket calibration buffer 变化不能进入正式选择。最终可执行容量为：

```text
x_safe_final = min(x_safe_price_risk, x_safe_fill)
```

若 fill 没有可行容量，则 `x_safe_final` 保持缺失，不伪造为零或可交易上限。

## 策略评价边界

B1 与 B3 分开报告；B3 是使用事后信息的 oracle 上限。B3 缺失边界行从风险和资金分母中排除，同时要求边界 coverage 至少 95%。候选约束为 cash-weighted exceedance 不高于 5%、unweighted 不高于 6%、样本数至少 100 的最差 material bucket 不高于 8%，并要求 price-only 与 fill 后均存在正容量/正资金。无候选通过时 fail-closed，不选择策略。
