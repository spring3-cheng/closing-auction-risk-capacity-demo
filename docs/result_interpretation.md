# 结果解读

## 先看结论

仓库内置 smoke run 只证明五阶段流程和业务合同可执行，不支持正式模型优劣或容量结论。完整 2,000,000 行数据尚未在本机跑完，任何正式 coverage、pinball、最差桶与容量比较都应以 full run 新产物为准。

## Smoke 验收结果

- 8,448 行、242 个交易日、8,366 个 anchor；两项标签重建最大差异均为 0。
- train 截止 2024-09-09，calibration 为 2024-09-10 至 2024-10-23，test 为 2024-10-24 至 2024-12-31；final month 不参与模型学习。
- 699 个 smoke B3 key 全部匹配 anchor，45 点模型评分曲线单调化后无下降。
- final test 的 direct q95/q99 coverage 为 97.6%/99.3%，shape-strength 为 96.8%/99.3%；这些数字只用于 smoke 诊断。
- q95/q99 最差桶为 `shape_strength q95 × sell × >5% Cash/ADV`，coverage 87.9%（91 行），说明尾部弱桶尚未解决。
- 正式候选的 `x_safe_final` 中位数均为 0；主要原因是校准后的 no-order 绝对风险已超过 10bps，不能把 smoke 结果解释为可交易容量。
- 12 个正式策略候选全部未通过约束，输出 `no_eligible_candidate_fail_closed`，未选择策略。
- smoke 的 12 月同时用于策略选择和评价，明确不是独立样本外策略验证。
- 本版 `shape_strength` 已改为 train-only anchor-level `A_label` 曲线拟合；由于公开面板没有真实 `x=0` 行，`B_q` 使用 direct 模型的 `x=0` 预测，不能把该基线当作观测反事实。

上述指标仅用于验证 contract。smoke 每个分层保留极少样本，coverage、pinball、最差桶和资金总量都可能与 full run 显著不同。

## Full run 必看项

1. 整体每个 family × quantile × stage 的 coverage、target coverage、pinball loss 和平均预测分位。
2. `side × ratio_bucket` coverage 及样本数，重点检查 buy 中低 Cash/ADV 桶和最差 material bucket。
3. crossing rate 及 crossing 幅度；crossing 是独立 quantile 候选的诊断，不应被静默排序覆盖。
4. calibration/floor 带来的 coverage 改善是否以过高分位水平或明显容量下降为代价。
5. raw ratio 曲线下降数、`cummax` 后单调性、零容量/全网格可行/插值 crossing 状态。
6. B3 boundary coverage、缺失行分母政策、cash-weighted 与 unweighted exceedance。
7. price-only 容量与 fill 约束后容量的差异，特别是 `no_feasible_fill_ratio` 占比。

只有 full run 通过合同后，才可称为“阶段性样本离线结果”。容量始终是 10bps 约束下的研究性估计，不是生产承诺。
