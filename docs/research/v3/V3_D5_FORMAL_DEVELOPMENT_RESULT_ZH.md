# V3-D5 联合可靠性正式开发结果

## 1. 正式结论

V3-D5 的正式状态是：

```text
NEGATIVE_V3_D5_DEVELOPMENT_GATE
```

这是一个有明显预测信号、效率覆盖良好、但 exact cluster 风险门未通过的 near-miss，不能写成 PASS：

```text
policy calls:                  6521
outer OOF candidate rows:      13042
task×episode clusters:         180
L11 / L13 / L27:               199 / 760 / 5562
early-exit calls:              959 (14.7063%)
safe clusters:                 179 / 180
false-safe calls/clusters:     4 / 4
false gripper calls:           0
false full-action calls:       4
exact CP-UCB95:                0.0504043022
frozen maximum:                0.05
```

门槛仅超出 `0.0004043022`，但冻结协议禁止把 5% 改成 5.1%，也禁止看完结果后收紧某几个 outer threshold 再把结果写成通过。

## 2. 训练与防泄漏审计

D5 使用 task 0--9、global episode 12--29，共 18 个 outer global-episode folds。每个 outer fold 内再做 17 个 inner folds：

```text
outer folds:                   18
inner folds per outer:         17
fits per outer:                52
total deterministic GLM fits:  936
infeasible inner thresholds:   0
```

normalizer、layer anchor、194 个线性参数、L2 lambda 和 full-action threshold 均未读取对应 outer episode。每个候选行恰好得到一次 outer-OOF 预测。calibration_v2 和 independent_test_v2 payload 均未打开，未执行 rollout 或 active control。

17 / 18 个 outer folds 选择 `lambda=0.01`，episode14 选择 `0.001`。inner threshold 范围为 `0.11343--0.25804`，最大/最小比为 `2.275`，说明阈值稳定性是下一阶段必须处理的问题。

## 3. 模型是否学到了有效信号

答案是“学到了，但 point score 的 cluster tail 仍不够可靠”：

| 目标 | 正例数 / 13042 | OOF AUROC | OOF log loss | OOF Brier |
|---|---:|---:|---:|---:|
| full-action unsafe | 2590 | 0.92868 | 0.28751 | 0.08482 |
| gripper-step unsafe | 5216 | 0.94510 | 0.29717 | 0.08915 |

两个 AUROC 均超过 0.92，说明 97D 当前候选因果特征中确实存在可靠性信息。Gripper-v2 在 959 次正式选择中保持 0 个错误，也再次支持“夹爪必须单独建模、不能被平移动作风险替代”的 D4 结论。

但正式门看的是 cluster tail，不是平均排序能力。AUROC 很高并不保证被选入的极少数低分样本全部安全。

## 4. 四个 false-safe 的具体结构

| task | episode | layer | p_full / threshold | 真实 distance / 0.00390625 | 解释 |
|---:|---:|---:|---:|---:|---|
| 0 | 14 | 13 | 0.0808 / 0.1134 | 17.03× | 严重低估，不是边界噪声 |
| 3 | 16 | 11 | 0.1438 / 0.2580 | 1.02× | 几乎贴着 truth 阈值 |
| 3 | 22 | 11 | 0.0805 / 0.2259 | 2.28× | 明显 point-score misranking |
| 9 | 29 | 13 | 0.1232 / 0.1275 | 1.34× | 接近运行时概率阈值 |

四个错误分布在四个不同 task×episode cluster，L11/L13 各两个，全部通过 A1 consistency，全部是 full-action unsafe，均没有 gripper mismatch。

最重要的诊断是：只有 task9/episode29 的 score 接近阈值；另外两个 score 仅为各自阈值的 35.6% 和 55.7%。所以“统一把阈值缩小一点”可减少错误，但不能解释或根治严重低估样本。

## 5. 事后阈值缩放只能是诊断

为了区分“纯阈值 near-miss”和“模型排序失败”，冻结结果之后做了明确标记为 post-hoc、runtime 未授权的缩放消融：

| outer threshold multiplier | early calls | safe clusters | false clusters | CP-UCB95 | 正式授权 |
|---:|---:|---:|---:|---:|---|
| 1.00 | 959 | 179 | 4 | 0.05040 | 否，正式 D5 失败 |
| 0.95 | 957 | 179 | 3 | 0.04274 | 否，post-hoc |
| 0.90 | 956 | 179 | 3 | 0.04274 | 否，post-hoc |
| 0.50 | 943 | 179 | 1 | 0.02623 | 否，post-hoc |

0.95 缩放只减少 2 次早退并会得到通过数值，但它是在看到 D5 outer truth 后才被观察到，不能替换冻结策略。即使缩到 0.5 仍有一个错误，也证明 point probability 的 tail misranking 确实存在。

## 6. 效率解释

按 RP-PEP 的解析 FM call 账本：

```text
behavior A1 observed FM calls:       67719
D5 shadow estimated FM calls:        43530
estimated reduction:                 35.7197%
```

该数值未包含双风险 head 延迟，也不是闭环端到端实测加速。它只能说明当前路由覆盖在计算账本上具有继续研究的价值。

D4 与 D5 使用不同数据角色，不能作严格同表 superiority 对比；定性上，D5 用直接 full-action head 取代 legacy motion/tail hard veto 后，覆盖率从 D4 calibration shadow 的 4.32% 提升到 D5 development OOF 的 14.71%，但也暴露出 4 个 full-action cluster tail 错误。

## 7. 下一阶段应解决什么

D6 只能做 protocol design，不能重写 D5。建议围绕三个明确问题：

1. **严重度感知。** 当前 binary full-action target 把 `1.02×` 和 `17.03×` 阈值违规视为同一正例。下一版应增加连续/ordinal severity 或 upper-quantile head，使严重低估比边界样本承担更高代价。
2. **point score 到风险上界。** 使用 inner-fold dispersion、jackknife 或 conformal residual 构造 full-action risk UCB，让不确定性高的样本 fail closed，而不是只比较均值概率。
3. **阈值稳定性。** 18 个 outer threshold 相差 2.275 倍。D6 应预注册跨 inner folds 的保守聚合或 shrinkage；D5 中观察到的 0.95 只能作为开发假设，必须在新的确认数据上验证。

episode 30--39 已被查看，不可用于 D6 修复后的“新验证”；episode 40--49 independent test 继续封存。D5 OOF 以后只能作为 development evidence，不能再次被称为 fresh confirmation。

## 8. 可核验证据

```text
D5 contract SHA:
e0a584e76f03d0f1b43cd5bbd3477ee2e3694f5425642868b3ec563edd52a29f

formal OOF report SHA:
bddd8fdbbf53f5d8270ee13012dc6f29d5481ca6c5e1c4dde4aacb85cd3ca2bf

formal attestation SHA:
f08e35e9588f44900d6e714dc45c7afb9e1cc7586e8bbbfade488f3ed783b6f8

negative analysis report SHA:
e4705fcbaa0e1a917df2a928ac1afc62c4921757ede682f8e6ca8c8df2aee9b4
```
