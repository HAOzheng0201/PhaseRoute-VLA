# 历史 M4.29：失败归因与 RP-PEP 发布选择

> 本文是旧 M4.28 router 的冻结负结果与当时的 RP-PEP 选择。后续 PhaseRoute V3
> five-head router 使用不同协议并已通过 D9 18/18 gates；本文不得被概括为所有
> learned routing 都不可行。当前状态见 `docs/RELEASE_STATUS_ZH.md`。

日期：2026-08-05

## 结论

1. M4.28 learned route-then-solve router 工程实现可复现，但一次性 sealed 科学门失败：`router_offline_gate=NOT_VIABLE`、`runtime_integration_allowed=false`。
2. 正式方法选择已有严格闭环证据的 opt-in RP-PEP。它不预测未来退出层，只裁剪已证明无生产性的 FM solve，并用同形 RNG burn 保持 A1 随机序列。

“代码跑通”不等于把未通过安全门的 router 包装为成功。正式可运行路径是 RP-PEP；learned router 作为负结果完整保留。

## Learned router 的错误浅退

对 1,314 行 sealed prediction 独立重建后，得到：

| task | episode | step | policy call | teacher | router | safe13 score | boundary probability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 28 | 42 | 4 | 27 | 13 | 0.989310 | 0.994488 |
| 4 | 20 | 186 | 22 | 27 | 13 | 0.985449 | 0.998436 |
| 4 | 20 | 226 | 27 | 27 | 13 | 0.993106 | 0.986831 |
| 7 | 28 | 66 | 7 | 27 | 13 | 0.993632 | 0.999779 |

准确表述是：**4 次错误浅退，分布在 3 个 task/episode group**。这是离线反事实安全错误；按照停止规则没有把 router 接入闭环，所以不能声称为“三个最终任务失败”。

冻结阈值为 `0.9840359290815168`，binary exact 为 54.03%。打开 sealed 数据后提高阈值虽能事后清除错误浅退，但属于测试集调参，不能作为新方法或新测试结果。四个错误点的 boundary probability 均很高，说明该信号不能独立充当安全兜底。

## RP-PEP 配对结果

| 指标 | 原 A1 Early Exit | RP-PEP | 差异 |
|---|---:|---:|---:|
| 成功 episode | 20/20 | 20/20 | 0 |
| action chunk SHA mismatch | — | 0 | 精确一致 |
| exit sequence mismatch | — | 0 | 精确一致 |
| trajectory mismatch | — | 0 | 精确一致 |
| FM solver calls | 2,002 | 1,179 | -41.11% |
| 平均 policy latency | 10,563.73 ms | 7,282.43 ms | -31.06% |
| 中位 policy latency | 9,561.36 ms | 6,701.43 ms | -29.91% |

RP-PEP 到 layer 13 退出需要 5 次真实 FM solve；继续到 layer 27 总计 7 次。它是冻结的生产性求解计划，不是一个免费的先验 router。

## 前四卡 state-30 smoke

LIBERO Spatial task 0–9 各运行 episode 30，seed `20261329`：

| 项目 | 结果 |
|---|---:|
| 完成 rollout | 10/10 |
| 成功 rollout | 10/10 |
| policy calls | 128 |
| FM calls | 589 |
| layer 11 / 13 / 27 exits | 65 / 56 / 7 |
| telemetry errors | 0 |

四个 worker 的 checkpoint、任务网格、episode、seed、RP-PEP 开关、GPU UUID 和 FM accounting 均通过汇总器检查。

## 科学限制

- 20 对严格等价是强工程证据，但不足以证明所有 LIBERO 状态的普遍等价。
- state-30 smoke 验证入口，不替代多 seed benchmark。
- M4.28 sealed 数据已经打开；新 router 必须预注册新的独立 sealed set。
- 当前发布声称效率 Pareto 改进，不声称成功率显著高于 A1。

机器可读证据：[`rp_pep_paired.json`](rp_pep_paired.json)、[`release_smoke_state30.json`](release_smoke_state30.json)、[`router_sealed.json`](router_sealed.json)、[`router_failure_analysis.json`](router_failure_analysis.json)。
