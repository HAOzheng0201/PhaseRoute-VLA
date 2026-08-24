# 阶段一第二工作块：五臂配对 smoke 结果

## 1. 实验结论

在同一 A1 checkpoint、物理 GPU、LIBERO task、官方初始状态和 episode seed 下，以下五个方法均成功完成了闭环 episode：

| 方法 | 成功 | policy calls | 路由分布 | 平均深度占 L27 | FM calls/次 | policy wall 均值 | episode wall |
|---|---:|---:|---|---:|---:|---:|---:|
| fixed L11 | 1/1 | 35 | L11: 35 | 42.86% | 1.00 | 448.20 ms | 54.51 s |
| fixed L13 | 1/1 | 33 | L13: 33 | 50.00% | 1.00 | 498.74 ms | 52.84 s |
| fixed L27 | 1/1 | 33 | L27: 33 | 100.00% | 1.00 | 858.91 ms | 66.89 s |
| original A1 | 1/1 | 34 | L11: 28, L27: 6 | 52.94% | 8.41 | 1098.79 ms | 73.20 s |
| PhaseRoute V3 | 1/1 | 35 | L13: 4, L27: 31 | 94.29% | 6.77 | 1518.36 ms | 92.75 s |

配对身份全部通过机器检查：

```text
suite              libero_10
task/state         0 / 0
seed               20260824
checkpoint SHA-256 dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f
physical GPU UUID  535e41e1-a1ac-af65-a015-fc281644709e
instruction hash   8405398832a6d730
```

## 2. 正面结果

1. 五个方法均成功，证明 fixed-depth、原始 A1 和 PhaseRoute 的输入到动作再到 LIBERO 控制链路均可运行。
2. PhaseRoute 的 FM action-head 调用从 original A1 的平均 8.41 次降到 6.77 次，描述性减少 19.50%。
3. PhaseRoute 的 35 个动作全部通过有限值和 `8 × 7` 维度审计，runtime、telemetry 和 measurement error 均为 0。
4. fixed L11/L13/L27 均只调用一次 FM head，因此建立了不同 transformer 深度下的单-head 延迟边界。
5. 在这个单 episode 中，fixed L13 的 episode 总时长最短。fixed L11 单次推理更快，但多产生了两个 policy call，说明单次浅层计算与闭环总耗时不能混为一谈。

## 3. 必须保留的负面结果

PhaseRoute 当前不能宣称比 original A1 更快：

| 描述性比较 | PhaseRoute 相对 original A1 |
|---|---:|
| FM calls/次 | 减少 19.50% |
| policy wall 均值 | 增加 38.19% |
| episode wall | 增加 26.71% |
| 成功结果 | 相同，均成功 |

该负结果不是运行错误。PhaseRoute preflight、sealed run attestation 和全部测量门禁均为 PASS。

## 4. 为什么 FM 调用减少了，wall time 却变慢

### 4.1 路由过于保守

original A1 的 34 次调用中有 28 次在 L11 退出，早退比例为 82.35%；PhaseRoute 的 35 次调用中只有 4 次在 L13 退出，早退比例仅为 11.43%，其余 31 次全部回退到 L27。

因此，PhaseRoute 虽减少了部分 FM 比较动作，但 transformer 平均执行深度从 original A1 的 52.94% 增加到了 94.29%。更深的 backbone 计算抵消并超过了 FM-call reduction。

### 4.2 仍继承多候选 FM 比较开销

fixed 三臂每次只调用一个 FM action head。PhaseRoute 当前仍基于 RP-PEP/A1 的候选比较路径：L11/L13 决策前需要动作一致性信息，回退 L27 时也已经支付了前面候选的 FM 计算。因此 PhaseRoute 平均仍有 6.77 次 FM 调用，而不是“选择一个深度后只生成一次动作”。

### 4.3 PhaseRoute 辅助模块有真实但非主要的开销

本轮外部 overlay 测得每个 policy call 的主要辅助开销均值：

| 组件 | 均值 |
|---|---:|
| visual feature capture | 93.52 ms |
| phase estimator | 12.88 ms |
| runtime prepare（包含部分上述工作） | 23.32 ms |
| router predict | 1.01 ms/候选 |
| candidate route | 2.70 ms/候选 |

visual capture 与 phase estimator 值得优化，但它们不足以单独解释全部延迟差异；更关键的问题仍是大量 L27 回退和多 FM 候选计算。

## 5. 对下一阶段的直接指导

在扩展到 10 tasks 之前，应先解决当前结构的计算矛盾：

1. 研究“先路由、后单次动作生成”的控制路径，使选定 L11/L13/L27 后只运行一次 FM head；
2. 分析当前 gripper veto 和 router 阈值为何使 task 0 的 L27 比例达到 88.57%；
3. 使用 ordinary engineering states 调整或重新训练路由参数，不触碰已经消费的 D9 state 40–49；
4. 先在小规模配对集证明 FM-call、平均深度和 wall time 同时改善，再扩展 10-task 正式矩阵；
5. fixed L13 的成功和较低延迟只说明该 task 存在更激进路由空间，不能据一个 episode 固定所有任务为 L13。

## 6. 结果边界

本轮是 ordinary engineering smoke，不是新的独立测试，不是 D9 重测，也不支持成功率、统计显著性或部署结论。所有百分比均为单个配对 episode 的描述性结果。

机器可读结果：`results/stage1/stage1_phase2_five_arm_smoke.json`。
