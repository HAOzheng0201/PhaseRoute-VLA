# Route-first Stage 7：工程留出验证预注册

## 状态

```text
READY_TO_COLLECT_ENGINEERING_HOLDOUT_STATES_10_11
```

本协议、评估实现和测试在查看 states 10–11 的 feature、teacher label 或 score 之前
冻结。Stage 7 不再训练模型、不移动阈值，也不允许重新启用已经失败的 L11 head。

冻结输入：

| 输入 | SHA-256 |
|---|---|
| Stage 6 calibrated router | `ae561b77c01bd4c7eee6cc0ff91e215733662544cc1af2e5039b0a8f02c60cc2` |
| Stage 6 result | `c599ffb8280368a1014f4de0827524c3e0bc5d9ccc172ede4062600fea9d5de5` |
| Stage 6 verification | `34b07c89be854df10f541b8b6a3eebffcd1c7f9a7cf021276d2850f81148c7bb` |
| Stage 7 protocol | `f7b79d0706f836bf6df56548b64cce076a258c183fddc19c6dac7188f9681a14` |

## 冻结路由

Stage 7 只审计以下 route-first 决策，任何结果都不能改变它：

```text
score11 head: disabled

score13 >= 0.9174261218080999 -> L13
otherwise                     -> L27
```

199D 输入、两个 affine score head、PCA64/L2=0.3 折叠权重和 Stage 6 阈值全部按文件
SHA 绑定。评估程序还逐字段检查 `enabled11=false`、`enabled13=true`、
`engineering_holdout_authorized=true` 和 `active_control_authorized=false`。

## 数据边界

```text
states 0–7   -> 已用于训练，本阶段禁止
states 8–9   -> 已用于阈值选择/确认，本阶段禁止
states 10–11 -> 本次唯一工程 holdout
states 40–49 -> 历史 D9，永久禁止新方法使用
```

数据必须是 LIBERO-10 全部 10 个 task × states 10–11，共 20 个 episode。冻结
PhaseRoute-V3 继续产生动作，route-first 仅 observation-only 记录 teacher context，不会
影响 rollout。task/state identity 只用于 grid 审计和 cell 等权，不进入 score model。

## 预注册门禁

主检查只针对仍启用的 safe13 head。每个 `(task,state)` cell 总权重相同，风险上界使用
90% one-sided weighted Wilson 与 Kish effective sample size。

| 范围 | min coverage | min effective rows | empirical false-safe | 90% upper bound |
|---|---:|---:|---:|---:|
| states 10–11 pooled | 1.5% | 16 | ≤20% | ≤40% |
| state 10 单独 | 1.5% | 8 | ≤20% | ≤40% |
| state 11 单独 | 1.5% | 8 | ≤20% | ≤40% |

每个单独 state 沿用 Stage 6 state-9 confirmation 标准；pooled 的有效行要求加倍为 16。
最终决策要求三组 gate 全部通过，防止一个 state 的好结果掩盖另一个 state 的退化。

以下内容只报告、不参与 pass/fail：

- score13 的固定分位数；
- 每个 task 的 coverage 与 false-safe；
- L13 选择行对应的 raw L11/L13/L27 teacher 数量；
- 冻结 V3 controller 的 task success。

task success 不是 router label。因为 route-first 没有控制动作，holdout 中的 rollout
失败不能归因于新路由；同样，rollout 成功也不能证明 active route-first 安全。

## Fail-closed 决策

1. L11 在任何情况下都保持关闭；
2. L13 阈值必须逐 bit 保持 Stage 6 值；
3. pooled、state 10、state 11 任一 gate 失败，即禁止 runtime integration；
4. 失败后不得用 states 10–11 重训、重校准或寻找替代阈值；
5. 通过只允许进入“实现 active runtime”阶段，不直接授权 active rollout；
6. active test 必须另行冻结 generated-state 配对协议，不能复用历史 D9 做新测试。

## 已完成代码门

新增 `route_first_holdout.py` 和 `evaluate_route_first_holdout.py`，实现：

- exact calibrated-router SHA 与 metadata 绑定；
- exact tasks 0–9 × states 10–11 grid 校验；
- pooled 与 per-state 固定阈值 gate；
- per-task、score quantile 和 raw confusion 诊断；
- exclusive-create 输出与 fail-closed 授权字段；
- active control 永久保持 false。

Holdout 与 calibration 定向测试合计 `12 passed, 1 warning`；全仓 CPU 回归为
`514 passed, 22 subtests passed, 3 warnings`，0 失败，用时 71.98 秒。此时 states
10–11 尚未打开，也没有使用 GPU。

下一步只能先提交并推送本预注册工作块，然后采集一次 exact 20-episode holdout，最后
运行一次评估程序。看过结果后不允许修改本协议再重算正式 Stage 7 结论。
