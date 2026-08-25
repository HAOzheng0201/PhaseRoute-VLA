# Route-first Stage 7：工程留出验证结果

## 1. 结论

Stage 7 已按提交 `693a0d8` 中的预注册协议一次性执行，状态为：

```text
PASS_ENGINEERING_HOLDOUT_RUNTIME_INTEGRATION_READY
```

冻结的 L13 阈值 `0.9174261218080999` 在 states 10–11 pooled、state 10 单独和
state 11 单独三组 gate 上全部通过。L11 继续关闭，模型权重、超参数和阈值均未改变。

主要结果：

| 范围 | selected rows | group-equal coverage | false-safe | 90% upper bound | gate |
|---|---:|---:|---:|---:|---|
| states 10–11 pooled | 67 | 10.56% | 6.65% | 11.79% | PASS |
| state 10 | 34 | 11.08% | 9.76% | 18.44% | PASS |
| state 11 | 33 | 10.03% | 3.22% | 10.02% | PASS |

这允许进入 route-first runtime integration 的**实现阶段**，但仍不授权 active control。
本阶段没有证明闭环成功率、端到端延迟或 wall-clock speedup。

## 2. 预注册完整性

在打开 states 10–11 前，以下内容已提交并推送：

| 内容 | 冻结值 |
|---|---|
| source commit | `693a0d895624cff5d30de2227aaeebe2c1b75b78` |
| calibrated router SHA-256 | `ae561b77c01bd4c7eee6cc0ff91e215733662544cc1af2e5039b0a8f02c60cc2` |
| Stage 7 protocol SHA-256 | `f7b79d0706f836bf6df56548b64cce076a258c183fddc19c6dac7188f9681a14` |
| L11 | disabled |
| L13 threshold | `0.9174261218080999` |
| decision | pooled、state 10、state 11 必须全部通过 |

执行结果记录 `threshold_changed=false`。没有根据 holdout 结果重新训练、重新校准或移动
阈值，也没有打开历史 D9 states 40–49。

## 3. 采集与运行证明

采集前 GPU 0–3 正被其他用户的 Isaac Sim 使用；物理 GPU 4–7 的利用率均为 0%，因此
只使用 GPU 4–7。每个 worker 严格绑定一个物理 UUID，冻结 PhaseRoute-V3 负责动作，
route-first collector 只记录 199D context 和 teacher layer。

| GPU | tasks | episodes | success | calls | teacher L11/L13/L27 | errors | attestation SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 4 | 0, 4, 8 | 6 | 5 | 253 | 0 / 19 / 234 | 0 | `dcc981428cc2e460c902475f7785d86bb9f11c99248b5f97fb97794067edb255` |
| 5 | 1, 5, 9 | 6 | 6 | 182 | 9 / 21 / 152 | 0 | `cc9f6432475e9bf049d307138c2abcc4f407e27c92b4cc81c3f0fff65cc6a2bf` |
| 6 | 2, 6 | 4 | 4 | 123 | 6 / 16 / 101 | 0 | `d9b150311add000ceee97cc4c60c832f141392976cb9e19238c5465bccd0b1fe` |
| 7 | 3, 7 | 4 | 4 | 123 | 11 / 16 / 96 | 0 | `ad5c5819274c7d34732cad04404684939cb5ba890c7fee0bcdacd7a4d42335f2` |
| **合计** | **0–9** | **20** | **19** | **681** | **26 / 72 / 583** | **0** | — |

唯一失败 rollout 为 `task8/state10`，在 65 calls 后失败；`task8/state11` 在 55 calls
后成功。失败 episode 被原样保留在 aggregate 中，没有重跑或筛除。因为 route-first
只 observation-only，不能把该失败归因于新路由。

首次沙箱内启动的四个 worker 在 preflight 阶段因 CUDA 不可见而停止，未加载模型、
未打开 episode、未生成 teacher shard；另有一个授权服务中断后留下的空目录。它们均
属于基础设施尝试，不进入科学 aggregate。获得用户明确授权后，四个沙箱外 run 的
CUDA/UUID preflight 全部通过。

## 4. Aggregate

只聚合四个 attestation 为 PASS 的 teacher shard：

| 项目 | 数值 |
|---|---:|
| exact grid | 10 tasks × states 10–11 |
| episodes | 20 |
| rows | 681 |
| feature shape | `[681, 199] float32` |
| teacher L11 / L13 / L27 | 26 / 72 / 583 |
| calls per episode | min 20 / mean 34.05 / max 65 |
| payload SHA-256 | `fe0b60bfc6412be345f3b442b59ca02724c05c74f6e3414d6852bcdefe08ff98` |
| file SHA-256 | `4948ae68b8b9eca77a2f0615ac6a8001f1eac9df4ae996a735b1288e443d2950` |

所有风险主指标使用 `(task,state)` cell 等总权重；20-call 和 65-call episode 对统计量的
总质量相同。

## 5. 固定阈值 gate

### 5.1 Pooled

| 指标 | 预注册要求 | 实际值 | 结果 |
|---|---:|---:|---|
| coverage | ≥1.5% | 10.56% | PASS |
| effective selected rows | ≥16 | 64.68 | PASS |
| empirical false-safe | ≤20% | 6.65% | PASS |
| 90% upper bound | ≤40% | 11.79% | PASS |

### 5.2 每个 state 独立检查

| state | coverage | effective rows | false-safe | 90% upper bound | 结果 |
|---:|---:|---:|---:|---:|---|
| 10 | 11.08% | 32.80 | 9.76% | 18.44% | PASS |
| 11 | 10.03% | 31.97 | 3.22% | 10.02% | PASS |

三个 gate 全部通过，因此结果允许实现 runtime integration。如果任何一个失败，协议会
直接给出 `FAIL_ENGINEERING_HOLDOUT_ROUTE_FIRST_DISABLED`，不允许用另一个 state 的好
结果抵消。

## 6. 必须保留的误路由与任务不均衡

固定 router 在 681 行中离线选择 L13 67 次，其 teacher 分布为：

```text
teacher L11: 24
teacher L13: 39
teacher L27:  4  <- raw false-safe
```

四条 raw false-safe 为：

| task | state | call ordinal | score13 |
|---:|---:|---:|---:|
| 1 | 10 | 19 | 0.999499 |
| 1 | 11 | 20 | 0.992739 |
| 5 | 10 | 0 | 0.945381 |
| 9 | 10 | 24 | 0.955921 |

因此本结果不是“零错误安全路由”。尤其是 task 5 只有两条 L13 选择，其中一条为
false-safe，描述性 group-equal false-safe 为 47.62%，有效样本量仅约 2；task 1 和
task 9 的描述性 false-safe 分别为 18.15% 和 12.47%。Per-task 指标预注册为 report-only，
不会在看过结果后升级成新的正式 gate，但它明确提示后续 active test 必须做 task-stratified
审计和 fail-closed telemetry。

与 state 9 confirmation 相比：

| split | coverage | false-safe | 90% upper bound |
|---|---:|---:|---:|
| state 9 | 11.29% | 2.68% | 9.03% |
| states 10–11 | 10.56% | 6.65% | 11.79% |

覆盖率稳定，但 holdout 经验误安全率更高，说明 router 并非完美泛化；通过的是宽松的
工程推进门禁，不是安全认证。

## 7. 当前能做什么、不能做什么

允许：

- 实现加载 calibrated router 的 route-first runtime；
- 编写 CPU/GPU parity、单次 FM、fail-closed 和 telemetry 测试；
- 在任何 active rollout 前预注册新的 generated-state 配对协议。

不允许：

- 直接在现有 states 10–11 上调阈值或训练；
- 把 observation-only 结果写成闭环成功率提升；
- 把离线 layer-count reduction 写成 wall-clock speedup；
- 复用 historical D9 states 40–49 做新方法选择或测试；
- 未经新协议直接运行 route-first active control。

## 8. Artifact 与验证

| artifact | SHA-256 |
|---|---|
| aggregate NPZ | `4948ae68b8b9eca77a2f0615ac6a8001f1eac9df4ae996a735b1288e443d2950` |
| holdout scores | `c6b8872a18ca44e7762f05cbef29967aea06cde5c9324da392d93d8398326a45` |
| published result | `d9780a5e4765ee9a80165eb790b99b4e9e85fcb1ae6d34ae006ddb72ce48f258` |

结果 checksum 和语义审计均通过。Holdout/calibration 定向测试为
`12 passed, 1 warning`；全仓 CPU 回归为
`514 passed, 22 subtests passed, 3 warnings`，0 失败，用时 74.36 秒。

## 9. 下一阶段

Stage 8 只做 runtime integration 和工程门禁：

1. runtime 必须同时验证 calibrated-router SHA、Stage 7 PASS result 和 active=false；
2. 在 action-free 199D context 上先路由到 L13/L27，再只对选定 hidden state 做一次 FM；
3. L11 代码路径保持关闭，异常或 artifact 不一致固定回退 L27；
4. 增加选择层、FM 调用次数、误差和 fallback 的逐调用 telemetry；
5. 先完成 CPU 单元测试、离线 replay 与单 GPU smoke；
6. active paired rollout 前另行冻结 generated-state protocol、baseline、seed 和非退化门禁。

Stage 7 PASS 不会自动跳过上述步骤。
