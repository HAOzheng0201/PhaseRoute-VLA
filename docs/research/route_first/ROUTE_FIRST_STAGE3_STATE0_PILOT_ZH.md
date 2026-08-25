# Route-First Stage 3：10-Task / State-0 Teacher Pilot

日期：2026-08-25

## 结论

冻结 V3 teacher 的 `10 tasks × state 0` pilot 完成。10 个 episode 中 9 个成功，唯一失败为
task 5；368 次 policy call 全部 prepared/committed，runtime error 为 0。三个 action-free
teacher shard 通过 exact-grid 聚合门禁，得到 `[368,199]` float32 数据，L11/L13/L27 标签为
`12/51/305`。

这个结论只说明 observation-only 数据链路可扩展到全部 10 个 task，并且三类 teacher 标签
都有覆盖。它不代表新 router 已训练，也不代表 single-FM route-first runtime 已获得加速。

## 实验配置

| 项目 | 固定值 |
|---|---|
| suite | LIBERO-10 |
| tasks | 0–9 |
| initial state | 0 |
| seed base | 20260824 |
| checkpoint SHA-256 | `dcafd9ee...631b7f` |
| GPU | 物理 4/5/6 |
| task partition | 4: `0,3,6,9`; 5: `1,4,7`; 6: `2,5,8` |
| context | 199D、action-free、identity-free |
| control | 冻结 V3；采集 overlay 不参与决策 |

## 每个 task 的结果

| task | success | calls | L11 | L13 | L27 |
|---:|:---:|---:|---:|---:|---:|
| 0 | 是 | 35 | 0 | 4 | 31 |
| 1 | 是 | 30 | 2 | 5 | 23 |
| 2 | 是 | 32 | 1 | 7 | 24 |
| 3 | 是 | 27 | 6 | 5 | 16 |
| 4 | 是 | 31 | 0 | 2 | 29 |
| 5 | 否 | 65 | 1 | 2 | 62 |
| 6 | 是 | 28 | 0 | 5 | 23 |
| 7 | 是 | 31 | 0 | 6 | 25 |
| 8 | 是 | 52 | 0 | 7 | 45 |
| 9 | 是 | 37 | 2 | 8 | 27 |
| **总计** | **9/10** | **368** | **12** | **51** | **305** |

early-exit 标签占 63/368（17.12%）。若只按选中层数估算，平均执行深度为 L27 的
90.88%，即层数缩减 9.12%。这仍是旧 V3 teacher 的行为，不是新方法的真实 FLOPs、延迟
或吞吐结果；teacher 当前还会为路由运行多次 FM，不能把 1503.84 ms/call 当作新方法速度。

## task 5 负结果如何解释

task 5 在 65 次调用后未完成任务，但 65 次调用中 62 次走 L27，L27 占 95.38%，且没有
runtime error、非有限 action 或采集错误。因此这个单样本失败不能被解释成“提前退出导致
失败”。更合理的结论是：冻结 teacher/backbone 在该 state 上本身存在任务波动，需要后续
与 full-depth 或 original A1 配对验证；该 episode 的 context/teacher 行仍然是有效训练数据。

## 数据门禁

聚合器对三个 shard 完成以下检查：

- task×state 集合与 `10×1` 网格完全相等；
- 每个 episode 的 call ordinal 从 0 连续递增；
- `(episode_id, call_ordinal)` 全局唯一；
- schema、feature group 与 199D 维度完全一致；
- 所有 feature 有限；
- 每个源 payload SHA-256 重新计算一致。

聚合结果：

```text
runs/route_first_teacher_pilot_state0/aggregate/route_first_teacher_state0.npz
rows: 368
payload SHA-256: ef1cac62c8feea17bc701868ca05991928bd2464201698d560ca9433835a0cd1
file SHA-256: afde9703e8cc6ea27229fb26284fb9395b7bd46de3e8096716e906795257a6dc
```

## 启动编排负结果

首次三 worker 同秒启动时，共享 `OUTPUT_ROOT` 的秒级目录名发生碰撞。GPU5/6 在 preflight
独占写入时安全退出，均未加载模型、未产生 policy call 或 teacher shard；GPU4 正常继续。
随后 GPU5/6 使用隔离输出根补启，task 划分、state 和 seed 均未改变。

runner 已改为目录名包含物理 GPU 编号和纳秒时间戳，后续共享输出根的多卡采集不会再依赖
“启动时间错开”来避免碰撞。

## 代码验证

route-first 定向测试为 7 passed；项目门禁为 `496 passed + 22 subtests`、0 failed、3 个
third-party warning。runner 通过 `bash -n`，结果 JSON 可严格解析，D9 五个冻结保护文件哈希
均未改变。

第一次直接对仓库根执行 pytest 时，上游 `a1/data/vla/test_dataloader.py` 被意外收集，并因
未设置 `DATA_DIR` 在项目测试开始前退出。随后按项目冻结范围 `pytest tests` 且设置
`DATA_DIR` 重跑，得到上述完整通过结果；前一次属于测试发现范围配置事件，不是方法失败。

## 下一步

按冻结协议采集训练 states 1–7，并与本次 state 0 合并为 `10 tasks × 8 states`。采集完成后
先做 exact-grid 和数据分布审计，再进行 grouped task/episode CV。states 8–9 只用于阈值
校准；states 10–11 在模型和阈值冻结前不得打开，D9 states 40–49 始终禁止用于新方法选择。
