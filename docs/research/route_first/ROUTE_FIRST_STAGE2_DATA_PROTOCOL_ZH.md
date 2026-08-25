# Route-First Stage 2：数据协议与聚合门禁

Stage 1 只证明单 episode 采集无干预。Stage 2 固定后续数据边界，并实现严格 shard
校验和 exact-grid 聚合，防止漏 task、漏 state、重复 policy call 或坏哈希悄悄进入训练。

## 固定划分

| split | initial-state indices | 用途 |
|---|---|---|
| train | 0–7 | 参数训练与 grouped CV |
| calibration | 8–9 | 冻结 L11/L13 安全阈值 |
| engineering holdout | 10–11 | 模型与阈值冻结后一次性打开 |
| historical D9 | 40–49 | 禁止新方法训练、校准和选择 |

固定配置位于 `configs/route_first_teacher_protocol.json`。task/episode/call identity 只用于
分组和数据审计，不进入 199D feature。

## 聚合器门禁

`scripts/aggregate_route_first_teacher.py` 对每个 NPZ 执行：

1. `allow_pickle=False` 严格读取；
2. schema、199D group names/widths 完全匹配；
3. 重新计算 payload SHA-256；
4. episode identity 与 task ID 一致；
5. 每个 episode 的 call ordinal 必须从 0 连续递增；
6. 拒绝重复 `(episode_id, call_ordinal)`；
7. 实际 task×state 集合必须与命令指定网格完全相等；
8. 按 task、state、call 排序后 exclusive-create 发布 aggregate。

真实 Stage-1 shard 已通过聚合器回放：输入和聚合均为 35 行，payload SHA-256 保持为
`f2881e133fc61b580e11b1a3240a964da4f6eef008d7a93c776737e39606e329`。

全量测试结果：`495 passed + 22 subtests`，0 failed。

跨 10 task 的 state-0 pilot 原计划使用空闲 GPU 4/5/6，但并行外部审批通道断流，三个
worker 均未启动，也未创建部分结果。获得明确多卡授权后应按原 task partition 执行，
不得因为审批失败更换 seed 或数据网格。
