# Route-first Stage 10：fresh-state 生成与封存结果

## 1. 结论

Stage 10 的 fresh-state 生成门禁一次通过：

```text
PASS_ROUTE_FIRST_STAGE10_FRESH_STATES_FROZEN
```

本结果只确认 60 个新 MuJoCo 状态已按预注册 schedule 确定性生成并封存，不包含策略
推理、任务成功率或延迟结论，也不代表 180 个三臂 active rollout 已完成。

## 2. 执行记录

| 项目 | 结果 |
|---|---:|
| source commit | `9acdadad0672dfbaec9697e4025147fc991b3243` |
| task × replicate | `10 × 6` |
| 独立生成进程 | `120` |
| 确定性生成遍数 | `2` |
| 两遍 byte-identical state | `60 / 60` |
| 每个 task 的唯一状态 | `6 / 6` |
| 初始已完成状态 | `0` |
| checkpoint 加载 | `0` |
| policy action 采样 | `0` |
| official states 0--49 访问 | `0` |
| V3-D8/D10 state 复用 | `0` |
| GPU 查询或初始化 | `0` |

两遍生成共耗时约 125.15 秒，使用 `CUDA_VISIBLE_DEVICES=-1`、OSMesa 与最多 8 个 CPU
worker。生成过程中没有失败记录，因此没有基础设施重试；协议也禁止为 duplicate、nonfinite、
initially-solved 或不确定性状态换 seed 补位。

## 3. 状态维度

MuJoCo flattened state 的维度由各 task 场景决定。同一 task 的 6 个 replicate 维度完全
一致：

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dimension | 123 | 123 | 47 | 51 | 84 | 45 | 71 | 84 | 47 | 47 |

维度不同不是异常：不同 LIBERO task 的对象数量和 MuJoCo 状态结构不同；门禁要求的是
task-local 一致，而不是跨 task 强制相同。

## 4. 封存哈希

| 对象 | SHA-256 |
|---|---|
| generation result | `30066ff0c0c3d8ddedc77c24a456165304c682df8947cb66305b3dbc535862e9` |
| local state attestation | `c8915722c1e7a6f05772d1d949cfcb8f24bfa4c93d3c70c2b63f0daff348a5ab` |
| `fresh_states.pt` | `a757248882334724fd29d29e5d4535d08f0424b4f9c47abee69131f93e84ad4a` |
| tracked state result | `4fa648bbeba2b91765517c31fb24ca3226fbd05cbdc051d901b6b06d8880861c` |
| tracked state binding | `e21d3536a8b93b627a7caec8f8c0d297fd13c2d10b6f069b5e6a7a70feb4ade3` |

原始 payload 和 120 条逐进程记录保存在被 Git 忽略的 `runs/`，避免把机器相关运行产物
塞进开源仓库；Git 跟踪的 result 与 binding 固定其路径、大小、SHA、source commit、协议
和 schedule。后续 runner 必须先同时验证 tracked binding、local attestation 和 payload，
任何一个字节漂移都 fail closed。

## 5. 下一门禁

本结果只授权实现三臂 active runner 与 CPU contract tests。真正打开 60 个 payload 并运行
Original A1、candidate-first V3、route-first Stage 8 之前，还必须：

1. 实现逐 triplet 全排列执行、同 state/policy seed/GPU UUID 绑定；
2. 实现三臂各自的 runtime attestation 与 route-first exactly-one-FM 审计；
3. 实现 60-triplet/180-arm 完整性聚合器；
4. 完成 CPU 合约测试、全回归和 clean runner commit；
5. active preflight 确认一进程一卡、显存门槛和外部占用门禁。

在这些条件完成前，`active_rollouts_started` 保持 `false`。
