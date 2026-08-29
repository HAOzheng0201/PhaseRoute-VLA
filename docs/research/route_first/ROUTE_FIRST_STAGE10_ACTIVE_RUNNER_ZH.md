# Route-First Stage 10 三臂主动验证运行器

## 本阶段结论

本阶段只实现、审计并冻结正式实验基础设施，不打开 fresh-state payload，也不执行主动控制。正式实验必须等运行器形成独立 clean commit、readiness 证据形成第二个 clean commit 后才能开始。

Stage 10 的科学问题是：在未用于 Stage 6–9 开发的新生成状态上，`route_first_stage8` 能否在成功数不明显下降的前提下，稳定降低每次策略调用延迟，并保持每个有效策略调用恰好一次 flow matching（FM）。

## 冻结实验单元

```text
60 个 fresh triplet
├── 10 个 LIBERO-10 task
├── 每 task 6 个独立 replicate
└── 每个 triplet 的同一 state / policy seed / GPU UUID
    ├── original_a1
    ├── candidate_first_v3
    └── route_first_stage8

总计：60 × 3 = 180 个 active rollout
```

每个 task 的 6 个 replicate 使用三种方法的全部 `3! = 6` 种执行顺序，抵消固定 arm position 带来的系统偏差。首个 arm 之后的轨迹自然分叉属于有效闭环结果，不能要求三条轨迹继续共享观测。

## 三种方法的控制差异

| 方法 | 可输出层 | 决策时机 | 动作头计算 |
|---|---:|---|---|
| Original A1 | L1, L3, …, L27 | 生成相邻候选动作后比较动作变化 | 可能多次 FM |
| candidate-first V3 | L11/L13/L27 | 先生成候选动作，再执行冻结风险门 | 可能多次 FM |
| route-first Stage 8 | L13/L27 | 根据动作无关 199D 上下文先选深度 | 每个有效调用严格一次 FM |

route-first 的一次 FM 完整性不能读取 telemetry 顶层 `fm_calls`。历史计数器会包含控制器内部经过的候选位置，顶层可能显示 3。运行器逐调用要求同时存在且只存在：

1. 一个 `evaluated=true` 的 `exit_candidate`，其 `fm_calls=1`；
2. 一个 `route_first_selected_action`，其 `fm_calls=1`；
3. 一个 `phase_route_decision`，其 `fm_calls=1`；
4. 三个事件选择同一 L13 或 L27，且没有 reject/error 事件。

## 运行链路

```text
tracked protocol + schedule
             │
             ▼
local state binding ──SHA/size──► fresh_states.pt
             │                         │
             │ preflight 只验字节      │ arm 进程首次安全反序列化
             ▼                         ▼
clean runner readiness          精确 task/replicate state
             │                         │
             └──────────┬──────────────┘
                        ▼
              per-arm GPU preflight
          clean / same UUID / no process / ≥40GB
                        │
                        ▼
                 独立模型进程运行
      telemetry + measurement + runtime + episode log
                        │
                        ▼
              进程退出后的 GPU postflight
                        │
                        ▼
                immutable arm attestation
                        │
              三臂均通过且配对身份一致
                        ▼
                immutable triplet record
                        │
                 完整 60 个 triplet
                        ▼
                一次性计算正式 aggregate
```

采用“每臂一个独立进程”，是为了让协议中的“每个 arm 前至少 40,000 MiB 空闲显存”成为真实可验证条件。若模型常驻同一进程，第二、第三臂开始前显存已经被本实验占用，无法诚实满足该门槛。

## 防止结果选择

- 任务失败是有效结果，必须封存，不能重跑换取成功。
- 基础设施错误允许重试，但 task、replicate、state SHA、policy seed、arm order、GPU UUID 和代码 commit 必须完全相同。
- `.attempts/` 永久保存失败尝试和 `abort.json`。
- 不允许换 seed、换 state、移动 threshold、重训 router 或删除不利 episode。
- 未完成 60 个 triplet 时，聚合器直接报错，不计算中期门槛。

## 正式门槛

所有条件为逻辑与：

- `route success ≥ candidate success − 6`；
- `route success ≥ Original A1 success − 6`；
- 60 个 triplet 内 `route/candidate episode-P50` 比值的中位数 `≤ 0.80`；
- 60 个 triplet 内 `route/Original-A1 episode-P50` 比值的中位数 `≤ 0.90`；
- route-first 每个有效 policy call 恰好一次 FM；
- 必须完整得到 60 个 triplet 和 180 个 arm 的有效证据。

延迟只使用 `stage1_measurement.jsonl` 的 `policy_wall_latency_ms`，与 Stage 9 保持同一测量口径。McNemar 精确检验只作描述性报告，不作为优越性门槛。

## 证据目录

```text
runs/route_first_stage10_active/
└── taskXX_replicateYY/
    ├── command.txt
    ├── .attempts/
    │   └── armP_METHOD/attempt_NNN.incomplete/
    ├── arm1_METHOD/
    │   ├── command.txt
    │   ├── preflight.json
    │   ├── preflight_stdout.log
    │   ├── stdout.log
    │   ├── policy_telemetry.jsonl
    │   ├── stage1_measurement.jsonl
    │   ├── phase_route_runtime.jsonl   # dynamic arm only
    │   ├── episode.log
    │   ├── result.json / result.sha256
    │   ├── gpu_postflight.json
    │   └── arm_attestation.json / .sha256
    ├── arm2_METHOD/
    ├── arm3_METHOD/
    └── triplet_record.json / .sha256
```

raw rollout 位于 Git 忽略的 `runs/`，只有冻结协议、readiness、最终精简结果和研究文档进入版本库。

## Readiness 之后的命令

先查看 GPU，明确选择当前无人使用且显存满足门槛的卡：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

单个 triplet：

```bash
cd /data3/haozheng/A1/PhaseRoute-VLA

PYTHONNOUSERSITE=1 \
/home/haozheng/.conda/envs/a1/bin/python \
  scripts/run_route_first_stage10_triplet.py \
  --task-id 0 \
  --replicate-id 0 \
  --physical-gpu-index <IDLE_GPU> \
  --expected-gpu-uuid <GPU_UUID> \
  --python-bin /home/haozheng/.conda/envs/a1/bin/python
```

使用一组已人工确认空闲的 GPU 执行完整冻结 schedule：

```bash
PYTHONNOUSERSITE=1 \
/home/haozheng/.conda/envs/a1/bin/python \
  scripts/launch_route_first_stage10_active.py \
  --gpu-indices 0,1,2,3 \
  --python-bin /home/haozheng/.conda/envs/a1/bin/python
```

launcher 会在当前终端逐行显示带 GPU 前缀的运行日志，同时写入独立 launch log。它不会计算中期指标。只有 60 个 triplet 全部完成后才能运行：

```bash
/home/haozheng/.conda/envs/a1/bin/python \
  scripts/aggregate_route_first_stage10_active.py
```

## 声明边界

即使 Stage 10 通过，也只证明这组 LIBERO-10 fresh generated states 上的工程确认通过；它不是有统计功效设计的非劣检验，不证明系统级端到端加速、跨 suite 泛化、真实机器人有效性或部署安全性。
