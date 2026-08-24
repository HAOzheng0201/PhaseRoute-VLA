# 阶段一第一工作块结果（2026-08-24）

## 结论

固定 L13 的测量基础设施、代码门禁和真实 GPU task 0/state 0 smoke 均已通过。当前状态为：

```text
PASS: fixed L11/L13/L27 plumbing
PASS: external timing overlay and exact-action identity
PASS: D9 protected source SHA-256 unchanged
PASS: fixed-layer controller targeted tests (17 passed)
PASS: 484 tests + 22 subtests
PASS: pip / shell / Python / diff checks
PASS: fixed L13 GPU smoke, task 0/state 0, success 1/1
PASS: 33/33 calls are L13, one FM call, finite 8x7 action and CUDA timing
```

## 本轮发现并解决的问题

正式 evaluator 的 `exit_layer_id` 配置没有传入 `predict_actions(exit_id=...)`。不能直接修改该 evaluator，因为它属于 D9 SHA-256 绑定证据。最初尝试在独立 utility 中直接走 legacy `exit_id`，真实 GPU 首次 policy call 报错 `ValueError: No exit action found`：该分支只返回 KV/hidden，flow-matching 路径需要 `outputs.exit_action`。最终修复为专用 `FixedLayerFlowMatchingController`：只在目标层调用一次 frozen FM action head，既不计算相邻层 comparison action，也不做原 A1 阈值判断，同时继续复用 frozen D9 evaluator。

第一次全量测试得到 `478 passed + 22 subtests，1 failed`。唯一失败不是模型或功能错误，而是新增计时代码改变了 D9 protected 文件哈希。没有更新旧哈希掩盖问题；代码随后改为外部 overlay，并恢复全部 protected 文件原始 SHA。加入专用 fixed-layer 控制器、动作审计和显式 all-8 GPU 实验 opt-in 后，最终全量结果为 `484 passed + 22 subtests`。

## 已保留的真实 GPU 负结果

| 运行目录 | 结果 | 原因 |
|---|---|---|
| `runs/stage1_fixed_baselines/fixed_l13_20260824_224621` | FAIL before model load | 普通沙箱内 EGL 不可见，`MUJOCO_EGL_DEVICE_ID must be between 0 and -1` |
| `runs/stage1_fixed_baselines/fixed_l13_20260824_224820` | FAIL at first policy call | legacy `exit_id` 没有产生 flow-matching `exit_action` |

第二次运行已成功加载 8,460,174,855 参数的 A1、峰值显存约 33,844 MiB，并进入 LIBERO task 0/state 0；失败发生在第一次动作推理，因此不能计作端到端通过。这些目录不会删除，也不会把失败改写成成功。

## fixed L13 权威 smoke 结果

权威运行目录为 `runs/stage1_fixed_baselines/fixed_l13_20260824_232431`。它在物理 GPU 4 上执行；启动前 GPU 4 无计算进程、显存占用 18 MiB、利用率 0%。同一代码修复后的首次成功运行 `fixed_l13_20260824_231424` 保留为预审计 smoke；随后增加只读 `action_finite/action_shape` 字段和 runner 硬门禁，并重跑得到下表结果。

| 项目 | 结果 |
|---|---:|
| LIBERO task/state | 0 / 0 |
| episode success | 1 / 1 |
| policy calls | 33 |
| L13 selections | 33 / 33 |
| FM calls per policy call | 1 |
| finite `8 × 7` action chunks | 33 / 33 |
| measurement errors | 0 |
| CUDA event records | 33 / 33 |
| policy wall latency mean / p50 / p95 | 498.74 / 482.84 / 590.42 ms |
| CUDA event latency mean / p50 / p95 | 498.72 / 482.83 / 590.41 ms |
| episode success wall time | 52.84 s |

原始产物 SHA-256：

```text
evaluation_summary.json  ee6e45f735b2ee89f1e854b886b5a35337bbef712b6b33d9de3c5b91d2fab906
policy_telemetry.jsonl   f4813a4ef0f2479a8afd1bd989bb45c7401c89f36ec0711262a46537d6fe64bc
stage1_measurement.jsonl a4251045e807cc28b6d4a7bd58a31b946650a239e9a3738b7b98db26b98e07da
```

这只证明 fixed-L13 的输入到输出链路、控制不变量和测量链路可运行；单个普通工程 episode 不能证明成功率提升，也不能把 50% 的层数减少直接等同于 50% wall-clock 加速。

## 历史 GPU 状态

2026-08-24 22:24（Asia/Shanghai）检查时：

| 物理 GPU | 显存占用 | 利用率 | 状态 |
|---:|---:|---:|---|
| 0 | 6230 MiB | 100% | occupied |
| 1 | 6085 MiB | 100% | occupied |
| 2 | 6021 MiB | 100% | occupied |
| 3 | 4618 MiB | 37% | occupied |

当时 0–3 均有 OceanGym/Isaac Sim 计算进程，因此没有抢占。当前约束已更新为允许使用 0–7，但每次仍须优先选择没有其他计算进程的卡。

## 下一工作块

继续在启动时动态选择空闲卡，完成同一 task/state/seed 的 fixed L11、fixed L27、original A1 和 PhaseRoute V3 配对 smoke：

```bash
FREE_GPU_INDEX=4  # 示例；运行前按 nvidia-smi 结果替换
GPU_INDEX="${FREE_GPU_INDEX}" \
EXIT_LAYER=11 \
TASK_IDS=0 \
EPISODE_INDICES=0 \
PYTHON_BIN=/home/haozheng/.conda/envs/a1/bin/python \
bash scripts/run_fixed_layer_baseline.sh

# 同卡再执行一次 EXIT_LAYER=27。
```

随后运行带测量 overlay 的 PhaseRoute：

```bash
GPU_INDEX="${FREE_GPU_INDEX}" \
TASK_IDS=0 \
EPISODE_INDICES=0 \
STAGE1_MEASUREMENT=1 \
ALLOW_ALL_GPUS=1 \
PYTHON_BIN=/home/haozheng/.conda/envs/a1/bin/python \
bash scripts/run_libero_phase_route_v3.sh
```

以上属于 ordinary engineering smoke，不重测 D9 state 40–49，也不产生新的独立性声明。
