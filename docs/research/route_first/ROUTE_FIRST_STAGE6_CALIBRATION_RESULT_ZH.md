# Route-first Stage 6：阈值校准与一次性确认结果

## 1. 结论

Stage 6 已按预注册协议完成，状态为：

```text
PASS_ONE_SHOT_CONFIRMATION_ENGINEERING_HOLDOUT_READY
```

结果不是“两个浅退头全部成功”，而是 fail-closed 的部分成功：

- L11 在 state 8 选出的阈值为 `0.9807427653`，但在 state 9 一次性确认失败，已关闭；
- L13 在 state 8 选出的阈值为 `0.9174261218`，在 state 9 原样确认通过，保持启用；
- state 9 的 group-equal 早退覆盖率为 11.29%，估算执行层数降低 5.85%；
- states 10–11 工程留出集可以打开，但 route-first active control 仍未获授权；
- 历史 D9 states 40–49 继续永久禁用。

当前校准后的确定性路由是：

```text
199D action-free context
        │
        ├─ score11（保留供审计，但 L11 head disabled）
        └─ score13
              │
              ├─ score13 >= 0.9174261218  -> L13
              └─ otherwise                -> L27
```

这证明 route-first 分数中的 L13 信号能在一次新 state 上通过预注册的小样本工程门禁，
但尚未证明闭环成功率非退化、wall-clock 加速或优于 A1/PhaseRoute-V3。

## 2. 预注册约束是否被遵守

校准前冻结的边界为：

| 项目 | 冻结值 |
|---|---|
| source commit | `8190a611f52aada2283fbcc8681723969d4c99eb` |
| score model | `PCA64 + L2=0.3`，训练后折叠为 199D affine heads |
| uncalibrated router SHA-256 | `38aaef193442a4b40e71b1d48bee42ffbe5f191cad64f99d20bd3f75df3ad3ae` |
| protocol SHA-256 | `c1ab2aef44595d3d86b04684155a74302d2bb70b91bf01c1722f86d0790ce1d1` |
| state 8 | 只能选阈值 |
| state 9 | 只能原样确认或关闭 head，禁止移动阈值 |
| states 10–11 | 校准 artifact 冻结前禁止打开 |
| states 40–49 | 永久禁止新方法使用 |

实际执行中，state 9 没有重新拟合模型、PCA、L2 或阈值，结果 JSON 明确记录
`thresholds_changed_from_state8=false`。L11 确认失败后没有寻找替代阈值，而是按协议关闭。

## 3. 数据采集与完整性

使用采集时空闲的物理 GPU 4–6。每个进程都只看一张卡，运行冻结 PhaseRoute-V3
controller；route-first collector 仅记录 199D context 与 teacher layer，不改变动作。

| GPU | tasks | episode | 成功 | calls | L11 / L13 / L27 | runtime error | attestation SHA-256 |
|---:|---|---:|---:|---:|---:|---:|---|
| 4 | 0, 4, 8 | 6 | 5 | 231 | 0 / 18 / 213 | 0 | `393f19b9e8ed3bd666d2e52edc9ea067402f289de94ffe02aef723196ea99956` |
| 5 | 1, 5, 9 | 6 | 6 | 167 | 7 / 26 / 134 | 0 | `d9b1c555330ad682ac6527671edc9a4d7dbbf5af9ffda2983342beac09183dac` |
| 6 | 2, 3, 6, 7 | 8 | 7 | 268 | 10 / 34 / 224 | 0 | `70525a29d5d5ea747ae7970933e0cd89c73f9e12121693e94d6ec8f5d0fd4ea0` |
| **合计** | **0–9** | **20** | **18** | **666** | **17 / 78 / 571** | **0** | — |

两个失败 episode 为 `task8/state8` 和 `task7/state8`，均达到 65 次策略调用后失败。
它们被原样保留在数据中，没有因 task success 失败而删除或重跑。全部 state 9 episode
均成功，但 task success 不作为路由阈值选择或确认标签；teacher layer 才是监督信号。

首次启动的四个进程误用了不含 Qwen2 tokenizer 的
`/data3/haozheng/A1/hf_cache`。它们都在模型加载阶段失败，没有打开 episode，也没有生成
teacher NPZ，因此从聚合中排除。随后固定使用仓库内正确缓存
`.cache/huggingface` 重启。这个基础设施负结果予以保留，不混入 20-episode 科学样本。

## 4. 聚合数据

只聚合三个 attestation 为 PASS 的正确 shard：

| 项目 | 数值 |
|---|---:|
| task × state cells | 10 × 2 = 20 |
| rows / policy calls | 666 |
| feature shape | `[666, 199] float32` |
| teacher L11 / L13 / L27 | 17 / 78 / 571 |
| calls per episode | min 20 / mean 33.3 / max 65 |
| aggregate payload SHA-256 | `da0912c44348d33ea52a6c56210f2c9c02778bccfa34555b6807e5b54ce361ad` |
| aggregate file SHA-256 | `896e1644c45b45168d9d3216863fe69f1c6114ee70f08e6a1a1fad28b6fb73f5` |

不同 episode 长度不等，所有 coverage 与 false-safe 主指标继续使用 `(task,state)` cell
等总权重，避免 65-call 失败轨迹支配统计量。

## 5. State 8：阈值选择

只枚举 state 8 中真实出现过的唯一 score，并按预注册规则最大化可行 group-equal
coverage。两个 head 在 selection split 上都有可行阈值：

| head | threshold | selected rows | coverage | empirical false-safe | 90% upper bound | 结果 |
|---|---:|---:|---:|---:|---:|---|
| L11 | 0.9807427653 | 7 | 2.18% | 23.40% | 48.16% | ENABLED FOR CONFIRMATION |
| L13 | 0.9174261218 | 37 | 12.33% | 8.55% | 16.54% | ENABLED FOR CONFIRMATION |

L11 的 357 个候选中只有 4 个可行；L13 有 24 个可行。阈值选出之后立即冻结，state 9
不允许再移动。

## 6. State 9：原阈值一次性确认

| head | threshold | selected rows | coverage | empirical false-safe | 90% upper bound | 决策 |
|---|---:|---:|---:|---:|---:|---|
| L11 | 0.9807427653 | 10 | 3.51% | **61.07%** | **77.99%** | **FAIL，关闭** |
| L13 | 0.9174261218 | 34 | 11.29% | **2.68%** | **9.03%** | **PASS，启用** |

L11 同时超过预注册的经验 false-safe 上限 50% 和 90% 上界上限 75%，因此必须关闭。
L13 的 34 个浅退调用中，teacher label 为 L11/L13/L27 的原始数量分别为 6/27/1，
即出现 1 条 raw false-safe；按 cell 等权后的 false-safe 为 2.68%。

关闭 L11 后，state 9 的最终离线路由描述为：

| 指标 | 数值 |
|---|---:|
| L11 / L13 / L27 | 0 / 34 / 275 |
| group-equal early-exit coverage | 11.29% |
| false-shallow among early exits | 2.68% |
| executed depth ratio to L27 | 94.15% |
| layer-count reduction ratio | 5.85% |

这里的 5.85% 是按选择层数估算的 transformer depth reduction，不是实测 wall-clock
speedup，也未包含 router、vision encoder、tokenizer 或 action head 的固定开销。

## 7. 负结果与解释边界

1. **L11 没有跨 state 稳定确认。** Stage 5 的高 AP lift 只证明排序信号，不保证某个
   固定阈值能控制风险；state 9 结果验证了此前对 L11 稀少标签和任务结构的担忧。
2. **L13 不是零风险。** state 9 有 1 条 raw L27 teacher call 被错误选为 L13。当前门禁
   是工程筛选标准，不是形式化安全保证。
3. **计算收益仍有限。** 在关闭 L11 后，确认集估算层数降低只有 5.85%；route-first
   的核心优势“每次 policy call 只做一次 FM”尚未在 active runtime 中实测。
4. **两个 rollout 失败不是 route-first 导致。** 采集时 route-first 仅 observation-only，
   实际动作仍来自冻结 V3 controller，因此不能把 `task8/state8` 或 `task7/state8` 的失败
   归因于新 router。
5. **没有闭环结论。** 本阶段没有让校准 router 控制机器人；不能报告 success rate
   non-degradation、端到端 latency 或 closed-loop improvement。

## 8. Artifact 与复现

聚合命令：

```bash
python scripts/aggregate_route_first_teacher.py \
  --input runs/route_first_teacher_calibration_states8_9/libero_10_gpu4_20260825_213223_905046799/route_first_teacher_context.npz \
  --input runs/route_first_teacher_calibration_states8_9/libero_10_gpu5_20260825_213357_954514361/route_first_teacher_context.npz \
  --input runs/route_first_teacher_calibration_states8_9/libero_10_gpu6_20260825_213543_327649228/route_first_teacher_context.npz \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episode-indices 8,9 \
  --output runs/route_first_teacher_calibration_states8_9/aggregate_states8_9.npz \
  --summary runs/route_first_teacher_calibration_states8_9/aggregate_states8_9.json
```

校准命令：

```bash
python scripts/calibrate_route_first_router.py \
  --aggregate runs/route_first_teacher_calibration_states8_9/aggregate_states8_9.npz \
  --router runs/route_first_router_stage5/router_uncalibrated.npz \
  --protocol configs/route_first_calibration_protocol.json \
  --output-dir runs/route_first_calibration_stage6 \
  --published-result results/route_first/route_first_stage6_calibration.json
```

| artifact | SHA-256 |
|---|---|
| aggregate NPZ | `896e1644c45b45168d9d3216863fe69f1c6114ee70f08e6a1a1fad28b6fb73f5` |
| calibrated router | `ae561b77c01bd4c7eee6cc0ff91e215733662544cc1af2e5039b0a8f02c60cc2` |
| calibration scores | `58c079f1b27018ffd2743cfd3898d66aa56ce8b6340712a7d9e2a3187a1360f0` |
| published result JSON | `c599ffb8280368a1014f4de0827524c3e0bc5d9ccc172ede4062600fea9d5de5` |

Artifact reload 后的最大分数误差为 0，metadata 记录 `enabled11=false`、
`enabled13=true`、`engineering_holdout_authorized=true` 和
`active_control_authorized=false`。

定向测试为 `12 passed, 1 warning`；全仓 CPU 回归为
`508 passed, 22 subtests passed, 3 warnings`，0 失败，用时 74.31 秒。warning 来自
现有 Google API Python 版本提示与 Pydantic 字段兼容提示，不是校准失败。

## 9. 下一阶段

Stage 7 是冻结校准器的 engineering holdout，不是直接宣称新方法有效：

1. 在查看 states 10–11 前冻结 holdout 协议和 pass/fail 门禁；
2. 固定当前 calibrated router，禁止修改权重和阈值；
3. 采集 LIBERO-10 tasks 0–9 × states 10–11 的 observation-only teacher 数据；
4. 检查 L13 的 coverage、false-safe、分布漂移和 artifact roundtrip；
5. 若 L13 失败则关闭 route-first active 路径并报告负结果；
6. 只有 holdout 通过后，才设计新的 active paired engineering test；历史 D9 数据仍不
   得打开或用于选择。
