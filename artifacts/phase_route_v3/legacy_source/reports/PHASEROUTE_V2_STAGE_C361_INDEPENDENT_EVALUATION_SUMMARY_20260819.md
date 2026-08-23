# PhaseRoute-VLA C3.61 独立评估结果记录

日期：2026-08-19  
状态：正式 one-shot 已完成；Motion 与 Tail 组件通过，Gripper 未评估  
结论边界：离线组件可行性成立；尚未证明闭环 LIBERO 成功率或实际推理效率优于 A1

## 1. 本次正式执行

- 仅使用物理 GPU 0–3；GPU 4–7 未参与计算。
- 7 个 worker 按 `context → shard0 → shard1 → shard2 → shard3 → aggregate → evaluate` 串行完成 marker 前 READY。
- 在两次 worker 存活检查之间人工提交唯一口令 `COMMIT_C361_ONE_SHOT`。
- 全局 marker 已以 `0444` 权限耐久落盘；不允许删除、替换、恢复或重跑。
- 四个 candidate shard 随后在 GPU 0–3 并行执行，最终全部成功发布。
- 运行结束后 GPU 0–7 均恢复为约 15 MiB 驱动基线，无残留计算进程。

正式执行前验证：

- C3.60–C3.61：315 tests passed。
- C3.57–C3.61：625 tests passed。
- AST：594 个 Python 文件解析通过。
- 关键冷导入：16 个模块通过。
- `/bin/sh -n`、`pip check`、`git diff --check` 均通过。
- 两次独立红队最终均给出 GO。
- 正式产物只读审计为 GO；282 个冻结 source-closure 文件零漂移。

## 2. 冻结身份与不可变哈希

| 对象 | SHA-256 |
|---|---|
| C3.59 result | `d4c2a9f29ebd30903ef5b63402521f076eb15c856f60771f54c00ea9867632e8` |
| Scientific contract | `fd60edbe4f76f252b3b97a09a75dd2f3c86826627ce90068316ff0ca8520869b` |
| C3.60 v3 canonical contract | `fb9fcf0ed10b1989ad8f385dda29c527cdaea49ebb8d01767f39f64053fa26a0` |
| C3.60 v3 result | `976d0ac55ad664427348909a14c94bf19652814ce380ec0ebb5a2a50200a3f03` |
| C3.61 global marker | `cc5d4872e0d6ee929ccf9398bad3d6b7f63520c9ad1f248bac1be5824e2c967a` |
| Context result | `91981f1bacbe18da183907152bf37a67107d3a48fd1501a3ce16cb9e475ba009` |
| Shard 0 result | `26dbd53b4719d28f5bae10f28f7424f0c7059a865d0eabf165c9172b30ef6e8e` |
| Shard 1 result | `e27a8c52e1ba521cf10d1c08a40d7240f5307acfce9151eed21f76cc02f06e32` |
| Shard 2 result | `ea3867e538b637c58bfca4c18af20caa2f7b3281c9791444b7bbd98e43c86fc4` |
| Shard 3 result | `e16322779d4c97a2137528667e77472ad37b06fd0e2a927e9f7d3faba4b935fd` |
| Aggregate result | `16a78d90c3eec9b4bed52756ddc30b8787fbe42f20415ab6fcf997c01b29f118` |
| Evaluation result | `b696c9df07b1d83282af8699440919e8f6a835cc68f9ebaba3a6339b23d3a7c2` |
| Evaluation payload | `b7d9a9743e1a563a54a98be8a8e275b2763bc7401b56907461350d7eb1263bb3` |
| Evaluation records | `0e892845c48c4154123c15e0d67d34cea736f56eb90f213c71017d83e5eea667` |

C3.60 v3 external validator 的最终摘要：

- `checks = PASS`
- implementation sources rehashed：15
- runner sources rehashed：6
- test sources rehashed：16
- runtime sources rehashed：245
- trust anchors rehashed：6
- prefreeze incident validations：2
- runtime-readiness incident validations：2
- protected access counts：全部为 0
- GPU probe：false
- marker created or attached：false

## 3. 独立集与张量语义

- 独立集共 1,192 个 action call，覆盖 10 个 task、9 个 episode-index 组。
- 四个 shard 行数为 `300 + 300 + 299 + 293 = 1192`。
- 每行同时回放候选层 L11、L13，以及仅作 consistency teacher 的完整深度 L27。
- `candidate_actions`：`[1192, 3, 8, 7]`。
- `full_depth_deltas`：`[1192, 3, 8, 7]`。
- `candidate_context_features`：`[1192, 2, 82]`。
- 每个 action chunk 为 8 个时间步、每步 7 维动作。
- 全部 1,192 行均为 `evidence_status = FINITE`，无非有限数值执行失败。

## 4. 主要结果

最终 family decision：

| 组件 | 判定 |
|---|---|
| Motion | `PASS` |
| Tail | `PASS` |
| Gripper | `NOT_EVALUATED_DUE_TO_FROZEN_DEVELOPMENT_FAILURE` |

### 4.1 Motion

| 指标 | Model SSE | Frozen baseline SSE | Ratio | 相对改善 |
|---|---:|---:|---:|---:|
| Translation RMS | 0.1708098902 | 0.2215781174 | 0.7708788766 | 22.91% |
| Rotation RMS | 0.2014802045 | 0.2479226009 | 0.8126738094 | 18.73% |

- 两项 pooled support 均为 2,384 个 row-layer pair。
- 严格门槛为 `ratio < 1.0`，无 epsilon 或 tolerance。
- Translation 与 Rotation 均在 9/9 个 episode-index 组改善；预注册要求为至少 7/9。
- 按 task 看，Translation 为 10/10 改善；Rotation 为 8/10 改善。Task 2 和 Task 5 是局部退化点。

### 4.2 Tail calibration

| 指标 | 结果 | Frozen baseline | Ratio/门槛 |
|---|---:|---:|---:|
| Coverage | 89.6393% | — | 必须位于闭区间 [85%, 98%] |
| q=0.9 Pinball sum | 57.78888285 | 58.68753357 | 0.9846875364，改善 1.53% |
| Upper-width sum | 270.19030589 | 272.97265625 | 0.9898072195，改善 1.02% |

- Tail pooled support 为 2,384。
- 按 task 看，Pinball 为 10/10 改善，Width 为 6/10 改善。
- 8/10 task 的 coverage 位于预注册区间；Task 0（84.07%）和 Task 8（83.20%）低于 85%。

## 5. “多少实验因提前退出而最终失败”

当前正式证据不能计算这个因果数量，不能写成 0，也不能把风险超界数直接称为任务失败。

原因：

- `runtime_threshold_defined = false`，没有真实的在线 selected route。
- `joint_shadow_or_active_control_authorized = false`，没有闭环控制。
- 记录中没有 `task_success`、`reward`、`done`、`selected_layer` 或 `failure_reason`。
- 当前语义明确为 `offline_layer27_teacher_consistency_only`。

可以直接统计的是 action-chunk 风险代理：

```text
tail_target = max_abs(A_early - A_layer27) over [8, 7]
tail_covered = tail_target <= calibrated_tail_upper
```

| 风险代理 | 数量 | 比例 |
|---|---:|---:|
| L11 超过校准上界 | 112 / 1192 | 9.3960% |
| L13 超过校准上界 | 135 / 1192 | 11.3255% |
| 总 row-layer 超界 | 247 / 2384 | 10.3607% |
| 任一浅层超界的 action call | 189 / 1192 | 15.8557% |
| 两个浅层均超界的 action call | 58 / 1192 | 4.8658% |

90 个 task×episode cell 中有 76 个至少出现一次超界，但这仍不能称为“76 个实验失败”。约 10% 的超界率与 q90 风险界预期一致，且 Tail 的总体 coverage 判定仍为 PASS。

## 6. 效率结论的边界

- L11/L27 与 L13/L27 对应的纯 Transformer 深度理论节省分别约为 59.26% 和 51.85%。
- 这不是实测 FLOPs、端到端 latency 或吞吐率。
- 本轮没有学习或启用 runtime threshold，也没有形成 L11/L13 的真实路由分布。
- 因此，当前不能声称“PhaseRoute-VLA 已在闭环任务中比 A1 更快且成功率更高”。

可严谨声称的是：在冻结的 1,192 条独立离线证据上，L11/L13 早层候选的 Motion 风险估计和 Tail 校准相对冻结 baseline 通过了预注册门槛，说明继续训练并验证真实路由具有依据。

## 7. 下一阶段建议

1. 在开发/训练划分上修复并重新训练 Gripper 分支；不得用本次独立测试集反向调参。
2. 只在开发/校准集上学习 runtime routing threshold，明确 L11、L13、L27 的选择规则。
3. 新建独立 protocol，进行 shadow routing，并实测 selected-layer 分布、latency、FLOPs、显存与风险超界率。
4. 完成闭环 LIBERO rollout，与 A1/full-depth 比较成功率、平均计算量和失败类型。
5. 对 Task 2、Task 5 的 Rotation 退化，以及 Task 0、Task 8 的 coverage 偏低做开发集诊断；禁止针对本次 test artifact 直接优化。

## 8. 正式 artifact 位置

- Marker：`reports/phase_route_v2_stage_c361_independent_evaluation_20260819_v1.global_attempt_consumed.json`
- Context：`reports/phase_route_v2_stage_c361_independent_context_20260819_v1/`
- Candidate shards：`reports/phase_route_v2_stage_c361_independent_candidate_shard00of04_gpu0_20260819_v1/` 至 `...shard03of04_gpu3.../`
- Aggregate：`reports/phase_route_v2_stage_c361_independent_aggregate_20260819_v1/`
- Evaluation：`reports/phase_route_v2_stage_c361_independent_evaluation_20260819_v1/`

所有正式目录均已原子发布；不存在对应 `.incomplete` 或 `abort.json`。

## 9. 非阻断运行提示

- 四个 candidate 日志均提示 `config.proprio_dim 8 does not match config.action_dim 7 for AffordVLA`；本轮未因此失败，但后续闭环实验前应确认这是预期的状态/动作定义，而不是配置遗漏。
- Google API 提示 Python 3.10 将在 2026-10-04 进入其支持终点后的停止支持窗口。
- Pydantic 输出了 `repr`/`frozen` 元数据兼容性 warning；本轮结果与哈希未受影响。
