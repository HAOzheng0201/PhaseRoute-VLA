# PhaseRoute-v2 C3.59：独立评估协议冻结与独立验签报告

日期：2026-08-19（Asia/Shanghai）

正式状态：`C359_INDEPENDENT_EVALUATION_PROTOCOL_FROZEN`

schema：`phase-route-vla.stage-c359-independent-evaluation-protocol.v1`

阶段性质：metadata-only、design-only、CPU-only、one-shot independent-evaluation
protocol freeze。

正式 `result.json` SHA-256：
`d4c2a9f29ebd30903ef5b63402521f076eb15c856f60771f54c00ea9867632e8`

独立验签状态：`PASS`

> **最重要的证据边界：** C3.59 冻结了 independent-test 的五字段元数据选择、未来输入构造、
> 指标、门、失败语义、一次性消费规则以及 C3.60 的授权边界；本阶段没有打开 independent-test
> source manifest，没有解析 `array_path`，没有解析 sample path，没有打开 sample payload，也没有计算
> 任何 independent-test 指标。本文不能被引用为 PhaseRoute-v2 优于 A1、CogVLA 或其他方法的证据。

## 1. 阶段目标、角色与结论

C3.59 位于 development 与 calibration 完成之后、独立测试真正执行之前。它解决的不是“模型成绩
是多少”，而是“在看见任何独立测试样本之前，把将来只能执行一次的评估规则完整写死”。具体包括：

1. 从 C3.49 已冻结的全局五字段 metadata index 中，确定 independent-test 的 1,192 行、90 个
   task×episode cells 及四 shard 分配；
2. 绑定 C3.55 development checkpoint、C3.58 tail correction 以及所有上游协议与源码 SHA；
3. 冻结 L11/L13 相对同噪声 L27 consistency teacher 的输入、82 维 causal context、motion/tail
   目标、基线、dtype、归约顺序和严格判定门；
4. 保留 gripper 的 development 负结果，不允许利用 independent test 修复或重新晋级；
5. 冻结 global one-shot attempt marker，使失败后不能在同一 test 上反复调参直到得到 PASS；
6. 只授权 C3.60 实现、合成测试、独立审查并 SHA 冻结 runner；C3.59 本身不授权打开 test
   payload。

本阶段可以正式声明：**独立评估协议及 metadata selection 已冻结、正式发布并通过独立验签。**
本阶段不能声明：motion/tail 已通过独立测试、任务成功率提升、提前退出效率提升、在线安全、可部署，
或相对 A1/CogVLA 更优。

## 2. 1,192 行、90 cells 与四 shard

independent-test episodes 固定为：

```text
[7, 10, 13, 16, 19, 22, 25, 28, 29]
```

覆盖 task `0--9`，所以完整分组为：

```text
10 tasks × 9 episodes = 90 task×episode cells
```

| 项目 | 冻结值 |
|---|---:|
| source rows | 1,192 |
| task×episode cells | 90 |
| 每 cell 行数范围 | 9--28 |
| shard 0 | 300 |
| shard 1 | 300 |
| shard 2 | 299 |
| shard 3 | 293 |

shard 规则固定为 `shard_assignment = dataset_index mod 4`。metadata selection 保持全局
`dataset_index` 的原始递增顺序，四 shard 无角色回退。持久化记录只有以下五个整数域：

```text
dataset_index, task_id, episode_index, call_ordinal, shard_assignment
```

`sample path`、`array_path`、`step_id`、行为退出层、图像、proprio、candidate action、risk target
与 task success 均未写入 C3.59 selection artifact。selection 的 ordered dataset-index identity SHA 为：

```text
4ca81655ca3943428315c371ed7fc734491ea07b049ee734616ab66860950236
```

## 3. 从输入到指标的预注册流程

下图描述的是 **C3.59 冻结、但尚未在 independent payload 上执行** 的未来流程。C3.60 只能实现
和合成测试 runner；真正的 one-shot 数据访问还必须等待后续 C3.61 同时绑定 C3.59 与 C3.60 的
外部 result SHA。

```mermaid
flowchart LR
    M[冻结 metadata selection<br/>1192 rows / 90 cells<br/>5 integer fields]
    P[未来受保护 payload access<br/>C3.59 阶段未发生]
    C[九个 causal context tensors<br/>past-only history length 8]
    A[同一 cached FM input<br/>A1 fixed-depth FM10<br/>L11 / L13 / L27]
    D[decision deltas<br/>L11,L13 minus L27<br/>FP32 1192×2×8×7]
    F[isolated current-candidate builder<br/>FP32 1192×2×82]
    MP[motion frozen predictor<br/>translation + rotation<br/>CPU FP64]
    TP[tail frozen q90 predictor<br/>FP64→FP32 + fixed correction]
    MM[motion SSE ratios<br/>pooled + per layer + episodes]
    TM[tail coverage + q90 pinball<br/>+ mean-width ratios]
    G[family-level PASS / FAIL / INCONCLUSIVE<br/>gripper fixed NOT_EVALUATED]

    M -.C3.60 后且 marker 已持久化.-> P
    P --> C
    P --> A
    A --> D
    C --> F
    A --> F
    D --> MM
    F --> MP --> MM
    D --> TM
    F --> TP --> TM
    MM --> G
    TM --> G
```

### 3.1 九个 causal context 输入

| tensor | dtype | future independent shape | 作用 |
|---|---|---:|---|
| `instruction_summary` | FP32 | `[1192,3584]` | 指令摘要；不直接拼进最终 82D，但参与上游 phase/输入校验 |
| `vision_crop_summary` | FP32 | `[1192,5,3584]` | 五个视觉 crop 摘要 |
| `vision_crop_mask` | bool | `[1192,5]` | crop 有效位 |
| `phase_embedding` | FP32 | `[1192,128]` | phase 表征 |
| `phase_scalars` | FP32 | `[1192,3]` | progress/boundary/uncertainty 类标量 |
| `normalized_proprio` | FP32 | `[1192,8]` | 当前归一化本体状态 |
| `proprio_history` | FP32 | `[1192,8,8]` | 长度 8 的 past-only proprio history |
| `action_history` | FP32 | `[1192,8,8,7]` | 长度 8、每项 `8×7` 的 past action chunk |
| `history_mask` | bool | `[1192,8]` | 右对齐历史有效位 |

history 的 state key 是 `(task_id, episode_index)`；每个 cell 的 `call_ordinal` 从 0 连续递增。
窗口先以 FP32 零填充并右对齐 materialize，再 commit 当前 `normalized_proprio [8]` 与 source
`teacher_normalized_action/behavior_action [8,7]`。因此当前行不会进入自己的 context；有效历史数恰为
`min(call_ordinal, 8)`，新 cell 第一行是全零值和全 false mask，禁止跨 cell 状态泄漏。

### 3.2 同噪声候选与 decision delta

候选层轴顺序固定为 `[11,13,27]`。每行同一个 cached `teacher_exit_input_x [8,7]` 被三层
bit-identical 复用，A1 fixed-depth replay 固定 FM10、CUDA BF16 autocast、deterministic algorithms
和 shard seed `20260817 + shard_index`。正式 A1 config/checkpoint SHA 分别为：

```text
config:     9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca
checkpoint: dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f
```

未来输出合同为：

```text
candidate_actions  : FP32 [1192,3,8,7]
full_depth_deltas  = candidate_actions - candidate_actions[:,2:3,:,:]
decision_delta     = full_depth_deltas[:,0:2,:,:]  # FP32 [1192,2,8,7]
full_depth_deltas[:,2,:,:] == 0                    # exact
```

L27 是同 observation、同 noise 下的 consistency teacher，不是 expert action、task-success label 或
安全 ground truth。

### 3.3 82 维 isolated current-candidate context

L11 与 L13 分别调用一次 isolated single-candidate builder；另一 shallow candidate 与 L27 teacher
均不可见。C3.40 原始 86D 的最后 `[82:86]` layer one-hot 被删除，candidate layer 仅通过冻结的
layer-specific preprocessing/anchor 体现。

| half-open slice | 维数 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | phase scalars |
| `[3:7]` | 4 | phase embedding mean、population std、RMS、max-abs |
| `[7:15]` | 8 | normalized proprio |
| `[15:23]` | 8 | proprio minus latest past proprio |
| `[23:30]` | 7 | latest past chunk first action |
| `[30:37]` | 7 | current candidate first action |
| `[37:44]` | 7 | current minus latest-past first action |
| `[44:51]` | 7 | current candidate horizon mean |
| `[51:58]` | 7 | current candidate horizon population std |
| `[58:65]` | 7 | past first-action mean |
| `[65:72]` | 7 | past first-action population std |
| `[72:78]` | 6 | history fraction、candidate temporal/total RMS、past/current-vs-past/history temporal RMS |
| `[78:82]` | 4 | pooled vision mean、population std、RMS、crop-dispersion RMS |

最终 `candidate_context_features` 为 contiguous finite FP32 `[1192,2,82]`，flatten 顺序固定为
source-row-major，并在每一行内先 L11、后 L13。

## 4. Motion：精确目标、基线、指标与门

### 4.1 目标与冻结推理

对 `delta = decision_delta`：

```text
translation_rms = sqrt(mean(delta[:,:,:,0:3]^2, axes=(horizon,component)))
rotation_rms    = sqrt(mean(delta[:,:,:,3:6]^2, axes=(horizon,component)))
```

目标先以 FP32 构造为 `[1192,2,2]`，再一次性转 CPU FP64 用于指标。冻结 C3.55 full-development
refit 的推理为：

```text
standardized = (feature - feature_mean_by_layer) / feature_scale_by_layer
residual     = matmul(standardized, motion_weight.T)
prediction   = motion_anchor[None,:,:] * exp(residual - correction[None,:,:])
```

prediction 为 CPU FP64 `[1192,2,2]`。基线是 checkpoint 内的 layer-specific
`motion_anchor.unsqueeze(0).expand(1192,-1,-1)`；禁止从 test 重新估计 anchor、mean、scale、weight
或 correction。

### 4.2 Primary gates

对 translation 与 rotation 分别计算：

```text
ratio = sum((prediction - target)^2) / sum((frozen_anchor - target)^2)
```

严格门为：

1. pooled ratio `< 1.0`；
2. L11 ratio `< 1.0`；
3. L13 ratio `< 1.0`；
4. 对该 target 的 9 个 test episodes，至少 7 个 episode 满足 model SSE 严格小于 baseline SSE。

episode SSE 汇总该 episode 的全部 tasks、calls 与两个 layers。比较不加 epsilon/tolerance；相等不算
improvement。每个 target 的 pooled support 是 2,384 个 row-layer pairs，每层 support 是 1,192。
分子与分母分别以 contiguous CPU FP64 单线程 `torch.sum` 直接求和，只做一次最终除法；不允许先算
per-row ratio、shard mean 或 group macro average。

完整、有限但没有跨过门时，科学状态是 `FAIL`，不是 `INCONCLUSIVE`；primary denominator 非有限或
非正时，该 family 为 `INCONCLUSIVE`。

## 5. Tail：精确目标、校准上界与三组门

Tail target 是每个 L11/L13 action chunk 相对 L27 的 `8×7` 最大绝对差：

```text
tail_target = max_abs_over_8x7(decision_delta)  # FP32 [1192,2]
```

冻结 C3.55 tail q90 predictor 在 CPU FP64 上使用与 motion 相同的 layer-specific standardization：

```text
q90_fp64 = tail_q90_anchor[None,:,:] * exp(residual - correction[None,:,:])
```

随后严格执行 FP64→FP32 cast，再加 C3.58 已冻结的 FP32 layer correction：

| layer | fixed correction |
|---:|---:|
| 11 | `0.009694024920463562` |
| 13 | `0.0036352351307868958` |

```text
upper_fp32 = float32(q90_prediction) + correction_fp32[None,:]
covered    = target_fp32 <= upper_fp32
```

禁止在 test 上重新 conformal fit、重新校准或选择 correction。Tail 必须同时通过以下全部门：

1. **coverage：**pooled、L11、L13 的 empirical coverage 都位于闭区间 `[0.85,0.98]`；
2. **q90 pinball：**使用
   `0.9*max(target-z,0) + 0.1*max(z-target,0)`，pooled、L11、L13 相对冻结
   C3.55 layer anchor 的 pinball ratio 都严格 `< 1.0`；
3. **mean width：**单侧区间下端固定为 0，width=`upper-0`；pooled、L11、L13 的 mean-width
   ratio 都严格 `< 1.0`。

FP32 upper 构造完成后，pinball 与 width 的数值累计使用 CPU FP64；同样禁止 epsilon、目标感知删行、
per-row ratio 或 group macro primary。

## 6. Gripper 固定负结果与 family decision

Gripper 的正式 disposition 固定为：

```text
NOT_EVALUATED_DUE_TO_FROZEN_DEVELOPMENT_FAILURE
```

因此 independent test 上不构造 gripper target、不运行 gripper predictor、不计算 gripper metric，
也不能以 test 结果修复或晋级该 family。joint shadow disposition 固定为：

```text
SHADOW_NOT_AUTHORIZED_GRIPPER_DEVELOPMENT_FAILURE
```

Motion 与 tail 分 family 报告 `PASS / FAIL / INCONCLUSIVE`。其中任一 family 的 PASS 最多只能作为
“相对同噪声 L27 teacher 的离线 component evidence”；它不授权 joint router scoring、shadow control、
active action control、部署或任务成功/安全声明。

所有 family 还必须报告完整 cube：pooled、by-layer、by-task、by-task-and-layer、by-episode、
by-episode-and-layer、by-task-and-episode、by-task-episode-and-layer。分组指标是 diagnostic，不能反向
参与模型、门或结论选择。

## 7. Global one-shot attempt marker

独立测试不是“失败后修一修再跑”的开发集。C3.59 冻结的消费点是：

```text
atomic exclusive marker create
→ marker file fsync
→ parent directory fsync
→ global scientific attempt irreversibly consumed
```

marker 必须先于以下四类任一事件：

1. independent-test source manifest open/parse；
2. independent-test `array_path` parse/follow；
3. independent-test sample path resolution；
4. independent-test sample payload open。

任一 shard 到达受保护访问即属于同一个 global attempt；marker 的 fsync 成功本身就消费 attempt，
即使随后一个 payload 都没打开。没有 marker 的受保护访问是 `ABORT_SECURITY_INCIDENT`。marker 不得删除、
替换或重建以获得重试机会。

只有在 marker 之前失败且四类受保护事件一个都未发生时，才是
`PRE_ATTEMPT_ABORT_MAY_RETRY_AFTER_FIX`。marker 之后任何完整性或执行故障固定为
`IMMUTABLE_INCONCLUSIVE_EXECUTION_FAILURE`；不允许 resume/rerun 改变结果。完整有限计算未过科学门则
是不可变 `FAIL`。`FAIL` 或 `INCONCLUSIVE` 都不能在同一 test 上重跑寻求 PASS，新协议版本也不会让
同一 test 恢复资格；方法或门发生变化时必须使用新的、从未触碰的 independent data。

## 8. 工程安全与访问审计

### 8.1 运行与 I/O 安全

正式 freezer 以 Python 3.10.20、`-I -B`、`PYTHONNOUSERSITE=1`、空
`CUDA_VISIBLE_DEVICES` 运行；未查询 GPU，CUDA 未初始化。freezer 为 stdlib-only，禁止导入 A1、
NumPy、PyTorch、TensorFlow、Transformers、Diffusers 等重模块。

输入读取使用 repository-relative `openat`、`O_NOFOLLOW`、regular-file 与字节上限检查，并在同一
file descriptor 上完成读取、身份稳定性检查和 SHA 验证；JSON 拒绝 duplicate key、NaN 与 Infinity。
正式发布采用 exclusive `.incomplete`、exclusive artifact create、file/directory fsync 和 Linux
`renameat2(RENAME_NOREPLACE)`，拒绝覆盖既有正式目录。

独立 validator 是 read-only frontend：先用外部传入的正式 result SHA 验证 `result.json`，再验证
implementation source SHA 后才编译已认证的 freezer source，并逐字段重建、复核 selection、source、
command、access receipts 与目录内容。

### 8.2 正式 access ledger

| 项目 | 结果 |
|---|---:|
| 正式 checks | `12/12 true` |
| recorded file paths | 256 |
| parent JSON documents | 6 |
| metadata documents | 1 |
| opaque SHA-bound receipts | 15 |
| visible GPU count | 0 |
| CUDA initialized | false |
| output writes | command `1`；selection `1`；result `1`；abort `0` |

以下正式 operation counts 全部为 0：

```text
source_manifests_opened
sample_payload_paths_resolved
sample_payload_files_opened
independent_test_sample_payloads_opened
array_paths_parsed_or_followed
numpy_payload_loads
torch_payload_deserializations
checkpoint_deserializations
gpu_queries
cuda_initializations
model_fits
optimizer_steps
conformal_fits
checkpoint_writes
runtime_threshold_selections
online_rollouts
active_action_controls
```

审计还重新验证了 C3.57 closed-world source closure：10 个 implementation sources、5 个 runners、
6 个 tests 与 227 个 runtime dependency sources 均保持当前 SHA。

需要诚实保留一个工程限制：access ledger 是冻结控制流和统一 I/O helper 的结构化证据，不是外部
kernel syscall trace；原子发布模型也仍假设 cooperative single publisher，不声称防御已打开输出目录
后的恶意同主机篡改。

## 9. 测试、独立验签与 SHA 交接

### 9.1 测试结果

冻结前最终验证记录为：

| 范围 | 结果 |
|---|---|
| C3.48--C3.59 累计回归（25 个 test files） | **549 passed** |
| C3.59 定向协议/freezer/validator 回归 | **27 passed** |

定向测试覆盖：科学 contract 总 SHA 与 mutation rejection、motion 动作轴、82D offsets、history、
same-noise/FM10、marker 语义、metadata selection、parent/artifact SHA、cold isolated import、重模块/GPU/
payload-loader AST 禁令、symlink/no-follow、防覆盖发布，以及 validator 的“先验 result SHA、再认证源码”
顺序。

独立验签返回：

```text
status           = C359_INDEPENDENT_EVALUATION_PROTOCOL_FROZEN
result_sha256    = d4c2a9f29ebd30903ef5b63402521f076eb15c856f60771f54c00ea9867632e8
selection_sha256 = c713cf8e2d4140747cdf119fa9cc552441fafe98b01679120f279608899b0ed8
rows             = 1192
checks           = PASS
```

### 9.2 正式目录三文件 SHA

| artifact | bytes | SHA-256 |
|---|---:|---|
| `command.txt` | 124 | `b570cd2b2c3cb913b2791cd4ad1516a4848c9ddab0de0af8b0abcb96cb082543` |
| `independent_test_metadata_index.jsonl` | 109,450 | `c713cf8e2d4140747cdf119fa9cc552441fafe98b01679120f279608899b0ed8` |
| `result.json` | 111,702 | `d4c2a9f29ebd30903ef5b63402521f076eb15c856f60771f54c00ea9867632e8` |

内部 scientific-contract canonical JSON SHA 为：

```text
fd60edbe4f76f252b3b97a09a75dd2f3c86826627ce90068316ff0ca8520869b
```

### 9.3 冻结实现与测试源码 SHA

| source | SHA-256 |
|---|---|
| `scripts/dynamic_compute/freeze_stage_c359_independent_evaluation_protocol.py` | `acac180a4aea2496d96c3d6d9bcc4f4d3b9cb12caf9e31e2a6eb8549322265a8` |
| `scripts/dynamic_compute/validate_stage_c359_independent_evaluation_protocol.py` | `499a9b6eaf9b0e1e862bbd1f1070735e2b4707c0b71e9deba8369b20d6961fb8` |
| `tests/dynamic_compute/test_c359_independent_evaluation_protocol.py` | `4ea7a8c6953616d7606608008ca30699de9b9bb61f4f3f54b296ffdc2330bfbf` |
| `tests/dynamic_compute/test_c359_independent_evaluation_protocol_freezer.py` | `d3ced10ebfb5035e8f11d3adfd9dd12472baebb33f60d6835cd36e531b3db010` |

## 10. 声明边界

### 10.1 当前允许声明

- C3.59 independent-test metadata selection 已冻结为 1,192 行、90 cells；
- future motion/tail 输入、目标、冻结预测器、基线、dtype、归约顺序与严格门已预注册；
- gripper development 负结果和 joint-shadow 禁令被保留；
- one-shot attempt consumption、失败状态与禁止 test-based repair 的政策已冻结；
- 正式三文件、implementation/test sources 与上游 artifacts 已由 SHA 连接；
- freezer 正式发布及独立 read-only validator 均 PASS。

### 10.2 当前明确禁止声明

- independent-test motion 或 tail 已 PASS/FAIL；
- independent-test coverage、pinball、width、SSE ratio 或任务成功率的任何数值；
- “有多少实验因提前退出而最终失败”的 independent 因果统计；
- PhaseRoute-v2 相对 A1、CogVLA 或其他 VLA 更准确、更高效、更安全或更强；
- layer-27 consistency 等同 expert、task success 或 safety ground truth；
- cluster/task/episode-level coverage guarantee；
- joint shadow、active control、online rollout、runtime threshold、可部署 checkpoint；
- GPU 延迟、吞吐、显存或端到端控制收益。

## 11. C3.60 授权与下一步

C3.59 只授权：

```text
C3.60_IMPLEMENT_TEST_REVIEW_AND_SHA_FREEZE_RUNNERS
```

C3.60 可以：

1. 实现 independent-test runner；
2. 仅使用 synthetic fixtures 做测试；
3. 独立审查 runner；
4. 冻结 runner source 与 runtime dependency SHA；
5. 重新 hash future runtime trust anchors；
6. 冻结确定性 CPU reduction，以及未来四 shard 的物理 GPU `0,1,2,3` allowlist。

C3.60 仍然不可以：打开或解析 test source manifest、解析 `array_path`、解析 sample path、打开 sample
payload、反序列化正式 checkpoint/tensor、查询或初始化 GPU、fit model/calibrator、选择 checkpoint/
threshold、执行 rollout 或 active control。物理 GPU `4,5,6,7` 明确禁止进入未来 C3.61 test 执行。

未来 C3.61 必须同时获得并外部绑定 C3.59 与 C3.60 的正式 result SHA；C3.59 单独不授权执行。
在 C3.61 global marker 持久化之前，independent payload 必须继续保持 sealed。

---

综上，C3.59 完成的是一次严谨的 **pre-data scientific commitment**：把 independent-test 的样本
身份、模型输入、目标、指标、门、失败语义、一次性访问纪律与工程实现边界在数据打开之前固定下来。
它是后续独立结论可信的必要条件，但本身不是独立性能结论。
