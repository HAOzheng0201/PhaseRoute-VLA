# PhaseRoute-v2 C3.60：独立评估 runner 红队复核

日期：2026-08-19（Asia/Shanghai）

复核结论：`REMEDIATION_IMPLEMENTED_SYNTHETIC_GATES_PASS_PENDING_C360_FREEZE`

阶段性质：source-only、synthetic-only、CPU-only red-team review。

> **证据边界：** 本次复核没有读取正式 independent selection、source manifest、
> `array_path`、sample path、NPZ/sample payload、checkpoint 或正式 C3.61 输出；没有查询 GPU、
> 初始化 CUDA、运行 freezer 或计算独立测试指标。因此本文不是模型效果证据，也不授权启动
> one-shot C3.61。

## 0. 2026-08-19 修复收口更新

第 3--7 节保留修复前红队证据。对应阻断项现已完成 source-only 修复：

1. 固定 live supervisor 在 marker 不存在时预启动 context、4 个 candidate、aggregate、
   evaluate 共 7 个 worker；唯一匿名 pipe capability 绑定 token SHA、role、supervisor/worker
   PID 与 start ticks，named FIFO 被拒绝。
2. worker 使用 `PR_SET_PDEATHSIG(SIGKILL)` 并复查 parent race。supervisor 死亡后现存
   worker 被内核终止；marker 后启动的新进程即使自行构造 pipe/token 也会被拒绝。
3. 协议升级为 `READY_BEFORE_GLOBAL_MARKER -> START`。7/7 READY 前无 START；upstream
   result SHA 只通过 live supervisor 的 role-bound START handoff 传递。
4. 7/7 READY 后仍要求 stdin 精确等于 `COMMIT_C361_ONE_SHOT\n`；EOF、缺少换行、前后
   空白及其他文本全部在 marker 前中止。
5. front4 改为可直接执行的 POSIX `/bin/sh` clean exec 与 `/usr/bin/env -i`。冻结真实常规
   解释器 `/home/haozheng/.conda/envs/a1/bin/python3.10` 和 `-B -s`；固定运行环境并拒绝
   `BASH_ENV/ENV/LD_PRELOAD/PYTHONPATH/LD_LIBRARY_PATH`。
6. marker 前 readiness 覆盖 phase checkpoint/model/CPU smoke、candidate GPU UUID/
   deterministic CUDA kernel/A1 model load、aggregate C3.54 builder shape smoke，以及 evaluator
   C3.55/C3.58 authenticated load/schema extraction/JSON-safe smoke。正式 independent 数据仍只在
   START 与 durable marker 之后访问。
7. marker 前创建同一 `0700` 私有日志目录；7 个 worker 各有独立日志。完成或失败均输出日志
   path、SHA-256、byte count 和 bounded 4 KiB UTF-8 tail。

最终 canonical contract SHA-256：

```text
6b356d4e5571e5c5b3566709d5188ae2f71c54717441fff8a8064d17e40f0200
```

最终非 freezer、CPU-only、synthetic-only 回归：

```text
203 passed, 3 warnings in 12.10s
```

新增协议定向集为 `31 passed`，覆盖第 7 个 READY 缺失不 START、伪造 READY 拒绝、named
FIFO 拒绝、marker 后新进程拒绝、SHA 仅经 START 传递、PDEATHSIG、clean-exec 和 commit
精确匹配。三条 warning 仅为 Python 3.10 Google API 生命周期提示及 Pydantic 元数据提示。

本更新只证明 R1--R4 实现及合成门禁通过。按约束没有运行 freezer、没有创建正式 marker、
没有读取正式 artifact 或查询 GPU；C3.61 仍未启动，须等待新的 C3.60 freeze/审计授权。

## 1. 本轮完成内容

冷导入检查发现 `transformers` 可能自动探测 TensorFlow 并触发 CUDA plugin 注册。为使正式
runtime 明确保持 PyTorch-only，C3.60 frozen environment 增加：

```text
USE_TORCH=1
USE_TF=0
USE_FLAX=0
```

三项已同步到合同、validator 和 synthetic tests。新合同 SHA-256 为：

```text
cfdc71884d14fd2d6f2c4914b93b7eef2b0a8a42d03441bad853e445527043a0
```

定向静态/合成回归：

```text
207 passed, 3 warnings in 11.90s
```

三条 warning 来自 Python 3.10 的 Google API 生命周期提示和 Pydantic 字段元数据提示，未造成
测试失败。合同计算 SHA 与固定 SHA 的硬断言通过。

## 2. 红队结论总览

| 严重度 | 发现 | 后果 | C3.61 前状态 |
|---|---|---|---|
| Critical | 跨进程 no-resume 不能被执行层证明 | marker 存在时，新进程仍可直接附着并继续未完成链 | 必须修复 |
| High | 调用者环境不是 closed world | `PYTHONPATH`、`LD_PRELOAD`、`PATH`、`TMPDIR` 等可改变导入、动态链接或 post-marker 行为 | 必须修复 |
| High | 多项可预防 readiness 错误在 marker 后才暴露 | 唯一科学 attempt 可能因 checkpoint/model/CUDA/import 问题被无意义消耗 | 必须修复 |
| Medium | 后续 adapter 接收调用者提供的 upstream result SHA | 直接调用 adapter 时，上游链来源依赖调用者，而不是仅依赖活着的唯一 supervisor | 与 no-resume 一并修复 |

当前判断不是“模型方法失败”，而是 **正式执行闭包尚不足以承担不可重试的独立测试**。

## 3. Critical：新进程可重复附着 marker

合同明确冻结：

```text
post_marker_rerun_or_resume_to_change_outcome = False
post_marker_resume_authorized = False
```

但 `OneShotAttemptGuard.attach_existing_durable_marker()` 的 single-use 状态只保存在当前 Python
对象内。每个新进程都能创建新的 `OneShotAttemptGuard`，重新读取同一个 marker 并获得新的
receipt：

- `stage_c360_one_shot_guard.py:589--648`
- `c361_bound_candidate_adapter.py:190--198`
- `aggregate_stage_c360_independent_candidates.py:85--92`
- `c361_bound_evaluate_adapter.py:166--180`

front4 入口确实在 `run_stage_c360_independent_front4.sh:77--87` 拒绝已有 marker；但是这只保护
重新执行 front4，无法阻止人工直接执行 candidate/aggregate/evaluate adapter。

纯 `/tmp` 合成复现使用一个 marker 和两个全新 guard，结果为：

```text
fresh_attach_1 = True
fresh_attach_2 = True
```

因此“正常 front4 后续子进程”与“front4 崩溃后的人工恢复进程”在现协议中不可区分。没有
`--resume` 参数并不等于执行层不存在 resume 能力。

## 4. High：运行环境不是 closed world

validator 只比较合同列出的变量：

- `stage_c360_independent_runner_validation.py:598--623`

front4 通过 `env "${common_env[@]}" ...` 覆盖部分变量，但不会清空其余调用者环境：

- `run_stage_c360_independent_front4.sh:89--109`
- `run_stage_c360_independent_front4.sh:118--121`
- `run_stage_c360_independent_front4.sh:168--178`
- `run_stage_c360_independent_front4.sh:217--237`

仍可继承的关键变量包括 `PYTHONPATH`、`LD_PRELOAD`、`PATH`、`TMPDIR` 及其他
Python/Torch/CUDA/动态链接器变量。具体风险包括：

1. `PYTHONNOUSERSITE=1` 不会忽略 `PYTHONPATH`，冻结源码可能导入调用者路径中的同名包；
2. `LD_PRELOAD` 可改变 Python、CUDA 和 shell 工具行为；
3. `nvidia-smi` 在 `stage_c360_independent_execution_models.py:204--231` 通过 `PATH` 查找；
4. shell 的 `env`、`sha256sum`、`awk`、`mktemp`、`tail`、`chmod` 也依赖 `PATH`；
5. `run_stage_c360_independent_front4.sh:139` 在 marker 和 context 已发布后才使用调用者
   `TMPDIR` 创建日志目录，错误 `TMPDIR` 会制造可预防的 post-marker failure。

## 5. High：readiness barrier 位于 marker 之后

context 在 `c361_bound_context_adapter.py:173--176` 创建并持久化 marker，之后才读取、反序列化
并构造 phase estimator（`178--189`）。以下错误会直接消耗唯一 attempt：checkpoint 损坏、
反序列化失败、state-dict/schema/geometry 不匹配、模型构造失败。

candidate 在附着 marker（`c361_bound_candidate_adapter.py:190--198`）之后才检查 PyTorch CUDA、
GPU UUID、确定性 runtime、A1 config/checkpoint 反序列化和模型上卡（`199--244`）。context 的
pre-marker `nvidia-smi` 仅证明四张物理卡可查询及 UUID/显存字段可解析，不证明 PyTorch CUDA
kernel 或 A1 模型能加载。

aggregate 的 C3.54 feature builder 使用 lazy import（
`c361_bound_aggregate_adapter.py:46--51`）；依赖或导入失败也可能在 marker 已消费后才发现。

协议允许真正不可预防的 post-marker 错误被记为 immutable inconclusive，但不应把可在不接触
independent payload 时完成的模型、依赖、GPU 和输出路径 readiness 检查推迟到消费点之后。

## 6. 已通过的关键控制

本轮没有发现下列控制的直接旁路：

1. candidate 参数与 CUDA runtime 均把 shard `0--3` 精确绑定物理 GPU `0--3`，并再次核对
   process-visible GPU 数量和 UUID；GPU `4--7` 没有参数映射入口；
2. aggregate 使用 `build_c354_candidate_context_features(runtime_inputs, actions)`，输入要求
   `[B,2,8,7]`，通过两次隔离的单候选调用得到 `[B,2,82]`；没有退回旧 paired builder，
   也没有把 L27 teacher 或另一 shallow candidate 泄漏进特征；
3. publication 使用 exclusive create、file/parent fsync 与 `RENAME_NOREPLACE`，并把 payload、
   records、marker、role 和固定输出路径绑定进 result；
4. upstream loader 会重验 result/payload/records SHA、固定 role/path、marker SHA 与
   `resume_authorized=False`；
5. source closure 的静态 import 复核没有发现遗漏的 `scripts.dynamic_compute` 本地依赖，
   `a1/**/*.py` 由 recursive runtime-source policy 覆盖。

这些通过项不能抵消第 3--5 节的阻断问题。

## 7. 必须采用的修复顺序

### Gate R1：单一存活 supervisor capability

把“可读取 marker”与“获准继续计算”分开。建议由一个固定、SHA-frozen 的 Python supervisor：

1. 在 marker 前启动并保有 context、四个 GPU candidate、aggregate/evaluate worker；
2. 通过匿名继承 FD/pipe 传递一次性、非落盘 capability；
3. worker 只接受来自该存活 supervisor 的 capability，不再允许仅凭 marker SHA 独立启动；
4. supervisor 或 pipe 消失后，任何新进程均不能恢复链；
5. upstream result SHA 由 supervisor 内部 handoff，移除直接 CLI 调用者的结果选择权。

仅增加另一个落盘 token、PID 文件或 stage-start marker 不足以解决问题：它们仍可被新进程读取，
且无法区分正常后续 stage 与崩溃后恢复。

### Gate R2：pre-marker readiness barrier

所有 worker 在 marker 前完成并向 supervisor 回报：

- 清洁环境和依赖/source SHA 重验；
- phase estimator 与 A1 config/checkpoint 的 authenticated load、schema/state-dict/model 构造；
- GPU 0--3 的 PyTorch CUDA、UUID、显存及最小 deterministic kernel smoke；
- C3.54 builder 与 evaluator 依赖导入；
- 固定输出路径均不存在，固定 `/tmp` 日志目录已成功创建；
- 四个 worker 都处于 ready-but-no-independent-access 状态。

只有全部 readiness receipt 通过后，supervisor 才能原子持久化全局 marker 并广播 start。

### Gate R3：clean exec

front4/supervisor 应使用清空环境的 exec，并显式恢复唯一允许的固定变量；至少需要固定绝对
`PATH`，拒绝 `PYTHONPATH`/`LD_PRELOAD`，固定 `TMPDIR=/tmp`，使用绝对 `nvidia-smi` 和
shell 工具路径。validator 需要验证环境允许集或明确的危险变量拒绝集，而不只是检查若干键值。

### Gate R4：合成故障与恢复拒绝测试

新增测试必须证明：

1. supervisor 存活时四 worker 可以完成唯一链；
2. supervisor 在 context、任一 shard、aggregate 后死亡时，新进程均不能继续；
3. marker 存在但无 capability 时，四个 adapter 全部拒绝；
4. checkpoint/model/CUDA/import/TMPDIR 故障全部在 marker 前失败；
5. 污染 `PYTHONPATH`、`LD_PRELOAD`、`PATH`、`TMPDIR` 不能进入执行闭包；
6. GPU 4--7 仍无任何查询、可见性或映射旁路。

## 8. 下一阶段判定

C3.61 当前保持 **未启动**。在 R1--R4 完成、全回归通过、重新独立审查并重新冻结 C3.60
result SHA 之前，不应创建正式 global attempt marker。

由于 runtime contract/source 已改变，任何此前生成的 C3.60 source hash/result 都应视为旧版本，
不能与新源码混用。本轮没有运行 freezer；下一次 freeze 必须发生在上述修复全部完成之后。
