# PhaseRoute-V3 D1：Gripper-v2 协议冻结报告

完成时间：2026-08-20

正式状态：`PASS_D1_GRIPPER_V2_PROTOCOL_FROZEN`

阶段性质：design-only、development-only、metadata-only、CPU-only

## 1. 阶段结论

D1 已完成 Gripper-v2 的预注册，但尚未训练任何参数，也没有产生方法效果。它冻结了：

1. 离散 step/transition count 目标和 timing 诊断；
2. 单候选、past-only 的 97D runtime 特征；
3. occurrence Bernoulli、zero-truncated binomial 基线和 ordinal cumulative-link 主方法；
4. fresh development-v2 上 18×17 episode-index LOEO；
5. development 指标、`FULL_PASS` / `FOCUSED_PASS_NON_DEPLOYABLE` / `FAIL` 门槛；
6. calibration 的 false-safe UCB 约束和独立 Tail-UCB veto；
7. development、calibration、independent-test 的访问顺序。

本阶段只授权进入 `V3-D2_FRESH_DEVELOPMENT_COLLECTION_AND_NESTED_OOF_ONLY`。它不授权打开 calibration/test，不授权 shadow rollout 或 active control，也不支持优于 A1、CogVLA 或 PhaseRoute-v2 的声明。

## 2. 为什么改 Gripper，而不是继续调旧模型

旧 C3.55 的 gripper family 失败不是因为 occurrence 不可预测，也不是因为正样本太少：

| 旧指标 | L11 | L13 | 结论 |
|---|---:|---:|---|
| step occurrence Brier skill | 0.619084 | 0.490728 | 明显有信号 |
| step occurrence AUROC | 0.909262 | 0.858106 | 明显有信号 |
| expected-risk SSE ratio | 0.595267 | 0.694465 | 均优于基线 |
| positive-magnitude step MAE ratio | 0.985955 | **1.0073668609606237** | L13 失败 |
| L13 positive support | — | **487** | 不是 support 缺失 |

旧方法把 (k/8) 或 (k/7) 当作连续 positive magnitude 回归，但执行语义实际是整数个 step/transition mismatch。旧 82D 特征还只保留当前候选动作的 first/mean/std 和整体 smoothness，丢掉了完整 8 步夹爪符号序列与 7 步 transition pattern。

D1 因此不去放宽旧门槛，而是预先修正监督变量和输入表示。旧负结果继续以 SHA-256 绑定并保留。

## 3. 从输入到输出的新流程

```mermaid
flowchart LR
    C[9 个 past-only context tensors] --> F[旧 82D causal summary]
    A[单一当前候选动作<br/>B x 8 x 7] --> S[8D gripper sign]
    A --> T[7D transition pattern]
    F --> X[97D feature]
    S --> X
    T --> X
    X --> O[Occurrence Bernoulli head]
    X --> B[ZT-Binomial count baseline]
    X --> R[Ordinal cumulative-link count head]
    O --> E[P(any) × E(count|positive) / N]
    R --> E
    E --> G[Gripper risk gate]
    U[Independent Tail UCB] --> V{AND veto}
    G --> V
    M[Motion gate] --> V
    V -->|任一失败/缺失/非有限| D[deeper compute]
    V -->|全部通过| Q[仅未来校准后才可考虑 early exit]
```

L27 teacher 只在离线阶段生成 label。它、另一候选层、full-depth delta、behavior action、task/episode/call identity、success、reward、done 和 future observation 均不能进入 runtime 输入。

## 4. 离散 target 定义和维度

动作 horizon 为 8，action dimension 为 7，gripper index 为 6，候选层为 L11/L13，同噪声 consistency teacher 为 L27。

二值 proxy 固定为：

```text
state(x) = 1 if x >= 0 else 0
```

因此精确零值归入 state 1；非有限值导致整个 partition fail closed。

令候选和 teacher 的二值状态分别为 (s_l[t]) 与 (s_{27}[t])：

```text
step_mismatch[t] = s_l[t] XOR s_27[t]                      # 8 bits
transition_l[t]  = s_l[t] XOR s_l[t-1]                    # 7 bits
transition_mismatch[t] = transition_l[t] XOR transition_27[t]
step_count       = sum(step_mismatch)                      # 0..8
transition_count = sum(transition_mismatch)                # 0..7
occurrence       = count > 0
```

| 输出 | shape | 支持集/作用 |
|---|---:|---|
| step state | `[B,2,8]` | 候选二值状态 |
| step mismatch bits | `[B,2,8]` | 保留逐步误差位置 |
| transition pattern | `[B,2,7]` | 保留候选状态变化位置 |
| transition mismatch bits | `[B,2,7]` | 保留逐 transition 误差 |
| occurrence | `[B,2,2]` | target axis 为 step/transition |
| count | `[B,2,2]` | step 0..8；transition 0..7 |
| first transition mismatch | `[B,2]` | 0 表示无误差，1..7 为首个位置 |

用于 routing 的派生量固定为：

```text
expected_fraction = P(count > 0) * E[count | count > 0] / support_max
```

不再定义 conditional positive continuous magnitude target。

## 5. Runtime 输入与 97D 特征

公开 API 每次只允许一个当前候选：

| 输入 | shape |
|---|---:|
| instruction summary | `[B,3584]` |
| vision crop summary / mask | `[B,5,3584]` / `[B,5]` |
| phase embedding / scalars | `[B,128]` / `[B,3]` |
| normalized proprio | `[B,8]` |
| proprio history | `[B,8,8]` |
| action history | `[B,8,8,7]` |
| history mask | `[B,8]` |
| current candidate action | `[B,8,7]` |
| candidate layer | scalar int，每次隔离调用只允许 11 或 13 |

特征布局固定为：

```text
[0:82]   legacy causal context summary
[82:90]  current candidate 8-step gripper sign sequence
[90:97]  current candidate 7-step transition pattern
```

新增的 15 维只来自当前候选自身，所以修复时序信息丢失的同时，不引入 teacher 或另一候选泄漏。

## 6. 预注册模型比较

三个统计问题保持独立：

| 组件 | 冻结模型 | 目标/loss |
|---|---|---|
| occurrence | anchored Bernoulli logistic GLM | unweighted BCE |
| count baseline | zero-truncated binomial GLM | positive-only conditional NLL |
| primary challenger | ordinal cumulative-link GLM | positive-only conditional NLL |

主方法在查看 fresh label 前已经固定为 ordinal cumulative-link，不能在 D2 结果出来后改成 beta-binomial、MLP 或其它更有利模型。所有模型无 hidden layer、linear feature head 无自由 bias、residual weight exact-zero 初始化；ordinal 模型允许且必须显式计入跨两层两目标共 26 个有序 cutpoints。layer anchor、feature normalization 和 correction 只能使用当前 fit partition。occurrence 加 primary count 的 trainable parameter cap 为 512。

训练契约预冻结为 CPU FP64、full-batch LBFGS strong-Wolfe、最多 500 iterations，L2 grid 为 `[0.001,0.01,0.1]`，使用 inner cell one-SE 选择更大的 lambda。D1 没有执行这些训练。

## 7. 18×17 grouped nested OOF

development-v2 episode index 固定为 12--29，共 18 个 index、10 个 task、180 个 canonical episode keys。

- outer：18 folds，每次完整留出一个 episode index 在全部 10 个 task 上的记录；
- inner：每个 outer-train 的剩余 17 个 episode index 各留出一次；
- 同一 task×episode 的全部 call 和 L11/L13 pair 必须一起移动；
- 禁止 random-row、candidate-layer 或 target-aware split；
- inner 指标以 17×10=170 个 episode-index×task cell 等权；
- outer validation 不能参与 lambda 或模型选择；
- 10-fold task jackknife 是必须报告的 secondary robustness，不作为事后替换 primary 的门。

完整 10 task × 18 episode × 2 layer assignment 的 canonical SHA-256 为：

```text
3bde764d07ecf14c1cd3494f21205a70cb714eefefed25c58d4e8bc5d7a91388
```

## 8. Development 科学门槛

### 8.1 支持度和 occurrence

每个 layer×target 至少需要 100 个 zero 和 100 个 positive；不足时为 `INCONCLUSIVE_NO_FIT_NO_ROW_DROP`。

step/transition 的 overall、L11、L13 均要求：

```text
Brier skill > 0
AUROC > 0.5
```

### 8.2 Conditional count

primary 为 ordinal conditional NLL，comparator 为 ZT-binomial conditional NLL：

- step 和 transition 的 overall ratio 都必须严格小于 1；
- 4 个 layer×target scope 至少 3 个严格小于 1；
- 最差 scope 不得大于 1.01；
- CRPS、count MAE 必须报告，但不能替换 primary NLL。

### 8.3 Expected fraction 与 group robustness

step/transition 的 overall、L11、L13 raw-SSE ratio 必须全部严格小于 1。group robustness 的每个 outer episode 值固定为：先在 10 task × 2 layer × 2 target 的 positive-only cell 上等权计算 ordinal conditional NLL，再与相同 cell 上的 ZT-binomial NLL 比较；任一 cell 缺 positive 时该 outer fold 为 `INCONCLUSIVE`。18 个 outer episode-index 中至少 13 个满足 ordinal NLL 严格更小；tie 计为不改善。13/18 对应单侧 exact sign test (p<0.05)，12/18 不够。

### 8.4 两个正结果等级

`FULL_PASS` 要求以上门槛全部通过，且 4/4 layer×target conditional NLL ratio 均小于 1。

`FOCUSED_PASS_NON_DEPLOYABLE` 允许预先定义的一项 near-neutral layer：至少 3/4 scope 改善、最差不超过 1.01，同时 occurrence、expected fraction、group robustness 和两个 overall count target 必须全部通过。这落实了“可以由某个预注册指标突出体现初步成功”，但不允许实验后随意挑一个好看的指标。

两者在 D2 都只是 non-deployable development candidate；其它情况统一冻结为 `NEGATIVE_RESULT_FROZEN_NO_CALIBRATION`。

## 9. Calibration 与 Tail-UCB 的未来边界

D3 只能使用 calibration-v2 冻结阈值。每个 canonical task-episode 是一个 cluster；只有至少一个 predicted-safe call 的 cluster 才进入 false-safe 分母。若该 cluster 内任一 predicted-safe call 发生 any mismatch，则 cluster event 为 1。对这些 cluster event 使用 one-sided exact Clopper-Pearson 95% UCB：

```text
cluster-level one-sided 95% UCB(false-safe) <= 5%
cluster safe coverage = clusters_with_any_safe_call / all_100_clusters >= 10%
```

threshold candidates 固定为冻结 gripper score 的有序唯一有限值；在满足 UCB 的候选中选择 cluster safe coverage 最大者，平票取更保守的小 threshold。禁止在 development 或 independent-test 上选阈值；always-defer 不算有效方法。

最终 route-safe 采用不可补偿的 AND：

```text
route_safe = motion_safe AND tail_ucb_safe AND gripper_safe
```

Tail UCB 缺失、非有限或超阈值时强制 deeper compute；gripper 的好分数不能抵消 tail 失败。Tail 仍只是相对 teacher 的一致性上界，不是碰撞、安全或成功率 certificate。

## 10. 正式访问账本

| 操作 | 次数 |
|---|---:|
| protocol JSON parse | 1 |
| D0 result JSON parse | 1 |
| legacy manifest hash pass | 1 |
| role selection JSON parse | 3 |
| legacy C3.55 result JSON parse | 1 |
| fresh development payload open | **0** |
| calibration payload open | **0** |
| independent-test payload open | **0** |
| C3.61 row payload open | **0** |
| model fit / GPU operation | **0 / 0** |

每个 bound JSON 还单独执行 raw SHA pass；表中的 parse 次数不把 raw hash pass 重复计为 JSON parse。

进程内 `torch/numpy/tensorflow/jax=[]`，`CUDA_VISIBLE_DEVICES=''`。

## 11. 完整性哈希

| 对象 | SHA-256 |
|---|---|
| 正式 validation JSON | `5f47a69bf6e4950aa31ba9889a5632850da2edc691efcb45a460188aef5c2cc6` |
| protocol JSON raw | `3a5f5ebe49ddee093dc352ab4d46f7bbfea66486bc94d12d925d4eb40d2eaad2` |
| protocol canonical JSON | `4bc9f362109da704b828b42d49f208e62cdee92a9fe8f13e0e4480500c3678a5` |
| synthetic target truth table | `2cf830527ed04b0a08e2c227c06419097b89e002de7be28a7c5f5eb0a68fa2e4` |
| grouped fold assignment | `3bde764d07ecf14c1cd3494f21205a70cb714eefefed25c58d4e8bc5d7a91388` |
| `gripper_v2_protocol.py` | `a9d56ccb759c082a27673a4f1297ce6bd0e705d43fa0a3e40b602569af234ec3` |
| validator CLI | `4835e1cfd369e99801607ba46de38ef1bb6cac7d4a3286bae86e1b2c3dcb6fd7` |
| test source | `f27eca8f44039fc8087564ee3aa2ac24311ccb41035d2c320014028b23d9a57a` |
| D0 result | `64d1159b3941fe1e7b806da981a0f47297758dcc2cad87d4e283d03db3a71c4b` |
| legacy C3.55 result | `ba9da228f2607a22b9839e630c332e88c032ae91fdaa0f62efbb7cbcca55e678` |

## 12. 测试和复现命令

当前定向结果：`19 passed`。

```bash
cd /data3/haozheng/A1/worktrees/phaseroute-v3

CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
/home/haozheng/.conda/envs/a1/bin/python -m pytest -p no:cacheprovider -q \
  tests/dynamic_compute/v3/test_gripper_v2_protocol.py
```

测试覆盖 protocol 篡改、duplicate/NaN JSON、symlink/traversal、threshold tie、identical/single-step/early-late/all-flip target、非法 layer/shape、runtime leakage、18×17 fold、三角色 grid、13/18 sign gate、Tail 不可补偿、relocated repo、CUDA 隔离和不可覆盖输出。

正式 validation 命令：

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
/home/haozheng/.conda/envs/a1/bin/python \
  scripts/dynamic_compute/v3/validate_v3_gripper_v2_protocol.py \
  --repo-root /data3/haozheng/A1/worktrees/phaseroute-v3 \
  --legacy-source-root /data3/haozheng/A1/source \
  --protocol configs/research/v3/gripper_v2/protocol.json \
  --output results/v3/v3_d1_gripper_v2_protocol_validation.json
```

正式输出使用 exclusive-create，不能静默覆盖。

## 13. 冻结前精度修正记录

第一次 pre-freeze validation 后，逐字段复核发现报告中的旧失败 ratio 只写到 8 位小数。正式冻结前将其从 `1.00736686` 修正为 legacy JSON 的精确值 `1.0073668609606237`，并在 validator 中新增 metric value、support 487 和 MAE loss 的显式检查。

随后 science red-team 又在正式提交前收紧了三处歧义：candidate layer 改为 isolated-call scalar；明确 ordinal 的 26 个 trainable cutpoints 不属于 linear feature bias；把 13/18 composite 和 calibration cluster-UCB 写成唯一公式。以上修正均发生在提交前，未读取新 payload、训练参数或改变预注册主方法。未冻结的 pre-freeze validation 只作为临时检查，最终不纳入项目产物。

## 14. 下一阶段

D2 才能首次访问 development-v2，并且只能完成：

1. 180 个 fresh episode key 的 context/candidate/target 构造；
2. target alignment、support 和 payload SHA 审计；
3. 18×17 nested OOF 的 occurrence、ZT-binomial、ordinal count 训练；
4. 按本报告预注册门槛输出 development-only disposition；
5. 保留全部负结果。

D2 仍不得打开 calibration-v2 或 independent-test-v2，也不得选 runtime threshold、跑 shadow 或控制机器人。若候选 collection 需要 GPU，只能使用 GPU 0--3；GPU 4--7 保持留给其他人。
