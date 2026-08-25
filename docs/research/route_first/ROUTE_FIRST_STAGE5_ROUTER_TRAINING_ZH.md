# Route-first Stage 5：无泄漏路由评分器训练

## 1. 结论

Stage 5 已完成，状态为：

```text
PASS_RANKING_GATES_UNCALIBRATED
```

这表示仅使用普通训练 states 0–7，199D action-free context 对冻结 V3 teacher
给出的 L11/L13/L27 路由存在可重复的排序信号。16 个预注册候选中选中
`PCA rank=64, L2=0.3`；最终 PCA 和标准化已折叠为两个 `199→1` 仿射头，运行时
只有 400 个参数、398 次乘加，不需要保存或执行 PCA。

这**不表示新方法已经可以进入闭环**。当前模型只输出 `safe11`、`safe13` 两个分数，
artifact 明确标记 `calibration_status=UNSET`，不包含 L11/L13 部署阈值。states 8–9、
states 10–11 和历史 D9 states 40–49 均未打开。

## 2. 数据与边界

训练输入固定为：

| 项目 | 数值 |
|---|---:|
| task | LIBERO-10 task 0–9 |
| training state | 0–7 |
| episode | 80 |
| policy call | 2,763 |
| feature | `[2763, 199] float32` |
| teacher L11/L13/L27 | 94 / 323 / 2,346 |

输入 aggregate payload SHA-256：

```text
ffbda4bcf8d47447e09a163d58a73c234e6731b8f370abca4e9dba5e2403fe92
```

输入 aggregate file SHA-256：

```text
e9f9a27090318d86ff2cd1ceb1949037fbf7c5d489a870898761a0e4b2f33683
```

协议把数据用途分为四个互斥区域：

```text
states 0–7   -> 模型训练与候选选择（本阶段）
states 8–9   -> 之后只能校准阈值（本阶段未打开）
states 10–11 -> 阈值冻结后的工程 holdout（本阶段未打开）
states 40–49 -> 历史 D9，禁止新方法训练、选择和校准
```

task ID 和 episode ID 只用于分组、划分和审计，不进入 199D 模型输入。语言指令的
连续统计量属于合法的当前观测上下文，仍保留在输入中。

## 3. 从 teacher label 到两个安全分数

teacher 路由标签被转换成两个有序二分类目标：

```text
y11 = 1[teacher_layer == 11]
y13 = 1[teacher_layer <= 13]
```

因为安全执行 L11 必然也应安全执行 L13，所以在线分数显式满足：

```text
raw11   = sigmoid(w11^T x + b11)
score13 = sigmoid(w13^T x + b13)
score11 = min(raw11, score13)
```

即始终有 `score11 <= score13`。这只保证分数的序关系，不会偷偷产生部署阈值。

### 3.1 fold 内训练

每个训练 fold 独立完成以下操作，validation 行从未参与 scaler 或 PCA：

```text
199D context
  -> task/episode cell 等权
  -> fold-local weighted mean/std
  -> fold-local weighted PCA
  -> class-balanced logistic head
  -> 折叠回原始 199D 的 affine weight/bias
```

不同 episode 的 policy-call 数为 18–65。先令每个 `(task, state)` cell 总权重相同，
再在拟合每个二分类头时平衡正负类，可以避免长 episode 和大量 L27 行支配损失。

### 3.2 候选选择与鲁棒性审计

预注册 grid 为：

```text
PCA rank: 8, 16, 32, 64
L2:       0.3, 1.0, 3.0, 10.0
```

模型选择使用 leave-one-episode-index-out：每次把同一个 state index 在全部 10 个
task 上留出，8 folds 的 OOF 分数覆盖每一行且只生成一次。主指标为 L11/L13
group-equal AP 的调和平均，tie-breaker 依次为 macro AUC、较低 rank、较高 L2。

选中候选后才运行 leave-one-task-out（LOTO）。LOTO 只用于检验跨任务鲁棒性，未参与
候选选择。

## 4. 真实训练结果

选中候选：

```text
pca64_l2_0.3
```

候选排名前五如下：

| candidate | harmonic AP | L11 AP | L13 AP |
|---|---:|---:|---:|
| pca64_l2_0.3 | 0.6446 | 0.5044 | 0.8926 |
| pca64_l2_1 | 0.6408 | 0.5003 | 0.8910 |
| pca64_l2_3 | 0.6287 | 0.4862 | 0.8894 |
| pca64_l2_10 | 0.5947 | 0.4471 | 0.8876 |
| pca32_l2_0.3 | 0.5417 | 0.3934 | 0.8699 |

### 4.1 主 OOF 结果

| 指标 | safe L11 | safe L13 |
|---|---:|---:|
| group-equal prevalence | 0.0383 | 0.1635 |
| AP | 0.5044 | 0.8926 |
| AP lift | **13.18×** | **5.46×** |
| ROC AUC | 0.9632 | 0.9750 |
| 预注册低覆盖率 | 1% | 5% |
| 实际覆盖率 | 1.010% | 5.006% |
| false-safe rate | 37.44% | **1.49%** |
| precision | 62.56% | **98.51%** |

L13 的低覆盖率排序很强：在约 5% 覆盖率内，false-safe 为 1.49%。L11 正例非常
稀少，虽然 AP lift 高，但 1% 覆盖率仍有 37.44% false-safe。

### 4.2 leave-one-task-out 结果

| 指标 | safe L11 | safe L13 |
|---|---:|---:|
| AP | 0.2102 | 0.6357 |
| AP lift | **5.49×** | **3.89×** |
| ROC AUC | 0.9038 | 0.9026 |
| 预注册低覆盖率 | 1% | 5% |
| 实际覆盖率 | 1.013% | 5.001% |
| false-safe rate | 67.73% | 35.07% |
| precision | 32.27% | 64.93% |

跨任务性能明显低于同任务、跨 state 的 OOF 性能，说明当前路由分数含有较强的任务
相关结构。它仍通过了预注册的最低门禁，但不能据此声称已经获得可靠的跨任务浅退
策略。

### 4.3 预注册门禁

8 个门禁全部通过，且看过结果后没有修改标准：

| gate | 标准 | 结果 | 状态 |
|---|---:|---:|---|
| episode L11 AP lift | >1.25 | 13.18 | PASS |
| episode L13 AP lift | >1.25 | 5.46 | PASS |
| task L11 AP lift | >1.05 | 5.49 | PASS |
| task L13 AP lift | >1.05 | 3.89 | PASS |
| episode L11 false-safe@1% | ≤0.50 | 0.3744 | PASS |
| episode L13 false-safe@5% | ≤0.50 | 0.0149 | PASS |
| task L11 false-safe@1% | ≤0.75 | 0.6773 | PASS |
| task L13 false-safe@5% | ≤0.65 | 0.3507 | PASS |

## 5. 必须保留的负结果与风险

1. **L11 标签极少。** 只有 94/2763 行，group-equal prevalence 仅 3.83%；task 0
   和 task 8 完全没有 L11 正例。
2. **跨任务 L11 风险仍高。** LOTO false-safe@1% 为 67.73%。通过探索阶段门禁只说明
   分数不是随机排序，不代表该风险可部署。
3. **最大 rank 胜出。** rank 64 明显优于 rank 32，说明当前信号并非极低维；轻量化
   来自训练后折叠为 affine heads，而不是证明 rank 8 已经足够。
4. **原始空间系数尺度较大。** L11/L13 weight L2 norm 分别约 33,958/5,931；最大
   系数来自方差很小的 `instruction_stats` 维度（最小训练标准差约 `7.79e-5`）。这是
   标准化后线性头折叠回原空间的预期结果，训练集 roundtrip 误差仅 `2.48e-7`，但它
   暗示后续 calibration/holdout 必须检查数值与分布漂移，不能只看训练 OOF。
5. **没有闭环结论。** 本阶段没有运行新路由控制，没有成功率、延迟或 speedup
   结论，也没有证明优于 A1/冻结 PhaseRoute-V3。

## 6. Artifact 与复现

运行命令：

```bash
python scripts/train_route_first_router.py \
  --aggregate runs/route_first_teacher_train_states1_7/aggregate_train_states0_7/route_first_teacher_train_states0_7.npz \
  --protocol configs/route_first_router_protocol.json \
  --output-dir runs/route_first_router_stage5 \
  --published-result results/route_first/route_first_stage5_router_training.json
```

生成物：

| artifact | SHA-256 |
|---|---|
| `router_uncalibrated.npz` | `38aaef193442a4b40e71b1d48bee42ffbe5f191cad64f99d20bd3f75df3ad3ae` |
| `oof_scores.npz` | `3cfa0b6f18e48cfa3ae8a8e7b9f718419b6eefe6ea82365f23ed120205a7ae3b` |
| published result JSON | `c73c475d47c7e82afb35c3bfa52ab58dd4eca8171fe77692a0c11f3b027d7a6d` |

router artifact 重新加载后与训练内存模型的最大绝对分数误差为
`2.4783e-7`，metadata 完全一致。压缩后 router 文件约 5.6 KB；其字段白名单中没有
`threshold11` 或 `threshold13`。

针对 dataset、router、指标、coverage、无泄漏 fold、嵌套分数和 artifact roundtrip
的 9 项定向测试全部通过。随后全仓 CPU 回归为 `502 passed, 22 subtests passed,
3 warnings`，用时 72.39 秒；3 个 warning 均来自现有依赖的 Python 版本/Pydantic
兼容性提示，不是本阶段失败。

## 7. 下一阶段

下一阶段是 Stage 6 calibration，而不是直接跑 active control：

1. 先冻结 states 8–9 的采集与阈值校准协议；
2. 采集 10 tasks × 2 states 的 observation-only context 和冻结 teacher label；
3. 固定候选模型，不再修改 PCA rank、L2 或权重；
4. 只在 calibration 数据上选择 L11/L13 阈值，并约束 false-safe 风险；
5. 阈值通过门禁后，才允许打开 states 10–11 做一次工程 holdout；
6. 历史 D9 states 40–49 继续永久禁止用于新方法选择。
