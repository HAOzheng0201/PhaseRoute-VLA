# Route-First Stage 4：完整训练集采集与审计

日期：2026-08-25

## 结论

训练范围 `LIBERO-10 × states 0–7` 已完整采集并通过 exact-grid 门禁。80 个 episode 共得到
2763 条 `[199]` action-free context，teacher 标签 L11/L13/L27 为 `94/323/2346`。全部
policy call 都已 prepared/committed，runtime error 为 0，所有 feature 有限且哈希可复算。

teacher 控制成功 72/80。8 个失败 episode 均执行到 65 calls，其中 483/520 calls 使用 L27
（92.88%），所以这些失败不能被现有证据解释为“过早退出导致”。数据采集有效性与 teacher
控制成功率必须分开报告。

## 数据规模和标签

| 指标 | 结果 |
|---|---:|
| task×state 网格 | 10×8 = 80 |
| 成功 episode | 72/80（90.0%） |
| policy calls / rows | 2763 |
| L11 | 94（3.40%） |
| L13 | 323（11.69%） |
| L27 | 2346（84.91%） |
| 任意提前退出 | 417（15.09%） |
| 平均 calls/episode | 34.5375 |
| calls 范围 | 18–65 |

若只按选中层数计算，teacher 的平均深度为 L27 的 91.92%，层数缩减约 8.08%。这是旧 V3
teacher 的标签统计，不是新 single-FM runtime 的速度或 FLOPs 结果。

## 控制失败

| task | state | calls | L11 | L13 | L27 |
|---:|---:|---:|---:|---:|---:|
| 2 | 7 | 65 | 1 | 8 | 56 |
| 4 | 2 | 65 | 0 | 2 | 63 |
| 5 | 0 | 65 | 1 | 2 | 62 |
| 8 | 1 | 65 | 0 | 4 | 61 |
| 8 | 3 | 65 | 0 | 3 | 62 |
| 8 | 4 | 65 | 0 | 3 | 62 |
| 9 | 3 | 65 | 2 | 3 | 60 |
| 9 | 4 | 65 | 2 | 6 | 57 |

task 8 是最明显的困难任务，仅 5/8 成功；task 9 为 6/8。task 0、1、3、6、7 均为
8/8 成功。后续若要研究失败归因，应做同 state、同 seed 的 full-depth/original A1 配对，
不能根据本批 teacher rollout 单独下因果结论。

## 分任务标签风险

标签高度不均衡，L27 占 84.91%。task 0 和 task 8 没有任何 L11 标签，task 4、5、6 的
L11 也分别只有 3、3、4 条。因此：

- 普通 accuracy 不能作为 router 主要指标，全部预测 L27 已有很高表面准确率；
- 两个 ordinal 安全头应分别判断 L11 和 L13，false-shallow 是首要错误；
- CV 必须按 task/episode 分组，不能把相邻调用随机拆到训练和验证；
- task/episode ID 只能用于分组审计，不能进入模型输入；
- 所有 task 都有 L13 标签，但 task 0/8 的 L11 正类缺失必须在 fold 报告中显式标记。

## Feature 审计

199D feature 的全局范围为 `[-3.0902, 3.5584]`，全部有限。191 个维度在本训练集上变化，
8 个维度为常数：3 个来自未启用 crop 的统计，5 个来自固定 crop mask。训练实现必须在每个
训练 fold 内拟合标准化器，并对常数维度使用安全 scale，不能先在全数据上标准化造成泄漏，
也不能产生除零。

## 聚合 artifact

```text
runs/route_first_teacher_train_states1_7/aggregate_train_states0_7/
  route_first_teacher_train_states0_7.npz

shape: [2763,199]
payload SHA-256: ffbda4bcf8d47447e09a163d58a73c234e6731b8f370abca4e9dba5e2403fe92
file SHA-256: e9f9a27090318d86ff2cd1ceb1949037fbf7c5d489a870898761a0e4b2f33683
```

聚合器已验证 80 个网格单元完整、call ordinal 连续、call identity 唯一、六个源 shard
payload 哈希一致，并以 exclusive-create 方式发布。raw rollout 和 NPZ 留在被忽略的 `runs/`；
可提交的不可变摘要位于 `results/route_first/`。

## 下一步和时间预估

下一阶段实现并训练 context-only ordinal router，进行 grouped task/episode CV 和
false-shallow 离线门禁。随后采集 states 8–9 仅用于阈值校准，再实现 single-FM active
runtime。到“完整跑通并形成第一版可信结论”预计仍需约 4 个阶段、3–6 天；若要补齐配对
基线、消融、重复种子和论文级统计，预计需要 1–2 周。states 10–11 在模型与阈值冻结前
仍不得打开，D9 states 40–49 始终禁止用于新方法选择。
