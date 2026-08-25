# Route-First Stage 1：Teacher 采集 Smoke

日期：2026-08-25

## 结论

observation-only 采集路径通过 task 0/state 0 闭环 smoke。冻结 V3 成功完成任务，35 次
policy call 全部 prepared/committed，runtime error 为 0。采集器得到 `[35,199]` float32
context-only features，全部有限，teacher route 分布为 L11/L13/L27 = `0/4/31`。

## 最重要的无干预证据

本次运行与采集前的 Stage-1 V3 reference 使用同一 checkpoint、task、initial state 和 seed。
逐调用比较结果为：

- 35/35 selected layer 完全一致；
- 35/35 normalized action SHA-256 完全一致；
- action 全部有限且 shape 均为 `[8,7]`；
- 两边均为 1/1 episode 成功。

因此在该配对 smoke 范围内，199D feature 提取和 NPZ 收集没有改变 V3 action、route 或
随机序列。

## 运行与 artifact

```text
runs/route_first_teacher_smoke/libero_10_20260825_170835
```

teacher NPZ：

```text
route_first_teacher_context.npz
rows: 35
shape: [35,199]
payload SHA-256: f2881e133fc61b580e11b1a3240a964da4f6eef008d7a93c776737e39606e329
file SHA-256: fac473f7c2fc48d4c48bc01274adb42b29cd787997200059a422cfff62fd3fdb
```

## 时间数据的边界

reference 与 collection 的 mean policy wall 分别为 1518.36 ms 和 1533.79 ms，描述性差值
约 1.02%。这是跨时段单 episode 测量，不能归因为采集开销，也不能用于效率结论。采集器
仅用于生成训练数据，最终 route-first runtime 不会写训练 NPZ。

## 下一步

冻结数据划分后，在仍未被 D9 使用的新工程范围采集：states 0–7 用于训练，8–9 用于
阈值校准，10–11 暂不打开作为工程留出。采集阶段仍运行冻结 V3 action；在数据审计、
分组训练和 false-shallow gate 完成之前，不接入 route-first active control。
