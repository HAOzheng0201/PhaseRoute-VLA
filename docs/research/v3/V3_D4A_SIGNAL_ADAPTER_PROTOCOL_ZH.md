# PhaseRoute-VLA v3 D4A：Motion/Tail 信号适配协议

D4A 在查看 v3 shadow 选择分布前固定旧 82D motion/tail 模型到新 97D Gripper-v2 feature 的连接方式。它只允许工程适配和逐元素验签，不允许训练、阈值搜索、active control 或 independent test。

## 1. 97D 到 82D 的唯一映射

```text
v3 feature [N, 97]
  ├─ [0:82]   legacy_causal_context -> frozen motion/tail heads
  ├─ [82:90]  current gripper sign sequence
  └─ [90:97]  current gripper transition pattern
```

适配器只能取 `[0:82]`，不得复制、重排或把 task/episode identity、L27 action 放入运行时特征。

## 2. 预先冻结的阈值

Motion 不在 D3 上搜索阈值，而是使用 C3.55 full-development layer anchor：

| Layer | translation RMS | rotation RMS |
|---:|---:|---:|
| 11 | 0.0321328071 | 0.0386499975 |
| 13 | 0.0146650289 | 0.0170458904 |

两个预测分量必须同时不超过各自 anchor。

Tail budget 固定为旧 q90 anchor 加旧 split-conformal correction：

| Layer | q90 anchor | correction | budget |
|---:|---:|---:|---:|
| 11 | 0.15234375 | 0.0096940249 | 0.1620377749 |
| 13 | 0.07666015625 | 0.0036352351 | 0.0802953914 |

这些值是预分布风险预算，不是机器人绝对安全阈值。是否可继续必须由 D4B 对 selected candidate 的完整 7D cosine consistency 和 gripper mismatch 做 false-safe cluster 审计决定。

## 3. 证据边界

旧 C3.55 checkpoint 是 development-only，C3.58 tail artifact 是 calibration-only；二者原本都没有 runtime threshold。D4A 只复用冻结参数、预处理和 correction，并单独预注册 runtime budget，不能把旧 artifact 的存在误写成已部署。

适配器通过后只授权在 episode 30--39 的冻结 artifacts 上运行 D4B formal shadow。episode 40--49、fresh rollout 和 active control 仍禁止。

## 4. 正式适配结果

适配器已在 clean commit `f6674996173b17aae4b6ad6468dd52518a3cedaa` 上执行并通过：

- 7,032 个 layer candidate，来自 3,516 个 policy call；
- D3 context、四个 candidate shard 和 dataset 的行身份精确一致；
- 从原始 context/candidate 重新构造的 97D feature 与冻结 dataset 逐元素相同；
- checkpoint 和 tail artifact 均先按 SHA 认证再 `weights_only` 加载；
- 没有 fit、阈值搜索、GPU、shadow selection、independent test 或 active control。

正式 signal payload SHA-256 为 `9fb66f57b004acfeb918f845adc39d33a73630abdcb3b96188f7ca65a7e6981c`。本结果只授权 D4B calibration shadow；它本身不披露或证明任何 L11/L13 选择比例。
