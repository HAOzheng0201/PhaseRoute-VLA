# Route-first Stage 6：阈值校准预注册

## 状态

```text
READY_TO_COLLECT_CALIBRATION_STATES_8_9
```

本协议和校准代码在查看 states 8–9 的 feature、teacher label 或路由分数之前冻结。
Stage 5 的 score model 权重、PCA rank 和 L2 从现在起均不可再改；Stage 6 只允许确定
L11/L13 阈值。

冻结输入：

| 输入 | SHA-256 |
|---|---|
| uncalibrated score model | `38aaef193442a4b40e71b1d48bee42ffbe5f191cad64f99d20bd3f75df3ad3ae` |
| Stage 5 result | `c73c475d47c7e82afb35c3bfa52ab58dd4eca8171fe77692a0c11f3b027d7a6d` |
| Stage 5 verification | `7446a7cf81ca3167c5ef3a5ce9d650e98db273b45a36982ffeaa0c049b5c3293` |
| calibration protocol | `c1ab2aef44595d3d86b04684155a74302d2bb70b91bf01c1722f86d0790ce1d1` |

## 数据职责

```text
state 8  -> threshold selection
state 9  -> one-shot confirmation
state 10–11 -> 在 calibration artifact 冻结前禁止打开
state 40–49 -> 历史 D9，永久禁止新方法使用
```

每个 split 覆盖 LIBERO-10 全部 10 个 task。采集继续使用冻结 PhaseRoute-V3 控制，
route-first collector 仅 observation-only 记录 `[199]` context 和 teacher layer，不改变
动作。task/state identity 仅用于 equal-cell weighting 和审计，不进入 score model。

## 统计规则

每个 head 的 false-safe 定义为：

```text
L11 false-safe: score11 >= t11 但 teacher_layer != 11
L13 false-safe: score13 >= t13 但 teacher_layer == 27
```

不同 episode 的调用长度不等，因此每个 `(task,state)` cell 具有相同总权重。除经验
false-safe 外，同时计算 90% one-sided weighted Wilson upper bound；加权样本量使用
Kish effective sample size。

state 8 只枚举真实出现过的唯一 score threshold，并从满足全部约束的候选中选择覆盖率
最大者；其次选择经验 false-safe 更低者，再其次选择更高阈值。若不存在可行阈值，
该 head 直接关闭。

### State 8 selection gate

| head | coverage | min effective rows | empirical false-safe | 90% upper bound |
|---|---:|---:|---:|---:|
| L11 | 0.5%–5% | 3 | ≤25% | ≤65% |
| L13 | 2.5%–15% | 10 | ≤10% | ≤25% |

### State 9 one-shot confirmation gate

| head | min coverage | min effective rows | empirical false-safe | 90% upper bound |
|---|---:|---:|---:|---:|
| L11 | 0.25% | 3 | ≤50% | ≤75% |
| L13 | 1.5% | 8 | ≤20% | ≤40% |

这些门禁是小样本工程门禁，不是形式化机器人安全保证。阈值较宽的 L11 上限反映其正例
极少，只能用于决定是否继续研究；后续 active paired test 仍必须验证成功率非退化。

## Fail-closed 规则

1. state 8 无可行阈值：关闭该 head；
2. state 9 只能检查 state 8 的**原阈值**，禁止重新拟合或移动阈值；
3. state 9 失败：只关闭相应 head，不能选择“看起来更好”的替代阈值；
4. L13 未确认：不授权进入 states 10–11 工程 holdout；
5. 两个 head 都关闭时固定走 L27；
6. 离线 confirmation 通过也不直接授权 active control。

## 已完成的代码门

`route_first_calibration.py` 已实现：

- weighted Wilson upper bound；
- 唯一 score-prefix 阈值搜索；
- state 9 exact-threshold confirmation；
- per-head fail-closed disable；
- L11→L13→L27 的确定性层选择。

12 项 calibration/router 定向测试全部通过；全仓 CPU 回归为 `508 passed,
22 subtests passed, 3 warnings`，0 失败，用时 73.45 秒。预注册后下一步才是采集
states 8–9，聚合成 exact 10 tasks × 2 states dataset，再运行一次校准与确认。
