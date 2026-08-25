# Route-first Stage 9：generated-state active paired pilot 预注册

## 1. 当前状态

本文件只冻结实验设计，状态为：

`PREREGISTERED_NOT_OPENED`

截至预注册时：

- official state 12 尚未用于本阶段 active smoke；
- official state 13 尚未用于本阶段 paired pilot；
- 本阶段尚未执行 route-first active control；
- 历史 D9 states 40--49 不会重新打开；
- L13 阈值固定为 `0.9174261218080999`，禁止移动。

机器可读协议为 `configs/route_first_active_pilot_protocol.json`，SHA-256：

`fcb1c2a1fdf7ea3f79343f72d25240449500a5eac3fad1372f0808023888db4d`

## 2. 为什么称为 generated-state paired

两臂从相同 official init state、相同 checkpoint、相同 episode seed 开始。第一次动作之后，
由于 controller 可能生成不同动作，两条轨迹的后续 observation 会自然分叉。因此只能配对初始条件，
不能再把后续 observation 当作共享离线输入。这正是本阶段必须做 active rollout 的原因。

## 3. 两个冻结方法

| 方法 | 路由时机 | FM 调用 |
|---|---|---|
| candidate-first V3 | 在 L11/L13 生成候选动作后 gate | 每 call 多次，取决于退出路径 |
| route-first Stage 8 | 199D context 先选 L13/L27 | 正常 call 固定 1 次 |

该 pilot 首先比较 route-first 与其直接 teacher V3，以隔离“路由提前到动作头之前”这一改动。
original A1 的最终三臂比较保留到 pilot 通过后的 fresh-state confirmation；项目已有冻结 D9
original-A1/V3 结果，但不会把历史 states 40--49 当作本阶段的新独立数据。

## 4. 执行顺序

### 4.1 不打开 episode 的 preflight

必须先完成：

1. Stage-8 全套测试和 D9 protected-code gate；
2. route-first runner/validator 的 CPU contract tests；
3. 动态选择当前无人使用的物理 GPU，并记录 UUID；
4. CUDA 和 artifact SHA preflight；
5. 输出目录 exclusive-create，禁止覆盖旧结果。

### 4.2 Engineering smoke

- task：0
- official init state：12
- 顺序：candidate-first V3，然后 route-first
- 任一 runtime integrity gate 失败，立即停止，不打开 state 13
- smoke 仅验证执行链，不用于性能结论

### 4.3 Paired pilot

- tasks：0--9
- official init state：13
- 每个 task 两臂使用同一 seed
- 偶数 task：V3 → route-first
- 奇数 task：route-first → V3
- 不用失败重跑替换结果

## 5. 冻结 gate

### 5.1 Runtime integrity

route-first 必须同时满足：

- runtime error = 0；
- nonfinite action = 0；
- prepared calls = committed calls = policy calls；
- L11 selected calls = 0；
- 100% valid calls 的 `fm_calls == 1`。

### 5.2 描述性成功率 guardrail

10 个 task 上 route-first 成功数不得比 V3 少超过 2。该阈值只是工程 pilot guardrail，样本量
不足以支持统计功效明确的非劣性结论。

### 5.3 真实延迟 gate

排除外部 GPU contention 后：

`route-first median policy wall ms / V3 median policy wall ms <= 0.90`

同时报告 mean、p50、p90、FM invocation count 和 decoder block count。未满足时不得用静态
FM 调用数宣称真实加速。

## 6. 失败处理

- 不移动阈值；
- 不替换失败 rollout；
- 保留所有失败日志；
- 先区分 runner/runtime bug、随机动作差异、任务分布风险和方法本身问题；
- 任何算法变更必须建立新协议并使用更新鲜的 init states。

## 7. 本 pilot 能与不能说明什么

通过后只允许进入 fresh-state confirmation，不能直接宣称：

- 最终闭环性能优于 original A1；
- 部署安全；
- 形式化提前退出可靠性；
- 跨 suite 泛化。

若通过，后续 confirmation 应加入 original A1 第三臂并扩大 fresh-state 数量。
