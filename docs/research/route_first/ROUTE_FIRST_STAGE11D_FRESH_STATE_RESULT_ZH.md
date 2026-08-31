# Route-first Stage 11D：fresh-state 生成与封存结果

## 1. 结论

Stage 11D 的 CPU-only fresh-state 生成门禁一次通过：

```text
PASS_ROUTE_FIRST_STAGE11D_STATES_FROZEN
```

这只证明 200 个新 MuJoCo 状态已经按照预注册 schedule 确定性生成并封存，不包含模型
推理、任务成功率、可靠性可预测性或速度提升结论。

## 2. 执行结果

| 项目 | 结果 |
|---|---:|
| source commit | `4e0b83bf38790abecb45c630b6b800db0960886a` |
| task × replicate | `10 × 20` |
| 独立生成进程 | `400` |
| 确定性生成遍数 | `2` |
| 两遍 byte-identical state | `200 / 200` |
| 每个 task 的唯一状态 | `20 / 20` |
| initially-solved state | `0` |
| checkpoint 加载 / policy action 采样 | `0 / 0` |
| official states 0--49 访问 | `0` |
| V3-D8 / Route-first Stage 10 state 复用 | `0` |
| GPU 查询或初始化 | `0` |
| generation wall time | `667.38 s` |

运行使用 `CUDA_VISIBLE_DEVICES=-1`、OSMesa 和最多 8 个 CPU worker。所有记录均在独立
进程中执行恰好一次 `env.reset()`；没有 outcome-based retry，也没有为状态质量换 seed。

## 3. 数据划分与状态维度

200 个 cluster 在生成前已经固定为 120 个 development-train、40 个 calibration 和 40 个
shadow-confirmation。它们不是 LIBERO official fixed-state episode identity。

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state dimension | 123 | 123 | 47 | 51 | 84 | 45 | 71 | 84 | 47 | 47 |

不同 task 的 MuJoCo flattened-state 维度不同是场景结构造成的；门禁要求 task 内维度一致，
不要求跨 task 相同。

## 4. 封存哈希

| 对象 | SHA-256 |
|---|---|
| generation result | `c9921ddf3f04ba2b47dd29d80456271767cba3bf691f3ccd5b5c78a8efe9d123` |
| local state attestation | `0ec5c7e9e5bd9a3b72d35488972e0a21f956d551cca618a287fed729f37dff7a` |
| `fresh_states.pt` | `2de72279a8dc60f7853ad698b2d710e6a73c83a625b26ed70e74e0d7d76856db` |
| tracked state result | `03fce084b7f46c64dc762cd0f9605981aad6ad51d056b23bd783c7ddcf1c4764` |
| tracked state binding | `0f1ffcf23310dbb782986cdf93cac8439054f249f36b8dc8118919f479f0434d` |

payload 为 197,546 bytes、200 条记录；local attestation 为 111,313 bytes。原始 payload 和
400 条逐进程证据保存在 Git 忽略的 `runs/`，避免把机器运行日志塞进开源仓库。Git 跟踪
的 result 与 binding 固定本地文件的路径、大小、SHA 和 source commit；下游在打开 tensor
前必须先验证这三层证据。

## 5. 当前授权边界

本结果与 binding 只解锁下一步：实现 original-A1 observation-only collection runner 及
其 CPU contract tests。它尚未授权：

- 执行 original-A1 observation collection；
- L13/L27 same-noise GPU replay；
- reliability model 训练或 calibration；
- 新方法 active control；
- 任何成功率、速度或优越性结论。

collection runner 必须形成独立 clean commit 和新的 machine-readable readiness，之后才能
在 preflight 通过时读取这 200 个状态。

## 6. 软件验证

新增 4 项 binding 测试覆盖 tracked evidence、真实本地 payload 加载以及 result/payload
篡改拒绝。Stage 11D 协议、state runner 与 binding 的定向测试为 `26 passed`；完整维护
测试为 `628 passed, 22 subtests passed`。三个 warning 均来自既有 Python 3.10 / Pydantic
依赖兼容提示，不影响本阶段结果。
