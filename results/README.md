# Frozen results

本目录包含 PhaseRoute-VLA 发布所需的小型机器可读证据。原始 rollout、teacher cache、hidden arrays 和视频不在 Git 中。

## 文件索引

| 文件 | 状态 | 说明 |
|---|---|---|
| `rp_pep_paired.json` | PASS | 20 对 baseline/RP-PEP 闭环等价性与效率 |
| `release_smoke_state30.json` | PASS | 物理 GPU 0–3、task 0–9、episode 30 smoke |
| `router_sealed.json` | 工程 PASS / 科学 NOT_VIABLE | 学习式 router 冻结 gate |
| `router_failure_analysis.json` | PASS | false-shallow 精确重建和发布选择 |
| `M4_29_summary.md` | — | 最终研究结论和错误归因 |
| `FINAL_VALIDATION.md` | — | 环境、测试、构建和 GPU 验证报告 |

发布副本的 SHA-256 见 `../artifacts/MANIFEST.json`。

## RP-PEP 配对结果

实验网格：10 个 LIBERO Spatial task × episode 27/28。每个配对使用相同 task、初始状态、seed 和 checkpoint。

```text
paired episodes:       20
total rollouts:        40
baseline successes:    20
RP-PEP successes:      20
action mismatches:     0
exit mismatches:       0
trajectory mismatches: 0
FM solve reduction:    41.11%
mean latency reduction:31.06%
median latency reduction: 29.91%
```

这是当前正式方法的主要证据。

## Release smoke

四张物理 GPU 0–3 对 task 0–9 各运行 episode 30：

```text
completed: 10 / 10
successes: 10 / 10
policy calls: 128
FM calls: 589
exit L11 / L13 / L27: 65 / 56 / 7
```

smoke 用于验证发布入口、任务分片、GPU 绑定和 telemetry 公式，不替代正式 benchmark 的多 seed 统计。

## Learned-router 负结果

一次性 sealed 评测：

```text
router_offline_gate: NOT_VIABLE
runtime_integration_allowed: false
science gates: 5 / 10
false-shallow records: 4
affected task/episode groups: 3
```

四次错误浅退均为 teacher 要求 layer 27，而 router 预测 layer 13。它们是离线安全错误；没有把失败 router 接入闭环，因此不能声称这些记录造成了三个最终任务失败。

## 结果读取示例

```python
import json

with open("results/rp_pep_paired.json") as file:
    result = json.load(file)

assert result["status"] == "PASS"
assert result["paired_episodes"] == 20
assert result["equivalence"]["action_chunk_sha256_mismatches"] == 0
```

## 可声明边界

可以声明：冻结 20-pair 网格上，RP-PEP 与 baseline 动作/退出/轨迹精确一致，并减少 41.11% FM solves 和 31.06% 平均策略延迟。

不能声明：

- 20-pair 等于完整 LIBERO 官方成功率；
- learned router 已可安全上线；
- 4 条 false-shallow 等于 4 次闭环失败；
- state-30 smoke 证明跨 seed 泛化；
- 当前 sealed set 可再次用作新 router 的 unseen test。
