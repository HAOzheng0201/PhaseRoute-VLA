# PhaseRoute-VLA 代码映射

本文给出改进代码的实际接入点。张量推导见 `PHASEROUTE_ARCHITECTURE_ZH.md`，目录发布策略见 `REPOSITORY_LAYOUT.md`。

## 正式运行路径

```text
scripts/run_libero_rp_pep.sh
└── robot_experiments/libero/eval_libero_early_exit.py
    ├── a1/vla/affordvla_early_exit.py
    ├── a1/vla/value_net.py
    ├── a1/vla/dynamic_compute/productive_exit.py
    ├── a1/vla/dynamic_compute/release.py
    └── robot_experiments/libero/exit_vla_utils.py
```

| 文件 | 当前职责 | 正式路径 |
|---|---|---|
| `affordvla_early_exit.py` | 中间层 KV、视觉压缩 hook、telemetry hook | 是 |
| `value_net.py` | candidate/reference FM solve、RNG burn、阈值判断 | 是 |
| `productive_exit.py` | 冻结 RP-PEP 候选层和随机流规则 | 是 |
| `release.py` | checkpoint、阈值、paired result 科学门 | 是 |
| `eval_libero_early_exit.py` | 配置合法性、模型/controller 初始化、闭环调度 | 是 |
| `exit_vla_utils.py` | observation→action、side-channel 与 postprocess | 是 |
| `run_libero_rp_pep.sh` | GPU 0–3 guard、preflight、正式命令 | 是 |

## 动态计算包

`a1/vla/dynamic_compute/` 按功能分为：

| 类别 | 文件 |
|---|---|
| 正式 RP-PEP | `productive_exit.py`, `release.py`, `device_guard.py` |
| 可观测性 | `telemetry.py`, `fm_diagnostics.py`, `vision_teacher_cache.py` |
| phase 信号与缓存 | `phase_cache.py`, `phase_observer.py`, `phase_dataset.py`, `weak_labels.py` |
| phase 模型 | `phase_estimator.py`, `phase_training.py`, `phase_depth_runtime.py` |
| 视觉聚合 | `vision_aggregation.py`, `learnable_vision_aggregation.py`, `*_runtime.py` |
| 预算与安全控制 | `budget_profiles.py`, `budget_controller.py`, `exit_policy.py`, `depth_hysteresis.py` |
| 学习式路由研究 | `causal_route_router.py`, `temporal_route_router.py`, `risk_route13_router.py`, `m427_task_jackknife_router.py` |

只有第一行进入当前正式 release。其他模块完整保留用于后续研究，但均默认关闭或离线运行。

## 数据契约

| 数据 | 形状 / 类型 | 产生位置 | 消费位置 |
|---|---|---|---|
| projected vision | `(B,5,144,3584)`；4 valid + 1 pad | A1 vision backbone | 主 VLM / phase cache |
| visual mapping | `(B,5,144)`；576 source / 288 unique slots | preprocessor/collator | 主 VLM indexed write |
| input sequence | `(B,680)` | preprocessor/collator | 主 VLM |
| proprio | `(B,1,1,8)` | `vla_utils` | FM state projector |
| layer KV | 28 × K/V `(B,4,680,128)` | 主 VLM | FM expert / early exit |
| candidate action | `(B,8,7)` | FM expert | delta / controller |
| LIBERO action | `(B,8,7)` | postprocess | action queue |
| telemetry | strict JSON object | rollout side channel | JSONL / analyzer |
| teacher cache | NPZ + manifest JSONL | collection scripts | offline router training |

这些形状对应正式 `model/libero_exit` checkpoint。`configs/models/libero.yaml` 是可选的
上游兼容训练配方，仍采用另一套 `10×32` 契约；它不能用来解释当前 RP-PEP checkpoint。

## 不变式

1. 所有新功能默认关闭时必须保持 A1 baseline 行为。
2. 日志、cache 和 callback 异常不得传播进机器人控制流。
3. route 不能比 teacher/原始提议更浅，除非有独立冻结安全证据。
4. 当前正式计划只接受 A1-FM10 固定候选层网格。
5. episode 边界必须重置所有时序状态。
6. 研究输出不得写入 Git 跟踪目录。
7. 物理 GPU 4–7 不得由项目 launcher 使用。

## 测试映射

每个 dynamic-compute 模块在 `tests/dynamic_compute/` 有对应测试。高风险发布路径额外由以下测试覆盖：

```text
test_productive_exit.py
test_m420b_paired_rollouts.py
test_m420b_replay_analysis.py
test_release_gate.py
test_release_smoke_summary.py
test_m429_failure_analysis.py
```

运行 `make test` 可以执行完整回归网格。
