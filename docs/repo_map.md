# PhaseRoute-VLA 代码映射

本文给出当前 PhaseRoute V3 的实际接入点。张量推导见
`PHASEROUTE_ARCHITECTURE_ZH.md`，发布策略见 `REPOSITORY_LAYOUT.md`。

## 正式研究运行路径

```text
scripts/run_libero_phase_route_v3.sh
├── scripts/validate_phase_route_v3_release.py
├── artifacts/phase_route_v3/
└── robot_experiments/libero/eval_libero_early_exit.py
    ├── a1/vla/affordvla_early_exit.py
    ├── a1/vla/value_net.py
    ├── a1/vla/dynamic_compute/productive_exit.py
    └── a1/vla/dynamic_compute/v3/
        ├── active_runtime.py
        ├── development_collection.py
        ├── final_router.py
        ├── runtime_adapter.py
        └── release.py
```

| 文件 | 当前职责 | V3 active path |
|---|---|---|
| `affordvla_early_exit.py` | 680-token/5-crop A1、layer KV、visual/phase callbacks | 是 |
| `value_net.py` | RP-PEP candidate/reference FM solve、V3 adapter hook | 是 |
| `productive_exit.py` | L3/L11/L13/L27 productive schedule 与 RNG 规则 | 是 |
| `phase_estimator.py` | causal phase embedding/progress/boundary/uncertainty | 是 |
| `v3/active_runtime.py` | live context、past-only history、phase、commit/fallback | 是 |
| `v3/development_collection.py` | 9-tensor context validation 与 exact 97D feature | 是 |
| `v3/final_router.py` | five-head payload loader 与 risk prediction | 是 |
| `v3/runtime_adapter.py` | L11→L13→L27 hierarchical selection | 是 |
| `v3/release.py` | small artifacts、A1 backbone、D9 result SHA gate | 是 |
| `eval_libero_early_exit.py` | task/state selection、闭环、telemetry/result output | 是 |
| `run_libero_phase_route_v3.sh` | GPU 0–3、UUID、overlay、non-overwrite | 是 |

## V3 包结构

`a1/vla/dynamic_compute/v3/` 按研究阶段保留完整代码：

| 类别 | 代表文件 |
|---|---|
| 数据泄漏与协议 | `data_lineage.py`, `gripper_v2_protocol.py`, `d9_protocol.py` |
| feature/target collection | `development_collection.py`, `same_noise_replay.py` |
| 模型与校准 | `gripper_v2_models.py`, `severity_reliability*.py`, `joint_reliability.py` |
| epistemic ensemble | `epistemic_ensemble*.py`, `final_router.py` |
| active runtime | `runtime_adapter.py`, `active_runtime.py` |
| paired test/aggregate | `paired_active_collection.py`, `d9_final_analysis.py` |
| release | `release.py` |

`scripts/dynamic_compute/v3/` 是 D0–D10 的可审计 runner；D11 迁移验收位于
`results/v3/` 与 `docs/research/v3/`。通用用户入口不调用硬绑定
D9 state 40–49 的研究 runner。

## 真实数据契约

| 数据 | 形状 / 类型 | 产生位置 | 消费位置 |
|---|---|---|---|
| raw image crops | `(B,5,576,588)`；4 valid + 1 pad | preprocessor/collator | ViT |
| projected vision | `(B,5,144,3584)` | vision backbone | main VLM / V3 runtime |
| visual mapping | `(B,5,144)`；576 source / 288 unique slots | collator | indexed write/pooling |
| input sequence | `(B,680)` | preprocessor/collator | main VLM |
| proprio | `(B,1,1,8)` | `exit_vla_utils` | FM state projector |
| layer KV | 28 × K/V `(B,4,680,128)` | main VLM | FM expert |
| candidate action | `(B,8,7)` | FM expert | A1 delta / V3 router |
| phase state | `(B,128)` + `(B,3)` | phase estimator | 82D feature |
| histories | `(B,8,8)`, `(B,8,8,7)`, `(B,8)` | active runtime | phase/97D feature |
| route feature | `(B,97)` per current candidate | feature builder | five-head router |
| selected action | `(B,8,7)` exact L11/L13/L27 | runtime adapter | postprocess/queue |
| policy telemetry | strict JSONL | rollout side channel | run audit |
| V3 runtime records | strict JSONL | active runtime | run attestation |

`configs/models/libero.yaml` 是另一套上游兼容 10×32 训练配方，不能用来解释当前
`model/libero_exit` checkpoint。

这些形状对应正式 `model/libero_exit` checkpoint。`configs/models/libero.yaml` 是可选的
上游兼容训练配方，仍采用另一套 `10×32` 契约；它不能用来解释当前 RP-PEP checkpoint。

## 不变式

1. V3 默认关闭时必须保持 A1 baseline 行为。
2. L11 feature 不看 L13/L27，L13 feature 不看 L27/future/outcome。
3. episode 开始清空 history；selected action 在 route 完成后才 commit。
4. task/episode identity 只能做 telemetry/order check，不能进入 97D。
5. missing/nonfinite/shape/order/artifact drift 一律 veto early exit 到 L27。
6. router 只选择 exact candidate，不修改 candidate action。
7. 正式 A1/FM10/threshold/router/phase/result 都必须通过 SHA gate。
8. 运行输出只能进入 ignored `runs/`，且禁止覆盖。
9. 项目 launcher 不得使用物理 GPU 4–7。
10. D9 states 40–49 不得再次用于模型/阈值选择。

## 历史模块

| 模块 | 状态 |
|---|---|
| RP-PEP | 保留的固定裁剪 baseline；仍提供 V3 productive/RNG path |
| M4.28 task-jackknife router | `NOT_VIABLE` 负结果；不进入 V3 |
| M3 phase-depth | legacy research；与 V3 互斥 |
| M4 static / M4.7 learned vision aggregation | research；D9/V3 active path 关闭 |
| CogVLA mapping | 设计参考；没有直接复制模型或 token compression |

## 测试映射

高风险路径由以下测试覆盖：

```text
tests/dynamic_compute/v3/test_active_runtime.py
tests/dynamic_compute/v3/test_runtime_adapter.py
tests/dynamic_compute/v3/test_development_collection.py
tests/dynamic_compute/v3/test_final_router.py
tests/dynamic_compute/v3/test_paired_active_collection.py
tests/dynamic_compute/v3/test_d9_final_analysis.py
tests/dynamic_compute/v3/test_release.py
```

`make test-v3-release` 执行快速 artifact/launcher gate，`make test` 执行完整历史与 V3
regression grid。
