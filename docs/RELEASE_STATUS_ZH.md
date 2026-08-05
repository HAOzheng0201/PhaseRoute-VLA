# PhaseRoute-VLA 发布状态

发布日期：2026-08-05  
发布方法：`rp_pep`  
运行时默认：关闭，必须显式启用  
学习式 router 运行许可：`false`

## 已验证能力

| 能力 | 状态 | 证据 |
|---|---|---|
| A1 + LIBERO import | PASS | release preflight |
| PyTorch 2.6 / CUDA 12.4 单卡计算 | PASS | CUDA smoke |
| dynamic-compute 回归测试 | PASS | `tests/dynamic_compute/` |
| RP-PEP 20-pair 闭环等价性 | PASS | `results/rp_pep_paired.json` |
| 物理 GPU 0–3 四卡发布 smoke | PASS | `results/release_smoke_state30.json` |
| checkpoint / threshold / result SHA gate | PASS | `a1/vla/dynamic_compute/release.py` |
| learned router 工程完整性 | PASS | `results/router_sealed.json` |
| learned router 科学安全门 | **NOT_VIABLE** | 10 门通过 5 门 |

## 正式 RP-PEP 结果

冻结网格：LIBERO Spatial task 0–9、episode indices 27 和 28、相同 seed/checkpoint，共 20 个配对 episode、40 次 rollout。

| 指标 | 结果 |
|---|---:|
| baseline successes | 20 / 20 |
| RP-PEP successes | 20 / 20 |
| success mismatches | 0 |
| action SHA mismatches | 0 |
| exit-layer sequence mismatches | 0 |
| policy-call count mismatches | 0 |
| trajectory mismatches | 0 |
| FM solve reduction | 41.11% |
| weighted mean latency reduction | 31.06% |
| median latency reduction | 29.91% |

可声明：在这组冻结配对实验中，RP-PEP 保持动作、退出序列和轨迹精确一致，同时降低 FM 调用和策略延迟。

不可声明：这些 20 个配对 episode 不能代替四个 LIBERO suite 的大规模官方 benchmark，也不能单独证明跨 checkpoint、跨硬件或真实机器人泛化。

## 学习式 router 负结果

M4.28 task-jackknife router 完成了数据协议、训练、校验和一次性 sealed evaluation，但：

```text
router_offline_gate: NOT_VIABLE
runtime_integration_allowed: false
science gates: 5 / 10
false-shallow records: 4
affected episode groups: 3
```

“3 个 episode group 受影响”不是“3 个闭环失败”。这批数据是离线 teacher-route 审计，不能把 false-shallow record 直接换算为闭环任务失败数。正式入口不会加载该 router。

## 冻结校验值

| Artifact | SHA-256 |
|---|---|
| `model/libero_exit/config.yaml` | `9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca` |
| `model/libero_exit/dataset_statistics.json` | `6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3` |
| `model/libero_exit/model.pt` | `dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f` |
| spatial threshold JSON | `5b3a0ee9f3851bf1b0c7f7e2b28bc61898ed0b4bd39f8752007719e9f26d7bd6` |
| curated paired result | `5e8c2fa2e50a30a5911b29bab796e50d624a2971da649b4aa82333ba9beefb16` |

完整 manifest：`artifacts/MANIFEST.json`。

## 正式入口

单卡：

```bash
GPU_INDEX=0 NUM_EPISODES=50 make run-rp-pep
```

前四卡 smoke：

```bash
make smoke-front4
```

两者均限制物理 GPU 0–3，使用 GPU UUID 绑定并拒绝覆盖已有运行目录。

## 仍需由新实验回答的问题

- 在四个 LIBERO suite 和更大 episode 网格上的成功率置信区间；
- 不同驱动、GPU 和 PyTorch patch 版本下的延迟可迁移性；
- 新 router 如何在不复用 M4.28 sealed set 的前提下建立独立验证集；
- 动态视觉压缩与 RP-PEP 组合后能否继续保持闭环等价性。

这些问题属于下一轮实验，不影响当前代码发布的工程完整性，但在论文中必须与已经验证的结论分开表述。
