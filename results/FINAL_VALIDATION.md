# PhaseRoute-VLA 最终验证报告

验证日期：2026-08-05  
发布结论：**PASS（RP-PEP release scope）**  
learned-router 结论：**NOT_VIABLE（禁止运行时集成）**

## 1. 发布范围

本报告对应独立、清理后的 PhaseRoute-VLA 项目，而不是原研究工作区。发布包含：

- 完整 A1 模型与 LIBERO 训练/评测路径；
- 29 个 `a1.vla.dynamic_compute` 模块；
- 完整研究 collection/training/replay/audit 脚本；
- 正式单卡 RP-PEP 与前四卡 smoke launcher；
- 全部动态计算测试、文档和冻结小型结果；
- checkpoint 下载、第三方补丁和 SHA release gate。

权重、原始 reports、teacher cache、hidden arrays、rollout 和视频不进入 Git。

## 2. 验证环境

| 项目 | 值 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA runtime | 12.4 |
| GPU | RTX 6000 Ada；launcher 仅允许物理 0–3 |
| transformers | 4.53.2 |
| datasets | 3.6.0 |
| NumPy | 1.25.0 |
| MuJoCo / robosuite | 2.3.7 / 1.4.1 |
| LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

## 3. 代码与构建验证

| 验证项 | 结果 |
|---|---|
| `python -m pip check` | PASS；无 broken requirements |
| shell 语法 | PASS；全部项目与 dynamic-compute shell |
| Python release 入口语法 | PASS |
| `git diff --check` | PASS |
| 快速 release gate | 5 passed，0 failed |
| 完整 dynamic-compute 回归 | **270 passed，0 failed** |
| 隐私化结果后的定向回归 | 11 passed，0 failed |
| PEP 517 wheel | PASS；`phase_route_vla-0.1.0-py3-none-any.whl` |
| Markdown 本地链接 | PASS |
| 冻结结果 manifest | PASS；4/4 hash 一致 |

完整回归的 4 条 warning 来自 Python 3.10 生命周期提示、受限进程中的 CUDA probe 和 Pydantic 第三方 schema 元数据；没有测试失败。

## 4. 完整 preflight

使用原研究归档中的 checkpoint 和 LIBERO 作为临时只读输入执行发布 preflight，结束后移除权重链接。结果：

```text
status: PASS
process exit code: 0
required packages: 12/12
required imports: 4/4
LIBERO submodule: PASS
release artifacts: PASS
```

release gate 的 expected/actual SHA 全部一致：

| Artifact | SHA-256 |
|---|---|
| checkpoint config | `9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca` |
| dataset statistics | `6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3` |
| `model.pt` | `dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f` |
| Spatial threshold | `5b3a0ee9f3851bf1b0c7f7e2b28bc61898ed0b4bd39f8752007719e9f26d7bd6` |
| curated paired result | `5e8c2fa2e50a30a5911b29bab796e50d624a2971da649b4aa82333ba9beefb16` |

## 5. RP-PEP 闭环证据

| 指标 | 原 A1 Early Exit | RP-PEP |
|---|---:|---:|
| 成功 episode | 20/20 | 20/20 |
| action / exit / trajectory mismatch | — | 0 / 0 / 0 |
| FM calls | 2,002 | 1,179 |
| 平均策略延迟 | 10,563.73 ms | 7,282.43 ms |
| 中位策略延迟 | 9,561.36 ms | 6,701.43 ms |

对应降幅为 FM 41.11%、平均延迟 31.06%、延迟中位数 29.91%。

## 6. 前四卡发布 smoke

冻结 state-30 汇总：

| 项目 | 结果 |
|---|---:|
| completed episodes | 10/10 |
| successes | 10/10 |
| policy calls | 128 |
| FM calls | 589 |
| exit L11 / L13 / L27 | 65 / 56 / 7 |
| telemetry errors | 0 |

发布副本对唯一 GPU UUID 进行了伪标识化，但保留四卡一一对应关系和全部一致性检查。

## 7. Learned router 的失败语义

M4.28 router 有 4 次错误浅退，分布于 3 个 episode group；这是离线 teacher-route 安全错误，不是 3 次已观测闭环失败。项目 fail closed：

- router 未连接到正式 forward；
- `runtime_integration_allowed=false`；
- release gate 固定 `learned_router_runtime_allowed=false`；
- 现有 sealed set 不得再次作为新模型的 unseen test。

## 8. 最终判定

项目达到以下发布标准：源码完整、目录聚焦、安装与 artifact 路径明确、默认行为安全、GPU 0–3 边界可审计、测试和 wheel 通过、结果可机器复核、负结果没有被包装成正结论。

PASS 仅属于 RP-PEP 工程发布与冻结的效率/等价性范围，不改变 learned router 的 `NOT_VIABLE` 结论，也不替代更大规模 LIBERO 多 seed benchmark。
