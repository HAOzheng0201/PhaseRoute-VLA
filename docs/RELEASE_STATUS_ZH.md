# PhaseRoute-VLA 发布状态

更新日期：2026-08-23

当前研究方法：`phase_route_v3`

历史正式 baseline：`rp_pep`

运行时默认：关闭，必须显式启用

授权范围：LIBERO 仿真研究复现；`deployment_authorized=false`

## 1. 当前结论

PhaseRoute V3 已完成从泄漏审计、development、independent calibration、fresh-state
shadow confirmation、active runtime parity、100-pair active independent test 到冻结
aggregate 的完整链路。D9 的 18/18 primary gates 全部通过，因此 V3 five-head router
可以作为当前正式研究方法发布。

这不会覆盖旧负结果：M4.28 task-jackknife router 的科学 gate 仍为 `NOT_VIABLE`，旧
RP-PEP 的 20-pair exact-equivalence 结果也仍有效。三者必须按名称、suite 与证据分开。

## 2. 已验证能力

| 能力 | 状态 | 冻结证据 |
|---|---|---|
| V3 router + phase payload 严格加载 | PASS | `artifacts/phase_route_v3/MANIFEST.json` |
| 97D causal context 与 fail-closed runtime | PASS | `tests/dynamic_compute/v3/` |
| 单卡 active runtime parity | PASS | V3-D9A/D9B readiness |
| LIBERO-10 100-pair active independent test | PASS | `results/v3/v3_d9_final_result.json` |
| primary science/engineering gates | 18 / 18 PASS | 同上 |
| clean-clone CPU release gate | PASS | `scripts/validate_phase_route_v3_release.py` |
| GPU 0–3 UUID 绑定与单卡 preflight | PASS | V3 launcher/preflight |
| 通用 task/state 选择与 non-overwrite 输出 | PASS | V3 launcher tests |
| 独立发布仓库自包含迁移 | PASS | `results/v3/v3_d11_release_migration.json` |
| 真实机器人部署 | **NOT VALIDATED** | 不在当前授权范围 |

## 3. PhaseRoute V3 正式结果

冻结测试：LIBERO-10 task 0–9 × official init-state indices 40–49，共 100 pair、200
rollout。每对使用相同 task、initial state 和预注册的 arm-specific seed。

| 指标 | Original A1 | PhaseRoute V3 |
|---|---:|---:|
| successes | 85 / 100 | 88 / 100 |
| success rate | 85% | 88% |
| FM calls | 39,996 | 24,776 |
| policy calls | 3,788 | 3,700 |
| FM calls / policy call | 10.5586 | 6.6962 |
| normalized FM-call reduction | — | 36.58% |
| L11 / L13 / L27 | — | 100 / 412 / 3,188 |
| early-exit calls | — | 512 / 3,700 = 13.84% |
| false-safe calls / clusters | — | 0 / 0 |
| exact CP-UCB95 | — | 2.951% |

成功率差为 `+3 pp`，task-stratified one-sided 95% bootstrap lower bound 为 `-2 pp`，
通过预注册的 non-degradation gate。McNemar equality `p=0.5078`，所以不能写成“显著
优于 A1”。FM 指标不包含 router latency，因此不能把 36.58% 直接称为 wall-clock
speedup。

正式 result SHA-256：

```text
4df77237e84ad82b05ae67145e52000b0e3430f34b6f69fcbee743687ac11952
```

## 4. 提前退出与失败

| 描述性统计 | 数量 |
|---|---:|
| PhaseRoute failures | 12 |
| failures with any early exit | 12 |
| failures with unsafe early call | 0 |
| A1 success / PhaseRoute failure | 3 |
| 其中 with unsafe early call | 0 |

这是共现统计，不是因果统计。same-noise L27 truth 没有发现 unsafe early call，但 L27
也不是 task-success certificate。只有在全新数据上对事前指定的 early call 做配对 L27
替换实验，才可能回答 causal effect。

## 5. 冻结运行 artifacts

| Artifact | 字节数 | SHA-256 |
|---|---:|---|
| A1 `model.pt`（外部下载） | 33,841,175,207 | `dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f` |
| V3 five-head router | 22,290 | `9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830` |
| phase estimator | 11,344,688 | `b601f8221d47136818d7a008eaf7cee06e1201bf514f371ae33f42cfb39515a1` |
| phase state | — | `8c0021be43d1cea28890833fd5e1faa8ee0191e809cbf3b1df0d3c36010d7598` |
| LIBERO-10 threshold JSON | 236 | `a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796` |

两个小模型与阈值随 Git release 提供；34 GB A1 backbone 从固定 Hugging Face revision
下载。launcher 在任何模型初始化前完成 byte size、SHA、payload schema、five-head
数量和 phase-state 校验。

## 6. 正式研究入口

CPU/clean-clone：

```bash
make preflight-v3
make test-v3-release
```

完整 artifact + 单 GPU preflight：

```bash
GPU_INDEX=0 PREFLIGHT_ONLY=1 make run-v3
```

普通仿真复现（不使用 D9 的 test schedule）：

```bash
GPU_INDEX=0 TASK_IDS=0 EPISODE_INDICES=0 make run-v3
```

launcher 只允许物理 GPU 0–3，用 GPU UUID 将进程限制为一张可见卡，固定
`libero_10 / FM10 / exit_interval=2 / L11-L13-L27`，并保存 preflight、command、
telemetry、runtime records、evaluation summary 与最终 attestation。

## 7. 历史结果如何保留

### RP-PEP

LIBERO Spatial 20-pair：20/20 vs 20/20 success、动作/退出/轨迹 mismatch 为 0、FM
solve reduction 41.11%、weighted mean latency reduction 31.06%。它是固定
RNG-preserving candidate-pruning baseline，不是 V3 learned router。

### M4.28 learned router

```text
router_offline_gate: NOT_VIABLE
runtime_integration_allowed: false
science gates: 5 / 10
false-shallow records: 4
affected episode groups: 3
```

该负结果对应不同 feature、router 和 sealed set，不能概括为“所有 learned routing 都
不可行”。四条 false-shallow 是离线记录，也不能写成四次闭环失败。

## 8. 数据与声明边界

LIBERO-10 official states 的角色已全部消耗：

```text
0–11   historical
12–29  development-v2
30–39  calibration-v2
40–49  consumed D9 independent-test-v2
```

因此：

- 不允许再用 40–49 选择模型、阈值或 ablation；
- 不允许把新的 official-state run 称为第二次 unseen test；
- 新 trainable ablation 必须各自重新训练 normalizer/router；
- 新 confirmation 必须先冻结 generated-state protocol；
- 旧 Spatial 20-pair 与 V3 LIBERO-10 100-pair 不可混表作直接排名；
- 当前没有真实机器人、跨 suite、wall-clock latency 或 deployment 结论。

消融和未来因果实验协议见
`docs/research/v3/V3_D10_PAPER_ABLATION_MIGRATION_ZH.md`；发布迁移的测试、CPU/GPU
attestation 与可移植性修复见
`docs/research/v3/V3_D11_RELEASE_MIGRATION_ZH.md`。

## 9. 当前发布资格复核记录

2026-08-23 在独立发布目录、冻结 V3 权重与阈值不变的前提下完成阶段二复核：

| 检查 | 结果 |
|---|---|
| `make test` | **472 passed**, 22 subtests passed，0 failed，72.57 s |
| `make test-v3-release` | **13 passed**，0 failed，6.43 s |
| `make check` | pip、Python/Shell 语法、`git diff --check` 全部 PASS |
| Python sdist/wheel | 构建成功，`twine check --strict` 与 wheel ZIP 完整性均 PASS |
| tracked 大文件 | 无超过 100 MB blob；唯一超过 10 MB blob 为已登记的 phase estimator |
| secrets 审计 | 文件名与常见 token/private-key 内容模式均无命中 |
| 通用入口绝对路径 | QuickStart、Makefile 与通用 launcher 无个人机器路径依赖 |
| broken symlink | 无 |
| README/Makefile 映射 | README 的 7 个 `make` target 与 Makefile 引用的脚本全部存在 |

复核首次运行发现两个 evidence-manifest 测试仍直接引用已删除的旧
`/data3/haozheng/A1/source`。根因是发布代码已经采用 bundled relocated evidence，测试
常量却未同步迁移；这不是模型、数值或冻结证据失败。修复后测试使用仓库内
`artifacts/phase_route_v3/legacy_source/`，显式启用 relocation，并继续逐文件验证原始
size/SHA 与 symlink fail-closed 语义；对应目标测试为 **14 passed + 22 subtests**。

本轮还把 `pyproject.toml` 的许可证元数据更新为标准 SPDX `MIT`，并显式登记
`LICENSE`/`NOTICE`，消除了 setuptools 2027 弃用警告。最终本地构建产物为：

```text
phase_route_vla-0.1.0-py3-none-any.whl  653168 bytes
SHA-256 94016191d9169853f07ae19fea49558a54cd668bd7fa6205a940f8af2e07e0a6

phase_route_vla-0.1.0.tar.gz            582588 bytes
SHA-256 0d0800acc4dcf7e37ba9dd6c57283bba70f18a3a9120ef0f6dfa34a7a2ffb041
```

这些 build SHA 仅记录本轮本地资格复核，不替代
`artifacts/phase_route_v3/MANIFEST.json` 中冻结 runtime artifacts 的正式 SHA。

### 9.1 Post-release GPU 工程验证

在 commit `e213844a3a6e78fcd2d876a1d29bac5c81c5c602` 的 clean worktree 上，只使用
物理 GPU 0 完成以下验证；GPU 4--7 未访问：

| 范围 | 结果 | 关键指标 |
|---|---|---|
| 完整 GPU preflight | PASS | UUID 单卡绑定、CUDA 12.4、backbone 与所有 V3 SHA/payload gate 通过 |
| task 0 / state 0 单 episode | PASS | 1/1 success，34 policy calls，L13/L27=4/30，0 error |
| 10 task × state 0 | PASS engineering smoke | 9/10 success，362 policy calls，L11/L13/L27=8/52/302，0 error |

10-task smoke 使用与冻结 runtime 一致的 FM10 配置，但属于普通 simulator engineering
run；它没有使用 D9 states 40--49，也不构成第二次 independent test。其描述性统计为：

```text
successes                         9 / 10
early exits                      60 / 362 = 16.57%
FM calls / policy call           2406 / 362 = 6.6464
policy-call latency mean         1422.65 ms
policy-call latency median       1538.10 ms
10-episode rollout wall time     681.18 s
```

唯一失败为 task 4/state 0：65 次调用中 L13 仅 2 次、L27 为 63 次，runtime error 为
0。该结果不能证明两次 L13 导致失败；在相同 state/seed 的 original A1 arm 完成前，也
不能形成 paired attribution。

本地、被 Git 忽略的原始记录及其 sealed SHA 为：

```text
runs/phase_route_v3/libero_10_20260824_001530/preflight.json
ef9d052d12dc68da8de52f649035510495399af71716783560ca0d5d34a66c5b

runs/phase_route_v3/libero_10_20260824_001738/run_attestation.json
b08b8bf2811bc712d10318354b000a066231b1db077065dcfc7a96541f4b09a4

runs/stage5_phase_route_v3/libero_10_20260824_003703/run_attestation.json
5519d737a7e1751f83be635ca7ee3261958295fdff0e3adaa0ab8b0bb54d00b1

runs/stage5_phase_route_v3/libero_10_20260824_003703/evaluation_summary.json
079b90281e3dd77d3f3bb2e6d66beb8f23a71c202dec8b2868316b98e955d155
```

original A1 的 GPU 1 preflight 已通过，但 rollout 尚未完成，因此本节不声明 paired
success difference、paired latency reduction 或 wall-clock speedup。

### 9.2 Clean-clone 复现

commit `28cac086d95a439d1fffb2a1b9775def8294dffd` 被重新克隆到 `/tmp` 的空目录，LIBERO
submodule 从本地 pinned object store 初始化到
`8f1084e3132a39270c3a13ebe37270a43ece2a01`。clone 中不存在原工作区的 `model/`、
`runs/`、`.cache/` 或旧 `/source` 内容。使用同一冻结 conda 环境得到：

```text
CPU V3 release gate             PASS (worktree_dirty=false)
full dynamic-compute tests      472 passed + 22 subtests, 0 failed
sdist + wheel build             PASS
twine check --strict            PASS / PASS
wheel ZIP integrity             PASS
final clone git status          clean
```

这证明源码、bundled runtime artifacts 与最小历史证据可从 clean Git clone 自包含验证。
它仍不等于“从零创建全新依赖环境”：当前 Python/CUDA 依赖继续来自已验证的 `a1` conda
环境；完全从零的环境安装验收需要在后续单独执行并记录。
