# Frozen results

本目录只保存支持结论所需的小型、机器可读、不可变证据。raw rollout、teacher cache、
hidden tensors、模型训练输出和视频分别留在 ignored `reports/` / `runs/` / `model/`。

## 当前结果索引

| 路径 | 状态 | 正确解释 |
|---|---|---|
| `v3/v3_d11_release_migration.json` | PASS | 研究分支到独立发布仓库的自包含迁移验收 |
| `v3/v3_d9_final_result.json` | **18/18 PASS** | PhaseRoute V3，LIBERO-10 100-pair active independent test |
| `v3/v3_d10_paper_analysis.json` | PASS | 从冻结 D9 结果导出论文资产并冻结消融边界 |
| `rp_pep_paired.json` | PASS | 历史 RP-PEP，LIBERO Spatial 20-pair exact equivalence |
| `release_smoke_state30.json` | PASS | 历史 RP-PEP 前四卡工程 smoke |
| `router_sealed.json` | 工程 PASS / 科学 NOT_VIABLE | M4.28 task-jackknife router 负结果 |
| `router_failure_analysis.json` | PASS | M4.28 false-shallow 精确重建 |

`results/v3/` 还包含 D0–D11 每个数据、拟合、校准、active test 与发布迁移阶段的
attestation。这些文件用于追溯完整研究链，正式主结果以 D9 final result 为唯一入口。

## PhaseRoute V3 D9

冻结网格：LIBERO-10 task 0–9 × official state 40–49，共 100 pair。D9 aggregate 只运行
一次，没有替换 episode/seed，也没有在结果已知后调模型或阈值。

```text
A1 successes:                   85 / 100
PhaseRoute successes:           88 / 100
paired delta:                   +3 pp
FM calls / policy call:         10.5586 -> 6.6962
normalized FM-call reduction:   36.58%
L11 / L13 / L27 calls:          100 / 412 / 3188
early-exit calls:               512 / 3700 = 13.84%
false-safe calls / clusters:    0 / 0
exact CP-UCB95:                 2.951%
primary gates:                  18 / 18 PASS
```

SHA-256：

```text
4df77237e84ad82b05ae67145e52000b0e3430f34b6f69fcbee743687ac11952
```

### failure association

12 个 PhaseRoute failure 都出现过至少一次 early exit；其中 unsafe early call 为 0。
A1-success/PhaseRoute-failure 有 3 pair，其中 unsafe early call 也为 0。这只能报告为
association，不能报告为“early exit 导致 12/3 次失败”。

## 历史 RP-PEP

实验网格：10 个 LIBERO Spatial task × episode 27/28。

```text
paired episodes:             20
baseline / RP-PEP success:   20 / 20, 20 / 20
action / exit / trajectory mismatches: 0 / 0 / 0
FM solve reduction:          41.11%
weighted mean latency reduction: 31.06%
```

它验证固定 RNG-preserving pruning 的 exact equivalence，不是 PhaseRoute V3 的 learned
routing 结果。suite 与 episode 网格不同，不能把 41.11% 与 36.58% 直接作优劣排名。

## 保留的 M4.28 负结果

```text
router_offline_gate: NOT_VIABLE
runtime_integration_allowed: false
science gates: 5 / 10
false-shallow records: 4
affected task/episode groups: 3
```

四条错误浅退均为离线 teacher-route record；该 router 没有接入闭环，不能把它们换算
为任务失败。V3 使用不同 feature、five-head ensemble、gripper veto、calibration 和
独立测试，所以旧负结果不能泛化为“learned router 全部失败”。

## 读取正式结果

```python
import json

with open("results/v3/v3_d9_final_result.json") as file:
    result = json.load(file)

assert result["status"] == "PASS_V3_D9_PAIRED_ACTIVE_INDEPENDENT_TEST"
assert result["success"]["pairs"] == 100
assert result["all_primary_gates_pass"] is True
assert all(result["gate_checks"].values())
```

## 可声明边界

可以声明：冻结 100-pair LIBERO-10 active independent test 中，V3 88/100 success，
通过预注册安全/非退化/效率门，并将 normalized FM calls/policy call 减少 36.58%。

不能声明：

- 88% 相对 85% 是统计显著优越（McNemar `p=0.5078`）；
- 0 observed false-safe 表示真实风险为 0；
- FM-call reduction 等于 wall-clock speedup；
- early exit 与 failure 共现证明因果；
- D9 states 40–49 还能作为新方法的 unseen test；
- 结果已覆盖其他 LIBERO suite 或真实机器人；
- M4.28 负结果已被删除或可以忽略。

所有发布副本 SHA 见 `../artifacts/MANIFEST.json`。
