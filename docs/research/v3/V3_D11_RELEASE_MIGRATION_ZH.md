# V3 D11：正式发布迁移与验收

## 结论

PhaseRoute V3 已从研究工作树正式合并到独立发布仓库，并在不依赖旧目录层级的条件下
通过完整测试、CPU artifact gate、GPU/backbone gate 和一次已有的闭环工程 smoke。
D11 状态为 `PASS_V3_D11_RELEASE_MIGRATION`。

本阶段验证的是“完整、整洁、可复现、可运行的研究发布”，不是重新打开 D9 的一次性
独立测试，也不是部署授权。

## 提交链

| 角色 | commit | 说明 |
|---|---|---|
| V3 研究发布实现 | `807fa5a` | 冻结 runtime artifact、通用 launcher、测试与文档 |
| 维度勘误保护 | `decbd15` | 保留 680 token、4+1 crops、8×7 action、8D proprio |
| 正式非快进合并 | `9b231dd` | 将两个父提交合并到发布仓库 `main` |
| 自包含迁移修复 | `0428a7a` | 消除完整测试对旧研究目录和 ignored reports 的依赖 |

合并后 evaluator 保持冻结 SHA：

```text
a4e3b1b49cdaf2021b3cd370d8a1e89c927906e7cbd5f8afdccd5ceb5b1826cd
robot_experiments/libero/eval_libero_early_exit.py
```

因此 D9B/D9 后续证据所绑定的 evaluator 字节没有变化。

## 迁移过程中真实发现的问题

研究工作树上的完整测试原先是 201 项通过；第一次在独立发布目录运行时出现：

```text
198 passed, 3 failed
```

三项失败分别来自：

1. 两个 relocated CLI 测试通过 `REPO_ROOT.parents[1]/source` 推测旧研究目录；
2. D8C prerequisite 测试读取被 `.gitignore` 排除的 D8A/D8B `reports/` 文件。

这不是模型、router 或数值计算失败，而是发布可移植性缺陷。D11 没有用 `skip` 隐藏
失败，而是归档了验证实际需要的最小证据：

- legacy manifest 的 28 个文件，共 1,737,937 bytes；
- D1 绑定的 C3.55 result；
- D8 的 200 个生成状态与 D8A/D8B result；
- 不包含 rollout、视频、图像、teacher cache 或 hidden state。

这些材料位于 `artifacts/phase_route_v3/`，保留原始 SHA；release gate 会逐项验证。
历史 manifest 中的绝对路径只作为 provenance，不再成为执行依赖。

## 干净提交上的验证

### 完整 V3 测试

在 `0428a7a`、clean worktree 上：

```text
201 passed
22 subtests passed
0 failed
1 Python 3.10 生命周期 warning
25.82 s
```

JUnit 位于 `/tmp/phaseroute_v3_d11_0428a7a_v3_tests.xml`，SHA-256：

```text
94d30f9b82a5fcd78d6adee3961511d9070aa6036443042f8930002dd4d274e5
```

### CPU release preflight

CPU preflight 为 `PASS`，记录的 commit 为 `0428a7a`，
`worktree_dirty=false`。它验证了：

- 三个正式 runtime artifact；
- 五个 router heads 与 phase estimator state；
- D9 formal result；
- 28 项 legacy evidence、D8A/D8B 与 C3.55 result；
- Python 3.10、必要包、import 与 LIBERO 子模块。

attestation SHA-256：

```text
0895b91073422fb905f5309a63911b34f7f25e6d5861a95ba1427d90ad420cef
```

### GPU/backbone preflight

只使用空闲物理 GPU 0，并通过 UUID 绑定使进程只看到一张卡：

```text
physical GPU:       0
visible GPU count:  1
CUDA runtime:       12.4
GPU UUID:           f52eda42-a640-8244-bcdb-e6201acae766
git commit:         0428a7a
worktree_dirty:     false
status:             PASS
```

33.8 GB A1 backbone 的 size/SHA、config 与 dataset statistics 均匹配。
preflight SHA-256：

```text
4fd58e9c3f3fad99dff664cf4b9fac46758e7c5299076fb7705f54bc9154fc3e
```

GPU 4–7 未被访问。

### 已有闭环工程 smoke

通用 launcher 在研究发布 commit `807fa5a` 上完成了 task 0、official state 0 的一次
闭环工程 smoke：

| 项目 | 数值 |
|---|---:|
| success | 1 / 1 |
| policy calls | 34 |
| prepared / committed | 34 / 34 |
| runtime errors | 0 |
| L11 / L13 / L27 | 0 / 4 / 30 |

13/13 run checks 通过，attestation SHA 为
`431b14d02403e402fd91e4849c060b7b3f80cd9818a46c9e524b2ed10fb21651`。
它的 scope 明确是 `general_simulator_run_not_D9_retest`，只能证明通用入口从输入到
输出可运行，不能当作新的准确率或独立测试结论。

## 科学边界

- 未重新运行或调参 D9 official states 40–49；
- 未删除 D5/D6 及 M4.28 的负结果；
- 不把失败 episode 与 early exit 的共现解释为因果；
- 不声称工程 smoke 是新的 independent test；
- `deployment_authorized=false` 保持不变；
- 主权重继续作为外部 SHA-pinned artifact，不进入 Git。

## 机器可读证据

完整字段、哈希和门禁见：

```text
results/v3/v3_d11_release_migration.json
```
