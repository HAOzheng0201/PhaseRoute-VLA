# Artifact 与数据管理策略

PhaseRoute-VLA 将“可审查的发布材料”和“体积很大的本地实验载荷”严格分开。仓库克隆后应保持轻量；完整实验通过固定代码、校验清单、冻结摘要和可重复命令重建。

## 目录策略

| 路径 | 内容 | Git 策略 |
|---|---|---|
| `a1/` | A1 主模型和 PhaseRoute-VLA 动态计算实现 | 提交 |
| `scripts/` | 正式 launcher 与完整研究管线 | 提交 |
| `tests/` | 单元、回归和 release-gate 测试 | 提交 |
| `docs/` | 架构、复现和研究说明 | 提交 |
| `results/` | 小型、冻结、机器可读结果与摘要 | 提交 |
| `artifacts/MANIFEST.json` | 权重、第三方代码和结果的 revision/SHA | 提交 |
| `model/` | 约 34 GB checkpoint 与训练权重 | 忽略 |
| `reports/` | 原始 result、teacher cache、hidden、NPZ | 忽略 |
| `runs/` | rollout、stdout、视频与临时 preflight | 忽略 |
| `.cache/` | Hugging Face、LIBERO 配置和工具缓存 | 忽略 |

## 发布内的冻结结果

`results/` 只保留支持结论所需的小型摘要：

| 文件 | 用途 |
|---|---|
| `rp_pep_paired.json` | 20-pair RP-PEP 与原始 early-exit 的严格配对证据 |
| `release_smoke_state30.json` | 前四卡、10-task、state-30 发布 smoke 汇总 |
| `router_sealed.json` | 学习式 router 一次性 sealed gate |
| `router_failure_analysis.json` | 4 条 false-shallow record 的后验诊断 |
| `FINAL_VALIDATION.md` | 发布级验证结论 |

这些 JSON 中的原始实验输入路径已规范化为仓库相对路径，唯一 GPU UUID 已替换为四个稳定的伪标识；原始输入内容的 SHA-256 仍保留。发布副本自身的 hash 记录在 `artifacts/MANIFEST.json`。

## checkpoint

checkpoint 由 `scripts/download_checkpoint.sh` 从以下不可变 revision 获取：

```text
repository: spatialtemporal-ai/a1-libero-exit
revision:   a014b84203c6fb981d3f6181dc3bc7207610b2a3
```

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `config.yaml` | 8,369 | `9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca` |
| `dataset_statistics.json` | 11,871 | `6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3` |
| `model.pt` | 33,841,175,207 | `dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f` |
| `exit_thresholds_libero_spatial_exp_1.0.json` | 241 | `5b3a0ee9f3851bf1b0c7f7e2b28bc61898ed0b4bd39f8752007719e9f26d7bd6` |

正式 release gate 同时验证 config、dataset statistics、checkpoint、阈值、paired result 的 hash 和科学门。只改文件名但不匹配内容不会通过。

## 为什么不提交原始 reports

研究归档包含约数十 GB 的以下数据：

- 每次 policy call 的 teacher action 与 FM trace；
- visual/language hidden summary 和 layer-13 hidden；
- episode 视频、日志、rollout 与临时缓存；
- router feature matrix、checkpoint 和中间拟合输出。

这些内容不适合作为 Git 仓库的一部分，也可能包含机器绝对路径。保留它们不会增强普通用户的代码复现，反而会使 clone、review 和版本管理不可用。因此发布仓库只保存生成它们的源码、测试、冻结统计量和哈希。

## 不可变与防覆盖约定

- 正式 launcher 为每次运行创建新时间戳目录；
- 四卡 launcher 遇到已存在的目标目录会退出；
- cache writer 使用原子安装并拒绝覆盖已有 shard；
- frozen result 的 SHA 改变会使 release gate 失败；
- 研究输出只能写入被忽略的 `reports/` 或 `runs/`。

## 大文件发布建议

若公开原始 artifacts，应使用带版本与 checksum 的对象存储、Hugging Face Dataset 或 Zenodo，而不是 Git history。建议每个外部 bundle 至少包含：

```text
MANIFEST.json
README.md
code_commit.txt
environment.txt
checksums.sha256
```

任何外部 artifact 发布都不应改变本仓库中已经冻结的科学结论。
