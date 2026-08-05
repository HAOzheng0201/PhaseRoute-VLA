# Repository Layout

PhaseRoute-VLA 是一个聚焦 A1 + LIBERO 动态计算的独立项目，不是原研究工作区的镜像。

```text
.
├── a1/
│   ├── data/                  A1 训练数据与多模态预处理
│   ├── eval/                  上游 A1 通用 evaluator
│   ├── model.py               主 Transformer 与视觉连接器
│   ├── train.py               A1 训练循环
│   └── vla/
│       ├── affordvla.py       A1 baseline VLA
│       ├── affordvla_early_exit.py
│       ├── action_heads.py    Flow Matching 等动作头
│       ├── value_net.py       Early-exit controller 接口
│       └── dynamic_compute/   PhaseRoute-VLA 改进模块
├── artifacts/                可下载 artifact 的不可变清单
├── configs/
│   ├── datasets/             LIBERO 与上游预训练数据配置
│   ├── experiments/          组合配置
│   └── models/               模型动作维度配置
├── docs/                     架构、复现、状态与设计文档
├── launch_scripts/           A1 训练 Python 入口
├── patches/                  固定第三方补丁
├── requirements/             已验证环境约束
├── results/                  冻结的小型发布证据
├── robot_experiments/libero/ LIBERO 闭环环境与策略适配
├── scripts/
│   ├── dynamic_compute/      完整研究里程碑管线
│   ├── download_checkpoint.sh
│   ├── setup_libero.sh
│   ├── run_libero_rp_pep.sh
│   └── validate_phase_route_release.py
└── tests/dynamic_compute/     单元、回归与 release gate
```

## 为什么保留 A1 主干

PhaseRoute-VLA 的改进发生在 A1 的 Early Exit、Flow-Matching candidate evaluation 和 LIBERO rollout 中，不能只发布一个外部 wrapper。完整保留 `a1/` 的原因是：

- checkpoint 需要原模型定义才能加载；
- RP-PEP 改变的是候选计算调度，不是替代动作模型；
- 训练和离线 teacher collection 依赖同一预处理与动作头；
- 单元测试需要验证默认关闭时不改变 A1 行为。

## 为什么不包含其他 benchmark 资产

RoboChallengeInference、VLABench 大型媒体和第三方环境不参与当前方法的冻结结论。复制它们会产生失效入口、重复第三方历史和不必要的大文件。因此本仓库只发布 LIBERO 路径；其他 benchmark 请使用上游 A1。

这不影响项目的“完整”：PhaseRoute-VLA 声明的训练、模型、推理、评测、研究和测试路径均在仓库内，外部只需要明确登记的依赖、数据和 checkpoint。

## 顶层文件原则

顶层只保留新用户第一小时会用到的文件：

```text
README.md
LICENSE
NOTICE
CITATION.cff
Makefile
pyproject.toml
requirements.txt
train_libero.sh
eval_libero.sh
eval_libero_exit.sh
```

具体研究命令统一进入 `scripts/dynamic_compute/`，结果统一进入 `results/`，大文件统一进入 ignored 目录。

## 发布前目录审计

```bash
rg '/home/|/mnt/|/data[0-9]/' .
find . -type f -size +10M -not -path './.git/*'
find . -name __pycache__ -o -name .pytest_cache -o -name '*.egg-info'
git diff --check
git status --short
```

正常发布仓库不应包含个人绝对路径、大于 10 MB 的跟踪文件、Python cache 或未解释的顶层脚本。
