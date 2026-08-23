# PhaseRoute-VLA 中文复现指南

本文从空环境开始复现 PhaseRoute V3 的通用 LIBERO-10 仿真入口。所有命令在项目
根目录执行；34 GB backbone、缓存和运行输出不会进入 Git。

## 1. 冻结环境

已验证组合：

```text
Linux x86_64
Python 3.10
PyTorch 2.6.0 + CUDA 12.4
torchvision 0.21.0
NVIDIA driver 570.133.07
RTX 6000 Ada, 48 GiB / GPU
```

launcher 只允许物理 GPU 0–3，并确保每个进程只看到一张卡。GPU 4–7 被明确保留。

## 2. 获取代码

```bash
git clone --recurse-submodules <your-phase-route-vla-repository>
cd PhaseRoute-VLA
git submodule status
```

LIBERO 应固定在：

```text
8f1084e3132a39270c3a13ebe37270a43ece2a01
```

## 3. 创建环境

```bash
conda create -n phase-route-vla python=3.10 -y
conda activate phase-route-vla
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

make install
make setup-libero
```

`make install` 使用 `requirements/constraints-cu124.txt` 固定关键版本；不要安装
`ai2-molmo[dev,serve,train]`。`make setup-libero` 会安装 pinned submodule、幂等应用
PyTorch 2.6 patch，并非交互创建 LIBERO config。

设置本地缓存：

```bash
export HF_HOME="$PWD/.cache/huggingface"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero"
export VLA_CONFIG_YAML=libero_simulation.yaml
```

## 4. 下载 A1 backbone

```bash
make download-checkpoint
```

预期主文件：

```text
model/libero_exit/config.yaml
model/libero_exit/dataset_statistics.json
model/libero_exit/model.pt
```

V3 的 router、phase estimator 和 LIBERO-10 threshold 已在
`artifacts/phase_route_v3/`，不需要另找未固定的外部链接。主要 SHA：

```text
dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f  model.pt
9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830  final_router.pt
b601f8221d47136818d7a008eaf7cee06e1201bf514f371ae33f42cfb39515a1  phase_estimator.pt
a98d9e2c79d83846f5a778b52fc32b4803bdaf2a49aab5e3d961d2e624139796  LIBERO-10 threshold
```

完整清单见 `artifacts/MANIFEST.json` 与
`artifacts/phase_route_v3/MANIFEST.json`。

## 5. 分层验收

### 5.1 不需要 GPU/34 GB 权重

```bash
make preflight-v3
make test-v3-release
```

该门禁验证三个随仓库发布的 artifact、payload schema、five-head 数量、phase-state
hash、D9 formal result 和科学声明边界。

### 5.2 CUDA 与完整 backbone

先查看 GPU 0–3：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

选择空闲卡，只做 preflight：

```bash
GPU_INDEX=0 \
PREFLIGHT_ONLY=1 \
PYTHON_BIN=python \
make run-v3
```

这一步会流式计算 34 GB `model.pt` 的 SHA，并检查：

- physical GPU index 只能是 0–3；
- UUID 绑定后 `torch.cuda.device_count()==1`；
- visible GPU UUID 与物理卡一致；
- A1 model/config/statistics 与 V3 小模型全部匹配；
- A1、V3 和 LIBERO import 完整。

## 6. 跑一个完整输入—输出闭环

不要用 D9 test states 40–49 做调参。普通 smoke 可选择 task 0、state 0：

```bash
GPU_INDEX=0 \
TASK_IDS=0 \
EPISODE_INDICES=0 \
SEED=20260823 \
OUTPUT_ROOT=runs/phase_route_v3 \
make run-v3
```

也可以显式选择多个 task/state：

```bash
GPU_INDEX=0 \
TASK_IDS=0,2-4 \
EPISODE_INDICES=0,3 \
make run-v3
```

语义是 task 与 state 的笛卡尔积。launcher 固定以下研究合同：

```text
suite                 libero_10
backbone              frozen A1 early-exit checkpoint
FM inference steps    10
candidate interval    2
productive schedule   RP-PEP compatibility path
V3 decisions          L11 -> L13 -> L27
missing/error policy  fail closed to exact L27
```

运行目录包含：

| 文件 | 作用 |
|---|---|
| `preflight.json` | 环境、GPU、artifact、Git provenance |
| `command.sh` | 可重新输入的核心命令参数 |
| `stdout.log` | 完整终端输出 |
| `episode_logs/*.log` | 每个 task/state 的冻结 evaluator 退出层记录 |
| `policy_telemetry.jsonl` | 每个 policy call 的通用 early-exit telemetry |
| `phase_route_runtime.jsonl` | causal context 准备、risk/route、fallback/error |
| `evaluation_summary.json` | episode success 与 runtime 汇总 |
| `run_attestation.json` | non-overwrite 最终完整性判定 |

只有 runtime record 数量与 policy calls 对齐、全部 prepared/committed、L11/L13/L27
计数完备且 error 为 0，最终 attestation 才会 PASS。

## 7. 如何确认模型张量契约

当前实现不是旧文档中的 600-token/10-action/2-crop 简化图。正式输入输出是：

```text
A1 multimodal prefix    680 tokens
visual crops            5 = 4 valid + 1 padded
projected crop bank     [1, 5, 144, 3584]
normalized proprio      [1, 8]
candidate action        [1, 8, 7]
causal route context    82D
candidate pattern       15D
router feature          97D
selected action chunk   8 × 7
```

逐模块输入输出见 `docs/PHASEROUTE_ARCHITECTURE_ZH.md` 和
`docs/A1_PROJECT_READING_GUIDE_ZH.md`。

## 8. 运行全部回归测试

```bash
make test-v3-release
make test
make check
```

`make test` 覆盖历史模块和 V3 D0–D10 合约；`make check` 还执行依赖、Python/shell
语法和 whitespace 检查。

## 9. 历史 RP-PEP

如果需要复现旧 LIBERO Spatial baseline：

```bash
GPU_INDEX=0 NUM_EPISODES=1 make run-rp-pep
```

它使用固定 candidate pruning，不加载 V3 router。两套结果不能混为一次对照实验。

## 10. 训练与研究扩展

A1 backbone 微调：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_ENTITY=<entity> \
WANDB_PROJECT=phase-route-vla \
bash train_libero.sh
```

V3 five-head router 已经是用独立 development/calibration 训练出来的参数，并非仅靠 A1
权重即可得到。若改变 phase、history、head aggregation 或 gripper gate，必须为该 arm
重新训练 normalizer/router 并单独 calibration；不能在 D9 test 上置零 feature 后选择
“最好”的版本。协议见 `configs/research/v3/post_d9/d10_ablation_protocol.json`。

## 11. 常见问题

### LIBERO import 要求交互输入

```bash
export LIBERO_CONFIG_PATH="$PWD/.cache/libero"
make setup-libero
```

### `ModuleNotFoundError: libero`

```bash
make setup-libero
python -c "from libero.libero import benchmark; print('LIBERO PASS')"
```

### GitHub/Hugging Face 中断

重复原命令。下载脚本先验证现有字节与 SHA，已完成文件不会重复下载。

### preflight 报 artifact hash 不一致

不要用 `--no-check` 绕过。删除或移走损坏的外部 checkpoint 后重新下载；Git 内 V3
artifact 若变化，应恢复与当前 commit 一致的文件。runtime 会 fail closed，launcher
会在模型加载前退出。

### 指定 GPU 4–7 被拒绝

这是项目合同，不是 bug。只在 GPU 0–3 中选择空闲卡。

### 如何读取正式结论

```bash
python - <<'PY'
import json
d = json.load(open("results/v3/v3_d9_final_result.json"))
print(d["status"])
print(d["success"])
print(d["efficiency"])
print(d["safety"])
print(d["early_exit_failure_association"])
PY
```

论文表述以 `docs/RELEASE_STATUS_ZH.md` 的边界为准。
