# PhaseRoute-VLA 中文复现指南

本指南从空环境开始复现正式 RP-PEP 路径。所有命令均在项目根目录执行；权重、缓存和运行结果不会提交到 Git。

## 1. 硬件与软件基线

已验证组合：

```text
Linux x86_64
Python 3.10
PyTorch 2.6.0 + CUDA 12.4
torchvision 0.21.0
NVIDIA driver 570.133.07
单卡显存约 48 GiB
```

正式 launcher 只允许物理 GPU 0–3。GPU 4–7 不会被使用。

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

如果 clone 时没有拉取 submodule，后续 `make setup-libero` 会补齐。

## 3. 创建环境

```bash
conda create -n phase-route-vla python=3.10 -y
conda activate phase-route-vla
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

安装项目与 LIBERO 依赖：

```bash
make install
make setup-libero
```

`make install` 会使用 `requirements/constraints-cu124.txt`，并从固定 Git commit 安装 `dlimp_openvla`。网络中断时可直接重试同一命令。

设置可复现的本地缓存位置：

```bash
export HF_HOME="$PWD/.cache/huggingface"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero"
export VLA_CONFIG_YAML=libero_simulation.yaml
```

`setup_libero.sh` 会非交互生成 `$LIBERO_CONFIG_PATH/config.yaml`，避免第一次 import 时等待输入。

## 4. 下载并校验权重

```bash
make download-checkpoint
```

预期文件：

```text
model/libero_exit/config.yaml
model/libero_exit/dataset_statistics.json
model/libero_exit/model.pt
model/libero_exit/exit_thresholds_libero_spatial_exp_1.0.json
```

手动复核：

```bash
sha256sum \
  model/libero_exit/model.pt \
  model/libero_exit/exit_thresholds_libero_spatial_exp_1.0.json
```

预期 SHA-256：

```text
dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f  model.pt
5b3a0ee9f3851bf1b0c7f7e2b28bc61898ed0b4bd39f8752007719e9f26d7bd6  exit_thresholds_libero_spatial_exp_1.0.json
```

## 5. 安装验收

CPU/文件级 preflight：

```bash
make preflight
```

CUDA smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(torch.cuda.get_device_name(0), "PASS")
PY
```

测试：

```bash
make test-release
make test
make check
```

## 6. 单卡正式运行

先做 1 episode/task 的完整 LIBERO Spatial 闭环验证：

```bash
GPU_INDEX=0 \
NUM_EPISODES=1 \
SEED=20260805 \
OUTPUT_ROOT=runs/rp_pep \
make run-rp-pep
```

launcher 会：

1. 拒绝 `GPU_INDEX=4..7`；
2. 查询物理卡 UUID，并用 UUID 设置 `CUDA_VISIBLE_DEVICES`；
3. 运行 checkpoint、阈值、冻结结果和可见 GPU 的 preflight；
4. 显式开启 `--rp_pep_enabled True`；
5. 把日志和评测结果写入带时间戳的新目录。

确认小规模运行正常后再增加：

```bash
GPU_INDEX=0 NUM_EPISODES=50 make run-rp-pep
```

## 7. 前四卡 release smoke

```bash
make smoke-front4
```

任务分片：

| 物理 GPU | task IDs |
|---:|---|
| 0 | 0, 4, 8 |
| 1 | 1, 5, 9 |
| 2 | 2, 6 |
| 3 | 3, 7 |

每个 worker 都记录并复核自己的物理 GPU UUID。汇总器要求 task 0–9 各出现一次、episode index/seed 一致、checkpoint SHA 一致且所有 worker 完整退出。

## 8. LIBERO 训练

准备：

- RLDS 数据：`data/libero_rlds`；
- A1 预训练 checkpoint：`model/pretrain`；
- 可选 W&B 配置。

使用前四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_ENTITY=<your-entity> \
WANDB_PROJECT=phase-route-vla \
bash train_libero.sh
```

训练输出默认进入 `model/checkpoints/`，不会提交 Git。训练配置是 `configs/experiments/libero_simulation.yaml`，动作输出契约为 `10×32`，LIBERO 数据仅监督前 7 维。

## 9. 常见问题

### GitHub 或 Hugging Face 超时

重复原命令即可。checkpoint 脚本会先校验现有文件，已完成且 hash 正确时不会重复下载。

### LIBERO 首次 import 要求交互输入

重新运行：

```bash
export LIBERO_CONFIG_PATH="$PWD/.cache/libero"
make setup-libero
```

### `ModuleNotFoundError: libero`

```bash
make setup-libero
python -c "from libero.libero import benchmark; print('LIBERO PASS')"
```

### 依赖被 pip 升级后冲突

```bash
python -m pip install -e ".[libero]" \
  -c requirements/constraints-cu124.txt
python -m pip check
```

不要安装 `transformers 5.x`、`numpy 2.x` 或 `mujoco 3.x` 来覆盖冻结环境。

### preflight 缺少 checkpoint

这是预期的安全失败。执行 `make download-checkpoint`，或将已验证的 checkpoint 放到 `model/libero_exit/` 后重试。

## 10. 如何核对论文结论

```bash
python - <<'PY'
import json
d = json.load(open("results/rp_pep_paired.json"))
print("status:", d["status"])
print("paired episodes:", d["paired_episodes"])
print("successes:", d["baseline_successes"], d["rp_pep_successes"])
print("FM reduction:", d["fm_solver_calls"]["reduction_fraction"])
print("mean latency reduction:", d["policy_latency"]["weighted_mean_reduction_fraction"])
print("equivalence:", d["equivalence"])
PY
```

科学声明必须以 `results/README.md` 中的边界为准。
