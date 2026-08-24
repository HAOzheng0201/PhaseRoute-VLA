# V3 D12：全新依赖环境验收

日期：2026-08-24  
状态：**PASS**  
范围：PhaseRoute-VLA 0.1.0 的安装、CPU release gate、LIBERO 数据读取与 Python 发布包  
不在范围：新的 GPU rollout、性能对比、真实机器人部署

## 1. 验收目的

D11 已证明 clean Git clone 可以在既有 `a1` conda 环境中复现 release gate，但没有
排除旧环境残留依赖的影响。本阶段新建独立 venv，从网络重新安装项目依赖，并验证：

- 项目约束能够解析为一致环境；
- pinned LIBERO 能在当前 setuptools 下被真正导入，而非只有 distribution metadata；
- LIBERO-10 benchmark 与 official init-state 可以读取；
- 动态计算全量测试、V3 release gate、sdist 和 wheel 都可复现；
- 整个资格验收不使用 GPU，也不改变冻结的科学结果。

## 2. 隔离条件与 provenance

```text
repository             /data3/haozheng/A1/PhaseRoute-VLA
qualification commit   d2853ad00a1f66381adcfcce7c29fab5117cab52
LIBERO commit           8f1084e3132a39270c3a13ebe37270a43ece2a01
venv                    .cache/qualification/fresh-venv-20260824
Python                  3.10.8
pip                     26.2.1
setuptools              84.0.0
CUDA visibility         empty
user site-packages      disabled
```

所有命令显式使用该 venv 的 Python；pytest 与 preflight 还设置
`PYTHONNOUSERSITE=1`。最终 preflight 记录 `worktree_dirty=false`、
`physical_gpu_index=null`、`cuda.required=false`。

## 3. 从零安装流程

实际验收按以下顺序执行；路径只表示本机证据位置，用户可改为自己的 venv：

```bash
python3.10 -m venv .cache/qualification/fresh-venv-20260824
FRESH=.cache/qualification/fresh-venv-20260824/bin/python

"$FRESH" -m pip install --upgrade pip setuptools wheel
"$FRESH" -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
make PYTHON="$FRESH" install

PYTHON_BIN="$FRESH" \
LIBERO_PATCHED_ROOT=.cache/qualification/libero-patched-20260824 \
LIBERO_CONFIG_PATH=.cache/qualification/libero-config-20260824 \
bash scripts/setup_libero.sh
```

随后安装 `.[dev]` 以执行构建与 twine 检查。`pip check` 最终为 PASS。关键版本：

| package | version | package | version |
|---|---:|---|---:|
| torch | 2.6.0+cu124 | torchvision | 0.21.0+cu124 |
| numpy | 1.25.0 | transformers | 4.53.2 |
| tokenizers | 0.21.4 | datasets | 3.6.0 |
| tensorflow | 2.15.0 | mujoco | 2.3.7 |
| robosuite | 1.4.1 | bddl | 1.0.1 |
| gym | 0.25.2 | dlimp | 0.0.1 |
| rich | 13.9.4 | twine | 6.2.0 |

## 4. 安装问题与修复

### 4.1 LIBERO distribution 存在但无法 import

固定 commit 的 `setup.py` 使用旧式包发现逻辑。setuptools 84 可以成功生成
`libero==0.1.0` editable metadata，却没有正确安装 `libero` namespace，表现为：

```text
ModuleNotFoundError: No module named 'libero'
```

修复由 `patches/libero-setuptools-editable.patch` 提供，显式声明顶层 `libero` 包并从
内层源码目录发现子包。`scripts/setup_libero.sh` 不修改 submodule，而是：

```text
pinned submodule (read-only source)
        │ copy + commit stamp
        ▼
.cache/libero-build (ignored isolated copy)
        │ apply PyTorch 2.6 + setuptools patches
        ▼
editable install into selected Python environment
```

脚本会拒绝没有正确 source-commit stamp 的旧 cache，避免静默复用来源不明的副本。
benchmark config 仍指向 pinned source 中的 assets、BDDL 与 init-state，保证数据路径稳定。

### 4.2 `twine` 与 `cached_path` 的依赖冲突

`twine 7` 要求 `rich>=14.3`，而 `cached_path 1.8.10` 要求 `rich<14`。开发依赖因此
固定为 `twine>=1.11.0,<7`，实际解析为 twine 6.2.0 与 rich 13.9.4；`pip check` 通过。

## 5. 验收结果

| 门禁 | 结果 | 证据摘要 |
|---|---|---|
| dependency consistency | PASS | `pip check` 无冲突 |
| A1 / LIBERO / dlimp import | PASS | 所有模块可导入 |
| LIBERO benchmark | PASS | `libero_10` 为 10 tasks |
| official init-state | PASS | 每 task 50 states；sample shape `(123,)` |
| dynamic-compute tests | PASS | 478 passed + 22 subtests |
| V3 release tests | PASS | 13 passed |
| repository checks | PASS | `make check` |
| CPU V3 preflight | PASS | clean commit；无可见 GPU |
| Python build | PASS | sdist + wheel |
| package metadata | PASS | `twine check --strict` 两个产物均通过 |
| wheel container | PASS | ZIP integrity |

最终 CPU preflight：

```text
.cache/qualification/final_preflight.json
sha256 af1e2c6b519e88454b87a74190ba88d7a2f2f7e51edcf956dff791cce5cdcda1
```

该 JSON 为本地、被 Git 忽略的运行证据；其内部记录 qualification commit、包版本、
release artifact SHA、全部 gate 与 `worktree_dirty=false`。

在代码提交 `d2853ad` 上生成的资格验收构建产物：

```text
772bb31c5eee33b326b59b7fd05c62fb7d979075b1bd59af5dfabd0b3d28901d  phase_route_vla-0.1.0-py3-none-any.whl
cba39e471e6959e8a8be03d10ec24ec6190c7c2376f4e0a00d2f0f0b6e9f723c  phase_route_vla-0.1.0.tar.gz
```

这些 SHA 对应 D12 资格验收快照，而不是承诺未来同版本重建具备字节级 reproducible
build；若 README、构建工具或归档时间变化，sdist/wheel 字节可能变化。

## 6. 非失败警告

CPU preflight 导入 TensorFlow 时输出 cuDNN/cuFFT/cuBLAS factory registration 与
TensorRT warning。这些是 cu124 环境加载 TensorFlow 插件时的 stderr 信息；门禁进程
设置 `CUDA_VISIBLE_DEVICES=`，输出 JSON 记录 `physical_gpu_index=null`，所有检查均
通过。Python 3.10 生命周期与 pydantic schema warning 同样不影响本次冻结环境验收。

## 7. 科学边界

本阶段证明的是“新机器按文档安装后，代码与仿真依赖可以被完整加载和验证”，不是
“PhaseRoute V3 获得新的成功率或速度提升”。没有加载 34 GB backbone，没有执行新的
episode，也没有重新选择阈值或训练 router。因此：

- D9 的 88% vs 85% 与 36.58% FM-call reduction 保持不变；
- stage-5 state-0 的 9/10 vs 10/10 与 34.30% FM-call reduction 保持不变；
- 不新增 wall-clock speedup、统计显著性或真实机器人结论。
