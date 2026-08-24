# V3 D13：v0.1.0-rc1 wheel 独立安装验收

日期：2026-08-24  
状态：**PASS**  
候选标签：`v0.1.0-rc1`  
范围：最终构建、第二个空 venv、仓库外 wheel smoke 与候选版本冻结  
不在范围：新 GPU rollout、重新训练、阈值选择、真实机器人部署

## 1. 目标

D12 已验证从源码 editable install 的全新环境。本阶段进一步排除工作区源码遮蔽：

1. 从 clean HEAD 重新构建 wheel 与 sdist；
2. 新建第二个空 Python 3.10.8 venv，不复用 D12 site-packages；
3. 从 wheel 安装 `phase-route-vla==0.1.0` 与冻结 LIBERO 依赖；
4. 切换到 `/tmp`，确认 `a1` 确实来自 wheel 环境的 site-packages；
5. 用 wheel 中的代码读取 LIBERO init-state 并验证 Git release tree 的 V3 bundle；
6. 固定构建哈希和本地 annotated rc1 tag。

所有 smoke 命令均设置 `CUDA_VISIBLE_DEVICES=` 与 `PYTHONNOUSERSITE=1`。

## 2. 构建门禁

最终 rc1 构建要求：

```text
python -m build --no-isolation     PASS
twine check --strict (wheel)       PASS
twine check --strict (sdist)       PASS
wheel ZIP integrity                PASS
```

精确 commit、wheel/sdist SHA-256 和 smoke 结论写入 annotated tag
`v0.1.0-rc1`；本地同样保存在被 Git 忽略的 rc1 qualification 目录。

## 3. 第二个空环境

```text
venv       .cache/qualification/rc1-wheel-venv-20260824
Python     3.10.8
PyTorch    2.6.0+cu124
NumPy      1.25.0
LIBERO     0.1.0 (isolated patched copy)
dlimp      0.0.1
GPU        hidden / visible count 0
pip check  PASS
```

LIBERO 使用新的：

```text
patched root  .cache/qualification/rc1-libero-patched-20260824
config root   .cache/qualification/rc1-libero-config-20260824
source commit 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

原 submodule 未被修改。

## 4. 网络中断与等价恢复

wheel metadata 把 dlimp 固定为：

```text
https://github.com/moojink/dlimp_openvla.git
040105d256bd28866cc6620621a3d5f7b6b91b46
```

GitHub smart-HTTP 连续两次出现 `gnutls_handshake()` TLS termination。没有改用浮动
branch 或其他版本，而是从 GitHub codeload 下载同一 commit tarball：

```text
bb92a601f7eeafc764d910b592bced13bcab4bc1140142d87cf0cecb1869aab1  dlimp-040105d.tar.gz
```

tarball 顶层目录包含完整 40-character commit，构建结果为 `dlimp 0.0.1`。其余依赖
继续按 wheel metadata 与 `requirements/constraints-cu124.txt` 解析，最终 `pip check`
无缺失或冲突。

## 5. 仓库外 smoke 结果

执行目录为 `/tmp`，因此源码根目录不会进入 Python import path。结果：

```text
WHEEL_INSTALL_SMOKE=PASS
phase_route_vla= 0.1.0
a1_path= .../rc1-wheel-venv-20260824/lib/python3.10/site-packages/a1/__init__.py
torch= 2.6.0+cu124
visible_gpu_count= 0
libero_tasks= 10
state_shape= (50, 123)
eligible_exit_layers= (3, 11, 13, 27)
v3_release= PASS
router_heads= 5
```

第一次 smoke 曾因验收命令漏传 RP-PEP 的 14 个原始 A1 exit layers 而得到
`TypeError`；补齐固定列表后通过。该错误发生在测试调用点，不是 wheel、依赖或模型
实现失败，未通过放宽断言规避。

## 6. wheel 与 Git release 的边界

内容审计显示 wheel 约 1.2 MB，只包含 `a1` Python package 和 distribution metadata，
不包含：

- `artifacts/phase_route_v3/` 的 router、phase estimator 与 threshold；
- `results/v3/` 的冻结 formal result；
- `configs/`、launcher、验证脚本与研究文档；
- LIBERO submodule 与 34 GB A1 backbone。

因此发布合同是：

```text
wheel                         Python code installation
Git tag/source tree           complete auditable research release
external pinned checkpoint    full GPU rollout prerequisite
```

本次 smoke 从 wheel 导入 V3 validation/runtime code，再让它验证 Git source tree 中的
冻结 artifacts 和 historical evidence。这证明 wheel 代码可被干净安装、并与 tag bundle
协同工作；不声称 wheel 单文件可独立运行 PhaseRoute V3。

## 7. 结论边界

本阶段没有使用 GPU、没有加载 backbone、没有执行 episode，也没有改变任何训练参数或
科学结果。D9 和 stage-5 的成功率、FM-call reduction 与既有限制保持不变。rc1 标签只
冻结已通过的工程复现状态，不把候选版本等同于正式论文结论或真实机器人发布。

