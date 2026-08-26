# Route-first Stage 9 主动实验执行链

## 当前结论

Stage 9 的独立执行链已经完成并通过正式 `tests/` 门禁，但截至本记录生成时，state 12 和 state 13 均未打开，也没有执行主动控制。当前授权边界只推进到“可以在提交并绑定代码后运行无 episode preflight”。

## 为什么新增独立执行链

Stage 8 已证明 route-first 能在动作生成前根据 199D action-free context 选择 L13/L27，但离线重放不能证明真实闭环可执行，也不能提供可信的 wall-clock 证据。Stage 9 因而新增独立入口，并继续保持历史 D9 evaluator、A1 `ExitController` 和 V3 runtime/controller 的精确字节不变。

执行路径如下：

```mermaid
flowchart LR
    A[预注册协议与 frozen SHA] --> B[无 episode preflight]
    B --> C[加载 frozen A1 backbone]
    C --> D[构造 frozen sparse controller]
    D --> E[克隆为 RouteFirstExitController]
    E --> F[199D context 在动作生成前选择 L13/L27]
    F --> G[仅在选中层执行一次 10-step FM]
    G --> H[输出 8×7 动作块]
    H --> I[逐调用 telemetry/latency/action audit]
    I --> J[fail-closed run attestation]
```

## 新增边界

- `route_first_active_protocol.py` 在导入 LIBERO 前验证协议 SHA、frozen artifact SHA、state 访问范围、seed 和 paired arm 顺序。
- `run_libero_route_first_active.sh` 当前硬限制为 task 0/state 12/第二臂；任何 state 13 请求立即退出。
- `validate_route_first_active_preflight.py` 不创建环境、不 reset 模拟器、不读取 init state，只检查代码、依赖、backbone、GPU UUID、外部 GPU 进程和 BF16 CUDA smoke。
- `run_route_first_active.py` 通过 `RouteFirstExitController.from_frozen_sparse_controller` 建立旁路控制器，不修改三份 D9 保护文件。
- `validate_route_first_active_run.py` 不信任 runner 的聚合计数，而是逐条重读 runtime event：每个有效调用必须只有一个 `evaluated=True` 的退出事件、`fm_calls == 1`、动作有限、L11 次数为 0。

## 测量修正

旧的 Stage-1 probe 只认识 candidate-first adapter 的 `consider_candidate`。如果直接用于 route-first，会在安装测量钩子时失败并导致重复包装。现在 probe 会根据 `adapter.route_first` 选择接口：

- candidate-first：继续测量 `consider_candidate` / `select_fallback`；
- route-first：测量 `probabilities` / `select_action`；
- 两条路径的测量结果都不进入控制输入。

## 测试结果

- 针对性合约测试：15 passed，1 warning。
- 正式 `tests/` 门禁：524 passed，22 subtests passed，3 warnings，73.62 秒。
- Python 编译、Shell 语法和 `git diff --check`：全部通过。
- 三份 D9 保护文件与两份 Stage-8 frozen 文件 SHA：全部保持不变。

根目录直接执行 `pytest` 会在收集既有示例 `a1/data/vla/test_dataloader.py` 时失败，因为它仍传入当前 `DataConfig` 不接受的旧参数 `rlds_dataset_name`。该文件不属于维护中的 `tests/` 门禁，本阶段没有修改它，也不会把这项既有问题误报为 Stage-9 通过。

## 下一步的严格顺序

1. 提交并推送本执行链，使 preflight 可以要求 clean worktree 并绑定 Git commit。
2. 重新检查 8 张 GPU，仅选择无外部计算进程且利用率低的物理卡。
3. 运行 `PREFLIGHT_ONLY=1`，确认不打开任何 simulator episode。
4. preflight 通过后，才按预注册顺序运行 candidate-first V3 的 task 0/state 12。
5. V3 臂完整封存后，运行 route-first 的 task 0/state 12。
6. 两臂 runtime、动作与测量门禁都通过后，另行生成 paired smoke 汇总；在此之前 state 13 保持关闭。

这些结果只支持工程可执行性判断，不构成最终闭环提升、wall-clock 加速或部署结论。

## Candidate-first 第一臂补充封装

在正式打开 state 12 前，进一步新增了 `run_libero_route_first_stage9_candidate.sh` 和 `validate_route_first_stage9_candidate_arm.py`。原因是通用 V3 runner 只保证“可运行的非 D9 state”，不会单独证明它属于 Stage-9 预注册 smoke。

新的第一臂封装同时要求：

- Stage-9 no-episode preflight 与通用 V3 preflight 均为 PASS；
- 两份 preflight 绑定同一物理 GPU UUID；
- task/state/seed 精确为 0/12/20260826；
- candidate-first 是第一臂；
- runtime prepared/committed/policy calls 完全一致且无错误；
- Stage-1 动作与延迟测量逐调用完整。

对应新增测试后，正式门禁更新为 527 passed、22 subtests passed、3 warnings。测试完成时 8 张 GPU 仍均有外部计算进程，因此没有运行 preflight，也没有打开 state 12。

## Stage-9 首次 GPU preflight 结果

在提交 `eb9335f` 的 clean worktree 上进行了三次 no-episode 尝试：

1. GPU 6：初始只读检查为空闲，但外部调度器在 preflight 窗口内加载约 10 GB，Stage-9 门禁 FAIL；
2. GPU 5：Stage-9 preflight 与 V3 preflight 均 PASS，协议、权重、UUID、CUDA、空闲显存和 D9 字节全部正确；
3. GPU 7：外部调度器在 preflight 前加载约 22 GB，Stage-9 门禁 FAIL。

GPU 5 的两套 preflight 通过后、主动启动前，该卡又被外部进程占用约 25 GB，因此没有复用已经失去“当前空闲”条件的结果。三次尝试均未创建 LIBERO 环境、未读取 state 12，也没有生成 `evaluation_summary.json` 或主动臂 attestation。state 12 与 state 13 继续保持未打开。

该结果同时验证了竞争条件下的 fail-closed 行为：资源门禁失败不会退化成共享 GPU 上的低可信延迟实验，也不会用失败目录替换正式结果。

## Stage-9 后续 GPU 竞争复核

在提交 `614ea12` 的 clean worktree 上又执行了两次严格相同的 candidate-first 第一臂启动：

1. GPU 6 在只读审计时无计算 PID、仅占用 563 MiB；进入 Stage-9 preflight 前，外部 PID `2698411` 加载约 10 GiB，门禁 FAIL；
2. GPU 4 随后恢复到 18 MiB、0% 利用率且无计算 PID；进入 preflight 前，外部 PID `2811685` 同样加载约 10 GiB，门禁 FAIL。

两次失败均满足 `simulator_episode_opened=false`，没有运行 V3 preflight、没有生成主动臂 attestation，也没有读取 state 12。累计 6 次 Stage-9 preflight 中，1 次曾同时通过 Stage-9 与 V3 no-episode preflight，5 次因共享 GPU 竞争被拒绝；candidate-first 与 route-first 主动臂执行次数仍均为 0。

这进一步定位出当前阻塞是“初始 GPU 审计到 preflight 采样之间被外部调度器抢占”的资源竞争，而不是协议、backbone、CUDA smoke、保护文件或 frozen artifact 验证失败。在没有独占卡或能够与外部调度器协调的资源租约前，不通过降低 40 GiB 门槛、允许共享 PID 或修改预注册实验来绕过该问题。
