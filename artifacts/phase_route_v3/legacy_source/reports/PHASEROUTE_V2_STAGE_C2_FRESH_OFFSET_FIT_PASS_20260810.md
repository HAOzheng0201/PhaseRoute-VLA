# PhaseRoute-v2 Stage-C2：fresh offset-fit 结果

实验 lineage 日期：2026-08-10  
正式拟合执行时间：2026-08-11（Asia/Shanghai）  
最终状态：`RECOVERY_OFFSET_FIT_PASS`  
运行边界：仅使用物理 GPU 0；GPU 4--7 未使用；episode 15--29 未访问；episode
40--49 尚未访问；runtime 尚未接入。

## 1. 结论

Stage-C2 独立恢复实验已完成 fresh offset-fit。冻结控制器权重来自 seed `20260823`，
checkpoint SHA-256 为：

```text
aed39f61295fc5adf5ae4780c3f6bff9d62a7401b2c0ed6dac5757258990556d
```

在 episode 30--39 的 1,345 条 fresh policy-call 记录上，按预注册网格
`[0, 8]`、步长 `0.05` 扫描后，首个通过点为：

| 指标 | 结果 | 冻结条件 |
|---|---:|---:|
| offset | 3.05 | 首个合格网格点 |
| underbudget | 18/1345 = 1.3383% | 经验值 <= 2% |
| 单侧 95% Clopper--Pearson 上界 | 1.9781% | <= 2% |
| 平均硬宽度 | 279.6253 | < 288 |
| full-width rate | 84.2379% | 诊断项 |
| accuracy | 35.1673% | 诊断项 |
| 反事实风险违例 | 0 | = 0 |

前一个网格点 `3.00` 的经验 underbudget 为 `19/1345 = 1.4126%`，但其单侧
95% 上界为 `2.0660%`，因此未通过。`3.05` 确为冻结规则下的最小合格点，未进行
事后扩大或修改搜索网格。

本结果只授权准备并执行一次 episode 40--49 fresh Gate：

```text
fresh_one_time_gate_authorized: true
fresh_one_time_gate_accessed: false
runtime_integrated: false
post_gate_tuning_authorized: false
sealed_test_accessed: false
```

它不等价于 Gate 已通过，也不授权 runtime integration。

## 2. fresh 数据与安全标签

fresh width-cache 的四卡重放和合并均已通过，随后复用预注册的 Stage-C v1 动作误差
阈值生成安全标签；没有在 episode 30--39 上重新拟合阈值。

```text
records: 1345
tasks: 0--9
episodes: 30--39
target widths 192/224/256/288: 519/139/291/396
mean target width: 237.4186
forbidden controller input fields present: none
all numeric controller inputs finite: true
```

关键数据 SHA-256：

```text
fresh width-cache arrays:
d91ede5e3d59e87ab6aff25e03ac646036e7c55068b6fbe649421455c07414e7

fresh width-cache records:
49253f2e9b91ff319038e6bf86695670f4a9d07e695b530634bd5daff0e3d3ff

fresh controller dataset arrays:
d74ccb9a2b6c4caaf7d8f08dc96c91f416c971d978ee13a4992d9dde81abf5ef

fresh controller dataset records:
e4282549e40fee4c84b63506c8029153bab3364df374ce632c11b008a7db264d
```

## 3. 权重与风险约束审计

恢复 checkpoint 仅替换旧 offset，没有改变控制器权重。独立逐 tensor 比较结果为：

```text
checkpoint_weights_unchanged: true
controller_state_dict bitwise equal: true
legacy offset discarded: 2.200000047683716
fresh offset: 3.0500000000000003
```

反事实检查保持同一条记录的 instruction、phase embedding、progress、proprio 和 history
不变，仅将 boundary/uncertainty 从 0 提升到 1。三组干预均为：

```text
continuation probability violation count: 0/1345
expected width violation count: 0/1345
minimum expected-width delta: +0.02298
mean expected-width delta: +7.02327
counterfactual risk monotonicity: PASS
```

## 4. 验证记录

针对 recovery 统计协议和 controller dataset 的单测：

```text
14 passed, 1 warning in 42.10s
```

warning 仅为 Python 3.10 后续支持期限提示，没有测试失败。

正式结果与 checkpoint：

```text
reports/phase_route_v2_stage_c2_fresh_offset_fit_20260810_v1/result.json
SHA-256: 9d069d70c0f5603983a5216b0d94ff944926714ede2c214aa0cf0ebbf6e001c9

reports/phase_route_v2_stage_c2_fresh_offset_fit_20260810_v1/recovery_width_controller.pt
SHA-256: 53a36a91293f85bb5501ba2ea33b3e9a87779f815f33a93dae85095ddeadd409
```

恢复协议 SHA-256：

```text
c0c4d9a44d6d6ada0b1fd6fab5066e2e2b60db0ec3715069dbdc01a5271353af
```

## 5. 下一阶段边界

下一阶段只能使用冻结的 recovery checkpoint 和 offset `3.05`，对 episode 40--49
执行一次 fresh Gate。Gate 必须同时满足经验 underbudget 不超过 2%、单侧 95% 精确
上界不超过 2%、平均硬宽度小于 288、反事实风险单调性通过。

Gate 完成后不得根据结果修改模型、offset、阈值或选择规则并重测。只有 Gate 明确通过，
才可以另行审查 runtime integration；项目级 sealed test episode 20--29 仍然禁止访问。
