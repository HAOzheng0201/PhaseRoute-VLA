# Configurations

当前发布聚焦 A1 + LIBERO。

```text
configs/
├── experiments/
│   ├── libero_simulation.yaml   # 正式 LIBERO 训练组合配置
│   └── pretrain.yaml            # 上游 A1 预训练组合配置
├── models/
│   ├── libero.yaml              # horizon=10, fixed action dim=32
│   └── pretrain.yaml
└── datasets/
    ├── libero_4_tasks.yaml       # 四个 LIBERO suite RLDS 训练集
    ├── libero_spatial.yaml       # Spatial-only 研究配置
    └── pretrain.yaml
```

组合配置引用 model 与 dataset 配置：

```yaml
model_config: models/libero.yaml
dataset_config: datasets/libero_4_tasks.yaml
```

`VLA_CONFIG_YAML` 接收相对于 `configs/experiments/` 的文件名：

```bash
export VLA_CONFIG_YAML=libero_simulation.yaml
```

LIBERO 默认数据位置为 `data/libero_rlds`。可以修改 dataset YAML 的 `path`，但不要把个人绝对路径提交到仓库。

RoboChallenge、Dobot 和 VLABench 配置属于上游 A1 的其他任务，不参与 PhaseRoute-VLA 发布，因此没有复制到当前项目。
