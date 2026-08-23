# PhaseRoute V3 artifacts

本目录包含两类不可变发布材料：

- `final_router.pt`、`phase_estimator.pt` 和 LIBERO-10 threshold 定义正式 V3
  research runtime；
- D8 生成状态、D8A/D8B 结果及 `legacy_source/` 只服务于自包含历史回归测试。

第二类材料共约 2.7 MB，不含 rollout、视频、图像、teacher cache 或 hidden state。
`legacy_source/` 保留 `docs/research/v3/legacy_evidence_manifest.json` 所列的原始
source-relative 布局，另含 D1 所绑定的 C3.55 结果。所有路径、字节数和 SHA-256
登记在本目录的 `MANIFEST.json`；历史 provenance 中的绝对路径不作为运行依赖。

34 GB A1 backbone 不进入 Git，由 manifest 中的 revision 和 SHA 单独验证。
