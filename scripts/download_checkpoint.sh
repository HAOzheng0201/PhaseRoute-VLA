#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="${CHECKPOINT_DIR:-${repo_root}/model/libero_exit}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
python_bin="${PYTHON_BIN:-python}"
hf_repo="spatialtemporal-ai/a1-libero-exit"
revision="a014b84203c6fb981d3f6181dc3bc7207610b2a3"
tokenizer_repo="Qwen/Qwen2-7B"
tokenizer_revision="453ed1575b739b5b03ce3758b23befdb0967f40e"
model_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
config_sha="9365d0a6ca6379a77877aaf46e170a7945f084c359560463edc14726965b04ca"
statistics_sha="6ec6ef68d0d5bae4cb5f9fc9acb715a22b9f4545e9e9b300d0d88695cd7afec3"
threshold_sha="5b3a0ee9f3851bf1b0c7f7e2b28bc61898ed0b4bd39f8752007719e9f26d7bd6"
model_bytes=33841175207
config_bytes=8369
statistics_bytes=11871
threshold_bytes=241

verify() {
  [[ -f "${checkpoint_dir}/model.pt" ]] || return 1
  [[ -f "${checkpoint_dir}/config.yaml" ]] || return 1
  [[ -f "${checkpoint_dir}/dataset_statistics.json" ]] || return 1
  [[ -f "${checkpoint_dir}/exit_thresholds_libero_spatial_exp_1.0.json" ]] || return 1
  [[ "$(stat -c '%s' "${checkpoint_dir}/model.pt")" == "${model_bytes}" ]] || return 1
  [[ "$(stat -c '%s' "${checkpoint_dir}/config.yaml")" == "${config_bytes}" ]] || return 1
  [[ "$(stat -c '%s' "${checkpoint_dir}/dataset_statistics.json")" == "${statistics_bytes}" ]] || return 1
  [[ "$(stat -c '%s' "${checkpoint_dir}/exit_thresholds_libero_spatial_exp_1.0.json")" == "${threshold_bytes}" ]] || return 1
  [[ "$(sha256sum "${checkpoint_dir}/model.pt" | awk '{print $1}')" == "${model_sha}" ]] || return 1
  [[ "$(sha256sum "${checkpoint_dir}/config.yaml" | awk '{print $1}')" == "${config_sha}" ]] || return 1
  [[ "$(sha256sum "${checkpoint_dir}/dataset_statistics.json" | awk '{print $1}')" == "${statistics_sha}" ]] || return 1
  [[ "$(sha256sum "${checkpoint_dir}/exit_thresholds_libero_spatial_exp_1.0.json" | awk '{print $1}')" == "${threshold_sha}" ]]
}

verify_tokenizer() {
  HF_HOME="${hf_home}" TRANSFORMERS_OFFLINE=1 \
    "${python_bin}" -c \
    'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", local_files_only=True)' \
    >/dev/null 2>&1
}

if verify; then
  echo "Checkpoint is already complete and verified: ${checkpoint_dir}"
else
  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required. Install huggingface_hub first." >&2
    exit 2
  fi

  mkdir -p "${checkpoint_dir}"
  HF_HOME="${hf_home}" HF_HUB_OFFLINE=0 hf download "${hf_repo}" \
    model.pt \
    config.yaml \
    dataset_statistics.json \
    exit_thresholds_libero_spatial_exp_1.0.json \
    --revision "${revision}" \
    --local-dir "${checkpoint_dir}"

  if ! verify; then
    echo "Checkpoint verification failed; expected hashes are in artifacts/MANIFEST.json." >&2
    exit 1
  fi
  echo "Checkpoint download and SHA-256 verification complete."
fi

if verify_tokenizer; then
  echo "Qwen2 tokenizer cache is already complete."
else
  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required to cache the frozen tokenizer." >&2
    exit 2
  fi
  HF_HOME="${hf_home}" HF_HUB_OFFLINE=0 hf download "${tokenizer_repo}" \
    config.json \
    generation_config.json \
    merges.txt \
    tokenizer.json \
    tokenizer_config.json \
    vocab.json \
    --revision "${tokenizer_revision}" >/dev/null
  if ! verify_tokenizer; then
    echo "Qwen2 tokenizer verification failed." >&2
    exit 1
  fi
  echo "Qwen2 tokenizer download and offline verification complete."
fi

if ! verify; then
  echo "Final checkpoint verification failed." >&2
  exit 1
fi
echo "All runtime artifacts are ready."
