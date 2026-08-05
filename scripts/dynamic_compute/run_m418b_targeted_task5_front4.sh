#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [EPISODE_START] [EPISODES_PER_SHARD] [REPO_ROOT] [SEED]}"
episode_start="${2:-3}"
episodes_per_shard="${3:-12}"
repo_root="${4:-${repo_default}}"
seed="${5:-20260804}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"

if [[ ! "${episode_start}" =~ ^[0-9]+$ ]]; then
  echo "EPISODE_START must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "${episodes_per_shard}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES_PER_SHARD must be a positive integer" >&2
  exit 2
fi
if (( episode_start + 2 * episodes_per_shard > 50 )); then
  echo "Requested task5 episode window exceeds the 50 default initial states" >&2
  exit 2
fi

declare -a policies=(early_exit early_exit full_depth full_depth)
declare -a gpus=(0 1 2 3)
declare -a starts=(
  "${episode_start}"
  "$((episode_start + episodes_per_shard))"
  "${episode_start}"
  "$((episode_start + episodes_per_shard))"
)
declare -a shards=(shard0 shard1 shard0 shard1)

for gpu in 0 1 2 3; do
  expected_uuid="$(
    nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader
  )"
  actual_uuid="$(
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONNOUSERSITE=1 \
      "${PYTHON_BIN:-python}" \
      -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)'
  )"
  if [[ "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for requested physical GPU${gpu}: expected ${expected_uuid}, got ${actual_uuid}" >&2
    exit 3
  fi
  echo "verified physical_gpu=${gpu} uuid=${expected_uuid} visible_device_count=1"
done

for index in 0 1 2 3; do
  run_dir="${output_root}/${policies[$index]}/${shards[$index]}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/episodes.jsonl" || -e "${run_dir}/eval_logs" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
done

pids=()
for index in 0 1 2 3; do
  policy="${policies[$index]}"
  gpu="${gpus[$index]}"
  start="${starts[$index]}"
  shard="${shards[$index]}"
  run_dir="${output_root}/${policy}/${shard}"
  mkdir -p "${run_dir}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DATA_DIR="${repo_root}" \
    HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLA_CONFIG_YAML=libero_simulation.yaml \
    TF_CPP_MIN_LOG_LEVEL=3 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN:-python}" \
    "${repo_root}/scripts/dynamic_compute/collect_m418_persistent_tasks.py" \
    --policy "${policy}" \
    --checkpoint "${repo_root}/model/libero_exit" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --task-suite libero_spatial \
    --task-id 5 \
    --episode-start-index "${start}" \
    --num-episodes "${episodes_per_shard}" \
    --seed "${seed}" \
    --fm-steps 10 \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  echo "started policy=${policy} task=5 episodes=${start}-$((start + episodes_per_shard - 1)) physical_gpu=${gpu} pid=${pids[-1]}"
done

status=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "complete policy=${policies[$index]} ${shards[$index]}"
  else
    echo "failed policy=${policies[$index]} ${shards[$index]}; inspect stdout.log" >&2
    status=1
  fi
done
exit "${status}"
