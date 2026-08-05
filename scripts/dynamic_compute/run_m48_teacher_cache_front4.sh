#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

repo_root="${1:-${repo_default}}"
output_root="${2:-${repo_root}/reports}"
seed="${3:-20260804}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
pids=()

for gpu in 0 1 2 3; do
  task_id="${gpu}"
  run_dir="${output_root}/m48_teacher_cache_v3_spatial_task${task_id}_1ep_20260802_v1"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/teacher_calls/manifest.jsonl" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
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
    "${PYTHON_BIN:-python}" \
    "${repo_root}/scripts/dynamic_compute/collect_m46_teacher_cache_task.py" \
    --checkpoint "${repo_root}/model/libero_exit" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --task-suite libero_spatial \
    --task-id "${task_id}" \
    --num-episodes 1 \
    --seed "${seed}" \
    --fm-steps 10 \
    --feature-dtype float16 \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  echo "started task=${task_id} physical_gpu=${gpu} pid=${pids[-1]}"
done

status=0
for task_id in 0 1 2 3; do
  if wait "${pids[$task_id]}"; then
    echo "complete task=${task_id}"
  else
    echo "failed task=${task_id}; inspect stdout.log" >&2
    status=1
  fi
done
exit "${status}"
