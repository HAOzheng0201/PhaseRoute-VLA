#!/usr/bin/env bash
set -uo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

repo_root="${1:-${repo_default}}"
output_root="${2:-${repo_root}/reports/m2_phase_collection_20260801}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "${output_root}"
pids=()
for gpu in 0 1 2 3; do
  task_id="${gpu}"
  task_output="${output_root}/gpu${gpu}_task${task_id}"
  mkdir -p "${task_output}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DATA_DIR="${repo_root}" \
    HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    HF_HUB_OFFLINE="1" \
    TRANSFORMERS_OFFLINE="1" \
    VLA_CONFIG_YAML="libero_simulation.yaml" \
    MUJOCO_GL="egl" \
    PYOPENGL_PLATFORM="egl" \
    TF_CPP_MIN_LOG_LEVEL="3" \
    PYTHONUNBUFFERED="1" \
    "${python_bin}" \
    "${repo_root}/scripts/dynamic_compute/collect_m2_phase_cache.py" \
    --checkpoint "${repo_root}/model/libero_exit" \
    --task-suite libero_spatial \
    --task-id "${task_id}" \
    --num-episodes 5 \
    --seed 20260801 \
    --fm-steps 10 \
    --summary-dtype float16 \
    --output-dir "${task_output}" \
    >"${task_output}/console.log" 2>&1 &
  pids+=("$!")
  echo "started gpu=${gpu} task=${task_id} pid=${pids[-1]} output=${task_output}"
done

status=0
for pid in "${pids[@]}"; do
  if wait "${pid}"; then
    echo "completed pid=${pid} status=0"
  else
    code=$?
    echo "completed pid=${pid} status=${code}"
    status=1
  fi
done
exit "${status}"
