#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT] [SEED]}"
repo_root="${2:-${repo_default}}"
seed="${3:-20260804}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
tasks=(4 5 6 7 8 9)

mkdir -p "${output_root}"
for task_id in "${tasks[@]}"; do
  run_dir="${output_root}/task${task_id}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/policy_calls.jsonl" || -e "${run_dir}/teacher_calls/manifest.jsonl" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
done

status=0
for batch_start in 0 4; do
  pids=()
  batch_tasks=()
  for offset in 0 1 2 3; do
    task_index=$((batch_start + offset))
    if (( task_index >= ${#tasks[@]} )); then
      break
    fi
    task_id="${tasks[$task_index]}"
    gpu="${offset}"
    run_dir="${output_root}/task${task_id}"
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
    batch_tasks+=("${task_id}")
    echo "started teacher-cache task=${task_id} physical_gpu=${gpu} pid=${pids[-1]}"
  done

  for index in "${!pids[@]}"; do
    task_id="${batch_tasks[$index]}"
    if wait "${pids[$index]}"; then
      echo "complete teacher-cache task=${task_id}"
    else
      echo "failed teacher-cache task=${task_id}; inspect task${task_id}/stdout.log" >&2
      status=1
    fi
  done
  if (( status != 0 )); then
    exit "${status}"
  fi
done

exit "${status}"
