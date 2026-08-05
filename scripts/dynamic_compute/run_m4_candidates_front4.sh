#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

repo_root="${1:-${repo_default}}"
output_root="${2:-${repo_root}/reports/m4_candidates_spatial_task0_20260802}"
num_episodes="${3:-3}"
task_id="${4:-0}"

mkdir -p "${output_root}"
modes=(baseline pool144_keep144 pool144_keep128 select288_keep128)
budgets=(0 144 128 128)
banks=(144 144 144 288)
pids=()

for gpu in 0 1 2 3; do
  mode="${modes[$gpu]}"
  keep_tokens="${budgets[$gpu]}"
  bank_tokens="${banks[$gpu]}"
  run_dir="${output_root}/${mode}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/policy_calls.jsonl" ]]; then
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
    "${repo_root}/scripts/dynamic_compute/collect_m4_static_task.py" \
    --checkpoint "${repo_root}/model/libero_exit" \
    --task-suite libero_spatial \
    --task-id "${task_id}" \
    --num-episodes "${num_episodes}" \
    --seed 20260802 \
    --fm-steps 10 \
    --keep-tokens "${keep_tokens}" \
    --bank-tokens "${bank_tokens}" \
    --min-tokens-per-crop 4 \
    --variant-name "${mode}" \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  echo "started mode=${mode} physical_gpu=${gpu} pid=${pids[-1]}"
done

status=0
for index in 0 1 2 3; do
  if wait "${pids[$index]}"; then
    echo "complete mode=${modes[$index]}"
  else
    echo "failed mode=${modes[$index]}" >&2
    status=1
  fi
done
exit "${status}"
