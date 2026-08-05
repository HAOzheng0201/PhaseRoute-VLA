#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

policy="${1:?usage: $0 POLICY OUTPUT_ROOT GPU_A GPU_B [EPISODES] [REPO_ROOT] [SEED]}"
output_root="${2:?usage: $0 POLICY OUTPUT_ROOT GPU_A GPU_B [EPISODES] [REPO_ROOT] [SEED]}"
gpu_a="${3:?GPU_A is required}"
gpu_b="${4:?GPU_B is required}"
episodes="${5:-3}"
repo_root="${6:-${repo_default}}"
seed="${7:-20260804}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"

if [[ "${policy}" != "early_exit" && "${policy}" != "full_depth" ]]; then
  echo "POLICY must be early_exit or full_depth" >&2
  exit 2
fi
if [[ ! "${gpu_a}" =~ ^[0-3]$ || ! "${gpu_b}" =~ ^[0-3]$ ]]; then
  echo "GPU ids must be physical GPU0–3" >&2
  exit 2
fi
if [[ "${gpu_a}" == "${gpu_b}" ]]; then
  echo "GPU_A and GPU_B must differ" >&2
  exit 2
fi

mkdir -p "${output_root}"
for shard in 0 1; do
  run_dir="${output_root}/shard${shard}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/episodes.jsonl" || -e "${run_dir}/eval_logs" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
done

pids=()
for shard in 0 1; do
  if [[ "${shard}" == "0" ]]; then
    gpu="${gpu_a}"
    tasks=(0 1 2 3 4)
  else
    gpu="${gpu_b}"
    tasks=(5 6 7 8 9)
  fi
  run_dir="${output_root}/shard${shard}"
  mkdir -p "${run_dir}"
  task_args=()
  for task_id in "${tasks[@]}"; do
    task_args+=(--task-id "${task_id}")
  done
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
    "${task_args[@]}" \
    --num-episodes "${episodes}" \
    --seed "${seed}" \
    --fm-steps 10 \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  echo "started policy=${policy} shard=${shard} tasks=${tasks[*]} physical_gpu=${gpu} pid=${pids[-1]}"
done

status=0
for shard in 0 1; do
  if wait "${pids[$shard]}"; then
    echo "complete policy=${policy} shard=${shard}"
  else
    echo "failed policy=${policy} shard=${shard}; inspect shard${shard}/stdout.log" >&2
    status=1
  fi
done
exit "${status}"
