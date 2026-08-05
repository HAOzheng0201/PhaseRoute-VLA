#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
repo_root="${2:-${repo_default}}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
seed=20265804
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to bind physical GPUs 0-3." >&2
  exit 2
fi
gpu_uuids=()
for gpu in 0 1 2 3; do
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu}" | tr -d '[:space:]')"
  if [[ ! "${uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
    echo "Could not resolve physical GPU ${gpu}: ${uuid}" >&2
    exit 2
  fi
  gpu_uuids+=("${uuid}")
done

# Cross-GPU swap relative to M4.20b: RP-PEP is now on GPU0/1.
policies=(rp_pep rp_pep early_exit early_exit)
shards=(shard0 shard1 shard0 shard1)
pids=()
run_dirs=()

for worker in 0 1 2 3; do
  policy="${policies[$worker]}"
  shard="${shards[$worker]}"
  run_dir="${output_root}/${policy}/${shard}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/episodes.jsonl" || -e "${run_dir}/eval_logs" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
  mkdir -p "${run_dir}"
  if [[ "${shard}" == "shard0" ]]; then
    tasks=(0 1 2 3 4)
  else
    tasks=(5 6 7 8 9)
  fi
  task_args=()
  for task_id in "${tasks[@]}"; do
    task_args+=(--task-id "${task_id}")
  done
  env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${gpu_uuids[$worker]}" \
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
    --num-episodes 3 \
    --episode-index 29 \
    --episode-index 30 \
    --episode-index 31 \
    --seed "${seed}" \
    --fm-steps 10 \
    --expected-gpu-uuid "${gpu_uuids[$worker]}" \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  run_dirs+=("${run_dir}")
  echo "started worker=${worker} policy=${policy} tasks=${tasks[*]} physical_gpu=${worker} pid=${pids[-1]}"
done

while true; do
  alive=0
  status_line="progress"
  for worker in 0 1 2 3; do
    if kill -0 "${pids[$worker]}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
    episodes=0
    if [[ -f "${run_dirs[$worker]}/episodes.jsonl" ]]; then
      episodes="$(wc -l < "${run_dirs[$worker]}/episodes.jsonl")"
    fi
    status_line+=" w${worker}=${episodes}/15"
  done
  echo "${status_line} alive=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

status=0
for worker in 0 1 2 3; do
  if wait "${pids[$worker]}"; then
    echo "complete worker=${worker} dir=${run_dirs[$worker]}"
  else
    echo "failed worker=${worker}; inspect ${run_dirs[$worker]}/stdout.log" >&2
    status=1
  fi
done
exit "${status}"
