#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT CACHE_ROOT [REPO_ROOT]}"
cache_root="${2:?usage: $0 OUTPUT_ROOT CACHE_ROOT [REPO_ROOT]}"
repo_root="${3:-${repo_default}}"
python_bin="${PYTHON_BIN:-python}"
checkpoint_sha=dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f

if [[ -e "${output_root}" ]]; then
  echo "Refusing to reuse existing output root: ${output_root}" >&2
  exit 1
fi
for task_id in 0 1 2 3 4 5 6 7 8 9; do
  result="${cache_root}/task${task_id}/result.json"
  manifest="${cache_root}/task${task_id}/teacher_calls/manifest.jsonl"
  if [[ ! -f "${result}" || ! -f "${manifest}" ]]; then
    echo "Missing frozen teacher cache for task${task_id}" >&2
    exit 2
  fi
done

nvidia_field() {
  local gpu="$1"
  local query="$2"
  local attempt value
  for attempt in 1 2 3 4 5; do
    if value="$(nvidia-smi -i "${gpu}" --query-gpu="${query}" --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s\n' "${value}"
      return 0
    fi
    sleep 2
  done
  echo "nvidia-smi failed after five attempts for GPU${gpu} field ${query}" >&2
  return 1
}

declare -a gpus=(0 1 2 3)
declare -a gpu_uuids=()
for gpu in "${gpus[@]}"; do
  expected_uuid="$(nvidia_field "${gpu}" uuid)"
  memory_used="$(nvidia_field "${gpu}" memory.used)"
  utilization="$(nvidia_field "${gpu}" utilization.gpu)"
  if (( memory_used > 100 || utilization > 5 )); then
    echo "Refusing busy physical GPU${gpu}: memory=${memory_used}MiB utilization=${utilization}%" >&2
    exit 3
  fi
  read -r visible_count actual_uuid < <(
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 \
      "${python_bin}" -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${gpu}: host=${expected_uuid} visible_count=${visible_count} visible_uuid=${actual_uuid}" >&2
    exit 4
  fi
  gpu_uuids+=("${expected_uuid}")
  echo "verified physical_gpu=${gpu} host_uuid=${expected_uuid} visible_uuid=${actual_uuid} memory_mib=${memory_used} utilization=${utilization}"
done

mkdir -p "${output_root}"
declare -a worker_pids=()

run_shard() {
  local gpu="$1"
  shift
  local run_dir="${output_root}/gpu${gpu}"
  mkdir -p "${run_dir}"
  local -a cache_args=()
  local -a task_args=()
  while (( $# > 0 )); do
    task_args+=(--expected-task-id "$1")
    cache_args+=(--cache-dir "${cache_root}/task$1/teacher_calls")
    shift
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
    "${python_bin}" \
      "${repo_root}/scripts/dynamic_compute/collect_m425_causal_route_features.py" \
      --checkpoint "${repo_root}/model/libero_exit" \
      --checkpoint-sha256 "${checkpoint_sha}" \
      "${cache_args[@]}" \
      "${task_args[@]}" \
      --output-dir "${run_dir}/artifact" \
      --expected-gpu-uuid "${gpu_uuids[$gpu]}" \
      --physical-gpu-index "${gpu}" \
      --seed 20260827 \
      >"${run_dir}/stdout.log" 2>&1
}

run_shard 0 0 4 & worker_pids+=("$!")
echo "started GPU0 tasks=0,4 pid=${worker_pids[-1]}"
run_shard 1 1 9 & worker_pids+=("$!")
echo "started GPU1 tasks=1,9 pid=${worker_pids[-1]}"
run_shard 2 2 7 8 & worker_pids+=("$!")
echo "started GPU2 tasks=2,7,8 pid=${worker_pids[-1]}"
run_shard 3 3 5 6 & worker_pids+=("$!")
echo "started GPU3 tasks=3,5,6 pid=${worker_pids[-1]}"

while true; do
  alive=0
  for pid in "${worker_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  completed="$(find "${output_root}" -type f -path '*/artifact/result.json' | wc -l)"
  echo "progress completed_shards=${completed}/4 alive_workers=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

status=0
for gpu in "${gpus[@]}"; do
  if wait "${worker_pids[$gpu]}"; then
    echo "worker complete physical_gpu=${gpu}"
  else
    echo "worker failed physical_gpu=${gpu}; inspect ${output_root}/gpu${gpu}/stdout.log" >&2
    status=1
  fi
done
exit "${status}"
