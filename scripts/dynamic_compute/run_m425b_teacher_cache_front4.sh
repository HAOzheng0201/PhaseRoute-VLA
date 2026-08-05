#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT] [SEED] [NUM_EPISODES]}"
repo_root="${2:-${repo_default}}"
seed="${3:-20260826}"
num_episodes="${4:-6}"
python_bin="${PYTHON_BIN:-python}"
checkpoint_sha=dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f

if [[ -e "${output_root}" ]]; then
  echo "Refusing to reuse existing output root: ${output_root}" >&2
  exit 1
fi
if [[ "${num_episodes}" != "6" ]]; then
  if [[ "${M426A_ALLOW_SEVEN_EPISODES:-0}" == "1" \
        && "${num_episodes}" == "7" \
        && "${seed}" == "20261026" ]]; then
    :
  elif [[ "${M427_ALLOW_FIFTEEN_EPISODES:-0}" == "1" \
        && "${num_episodes}" == "15" \
        && "${seed}" == "20261127" ]]; then
    :
  elif [[ "${M428_ALLOW_THIRTY_EPISODES:-0}" == "1" \
        && "${num_episodes}" == "30" \
        && "${seed}" == "20261228" ]]; then
    :
  else
    echo "unsupported seed/episode protocol for frozen front4 collection" >&2
    exit 2
  fi
fi

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
status=0
for batch_start in 0 4 8; do
  declare -a pids=()
  declare -a batch_tasks=()
  for offset in 0 1 2 3; do
    task_id=$((batch_start + offset))
    if (( task_id >= 10 )); then
      break
    fi
    gpu="${offset}"
    run_dir="${output_root}/task${task_id}"
    if [[ -e "${run_dir}" ]]; then
      echo "Refusing to reuse ${run_dir}" >&2
      exit 5
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
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${python_bin}" \
      "${repo_root}/scripts/dynamic_compute/collect_m46_teacher_cache_task.py" \
      --checkpoint "${repo_root}/model/libero_exit" \
      --checkpoint-sha256 "${checkpoint_sha}" \
      --task-suite libero_spatial \
      --task-id "${task_id}" \
      --num-episodes "${num_episodes}" \
      --seed "${seed}" \
      --fm-steps 10 \
      --feature-dtype float16 \
      --expected-gpu-uuid "${gpu_uuids[$gpu]}" \
      --physical-gpu-index "${gpu}" \
      --output-dir "${run_dir}" \
      >"${run_dir}/stdout.log" 2>&1 &
    pids+=("$!")
    batch_tasks+=("${task_id}")
    echo "started task=${task_id} episodes=${num_episodes} physical_gpu=${gpu} pid=${pids[-1]}"
  done

  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "batch_start=${batch_start} alive_workers=${alive}"
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    sleep 30
  done

  for index in "${!pids[@]}"; do
    task_id="${batch_tasks[$index]}"
    if wait "${pids[$index]}"; then
      echo "complete task=${task_id}"
    else
      echo "failed task=${task_id}; inspect ${output_root}/task${task_id}/stdout.log" >&2
      status=1
    fi
  done
  if (( status != 0 )); then
    exit "${status}"
  fi
done

exit "${status}"
