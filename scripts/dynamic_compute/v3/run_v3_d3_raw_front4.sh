#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin=/home/haozheng/.conda/envs/a1/bin/python
output_root="${repo_root}/reports/v3_d3_calibration_raw"
log_root="${repo_root}/reports/v3_d3_calibration_launch_logs"
checkpoint="${repo_root}/model/v3_d2/libero_exit"
attestation=/data3/haozheng/A1/source/reports/model_libero_exit_sha256_attestation_20260809_v1.json
data_dir=/data3/haozheng/A1/source

if [[ -n "$(git -C "${repo_root}" status --porcelain=v1)" ]]; then
  echo "V3-D3 raw collection requires a clean worktree" >&2
  exit 2
fi
"${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS_V3_D3_CALIBRATION_CONTRACT_FROZEN"' \
  "${repo_root}/results/v3/v3_d3_calibration_contract_validation.json"
if [[ -e "${output_root}" || -e "${log_root}" ]]; then
  echo "V3-D3 refuses to reuse raw output or log roots" >&2
  exit 3
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
  echo "nvidia-smi failed for GPU${gpu} field ${query}" >&2
  return 1
}

declare -a gpu_uuids=()
for gpu in 0 1 2 3; do
  expected_uuid="$(nvidia_field "${gpu}" uuid)"
  memory_used="$(nvidia_field "${gpu}" memory.used)"
  utilization="$(nvidia_field "${gpu}" utilization.gpu)"
  if (( memory_used > 100 || utilization > 5 )); then
    echo "Refusing busy physical GPU${gpu}: memory=${memory_used}MiB utilization=${utilization}%" >&2
    exit 4
  fi
  read -r visible_count actual_uuid < <(
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 \
      "${python_bin}" -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${gpu}" >&2
    exit 5
  fi
  gpu_uuids+=("${expected_uuid}")
done

mkdir -p "${output_root}" "${log_root}"
git -C "${repo_root}" rev-parse HEAD >"${log_root}/source_git_commit.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader >"${log_root}/gpu_preflight.csv"

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
    console_log="${log_root}/task${task_id}.log"
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      MUJOCO_EGL_DEVICE_ID="${gpu}" \
      DATA_DIR="${data_dir}" \
      HF_HOME=/data3/haozheng/A1/hf_cache \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      VLA_CONFIG_YAML=libero_simulation.yaml \
      TF_CPP_MIN_LOG_LEVEL=3 \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH="${data_dir}/robot_experiments/libero/LIBERO:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" \
      "${repo_root}/scripts/dynamic_compute/v3/collect_v3_d3_calibration_task.py" \
      --task-id "${task_id}" \
      --physical-gpu-index "${gpu}" \
      --expected-gpu-uuid "${gpu_uuids[$gpu]}" \
      --checkpoint "${checkpoint}" \
      --model-attestation "${attestation}" \
      --output-dir "${run_dir}" \
      >"${console_log}" 2>&1 &
    pids+=("$!")
    batch_tasks+=("${task_id}")
    echo "started calibration task=${task_id} physical_gpu=${gpu} pid=${pids[-1]} log=${console_log}"
  done

  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "calibration batch_start=${batch_start} alive_workers=${alive}"
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    sleep 30
  done

  failed=0
  for index in "${!pids[@]}"; do
    task_id="${batch_tasks[$index]}"
    if wait "${pids[$index]}"; then
      echo "completed calibration task=${task_id}"
    else
      echo "failed calibration task=${task_id}; inspect ${log_root}/task${task_id}.log" >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    exit 6
  fi
done

echo "PASS_V3_D3_RAW_FRONT4"
