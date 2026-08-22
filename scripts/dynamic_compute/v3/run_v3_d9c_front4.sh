#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "${mode}" != "preflight" && "${mode}" != "execute" && "${mode}" != "resume" ]]; then
  echo "usage: $0 {preflight|execute|resume}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin=/home/haozheng/.conda/envs/a1/bin/python
output_root="${repo_root}/reports/v3_d9c_paired_active"
checkpoint="${repo_root}/model/v3_d2/libero_exit"
router="${repo_root}/reports/v3_d8_final_router/final_router.pt"
phase_checkpoint=/data3/haozheng/A1/source/reports/m2_phase_estimator_v1_seed20260803/phase_estimator.pt
attestation=/data3/haozheng/A1/source/reports/model_libero_exit_sha256_attestation_20260809_v1.json
data_dir=/data3/haozheng/A1/source

if [[ -n "$(git -C "${repo_root}" status --porcelain=v1)" ]]; then
  echo "D9C requires a clean frozen-runner worktree" >&2
  exit 3
fi

if [[ "${mode}" == "execute" ]]; then
  if [[ -e "${output_root}" || -e "${repo_root}/reports/v3_d9c_launch_logs" ]]; then
    echo "D9C execute refuses existing formal output or launch logs" >&2
    exit 4
  fi
  log_root="${repo_root}/reports/v3_d9c_launch_logs"
elif [[ "${mode}" == "resume" ]]; then
  if [[ ! -d "${output_root}" ]]; then
    echo "D9C resume requires the existing formal output root" >&2
    exit 5
  fi
  log_root="${repo_root}/reports/v3_d9c_launch_logs/resume_$(date -u +%Y%m%dT%H%M%SZ)"
else
  if [[ -e "${repo_root}/reports/v3_d9c_preflight_logs" ]]; then
    echo "D9C preflight refuses to overwrite its log root" >&2
    exit 6
  fi
  log_root="${repo_root}/reports/v3_d9c_preflight_logs"
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

declare -a gpus=(0 1 2 3)
declare -a gpu_uuids=()
for gpu in "${gpus[@]}"; do
  expected_uuid="$(nvidia_field "${gpu}" uuid)"
  memory_used="$(nvidia_field "${gpu}" memory.used)"
  utilization="$(nvidia_field "${gpu}" utilization.gpu)"
  if (( memory_used > 100 || utilization > 5 )); then
    echo "Refusing busy physical GPU${gpu}: memory=${memory_used}MiB utilization=${utilization}%" >&2
    exit 7
  fi
  read -r visible_count actual_uuid < <(
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 \
      "${python_bin}" -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${gpu}" >&2
    exit 8
  fi
  gpu_uuids+=("${expected_uuid}")
  echo "verified physical_gpu=${gpu} uuid=${expected_uuid} memory_mib=${memory_used} utilization=${utilization}"
done

mkdir -p "${log_root}"
if [[ "${mode}" == "execute" ]]; then
  mkdir -p "${output_root}"
fi
git -C "${repo_root}" rev-parse HEAD >"${log_root}/source_git_commit.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader >"${log_root}/gpu_preflight.csv"

runner_flags=()
if [[ "${mode}" == "preflight" ]]; then
  runner_flags+=(--preflight-only)
elif [[ "${mode}" == "resume" ]]; then
  runner_flags+=(--resume)
fi

for batch_start in 0 4 8; do
  declare -a pids=()
  declare -a batch_tasks=()
  batch_limit=$((batch_start + 4))
  if (( batch_limit > 10 )); then
    batch_limit=10
  fi
  for ((task_id=batch_start; task_id<batch_limit; task_id++)); do
    gpu=$((task_id % 4))
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
      "${repo_root}/scripts/dynamic_compute/v3/run_v3_d9c_task.py" \
      --task-id "${task_id}" \
      --physical-gpu-index "${gpu}" \
      --expected-gpu-uuid "${gpu_uuids[$gpu]}" \
      --checkpoint "${checkpoint}" \
      --model-attestation "${attestation}" \
      --router "${router}" \
      --phase-checkpoint "${phase_checkpoint}" \
      --output-dir "${run_dir}" \
      "${runner_flags[@]}" \
      >"${console_log}" 2>&1 &
    pids+=("$!")
    batch_tasks+=("${task_id}")
    echo "started mode=${mode} task=${task_id} physical_gpu=${gpu} pid=${pids[-1]} log=${console_log}"
  done

  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "mode=${mode} batch_start=${batch_start} alive_workers=${alive}"
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    sleep 30
  done

  failed=0
  for index in "${!pids[@]}"; do
    task_id="${batch_tasks[$index]}"
    if wait "${pids[$index]}"; then
      echo "completed mode=${mode} task=${task_id}"
    else
      echo "failed mode=${mode} task=${task_id}; inspect ${log_root}/task${task_id}.log" >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "D9C remains INCOMPLETE; resume only with the frozen tuple and commit" >&2
    exit 9
  fi
done

if [[ "${mode}" == "preflight" ]]; then
  echo "PASS_V3_D9C_FRONT4_PREFLIGHT"
else
  echo "COMPLETE_V3_D9C_FRONT4_RAW_COLLECTION"
fi
