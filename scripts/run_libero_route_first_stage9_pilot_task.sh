#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPU_INDEX:?GPU_INDEX must name one currently idle physical GPU in 0..7}"
: "${TASK_ID:?TASK_ID must name one frozen pilot task in 0..9}"
gpu_index="${GPU_INDEX}"
task_id="${TASK_ID}"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/route_first_stage9_pilot}"

if [[ ! "${gpu_index}" =~ ^[0-7]$ || ! "${task_id}" =~ ^[0-9]$ ]]; then
  echo "GPU_INDEX must be in 0..7 and TASK_ID must be in 0..9." >&2
  exit 2
fi
gpu_uuid="${GPU_UUID:-}"
if [[ -z "${gpu_uuid}" ]]; then
  gpu_uuid="$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
fi
if [[ ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "Could not resolve GPU UUID for physical GPU ${gpu_index}." >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S_%N)"
pair_dir="${PAIR_DIR:-${output_root}/task${task_id}_state13_gpu${gpu_index}_${timestamp}}"
if [[ -e "${pair_dir}" ]]; then
  echo "Refusing to overwrite task pair directory: ${pair_dir}" >&2
  exit 2
fi
mkdir -p "$(dirname "${pair_dir}")"
mkdir "${pair_dir}"

candidate_arm=2
route_arm=1
if (( task_id % 2 == 0 )); then
  candidate_arm=1
  route_arm=2
fi
candidate_dir="${pair_dir}/arm${candidate_arm}_candidate_first"
route_dir="${pair_dir}/arm${route_arm}_route_first"

{
  printf 'GPU_INDEX=%q GPU_UUID=%q TASK_ID=%q PAIR_DIR=%q PYTHON_BIN=%q bash %q\n' \
    "${gpu_index}" "${gpu_uuid}" "${task_id}" "${pair_dir}" "${python_bin}" \
    "${repo_root}/scripts/run_libero_route_first_stage9_pilot_task.sh"
} > "${pair_dir}/command.sh"

echo "[Stage9-Pilot-Task] task ${task_id}, GPU ${gpu_index} (${gpu_uuid})"
if (( task_id % 2 == 0 )); then
  env GPU_INDEX="${gpu_index}" GPU_UUID="${gpu_uuid}" TASK_ID="${task_id}" \
    ARM_POSITION="${candidate_arm}" RUN_DIR="${candidate_dir}" \
    PYTHON_BIN="${python_bin}" HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}" \
    bash "${repo_root}/scripts/run_libero_route_first_stage9_pilot_candidate.sh"
  env GPU_INDEX="${gpu_index}" GPU_UUID="${gpu_uuid}" TASK_ID="${task_id}" \
    ARM_POSITION="${route_arm}" RUN_DIR="${route_dir}" \
    PYTHON_BIN="${python_bin}" HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}" \
    bash "${repo_root}/scripts/run_libero_route_first_stage9_pilot_route.sh"
else
  env GPU_INDEX="${gpu_index}" GPU_UUID="${gpu_uuid}" TASK_ID="${task_id}" \
    ARM_POSITION="${route_arm}" RUN_DIR="${route_dir}" \
    PYTHON_BIN="${python_bin}" HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}" \
    bash "${repo_root}/scripts/run_libero_route_first_stage9_pilot_route.sh"
  env GPU_INDEX="${gpu_index}" GPU_UUID="${gpu_uuid}" TASK_ID="${task_id}" \
    ARM_POSITION="${candidate_arm}" RUN_DIR="${candidate_dir}" \
    PYTHON_BIN="${python_bin}" HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}" \
    bash "${repo_root}/scripts/run_libero_route_first_stage9_pilot_candidate.sh"
fi

"${python_bin}" "${repo_root}/scripts/summarize_route_first_stage9_pilot_task.py" \
  --candidate-dir "${candidate_dir}" \
  --route-dir "${route_dir}" \
  --output "${pair_dir}/task_pair.json"
echo "[Stage9-Pilot-Task] completed and sealed: ${pair_dir}"
