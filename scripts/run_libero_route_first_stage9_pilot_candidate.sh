#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPU_INDEX:?GPU_INDEX must name one currently idle physical GPU in 0..7}"
: "${TASK_ID:?TASK_ID must name one frozen pilot task in 0..9}"
gpu_index="${GPU_INDEX}"
task_id="${TASK_ID}"
episode_index="${EPISODE_INDEX:-13}"
seed="${SEED:-20260826}"
arm_position="${ARM_POSITION:-}"
checkpoint="${CHECKPOINT:-${repo_root}/model/libero_exit}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/route_first_stage9_pilot}"
python_bin="${PYTHON_BIN:-python}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
libero_config="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
protocol="${repo_root}/configs/route_first_active_pilot_protocol.json"
router="${repo_root}/artifacts/phase_route_v3/final_router.pt"
phase_checkpoint="${repo_root}/artifacts/phase_route_v3/phase_estimator.pt"
thresholds="${repo_root}/artifacts/phase_route_v3/exit_thresholds_libero_10_exp_1.0.json"

if [[ ! "${gpu_index}" =~ ^[0-7]$ ]]; then
  echo "GPU_INDEX must be in 0..7." >&2
  exit 2
fi
if [[ ! "${task_id}" =~ ^[0-9]$ ]]; then
  echo "TASK_ID must be in 0..9." >&2
  exit 2
fi
expected_arm=2
if (( task_id % 2 == 0 )); then
  expected_arm=1
fi
if [[ -z "${arm_position}" ]]; then
  arm_position="${expected_arm}"
fi
if [[ "${arm_position}" != "${expected_arm}" || "${episode_index}" != "13" || "${seed}" != "20260826" ]]; then
  echo "Candidate pilot selection differs from the frozen alternating state-13 schedule." >&2
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
run_dir="${RUN_DIR:-${output_root}/task${task_id}_arm${arm_position}_candidate_first_gpu${gpu_index}_${timestamp}}"
if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite run directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "$(dirname "${run_dir}")"
mkdir "${run_dir}"

echo "[Stage9-Pilot-Candidate] task ${task_id}, state 13, arm ${arm_position}, GPU ${gpu_index} (${gpu_uuid})"
echo "[Stage9-Pilot-Candidate] output: ${run_dir}"

common_env=(
  CUDA_DEVICE_ORDER="PCI_BUS_ID"
  CUDA_VISIBLE_DEVICES="${gpu_uuid}"
  MUJOCO_EGL_DEVICE_ID="0"
  DATA_DIR="${repo_root}"
  HF_HOME="${hf_home}"
  LIBERO_CONFIG_PATH="${libero_config}"
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  VLA_CONFIG_YAML="libero_simulation.yaml"
  MUJOCO_GL="egl"
  PYOPENGL_PLATFORM="egl"
  TF_CPP_MIN_LOG_LEVEL="3"
  PYTHONNOUSERSITE="1"
)

env "${common_env[@]}" \
  "${python_bin}" "${repo_root}/scripts/validate_route_first_active_preflight.py" \
    --repo-root "${repo_root}" \
    --checkpoint "${checkpoint}" \
    --protocol "${protocol}" \
    --physical-gpu-index "${gpu_index}" \
    --expected-gpu-uuid "${gpu_uuid}" \
    --output "${run_dir}/stage9_preflight.json"

env "${common_env[@]}" \
  "${python_bin}" "${repo_root}/scripts/validate_phase_route_v3_release.py" \
    --repo-root "${repo_root}" \
    --checkpoint "${checkpoint}" \
    --require-backbone \
    --require-cuda \
    --physical-gpu-index "${gpu_index}" \
    --allowed-gpu-count 8 \
    --expected-gpu-uuid "${gpu_uuid}" \
    --output "${run_dir}/preflight.json"

"${python_bin}" "${repo_root}/scripts/validate_route_first_stage9_pilot_prelaunch.py" \
  --repo-root "${repo_root}" \
  --protocol "${protocol}" \
  --method candidate_first_v3 \
  --task-id "${task_id}" \
  --episode-index "${episode_index}" \
  --arm-position "${arm_position}" \
  --seed "${seed}" \
  --physical-gpu-index "${gpu_index}" \
    --expected-gpu-uuid "${gpu_uuid}" \
    --stage9-preflight "${run_dir}/stage9_preflight.json" \
    --v3-preflight "${run_dir}/preflight.json" \
    --output "${run_dir}/prelaunch.json"

overlay="${run_dir}/checkpoint"
mkdir "${overlay}"
ln -s "$(realpath "${checkpoint}/model.pt")" "${overlay}/model.pt"
ln -s "$(realpath "${checkpoint}/config.yaml")" "${overlay}/config.yaml"
ln -s "$(realpath "${checkpoint}/dataset_statistics.json")" "${overlay}/dataset_statistics.json"
ln -s "$(realpath "${thresholds}")" "${overlay}/exit_thresholds_libero_10_exp_1.0.json"

{
  printf 'GPU_INDEX=%q GPU_UUID=%q TASK_ID=%q ARM_POSITION=%q ' \
    "${gpu_index}" "${gpu_uuid}" "${task_id}" "${arm_position}"
  printf 'EPISODE_INDEX=%q SEED=%q RUN_DIR=%q PYTHON_BIN=%q bash %q\n' \
    "${episode_index}" "${seed}" "${run_dir}" "${python_bin}" \
    "${repo_root}/scripts/run_libero_route_first_stage9_pilot_candidate.sh"
} > "${run_dir}/command.sh"

env "${common_env[@]}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "${python_bin}" "${repo_root}/scripts/run_phase_route_v3.py" \
    --checkpoint "${overlay}" \
    --router "${router}" \
    --phase-checkpoint "${phase_checkpoint}" \
    --task-ids "${task_id}" \
    --episode-indices "${episode_index}" \
    --seed "${seed}" \
    --output-dir "${run_dir}" \
    --measurement-output "${run_dir}/stage1_measurement.jsonl" \
    2>&1 | tee "${run_dir}/stdout.log"

"${python_bin}" "${repo_root}/scripts/validate_route_first_stage9_pilot_gpu_postflight.py" \
  --physical-gpu-index "${gpu_index}" \
  --expected-gpu-uuid "${gpu_uuid}" \
  --output "${run_dir}/gpu_postflight.json"
"${python_bin}" "${repo_root}/scripts/validate_phase_route_v3_run.py" "${run_dir}"
"${python_bin}" "${repo_root}/scripts/validate_route_first_stage9_pilot_arm.py" \
  "${run_dir}" --repo-root "${repo_root}"
echo "[Stage9-Pilot-Candidate] completed and sealed: ${run_dir}"
