#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPU_INDEX:?GPU_INDEX must name one currently idle physical GPU in 0..7}"
gpu_index="${GPU_INDEX}"
checkpoint="${CHECKPOINT:-${repo_root}/model/libero_exit}"
task_ids="${TASK_IDS:-0}"
episode_indices="${EPISODE_INDICES:-12}"
seed="${SEED:-20260826}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/route_first_stage9_active}"
python_bin="${PYTHON_BIN:-python}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
libero_config="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
preflight_only="${PREFLIGHT_ONLY:-0}"
protocol="${repo_root}/configs/route_first_active_pilot_protocol.json"
router="${repo_root}/artifacts/phase_route_v3/final_router.pt"
phase_checkpoint="${repo_root}/artifacts/phase_route_v3/phase_estimator.pt"
thresholds="${repo_root}/artifacts/phase_route_v3/exit_thresholds_libero_10_exp_1.0.json"

if [[ ! "${gpu_index}" =~ ^[0-7]$ ]]; then
  echo "GPU_INDEX must be one physical GPU in 0..7." >&2
  exit 2
fi
if [[ "${preflight_only}" != "0" && "${preflight_only}" != "1" ]]; then
  echo "PREFLIGHT_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ "${task_ids}" != "0" || "${episode_indices}" != "12" || "${seed}" != "20260826" ]]; then
  echo "Candidate-first Stage-9 smoke is frozen to task 0, state 12, seed 20260826." >&2
  exit 2
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to bind and audit a physical GPU." >&2
  exit 2
fi

gpu_uuid="${GPU_UUID:-}"
if [[ -z "${gpu_uuid}" ]]; then
  gpu_uuid="$(
    nvidia-smi -i "${gpu_index}" --query-gpu=uuid --format=csv,noheader \
      | tr -d '[:space:]'
  )"
fi
if [[ ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "Could not resolve a unique UUID for physical GPU ${gpu_index}: ${gpu_uuid}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S_%N)"
run_dir="${output_root}/engineering_smoke_candidate_first_gpu${gpu_index}_${timestamp}"
if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

echo "[Stage9-Candidate] research simulator runtime; deployment is not authorized"
echo "[Stage9-Candidate] physical GPU ${gpu_index} (${gpu_uuid})"
echo "[Stage9-Candidate] frozen selection: task 0, state 12, arm position 1"
echo "[Stage9-Candidate] output: ${run_dir}"

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

if [[ "${preflight_only}" == "1" ]]; then
  echo "[Stage9-Candidate] both no-episode preflights passed: ${run_dir}"
  exit 0
fi

overlay="${run_dir}/checkpoint"
mkdir "${overlay}"
ln -s "$(realpath "${checkpoint}/model.pt")" "${overlay}/model.pt"
ln -s "$(realpath "${checkpoint}/config.yaml")" "${overlay}/config.yaml"
ln -s "$(realpath "${checkpoint}/dataset_statistics.json")" "${overlay}/dataset_statistics.json"
ln -s "$(realpath "${thresholds}")" "${overlay}/exit_thresholds_libero_10_exp_1.0.json"

{
  printf 'GPU_INDEX=%q GPU_UUID=%q TASK_IDS=%q EPISODE_INDICES=%q SEED=%q ' \
    "${gpu_index}" "${gpu_uuid}" "${task_ids}" "${episode_indices}" "${seed}"
  printf 'CHECKPOINT=%q OUTPUT_ROOT=%q PYTHON_BIN=%q bash %q\n' \
    "${checkpoint}" "${output_root}" "${python_bin}" \
    "${repo_root}/scripts/run_libero_route_first_stage9_candidate.sh"
} > "${run_dir}/command.sh"

env "${common_env[@]}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "${python_bin}" "${repo_root}/scripts/run_phase_route_v3.py" \
    --checkpoint "${overlay}" \
    --router "${router}" \
    --phase-checkpoint "${phase_checkpoint}" \
    --task-ids "${task_ids}" \
    --episode-indices "${episode_indices}" \
    --seed "${seed}" \
    --output-dir "${run_dir}" \
    --measurement-output "${run_dir}/stage1_measurement.jsonl" \
    2>&1 | tee "${run_dir}/stdout.log"

"${python_bin}" "${repo_root}/scripts/validate_phase_route_v3_run.py" "${run_dir}"
"${python_bin}" "${repo_root}/scripts/validate_route_first_stage9_candidate_arm.py" \
  "${run_dir}" --repo-root "${repo_root}"
echo "[Stage9-Candidate] completed and sealed: ${run_dir}"
