#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index="${GPU_INDEX:-0}"
checkpoint="${CHECKPOINT:-${repo_root}/model/libero_exit}"
task_ids="${TASK_IDS:-0}"
episode_indices="${EPISODE_INDICES:-0}"
seed="${SEED:-20260823}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/phase_route_v3}"
python_bin="${PYTHON_BIN:-python}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
libero_config="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
preflight_only="${PREFLIGHT_ONLY:-0}"
stage1_measurement="${STAGE1_MEASUREMENT:-0}"
allow_all_gpus="${ALLOW_ALL_GPUS:-0}"
router="${repo_root}/artifacts/phase_route_v3/final_router.pt"
phase_checkpoint="${repo_root}/artifacts/phase_route_v3/phase_estimator.pt"
thresholds="${repo_root}/artifacts/phase_route_v3/exit_thresholds_libero_10_exp_1.0.json"

if [[ "${allow_all_gpus}" != "0" && "${allow_all_gpus}" != "1" ]]; then
  echo "ALLOW_ALL_GPUS must be 0 or 1." >&2
  exit 2
fi
allowed_gpu_count=4
gpu_pattern='^[0-3]$'
gpu_scope='0-3 (historical default)'
if [[ "${allow_all_gpus}" == "1" ]]; then
  allowed_gpu_count=8
  gpu_pattern='^[0-7]$'
  gpu_scope='0-7 (explicit experiment opt-in)'
fi
if [[ ! "${gpu_index}" =~ ${gpu_pattern} ]]; then
  echo "GPU_INDEX is outside the allowed scope: ${gpu_scope}." >&2
  exit 2
fi
if [[ "${preflight_only}" != "0" && "${preflight_only}" != "1" ]]; then
  echo "PREFLIGHT_ONLY must be 0 or 1." >&2
  exit 2
fi
if [[ "${stage1_measurement}" != "0" && "${stage1_measurement}" != "1" ]]; then
  echo "STAGE1_MEASUREMENT must be 0 or 1." >&2
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
  gpu_uuid="$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
fi
if [[ ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "Could not resolve a unique UUID for physical GPU ${gpu_index}: ${gpu_uuid}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${output_root}/libero_10_${timestamp}"
if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

echo "[PhaseRoute-V3] research simulator runtime; deployment is not authorized"
echo "[PhaseRoute-V3] physical GPU: ${gpu_index} (${gpu_uuid}; allowed ${gpu_scope})"
echo "[PhaseRoute-V3] checkpoint: ${checkpoint}"
echo "[PhaseRoute-V3] task ids: ${task_ids}"
echo "[PhaseRoute-V3] official init-state indices: ${episode_indices}"
echo "[PhaseRoute-V3] output: ${run_dir}"

env \
  CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  MUJOCO_EGL_DEVICE_ID="0" \
  DATA_DIR="${repo_root}" \
  HF_HOME="${hf_home}" \
  LIBERO_CONFIG_PATH="${libero_config}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  VLA_CONFIG_YAML="libero_simulation.yaml" \
  MUJOCO_GL="egl" \
  PYOPENGL_PLATFORM="egl" \
  TF_CPP_MIN_LOG_LEVEL="3" \
  PYTHONNOUSERSITE="1" \
  "${python_bin}" "${repo_root}/scripts/validate_phase_route_v3_release.py" \
    --repo-root "${repo_root}" \
    --checkpoint "${checkpoint}" \
    --require-backbone \
    --require-cuda \
    --physical-gpu-index "${gpu_index}" \
    --allowed-gpu-count "${allowed_gpu_count}" \
    --expected-gpu-uuid "${gpu_uuid}" \
    --output "${run_dir}/preflight.json"

if [[ "${preflight_only}" == "1" ]]; then
  echo "[PhaseRoute-V3] preflight complete: ${run_dir}/preflight.json"
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
  printf 'CHECKPOINT=%q OUTPUT_ROOT=%q PYTHON_BIN=%q STAGE1_MEASUREMENT=%q ALLOW_ALL_GPUS=%q ' \
    "${checkpoint}" "${output_root}" "${python_bin}" "${stage1_measurement}" \
    "${allow_all_gpus}"
  printf 'bash %q\n' "${repo_root}/scripts/run_libero_phase_route_v3.sh"
} > "${run_dir}/command.sh"

measurement_args=()
if [[ "${stage1_measurement}" == "1" ]]; then
  measurement_args=(--measurement-output "${run_dir}/stage1_measurement.jsonl")
fi

env \
  CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
  MUJOCO_EGL_DEVICE_ID="0" \
  DATA_DIR="${repo_root}" \
  HF_HOME="${hf_home}" \
  LIBERO_CONFIG_PATH="${libero_config}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  VLA_CONFIG_YAML="libero_simulation.yaml" \
  MUJOCO_GL="egl" \
  PYOPENGL_PLATFORM="egl" \
  TF_CPP_MIN_LOG_LEVEL="3" \
  PYTHONNOUSERSITE="1" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "${python_bin}" "${repo_root}/scripts/run_phase_route_v3.py" \
    --checkpoint "${overlay}" \
    --router "${router}" \
    --phase-checkpoint "${phase_checkpoint}" \
    --task-ids "${task_ids}" \
    --episode-indices "${episode_indices}" \
    --seed "${seed}" \
    --output-dir "${run_dir}" \
    "${measurement_args[@]}" \
    2>&1 | tee "${run_dir}/stdout.log"

"${python_bin}" "${repo_root}/scripts/validate_phase_route_v3_run.py" "${run_dir}"
echo "[PhaseRoute-V3] completed and sealed: ${run_dir}"
