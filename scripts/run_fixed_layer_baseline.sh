#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index="${GPU_INDEX:-0}"
exit_layer="${EXIT_LAYER:-27}"
checkpoint="${CHECKPOINT:-${repo_root}/model/libero_exit}"
task_ids="${TASK_IDS:-0}"
episode_indices="${EPISODE_INDICES:-0}"
seed="${SEED:-20260824}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/stage1_fixed_baselines}"
python_bin="${PYTHON_BIN:-python}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
libero_config="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"

if [[ ! "${gpu_index}" =~ ^[0-7]$ ]]; then
  echo "GPU_INDEX must be one of 0,1,2,3,4,5,6,7." >&2
  exit 2
fi
if [[ ! "${exit_layer}" =~ ^(11|13|27)$ ]]; then
  echo "EXIT_LAYER must be one of 11,13,27." >&2
  exit 2
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

gpu_uuid="${GPU_UUID:-}"
if [[ -z "${gpu_uuid}" ]]; then
  gpu_uuid="$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
fi
if [[ ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "Could not resolve physical GPU ${gpu_index}: ${gpu_uuid}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${output_root}/fixed_l${exit_layer}_${timestamp}"
if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

{
  printf 'GPU_INDEX=%q GPU_UUID=%q EXIT_LAYER=%q TASK_IDS=%q EPISODE_INDICES=%q SEED=%q ' \
    "${gpu_index}" "${gpu_uuid}" "${exit_layer}" "${task_ids}" "${episode_indices}" "${seed}"
  printf 'CHECKPOINT=%q OUTPUT_ROOT=%q PYTHON_BIN=%q bash %q\n' \
    "${checkpoint}" "${output_root}" "${python_bin}" \
    "${repo_root}/scripts/run_fixed_layer_baseline.sh"
} > "${run_dir}/command.sh"

echo "[fixed-L${exit_layer}] physical GPU ${gpu_index} (${gpu_uuid})"
echo "[fixed-L${exit_layer}] task ids ${task_ids}; states ${episode_indices}"
echo "[fixed-L${exit_layer}] output ${run_dir}"

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
  "${python_bin}" "${repo_root}/scripts/run_fixed_layer_baseline.py" \
    --checkpoint "${checkpoint}" \
    --exit-layer "${exit_layer}" \
    --task-ids "${task_ids}" \
    --episode-indices "${episode_indices}" \
    --seed "${seed}" \
    --output-dir "${run_dir}" \
    2>&1 | tee "${run_dir}/stdout.log"

echo "[fixed-L${exit_layer}] completed: ${run_dir}"
