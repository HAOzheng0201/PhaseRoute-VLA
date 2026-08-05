#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_index="${GPU_INDEX:-0}"
episodes="${NUM_EPISODES:-1}"
seed="${SEED:-20260805}"
checkpoint="${CHECKPOINT:-${repo_root}/model/libero_exit}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/rp_pep}"
python_bin="${PYTHON_BIN:-python}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
libero_config="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"

if [[ ! "${gpu_index}" =~ ^[0-3]$ ]]; then
  echo "GPU_INDEX must be one of 0,1,2,3; GPUs 4-7 are reserved." >&2
  exit 2
fi
if [[ ! "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_EPISODES must be a positive integer." >&2
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
gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu_index}" | tr -d '[:space:]')"
if [[ ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "Could not resolve a unique UUID for physical GPU ${gpu_index}: ${gpu_uuid}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${output_root}/libero_spatial_${timestamp}"
mkdir -p "${run_dir}"

echo "[PhaseRoute] validated runtime: RP-PEP"
echo "[PhaseRoute] physical GPU: ${gpu_index} (${gpu_uuid}; front-four guard active)"
echo "[PhaseRoute] checkpoint: ${checkpoint}"
echo "[PhaseRoute] episodes per task: ${episodes}"
echo "[PhaseRoute] output: ${run_dir}"

env \
  CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
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
  "${python_bin}" "${repo_root}/scripts/validate_phase_route_release.py" \
    --repo-root "${repo_root}" \
    --require-cuda \
    --physical-gpu-index "${gpu_index}" \
    --expected-gpu-uuid "${gpu_uuid}" \
    --output "${run_dir}/preflight.json"

env \
  CUDA_DEVICE_ORDER="PCI_BUS_ID" \
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
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
  "${python_bin}" "${repo_root}/robot_experiments/libero/eval_libero_early_exit.py" \
    --task_suite_name libero_spatial \
    --pretrained_checkpoint "${checkpoint}" \
    --rp_pep_enabled True \
    --exit_ratio 1.0 \
    --action_head_flow_matching_inference_steps 10 \
    --exit_interval 2 \
    --steps_per_stage 1 \
    --threshold_type cosine \
    --exit_dist exp \
    --num_trials_per_task "${episodes}" \
    --seed "${seed}" \
    --reseed_each_episode True \
    --save_rollout_video False \
    --use_wandb False \
    --local_log_dir "${run_dir}/eval_logs" \
    --save_rollout_video_path "${run_dir}" \
    2>&1 | tee "${run_dir}/stdout.log"

echo "[PhaseRoute] completed: ${run_dir}"
