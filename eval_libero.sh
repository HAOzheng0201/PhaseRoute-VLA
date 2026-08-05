#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
checkpoint="${CHECKPOINT:-${repo_root}/model/libero}"
gpu_index="${GPU_INDEX:-0}"
num_episodes="${NUM_EPISODES:-50}"
task_suites="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
output_root="${OUTPUT_ROOT:-${repo_root}/runs/a1_baseline}"

if [[ ! "${gpu_index}" =~ ^[0-3]$ ]]; then
  echo "GPU_INDEX must be one of 0,1,2,3." >&2
  exit 2
fi
if [[ ! -f "${checkpoint}/model.pt" ]]; then
  echo "Missing baseline checkpoint: ${checkpoint}/model.pt" >&2
  exit 2
fi
gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${gpu_index}" | tr -d '[:space:]')"
if [[ ! "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "Could not resolve physical GPU ${gpu_index}." >&2
  exit 2
fi

for task_suite in ${task_suites}; do
  run_dir="${output_root}/${task_suite}"
  mkdir -p "${run_dir}"
  env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    DATA_DIR="${repo_root}" \
    HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}" \
    VLA_CONFIG_YAML=libero_simulation.yaml \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONNOUSERSITE=1 \
    "${python_bin}" "${repo_root}/robot_experiments/libero/eval_libero.py" \
      --task_suite_name "${task_suite}" \
      --pretrained_checkpoint "${checkpoint}" \
      --num_trials_per_task "${num_episodes}" \
      --save_rollout_video False \
      --use_wandb False \
      --local_log_dir "${run_dir}/eval_logs" \
      --save_rollout_video_path "${run_dir}"
done
