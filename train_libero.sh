#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
checkpoint="${CHECKPOINT:-${repo_root}/model/pretrain}"
experiment_name="${EXP_NAME:-phase_route_vla_libero}"
save_folder="${SAVE_FOLDER:-${repo_root}/model/checkpoints/${experiment_name}}"
batch_per_gpu="${BATCH_PER_GPU:-32}"
state_mask_prob="${STATE_MASK_PROB:-0.0}"
rdzv_port="${RDZV_PORT:-13600}"

IFS=',' read -r -a devices <<<"${gpu_devices}"
if [[ "${#devices[@]}" -lt 1 ]]; then
  echo "At least one GPU is required." >&2
  exit 2
fi
seen=()
for gpu in "${devices[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-3]$ ]]; then
    echo "CUDA_VISIBLE_DEVICES may contain only physical GPU indices 0-3." >&2
    exit 2
  fi
  for prior in "${seen[@]}"; do
    if [[ "${gpu}" == "${prior}" ]]; then
      echo "Duplicate GPU index: ${gpu}" >&2
      exit 2
    fi
  done
  seen+=("${gpu}")
done
if [[ ! "${batch_per_gpu}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_PER_GPU must be a positive integer." >&2
  exit 2
fi
if [[ ! -d "${checkpoint}" ]]; then
  echo "Missing A1 pretraining checkpoint directory: ${checkpoint}" >&2
  exit 2
fi

nproc_per_node="${#devices[@]}"
global_batch_size=$((nproc_per_node * batch_per_gpu))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_devices}"
export DATA_DIR="${DATA_DIR:-${repo_root}}"
export HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
export VLA_CONFIG_YAML=libero_simulation.yaml
export PYTHONNOUSERSITE=1

echo "[PhaseRoute] LIBERO training"
echo "[PhaseRoute] physical GPUs: ${gpu_devices}"
echo "[PhaseRoute] global batch: ${global_batch_size}"
echo "[PhaseRoute] checkpoint: ${checkpoint}"
echo "[PhaseRoute] output: ${save_folder}"

"${python_bin}" -m torch.distributed.run \
  --nproc-per-node="${nproc_per_node}" \
  --rdzv-endpoint="localhost:${rdzv_port}" \
  "${repo_root}/launch_scripts/train_vla.py" \
  qwen2_7b \
  --checkpoint "${checkpoint}" \
  save_folder="${save_folder}" \
  --vision_backbone openai \
  --action_head flow_matching \
  --seq_len 600 \
  --state_mask_prob "${state_mask_prob}" \
  --device_train_microbatch_size "${batch_per_gpu}" \
  --global_batch_size "${global_batch_size}" \
  --dataset libero_simulation.yaml \
  --ft_llm \
  --llm_learning_rate 5e-6 \
  --action_head_learning_rate 5e-5 \
  --vit_learning_rate 2e-6 \
  --connector_learning_rate 2e-6 \
  --warmup_steps 2000 \
  --freeze_steps 1000 \
  --save_interval_unsharded 1000 \
  --save_interval 1000 \
  --crop_mode resize \
  --max_crops 2 \
  --train_steps 500000 \
  --vla_config_path libero_simulation.yaml \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_project "${WANDB_PROJECT:-phase-route-vla}" \
  --wandb_run_name "${experiment_name}" \
  --log_interval 50 \
  --num_workers 2
