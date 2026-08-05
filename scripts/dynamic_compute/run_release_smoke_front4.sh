#!/usr/bin/env bash
set -euo pipefail
output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="${2:-${script_root}}"
repo_root="$(cd "${repo_root}" && pwd)"
python_bin="${PYTHON_BIN:-python}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
libero_config="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
seed=20261329
episode_index=30

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to bind and audit physical GPUs 0-3." >&2
  exit 2
fi

gpu_uuids=()
for worker in 0 1 2 3; do
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${worker}" | tr -d '[:space:]')"
  if [[ ! "${uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
    echo "Could not resolve a unique UUID for physical GPU ${worker}: ${uuid}" >&2
    exit 2
  fi
  gpu_uuids+=("${uuid}")
done

if [[ -e "${output_root}" ]]; then
  echo "Refusing to overwrite ${output_root}" >&2
  exit 1
fi
mkdir -p "${output_root}"

pids=()
run_dirs=()
for worker in 0 1 2 3; do
  case "${worker}" in
    0) tasks=(0 4 8) ;;
    1) tasks=(1 5 9) ;;
    2) tasks=(2 6) ;;
    3) tasks=(3 7) ;;
  esac
  run_dir="${output_root}/gpu${worker}"
  mkdir -p "${run_dir}"
  task_args=()
  for task_id in "${tasks[@]}"; do
    task_args+=(--task-id "${task_id}")
  done
  env \
    CUDA_DEVICE_ORDER="PCI_BUS_ID" \
    CUDA_VISIBLE_DEVICES="${gpu_uuids[$worker]}" \
    DATA_DIR="${repo_root}" \
    HF_HOME="${hf_home}" \
    LIBERO_CONFIG_PATH="${libero_config}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLA_CONFIG_YAML=libero_simulation.yaml \
    TF_CPP_MIN_LOG_LEVEL=3 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONNOUSERSITE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${python_bin}" \
    "${repo_root}/scripts/dynamic_compute/collect_m418_persistent_tasks.py" \
      --policy rp_pep \
      --checkpoint "${repo_root}/model/libero_exit" \
      --checkpoint-sha256 "${checkpoint_sha}" \
      --task-suite libero_spatial \
      "${task_args[@]}" \
      --num-episodes 1 \
      --episode-index "${episode_index}" \
      --seed "${seed}" \
      --fm-steps 10 \
      --expected-gpu-uuid "${gpu_uuids[$worker]}" \
      --output-dir "${run_dir}" \
      >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  run_dirs+=("${run_dir}")
  echo "started worker=${worker} tasks=${tasks[*]} physical_gpu=${worker} pid=${pids[-1]}"
done

while true; do
  alive=0
  status_line="release-smoke"
  for worker in 0 1 2 3; do
    if kill -0 "${pids[$worker]}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
    episodes=0
    if [[ -f "${run_dirs[$worker]}/episodes.jsonl" ]]; then
      episodes="$(wc -l < "${run_dirs[$worker]}/episodes.jsonl")"
    fi
    expected=2
    if [[ "${worker}" -lt 2 ]]; then expected=3; fi
    status_line+=" gpu${worker}=${episodes}/${expected}"
  done
  echo "${status_line} alive=${alive}"
  if [[ "${alive}" == "0" ]]; then break; fi
  sleep 30
done

status=0
for worker in 0 1 2 3; do
  if wait "${pids[$worker]}"; then
    echo "complete worker=${worker} dir=${run_dirs[$worker]}"
  else
    echo "failed worker=${worker}; inspect ${run_dirs[$worker]}/stdout.log" >&2
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then exit "${status}"; fi

"${python_bin}" "${repo_root}/scripts/dynamic_compute/summarize_release_smoke.py" \
  --input-root "${output_root}" \
  --episode-index "${episode_index}" \
  --seed "${seed}" \
  --expected-gpu-uuid "0=${gpu_uuids[0]}" \
  --expected-gpu-uuid "1=${gpu_uuids[1]}" \
  --expected-gpu-uuid "2=${gpu_uuids[2]}" \
  --expected-gpu-uuid "3=${gpu_uuids[3]}" \
  --output "${output_root}/result.json"
