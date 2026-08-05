#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mode="${1:?usage: $0 MODE OUTPUT_ROOT [REPO_ROOT] [EFA_CHECKPOINT] [PHASE_CHECKPOINT] [SEED]}"
output_root="${2:?usage: $0 MODE OUTPUT_ROOT [REPO_ROOT] [EFA_CHECKPOINT] [PHASE_CHECKPOINT] [SEED]}"
repo_root="${3:-${repo_default}}"
efa_checkpoint="${4:-${repo_root}/reports/m48_candidate_distill_spatial_tasks0_3_60step_20260802_v2_ac/efa_frozen_a1_distilled.pt}"
phase_checkpoint="${5:-${repo_root}/reports/m2_phase_estimator_v1_20260801/phase_estimator.pt}"
seed="${6:-20260804}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"

case "${mode}" in
  baseline|learned_efa144|joint_learned_efa144|joint_risk_full_token_efa144|joint_contact_full_token_efa144|phase_width_contact_full_token_efa144|phase_width_hysteresis_full_token_efa144|phase_width_uncertainty_hysteresis_full_token_efa144) ;;
  *)
    echo "unsupported mode: ${mode}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_root}"
pids=()

for gpu in 0 1 2 3; do
  task_id="${gpu}"
  run_dir="${output_root}/task${task_id}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/policy_calls.jsonl" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
  mkdir -p "${run_dir}"
  extra_args=()
  if [[ "${mode}" != "baseline" ]]; then
    extra_args+=(--efa-checkpoint "${efa_checkpoint}")
  fi
  if [[ "${mode}" == "joint_learned_efa144" || "${mode}" == "joint_risk_full_token_efa144" || "${mode}" == "joint_contact_full_token_efa144" || "${mode}" == "phase_width_contact_full_token_efa144" || "${mode}" == "phase_width_hysteresis_full_token_efa144" || "${mode}" == "phase_width_uncertainty_hysteresis_full_token_efa144" ]]; then
    extra_args+=(--phase-checkpoint "${phase_checkpoint}")
  fi
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DATA_DIR="${repo_root}" \
    HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLA_CONFIG_YAML=libero_simulation.yaml \
    TF_CPP_MIN_LOG_LEVEL=3 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONNOUSERSITE=1 \
    "${PYTHON_BIN:-python}" \
    "${repo_root}/scripts/dynamic_compute/collect_m47_learned_efa_task.py" \
    --mode "${mode}" \
    --checkpoint "${repo_root}/model/libero_exit" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --task-suite libero_spatial \
    --task-id "${task_id}" \
    --num-episodes 1 \
    --seed "${seed}" \
    --fm-steps 10 \
    --output-dir "${run_dir}" \
    "${extra_args[@]}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  echo "started mode=${mode} task=${task_id} physical_gpu=${gpu} pid=${pids[-1]}"
done

status=0
for task_id in 0 1 2 3; do
  if wait "${pids[$task_id]}"; then
    echo "complete mode=${mode} task=${task_id}"
  else
    echo "failed mode=${mode} task=${task_id}; inspect task${task_id}/stdout.log" >&2
    status=1
  fi
done
exit "${status}"
