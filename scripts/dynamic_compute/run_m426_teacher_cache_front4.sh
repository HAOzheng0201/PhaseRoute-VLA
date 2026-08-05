#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
repo_root="${2:-${repo_default}}"

# Reuse the audited four-GPU collector and freeze only the new data identity.
exec bash "${repo_root}/scripts/dynamic_compute/run_m425b_teacher_cache_front4.sh" \
  "${output_root}" \
  "${repo_root}" \
  20260926 \
  6
