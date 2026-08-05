#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT CACHE_ROOT [REPO_ROOT]}"
cache_root="${2:?usage: $0 OUTPUT_ROOT CACHE_ROOT [REPO_ROOT]}"
repo_root="${3:-${repo_default}}"

exec bash "${repo_root}/scripts/dynamic_compute/run_m425b_feature_collection_front4.sh" \
  "${output_root}" \
  "${cache_root}" \
  "${repo_root}"
