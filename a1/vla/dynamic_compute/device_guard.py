"""Small device-audit helpers with no CUDA or simulator import side effects."""

from __future__ import annotations


def normalize_gpu_uuid(value: str) -> str:
    """Normalize NVIDIA and PyTorch UUID spellings for exact comparison."""

    normalized = str(value).strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized
