from __future__ import annotations

import pytest

from a1.vla.dynamic_compute.device_guard import normalize_gpu_uuid


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("GPU-00000000-0000-0000-0000-000000000000", "00000000-0000-0000-0000-000000000000"),
        ("00000000-0000-0000-0000-000000000000", "00000000-0000-0000-0000-000000000000"),
        (" GPU-00000000-0000-0000-0000-000000000000 ", "00000000-0000-0000-0000-000000000000"),
    ],
)
def test_normalize_gpu_uuid(value: str, expected: str) -> None:
    assert normalize_gpu_uuid(value) == expected
