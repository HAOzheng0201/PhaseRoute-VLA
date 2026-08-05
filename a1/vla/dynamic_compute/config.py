"""Configuration contracts for PhaseRoute-VLA.

All behavior-changing switches default to ``False`` so old configurations and
checkpoints keep the original A1 behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for the JSONL telemetry side channel."""

    enabled: bool = False
    output_path: Optional[Path] = None
    flush_every: int = 100

    def __post_init__(self) -> None:
        if self.flush_every < 1:
            raise ValueError("telemetry.flush_every must be at least 1")
        if self.enabled and self.output_path is None:
            raise ValueError("telemetry.output_path is required when telemetry is enabled")


@dataclass(frozen=True)
class PhaseCacheConfig:
    """Configuration for the opt-in M2 phase-signal cache."""

    enabled: bool = False
    output_dir: Optional[Path] = None
    summary_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.enabled and self.output_dir is None:
            raise ValueError("phase_cache.output_dir is required when phase cache is enabled")
        if self.summary_dtype not in {"float16", "float32"}:
            raise ValueError("phase_cache.summary_dtype must be float16 or float32")


@dataclass(frozen=True)
class DynamicComputeConfig:
    """Top-level opt-in switches shared by future PhaseRoute modules."""

    enabled: bool = False
    phase_enabled: bool = False
    vision_aggregation_enabled: bool = False
    joint_budget_enabled: bool = False
    reliable_exit_enabled: bool = False
    lfp_enabled: bool = False
    dynamic_fm_steps_enabled: bool = False

    max_agg_tokens: int = 64
    profile_names: Tuple[str, ...] = ("B0", "B1", "B2", "B3")
    default_profile: str = "B3"
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    phase_cache: PhaseCacheConfig = field(default_factory=PhaseCacheConfig)

    def __post_init__(self) -> None:
        if self.max_agg_tokens < 1:
            raise ValueError("max_agg_tokens must be at least 1")
        if not self.profile_names:
            raise ValueError("profile_names cannot be empty")
        if len(set(self.profile_names)) != len(self.profile_names):
            raise ValueError("profile_names must be unique")
        if self.default_profile not in self.profile_names:
            raise ValueError("default_profile must be present in profile_names")
