"""PhaseRoute-VLA dynamic-compute building blocks.

Modules in this package are opt-in. Importing the package must not change A1's
model construction, checkpoint schema, random state, or inference behavior.
"""

from .config import DynamicComputeConfig, PhaseCacheConfig, TelemetryConfig
from .depth_hysteresis import (
    ExitDepthHysteresis,
    ExitDepthHysteresisConfig,
    ExitDepthHysteresisDecision,
)
from .phase_cache import (
    PHASE_CACHE_SCHEMA_VERSION,
    NullPhaseCacheWriter,
    PhaseCacheCallMetadata,
    SafePhaseCacheWriter,
    emit_phase_signal_summary,
    summarize_instruction_embeddings,
    summarize_visual_embeddings,
)
from .productive_exit import ProductiveExitPlan, a1_fm10_rp_pep_plan
from .phase_dataset import (
    PHASE_DATASET_SCHEMA_VERSION,
    PhaseDatasetConfig,
    assign_episode_splits,
    build_phase_dataset_arrays,
)
from .phase_estimator import (
    PhaseEstimatorConfig,
    PhaseLossConfig,
    PhaseState,
    PhaseStateEstimator,
    phase_estimator_loss,
)
from .phase_observer import (
    PHASE_OBSERVER_SCHEMA_VERSION,
    SafePhaseObserver,
)
from .phase_vision_runtime import (
    PhaseProfileVisionRouter,
    PhaseRoutedVisionAggregation,
    PhaseVisionProfile,
    make_exit_controller_profile_provider,
    make_phase_runtime_profile_provider,
)
from .telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    DynamicComputeTelemetry,
    NullTelemetryLogger,
    SafeJSONLTelemetryLogger,
    build_policy_call_telemetry,
    command_motion_summary,
    emit_telemetry_event,
    instruction_hash,
    summarize_vector,
)
from .vision_aggregation import (
    AggregatedVision,
    CompactedSequence,
    StaticVisionAggregationConfig,
    aggregate_projected_vision,
    compact_multimodal_sequence,
    rank_token_bank,
)
from .weak_labels import (
    PHASE_LABEL_SCHEMA_VERSION,
    BoundaryLabelConfig,
    PhaseWeakLabel,
    build_episode_weak_labels,
    build_weak_labels,
)

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "DynamicComputeConfig",
    "DynamicComputeTelemetry",
    "ExitDepthHysteresis",
    "ExitDepthHysteresisConfig",
    "ExitDepthHysteresisDecision",
    "NullTelemetryLogger",
    "NullPhaseCacheWriter",
    "PHASE_LABEL_SCHEMA_VERSION",
    "PHASE_CACHE_SCHEMA_VERSION",
    "PHASE_DATASET_SCHEMA_VERSION",
    "PHASE_OBSERVER_SCHEMA_VERSION",
    "BoundaryLabelConfig",
    "AggregatedVision",
    "CompactedSequence",
    "PhaseCacheCallMetadata",
    "PhaseCacheConfig",
    "PhaseDatasetConfig",
    "PhaseEstimatorConfig",
    "PhaseLossConfig",
    "PhaseState",
    "PhaseStateEstimator",
    "PhaseWeakLabel",
    "ProductiveExitPlan",
    "PhaseProfileVisionRouter",
    "PhaseRoutedVisionAggregation",
    "PhaseVisionProfile",
    "SafePhaseCacheWriter",
    "SafePhaseObserver",
    "SafeJSONLTelemetryLogger",
    "StaticVisionAggregationConfig",
    "TelemetryConfig",
    "build_policy_call_telemetry",
    "build_phase_dataset_arrays",
    "build_episode_weak_labels",
    "build_weak_labels",
    "command_motion_summary",
    "emit_telemetry_event",
    "emit_phase_signal_summary",
    "instruction_hash",
    "make_exit_controller_profile_provider",
    "make_phase_runtime_profile_provider",
    "phase_estimator_loss",
    "assign_episode_splits",
    "a1_fm10_rp_pep_plan",
    "aggregate_projected_vision",
    "compact_multimodal_sequence",
    "rank_token_bank",
    "summarize_instruction_embeddings",
    "summarize_visual_embeddings",
    "summarize_vector",
]
