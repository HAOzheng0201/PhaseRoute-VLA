"""Frozen V3-D3 cluster calibration for the Gripper-v2 risk head.

The calibration stage never refits the D2 GLMs.  It restores the attested
full-development state, computes one pre-declared score per L11/L13 candidate,
and selects one global threshold using 100 task-episode clusters.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import numpy as np
from scipy.stats import beta
import torch

from .development_collection import (
    D2_CHECKPOINT_SHA256,
    DevelopmentCall,
    stream_sha256,
)
from .gripper_v2_models import (
    COUNT_SUPPORT_MAX,
    FeatureNormalizer,
    OccurrenceFit,
    OrdinalFit,
    expected_positive_count,
)
from .gripper_v2_oof import HEAD_NAMES, OOF_SCHEMA_VERSION
from .gripper_v2_protocol import (
    CALIBRATION_EPISODES,
    DECISION_LAYERS,
    FEATURE_DIMENSION,
    decode_json_bytes,
    validate_selection_document,
)
from a1.vla.dynamic_compute.vision_teacher_cache import (
    VISION_TEACHER_CACHE_SCHEMA_VERSION,
    has_complete_candidate_fm_traces,
)


D3_SCHEMA_VERSION = "phase-route-vla.v3.d3-calibration-contract.v1"
D3_STATUS = "D3_CALIBRATION_CONTRACT_FROZEN"
D3_ROLE = "calibration_v2"
D3_SUITE = "libero_10"
D3_TASK_IDS = tuple(range(10))
D3_EPISODES = tuple(CALIBRATION_EPISODES)
D3_CLUSTER_COUNT = 100
D3_CONTRACT_RELATIVE_PATH = Path(
    "configs/research/v3/gripper_v2/d3_calibration_contract.json"
)
D3_CONTRACT_SHA256 = (
    "14ac58b6f41fb0f9b2ccadad4fde6fdebb952ca1d59a01104811bde3b9318817"
)
D3_SELECTION_RELATIVE_PATH = Path(
    "configs/research/v3/data_lineage/calibration_v2.json"
)
D3_SELECTION_SHA256 = (
    "6f2b2817985740298a06c4412b2f857624ac16c98d174d3ad03f1acca238f79e"
)
D3_TEST_SELECTION_RELATIVE_PATH = Path(
    "configs/research/v3/data_lineage/independent_test_v2.json"
)
D3_TEST_SELECTION_SHA256 = (
    "e2c1b2a11f84af9b71d588bf638d794c5a29870ace87b46b65960749e0f9bdf4"
)
D3_D2_ATTESTATION_RELATIVE_PATH = Path(
    "results/v3/v3_d2_formal_development_result.json"
)
D3_D2_ATTESTATION_SHA256 = (
    "bc6841b0a92eb0a66e1e60ad13d58ec15a0f4abb3f6750ad3898dee3c3c41092"
)
D3_D2_RESULT_RELATIVE_PATH = Path("reports/v3_d2_development_oof/result.json")
D3_D2_RESULT_SHA256 = (
    "09c5e71e656bc434f951b99fe2371abf8d5a4e42afcbbf13047d52520974fa79"
)
D3_D2_PAYLOAD_RELATIVE_PATH = Path(
    "reports/v3_d2_development_oof/development_gripper_v2_nested_oof.pt"
)
D3_D2_PAYLOAD_SHA256 = (
    "8e5a6d14b1b60c4e984fa68f32be063335728bd93678f01b2e9926f3741d3d6b"
)
D3_D2_DATASET_PAYLOAD_SHA256 = (
    "d2e21932cadbf683f8607791627a825efc352fd6677a046670de19d65e51433d"
)
D3_PRIMARY_PARAMETER_COUNT = 414
D3_FINAL_LAMBDAS = {
    "occurrence": 0.01,
    "zt_step": 0.1,
    "zt_transition": 0.1,
    "ordinal_step": 0.1,
    "ordinal_transition": 0.1,
}
D3_UCB_CONFIDENCE = 0.95
D3_FALSE_SAFE_UCB_MAX = 0.05
D3_MINIMUM_SAFE_COVERAGE = 0.10
D3_CONTEXT_SCHEMA_VERSION = "phase-route-vla.v3.d3-context.v1"
D3_CANDIDATE_SCHEMA_VERSION = "phase-route-vla.v3.d3-candidates.v1"
D3_DATASET_SCHEMA_VERSION = "phase-route-vla.v3.d3-gripper-dataset.v1"
_D3_EPISODE_ID = re.compile(r"^libero_10:task([0-9]+):episode([0-9]+)$")


class D3CalibrationError(ValueError):
    """Raised when calibration evidence or selection violates the contract."""


@dataclass(frozen=True)
class CalibrationEpisode:
    task_id: int
    episode_index: int
    seed: int

    @property
    def group_key(self) -> str:
        return f"{D3_SUITE}:task{self.task_id}:episode{self.episode_index}"


class CalibrationInitialStateWindowTaskSuite:
    """Expose exactly calibration episode initial states 30--39."""

    def __init__(self, base_suite: Any) -> None:
        self._base_suite = base_suite

    def get_task_init_states(self, task_id: int):
        states = self._base_suite.get_task_init_states(task_id)
        if len(states) < 40:
            raise D3CalibrationError("LIBERO task has fewer than 40 initial states")
        return states[30:40]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_suite, name)


def global_calibration_episode_index(local_episode_index: int) -> int:
    if (
        type(local_episode_index) is not int
        or not 0 <= local_episode_index < len(D3_EPISODES)
    ):
        raise D3CalibrationError("D3 local episode index must be in 0..9")
    return D3_EPISODES[0] + local_episode_index


def expected_calibration_seed(task_id: int, episode_index: int) -> int:
    if type(task_id) is not int or task_id not in D3_TASK_IDS:
        raise D3CalibrationError("D3 task id must be in 0..9")
    if type(episode_index) is not int or episode_index not in D3_EPISODES:
        raise D3CalibrationError("D3 episode index must be in 30..39")
    return 20260811 + task_id * 10_000 + episode_index


def _json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = decode_json_bytes(path.read_bytes(), context=context)
    except OSError as error:
        raise D3CalibrationError(f"{context} cannot be read") from error
    if not isinstance(value, dict):
        raise D3CalibrationError(f"{context} must be an object")
    return value


def load_d3_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D3_CONTRACT_RELATIVE_PATH
    if stream_sha256(path) != D3_CONTRACT_SHA256:
        raise D3CalibrationError("D3 contract SHA-256 differs")
    contract = _json(path, context="D3 calibration contract")
    if (
        contract.get("schema_version") != D3_SCHEMA_VERSION
        or contract.get("status") != D3_STATUS
        or contract.get("stage") != "V3-D3"
    ):
        raise D3CalibrationError("D3 contract header differs")
    if contract.get("score", {}).get("name") != "step_any_mismatch_probability":
        raise D3CalibrationError("D3 score family differs")
    if contract.get("threshold_selection", {}).get("ucb_confidence") != 0.95:
        raise D3CalibrationError("D3 confidence level differs")
    return contract


def load_calibration_selection(
    repo_root: str | Path,
) -> tuple[CalibrationEpisode, ...]:
    root = Path(repo_root).resolve(strict=True)
    path = root / D3_SELECTION_RELATIVE_PATH
    if stream_sha256(path) != D3_SELECTION_SHA256:
        raise D3CalibrationError("D3 calibration selection SHA-256 differs")
    value = _json(path, context="D3 calibration selection")
    validate_selection_document(
        value,
        role=D3_ROLE,
        episodes=D3_EPISODES,
        expected_count=D3_CLUSTER_COUNT,
    )
    records = tuple(
        CalibrationEpisode(
            task_id=int(record["task_id"]),
            episode_index=int(record["episode_index"]),
            seed=int(record["seed"]),
        )
        for record in value["records"]
    )
    expected_order = tuple(
        (task, episode) for task in D3_TASK_IDS for episode in D3_EPISODES
    )
    if tuple((record.task_id, record.episode_index) for record in records) != expected_order:
        raise D3CalibrationError("D3 calibration selection order differs")
    if any(
        record.seed != expected_calibration_seed(
            record.task_id, record.episode_index
        )
        for record in records
    ):
        raise D3CalibrationError("D3 calibration seed formula differs")
    return records


def task_calibration_window(
    selection: tuple[CalibrationEpisode, ...], task_id: int
) -> tuple[CalibrationEpisode, ...]:
    if type(task_id) is not int or task_id not in D3_TASK_IDS:
        raise D3CalibrationError("D3 task id must be in 0..9")
    selected = tuple(record for record in selection if record.task_id == task_id)
    if tuple(record.episode_index for record in selected) != D3_EPISODES:
        raise D3CalibrationError("D3 task calibration window differs")
    return selected


def _regular_file(path: Path, *, context: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise D3CalibrationError(f"{context} contains a symlink")
    try:
        metadata = absolute.stat()
    except FileNotFoundError as error:
        raise D3CalibrationError(f"{context} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise D3CalibrationError(f"{context} must be a regular file")
    return absolute.resolve(strict=True)


def load_calibration_task_calls(
    task_output_directory: str | Path,
    *,
    task_id: int,
    dataset_index_start: int = 0,
) -> tuple[DevelopmentCall, ...]:
    """Read one D3 manifest without opening any NPZ payload."""

    if task_id not in D3_TASK_IDS:
        raise D3CalibrationError("D3 task id must be in 0..9")
    if type(dataset_index_start) is not int or dataset_index_start < 0:
        raise D3CalibrationError("D3 dataset index start must be non-negative")
    output = Path(task_output_directory).resolve(strict=True)
    manifest = _regular_file(
        output / "teacher_calls" / "manifest.jsonl",
        context="D3 teacher manifest",
    )
    cache_directory = manifest.parent
    counters: dict[int, int] = defaultdict(int)
    previous_step: dict[int, int] = {}
    rows: list[DevelopmentCall] = []
    with manifest.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise D3CalibrationError("D3 manifest contains an empty line")
            try:
                source = json.loads(line)
            except json.JSONDecodeError as error:
                raise D3CalibrationError("D3 manifest row is invalid JSON") from error
            if not isinstance(source, dict):
                raise D3CalibrationError("D3 manifest row must be an object")
            match = _D3_EPISODE_ID.fullmatch(str(source.get("episode_id")))
            if match is None:
                raise D3CalibrationError("D3 manifest episode ID is not canonical")
            source_task, episode = map(int, match.groups())
            if source_task != task_id or source.get("task_id") != task_id:
                raise D3CalibrationError("D3 manifest task identity differs")
            if episode not in D3_EPISODES:
                raise PermissionError("D3 manifest contains a sealed episode")
            if source.get("schema_version") != VISION_TEACHER_CACHE_SCHEMA_VERSION:
                raise D3CalibrationError("D3 raw cache schema differs")
            if source.get("checkpoint_sha256") != D2_CHECKPOINT_SHA256:
                raise D3CalibrationError("D3 raw cache checkpoint differs")
            if source.get("teacher_kind") != "a1_early_exit":
                raise D3CalibrationError("D3 raw cache teacher kind differs")
            if not has_complete_candidate_fm_traces(source):
                raise D3CalibrationError("D3 raw cache FM trace is incomplete")
            step = source.get("step_id")
            if (
                type(step) is not int
                or step < 0
                or step <= previous_step.get(episode, -1)
            ):
                raise D3CalibrationError("D3 episode steps are not increasing")
            previous_step[episode] = step
            ordinal = counters[episode]
            counters[episode] += 1
            rows.append(
                DevelopmentCall(
                    dataset_index=dataset_index_start + len(rows),
                    task_id=task_id,
                    episode_index=episode,
                    call_ordinal=ordinal,
                    step_id=step,
                    behavior_exit_layer=int(source["teacher_exit_layer"]),
                    cache_directory=cache_directory,
                    array_path=str(source["array_path"]),
                    source_manifest_line=line_number,
                )
            )
    if tuple(sorted(counters)) != D3_EPISODES or any(
        counters[episode] < 1 for episode in D3_EPISODES
    ):
        raise D3CalibrationError(
            "D3 task manifest does not cover all episodes 30..39"
        )
    return tuple(rows)


def resolve_calibration_call_payload(call: DevelopmentCall) -> Path:
    if call.task_id not in D3_TASK_IDS or call.episode_index not in D3_EPISODES:
        raise PermissionError("D3 cannot resolve a non-calibration call")
    root = call.cache_directory.resolve(strict=True)
    if call.cache_directory.is_symlink() or not root.is_dir():
        raise D3CalibrationError("D3 cache directory must be regular")
    relative = Path(call.array_path)
    if (
        relative.is_absolute()
        or relative.suffix != ".npz"
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise D3CalibrationError("D3 cache payload path is unsafe")
    path = _regular_file(root / relative, context="D3 cache payload")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise D3CalibrationError("D3 cache payload escapes its directory") from error
    return path


def validate_d3_prerequisites(repo_root: str | Path) -> dict[str, Any]:
    """Validate D2 authorization without touching calibration/test payloads."""

    root = Path(repo_root).resolve(strict=True)
    contract = load_d3_contract(root)
    selection = load_calibration_selection(root)
    test_selection = root / D3_TEST_SELECTION_RELATIVE_PATH
    if stream_sha256(test_selection) != D3_TEST_SELECTION_SHA256:
        raise D3CalibrationError("D3 independent-test selection metadata differs")
    attestation_path = root / D3_D2_ATTESTATION_RELATIVE_PATH
    if stream_sha256(attestation_path) != D3_D2_ATTESTATION_SHA256:
        raise D3CalibrationError("D3 D2 attestation SHA-256 differs")
    attestation = _json(attestation_path, context="D3 D2 attestation")
    if (
        attestation.get("status") != "PASS_V3_D2_FULL_DEVELOPMENT_GATE"
        or attestation.get("authorization", {}).get(
            "d3_calibration_authorized"
        )
        is not True
        or attestation.get("authorization", {}).get(
            "independent_test_authorized"
        )
        is not False
    ):
        raise D3CalibrationError("D2 attestation does not authorize D3")
    result_path = root / D3_D2_RESULT_RELATIVE_PATH
    payload_path = root / D3_D2_PAYLOAD_RELATIVE_PATH
    if stream_sha256(result_path) != D3_D2_RESULT_SHA256:
        raise D3CalibrationError("D3 D2 result SHA-256 differs")
    if stream_sha256(payload_path) != D3_D2_PAYLOAD_SHA256:
        raise D3CalibrationError("D3 D2 final-model payload SHA-256 differs")
    result = _json(result_path, context="D3 D2 nested OOF result")
    if (
        result.get("status") != "PASS_V3_D2_FULL_DEVELOPMENT_GATE"
        or result.get("source_worktree_dirty") is not False
        or result.get("primary_parameter_count") != D3_PRIMARY_PARAMETER_COUNT
        or result.get("final_lambdas") != D3_FINAL_LAMBDAS
        or result.get("next_stage", {}).get("d3_calibration_authorized")
        is not True
    ):
        raise D3CalibrationError("D3 D2 nested OOF result differs")
    return {
        "status": D3_STATUS,
        "contract_sha256": D3_CONTRACT_SHA256,
        "d2_attestation_sha256": D3_D2_ATTESTATION_SHA256,
        "d2_result_sha256": D3_D2_RESULT_SHA256,
        "d2_payload_sha256": D3_D2_PAYLOAD_SHA256,
        "calibration_selection_sha256": D3_SELECTION_SHA256,
        "calibration_clusters": len(selection),
        "independent_test_selection_sha256": D3_TEST_SELECTION_SHA256,
        "calibration_payload_opened": False,
        "independent_test_payload_opened": False,
        "contract_id": contract["contract_id"],
    }


def _state_tensor(
    state: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = state.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or tuple(value.shape) != shape
        or not bool(torch.isfinite(value).all())
    ):
        raise D3CalibrationError(f"D3 final state {name} differs")
    return value.detach().contiguous()


def _normalizer(state: Mapping[str, Any]) -> FeatureNormalizer:
    mean = _state_tensor(state, "normalizer_mean", (FEATURE_DIMENSION,))
    scale = _state_tensor(state, "normalizer_scale", (FEATURE_DIMENSION,))
    if not bool((scale > 0).all()):
        raise D3CalibrationError("D3 final normalizer scale is not positive")
    return FeatureNormalizer(mean=mean, scale=scale)


def _occurrence_model(state: Mapping[str, Any]) -> OccurrenceFit:
    if float(state.get("l2_lambda", -1.0)) != D3_FINAL_LAMBDAS["occurrence"]:
        raise D3CalibrationError("D3 occurrence lambda differs")
    return OccurrenceFit(
        normalizer=_normalizer(state),
        anchor_probability=_state_tensor(state, "anchor_probability", (2, 2)),
        weight=_state_tensor(state, "weight", (2, FEATURE_DIMENSION)),
        l2_lambda=float(state["l2_lambda"]),
        final_loss=float(state["final_loss"]),
    )


def _ordinal_model(state: Mapping[str, Any], target_index: int) -> OrdinalFit:
    name = ("step", "transition")[target_index]
    support_max = COUNT_SUPPORT_MAX[target_index]
    if (
        int(state.get("target_index", -1)) != target_index
        or int(state.get("support_max", -1)) != support_max
        or float(state.get("l2_lambda", -1.0))
        != D3_FINAL_LAMBDAS[f"ordinal_{name}"]
    ):
        raise D3CalibrationError(f"D3 ordinal {name} metadata differs")
    increments = support_max - 2
    model = OrdinalFit(
        target_index=target_index,
        support_max=support_max,
        normalizer=_normalizer(state),
        weight=_state_tensor(state, "weight", (FEATURE_DIMENSION,)),
        raw_base=_state_tensor(state, "raw_base", (2,)),
        raw_increments=_state_tensor(state, "raw_increments", (2, increments)),
        l2_lambda=float(state["l2_lambda"]),
        final_loss=float(state["final_loss"]),
    )
    stored_cutpoints = _state_tensor(state, "cutpoints", (2, support_max - 1))
    if not torch.equal(model.cutpoints, stored_cutpoints):
        raise D3CalibrationError(f"D3 ordinal {name} cutpoints differ")
    return model


def load_frozen_d2_final_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    validate_d3_prerequisites(root)
    payload = torch.load(
        root / D3_D2_PAYLOAD_RELATIVE_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != OOF_SCHEMA_VERSION
        or payload.get("role") != "development_v2"
        or payload.get("suite") != D3_SUITE
        or payload.get("dataset_payload_sha256")
        != D3_D2_DATASET_PAYLOAD_SHA256
        or payload.get("primary_parameter_count") != D3_PRIMARY_PARAMETER_COUNT
        or payload.get("final_lambdas") != D3_FINAL_LAMBDAS
        or payload.get("calibration_or_test_payload_accessed") is not False
        or payload.get("runtime_threshold_selected") is not False
    ):
        raise D3CalibrationError("D3 D2 final-model payload header differs")
    state = payload.get("final_model_state")
    if not isinstance(state, dict) or tuple(state) != HEAD_NAMES:
        raise D3CalibrationError("D3 final-model head order differs")
    _occurrence_model(state["occurrence"])
    _ordinal_model(state["ordinal_step"], 0)
    _ordinal_model(state["ordinal_transition"], 1)
    return state


def score_calibration_features(
    final_state: Mapping[str, Any],
    features: torch.Tensor,
    candidate_layer: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not isinstance(final_state, Mapping) or tuple(final_state) != HEAD_NAMES:
        raise D3CalibrationError("D3 final-state mapping differs")
    occurrence = _occurrence_model(final_state["occurrence"])
    ordinal = (
        _ordinal_model(final_state["ordinal_step"], 0),
        _ordinal_model(final_state["ordinal_transition"], 1),
    )
    probabilities = occurrence.predict(features, candidate_layer)
    conditional = (
        ordinal[0].probabilities(features, candidate_layer),
        ordinal[1].probabilities(features, candidate_layer),
    )
    expected_fraction = torch.stack(
        [
            probabilities[:, target]
            * expected_positive_count(conditional[target])
            / COUNT_SUPPORT_MAX[target]
            for target in range(2)
        ],
        dim=1,
    )
    score = probabilities[:, 0].contiguous()
    if (
        not bool(torch.isfinite(expected_fraction).all())
        or not bool(((score > 0.0) & (score < 1.0)).all())
    ):
        raise D3CalibrationError("D3 calibration prediction is invalid")
    return {
        "score": score,
        "occurrence_probability": probabilities.contiguous(),
        "ordinal_step_probability": conditional[0].contiguous(),
        "ordinal_transition_probability": conditional[1].contiguous(),
        "ordinal_expected_fraction": expected_fraction.contiguous(),
    }


def clopper_pearson_upper(
    events: int,
    trials: int,
    *,
    confidence: float = D3_UCB_CONFIDENCE,
) -> float:
    if (
        type(events) is not int
        or type(trials) is not int
        or events < 0
        or trials < 0
        or events > trials
        or confidence != D3_UCB_CONFIDENCE
    ):
        raise D3CalibrationError("D3 Clopper-Pearson arguments differ")
    if trials == 0 or events == trials:
        return 1.0
    value = float(beta.ppf(confidence, events + 1, trials - events))
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise D3CalibrationError("D3 Clopper-Pearson result is invalid")
    return value


def _calibration_vectors(
    score: torch.Tensor,
    step_mismatch: torch.Tensor,
    transition_mismatch: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
    candidate_layer: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    rows = score.numel() if isinstance(score, torch.Tensor) else -1
    if (
        not isinstance(score, torch.Tensor)
        or score.device.type != "cpu"
        or score.dtype != torch.float64
        or score.ndim != 1
        or not bool(torch.isfinite(score).all())
        or not bool(((score > 0.0) & (score < 1.0)).all())
    ):
        raise D3CalibrationError("D3 score must be finite FP64 [N] in (0,1)")
    for value, name in (
        (step_mismatch, "step mismatch"),
        (transition_mismatch, "transition mismatch"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or value.shape != (rows,)
        ):
            raise D3CalibrationError(f"D3 {name} must be bool [N]")
    for value, name in (
        (task_id, "task id"),
        (episode_index, "episode index"),
        (candidate_layer, "candidate layer"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.long
            or value.shape != (rows,)
        ):
            raise D3CalibrationError(f"D3 {name} must be int64 [N]")
    if rows < D3_CLUSTER_COUNT:
        raise D3CalibrationError("D3 calibration rows are insufficient")
    if bool((transition_mismatch & ~step_mismatch).any()):
        raise D3CalibrationError("D3 transition mismatch is not step-contained")
    if not bool(((task_id >= 0) & (task_id <= 9)).all()):
        raise D3CalibrationError("D3 task id is outside 0..9")
    if not bool(((episode_index >= 30) & (episode_index <= 39)).all()):
        raise D3CalibrationError("D3 episode index is outside 30..39")
    if not bool(((candidate_layer == 11) | (candidate_layer == 13)).all()):
        raise D3CalibrationError("D3 candidate layer is outside 11/13")
    cluster = task_id * 10 + (episode_index - 30)
    if not torch.equal(
        torch.unique(cluster, sorted=True), torch.arange(D3_CLUSTER_COUNT)
    ):
        raise D3CalibrationError("D3 does not contain all 100 calibration clusters")
    return tuple(
        value.detach().cpu().contiguous()
        for value in (
            score,
            step_mismatch,
            transition_mismatch,
            task_id,
            episode_index,
            candidate_layer,
            cluster,
        )
    )


def select_global_threshold(
    *,
    score: torch.Tensor,
    step_mismatch: torch.Tensor,
    transition_mismatch: torch.Tensor,
    task_id: torch.Tensor,
    episode_index: torch.Tensor,
    candidate_layer: torch.Tensor,
) -> dict[str, Any]:
    """Select the maximum-coverage feasible threshold, with smaller tie-break."""

    (
        scores,
        unsafe,
        _,
        _,
        _,
        _,
        cluster,
    ) = _calibration_vectors(
        score,
        step_mismatch,
        transition_mismatch,
        task_id,
        episode_index,
        candidate_layer,
    )
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_unsafe = unsafe[order]
    sorted_cluster = cluster[order]
    safe_cluster = torch.zeros(D3_CLUSTER_COUNT, dtype=torch.bool)
    false_cluster = torch.zeros(D3_CLUSTER_COUNT, dtype=torch.bool)
    raw_curve: list[tuple[float, int, int]] = []
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and bool(
            sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        group_cluster = sorted_cluster[start:end]
        safe_cluster[group_cluster] = True
        false_cluster[group_cluster[sorted_unsafe[start:end]]] = True
        raw_curve.append(
            (
                float(sorted_scores[start]),
                int(safe_cluster.sum()),
                int((false_cluster & safe_cluster).sum()),
            )
        )
        start = end
    ucb_cache: dict[tuple[int, int], float] = {}
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold, safe_count, false_count in raw_curve:
        key = (false_count, safe_count)
        upper = ucb_cache.setdefault(
            key, clopper_pearson_upper(false_count, safe_count)
        )
        coverage = safe_count / D3_CLUSTER_COUNT
        rate = false_count / safe_count if safe_count else None
        feasible = (
            coverage >= D3_MINIMUM_SAFE_COVERAGE
            and upper <= D3_FALSE_SAFE_UCB_MAX
        )
        record = {
            "threshold": threshold,
            "safe_clusters": safe_count,
            "false_safe_clusters": false_count,
            "safe_cluster_coverage": coverage,
            "false_safe_cluster_rate": rate,
            "false_safe_cluster_ucb95": upper,
            "feasible": feasible,
        }
        curve.append(record)
        if feasible and (best is None or safe_count > best["safe_clusters"]):
            best = record
    status = (
        "PASS_V3_D3_CALIBRATION_GATE"
        if best is not None
        else "NEGATIVE_V3_D3_CALIBRATION_GATE"
    )
    return {
        "status": status,
        "score_name": "step_any_mismatch_probability",
        "candidate_threshold_count": len(curve),
        "feasible_threshold_count": sum(record["feasible"] for record in curve),
        "selected": None if best is None else dict(best),
        "curve": curve,
        "checks": {
            "all_100_calibration_clusters_present": True,
            "transition_mismatch_implies_step_mismatch": True,
            "single_global_threshold_only": True,
            "exact_cluster_clopper_pearson": True,
            "always_defer_not_accepted": best is not None,
        },
    }


__all__ = [
    "CalibrationEpisode",
    "CalibrationInitialStateWindowTaskSuite",
    "D3CalibrationError",
    "D3_CLUSTER_COUNT",
    "D3_CONTRACT_SHA256",
    "D3_CONTEXT_SCHEMA_VERSION",
    "D3_CANDIDATE_SCHEMA_VERSION",
    "D3_DATASET_SCHEMA_VERSION",
    "D3_EPISODES",
    "D3_FALSE_SAFE_UCB_MAX",
    "D3_MINIMUM_SAFE_COVERAGE",
    "D3_ROLE",
    "D3_STATUS",
    "D3_SUITE",
    "D3_UCB_CONFIDENCE",
    "clopper_pearson_upper",
    "expected_calibration_seed",
    "global_calibration_episode_index",
    "load_calibration_selection",
    "load_calibration_task_calls",
    "load_d3_contract",
    "load_frozen_d2_final_state",
    "resolve_calibration_call_payload",
    "score_calibration_features",
    "select_global_threshold",
    "task_calibration_window",
    "validate_d3_prerequisites",
]
