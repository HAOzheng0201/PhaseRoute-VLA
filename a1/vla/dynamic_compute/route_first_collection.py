"""Observation-only distillation collection for route-first routing.

The collector attaches to a live frozen PhaseRoute-V3 runtime instance.  It
copies the pre-action context passed to the frozen adapter and pairs the
derived action-free feature with V3's eventual selected layer.  Wrappers are
installed on the Python objects only; no frozen source file or model state is
modified, and collection failures never participate in control.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .route_first_features import (
    ROUTE_FIRST_FEATURE_DIMENSION,
    ROUTE_FIRST_FEATURE_GROUPS,
    ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
    ROUTE_FIRST_LAYERS,
    build_route_first_context_features,
)


ROUTE_FIRST_COLLECTION_SCHEMA_VERSION = (
    "phase-route-vla.route-first-teacher-collection.v1"
)


class RouteFirstCollectionError(ValueError):
    """Raised when collector installation or publication is invalid."""


def _safe_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for name in ("episode_id", "task_id", "step_id", "call_ordinal"):
        item = value.get(name)
        if item is None or isinstance(item, (str, bool, int, float)):
            result[name] = item
    return result


class RouteFirstTeacherCollector:
    """Collect context-only features and frozen V3 depth labels in memory."""

    control_influence = False

    def __init__(self, runtime: Any) -> None:
        adapter = getattr(runtime, "adapter", None)
        if adapter is None or not callable(getattr(adapter, "begin_policy_call", None)):
            raise RouteFirstCollectionError("runtime has no compatible adapter")
        if not callable(getattr(runtime, "record_route_event", None)):
            raise RouteFirstCollectionError("runtime has no route event callback")
        self.runtime = runtime
        self.adapter = adapter
        self._original_adapter_begin = None
        self._original_record_route_event = None
        self._installed = False
        self._pending: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._errors: list[str] = []

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self._rows)

    @property
    def error_count(self) -> int:
        return len(self._errors)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def _record_error(self, stage: str, error: Exception | str) -> None:
        text = str(error) if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self._errors.append(f"{stage}: {text}")

    def install(self) -> None:
        if self._installed:
            raise RouteFirstCollectionError("collector is already installed")
        self._original_adapter_begin = self.adapter.begin_policy_call
        self._original_record_route_event = self.runtime.record_route_event
        original_begin = self._original_adapter_begin
        original_record = self._original_record_route_event

        def observed_begin(runtime_inputs: Any) -> Any:
            result = original_begin(runtime_inputs)
            self._pending = None
            # ActivePhaseRouteRuntime deliberately installs a fail-closed
            # placeholder at the start of every call, then replaces it with
            # the valid context after visual/phase preparation.  Only the
            # second adapter begin is a collectable teacher input.
            if runtime_inputs is None:
                return result
            try:
                feature = build_route_first_context_features(runtime_inputs)
                if feature.shape != (1, ROUTE_FIRST_FEATURE_DIMENSION):
                    raise RouteFirstCollectionError("collector requires batch size 1")
                current = getattr(self.runtime, "_current", None)
                context = _safe_context(
                    current.get("context") if isinstance(current, Mapping) else None
                )
                self._pending = {
                    "feature": feature[0].detach().cpu().float().numpy().copy(),
                    "context": context,
                }
            except Exception as error:
                self._record_error("adapter_begin", error)
            return result

        def observed_record(event_name: str, payload: Mapping[str, Any]) -> Any:
            result = original_record(event_name, payload)
            if event_name != "phase_route_decision":
                return result
            try:
                selected = payload.get("selected_layer")
                if type(selected) is not int or selected not in ROUTE_FIRST_LAYERS:
                    raise RouteFirstCollectionError("teacher selected an invalid layer")
                if self._pending is None:
                    raise RouteFirstCollectionError("teacher decision has no context feature")
                context = dict(self._pending["context"])
                required = ("episode_id", "task_id", "step_id", "call_ordinal")
                if any(context.get(name) is None for name in required):
                    raise RouteFirstCollectionError("teacher context metadata is incomplete")
                self._rows.append(
                    {
                        **context,
                        "feature": self._pending["feature"],
                        "teacher_layer": selected,
                        "teacher_fallback": bool(payload.get("fallback", selected == 27)),
                    }
                )
                self._pending = None
            except Exception as error:
                self._record_error("route_decision", error)
            return result

        self.adapter.begin_policy_call = observed_begin
        self.runtime.record_route_event = observed_record
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.adapter.begin_policy_call = self._original_adapter_begin
        self.runtime.record_route_event = self._original_record_route_event
        self._original_adapter_begin = None
        self._original_record_route_event = None
        self._pending = None
        self._installed = False

    def summary(self) -> dict[str, Any]:
        counts = Counter(int(row["teacher_layer"]) for row in self._rows)
        return {
            "schema_version": ROUTE_FIRST_COLLECTION_SCHEMA_VERSION,
            "control_influence": False,
            "feature_schema_version": ROUTE_FIRST_FEATURE_SCHEMA_VERSION,
            "feature_dimension": ROUTE_FIRST_FEATURE_DIMENSION,
            "rows": len(self._rows),
            "teacher_layer_counts": {
                str(layer): int(counts.get(layer, 0)) for layer in ROUTE_FIRST_LAYERS
            },
            "error_count": self.error_count,
            "errors": list(self._errors),
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        temporary = target.with_name(target.name + ".incomplete")
        if target.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        if not self._rows:
            raise RouteFirstCollectionError("cannot publish an empty teacher collection")
        if self.error_count:
            raise RouteFirstCollectionError(
                f"cannot publish collection with {self.error_count} errors"
            )
        features = np.stack([row["feature"] for row in self._rows]).astype(
            np.float32, copy=False
        )
        teacher = np.asarray(
            [row["teacher_layer"] for row in self._rows], dtype=np.int16
        )
        episode_id = np.asarray([row["episode_id"] for row in self._rows])
        task_id = np.asarray([row["task_id"] for row in self._rows], dtype=np.int16)
        step_id = np.asarray([row["step_id"] for row in self._rows], dtype=np.int32)
        call_ordinal = np.asarray(
            [row["call_ordinal"] for row in self._rows], dtype=np.int32
        )
        fallback = np.asarray(
            [row["teacher_fallback"] for row in self._rows], dtype=np.bool_
        )
        if features.shape != (len(self._rows), ROUTE_FIRST_FEATURE_DIMENSION):
            raise RouteFirstCollectionError("published feature matrix geometry differs")
        identity_rows = [
            [str(episode_id[i]), int(task_id[i]), int(step_id[i]), int(call_ordinal[i])]
            for i in range(len(self._rows))
        ]
        digest = hashlib.sha256()
        digest.update(features.tobytes(order="C"))
        digest.update(teacher.tobytes(order="C"))
        digest.update(
            json.dumps(identity_rows, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        payload_sha256 = digest.hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("xb") as output_file:
                np.savez_compressed(
                    output_file,
                    schema_version=np.asarray(ROUTE_FIRST_COLLECTION_SCHEMA_VERSION),
                    feature_schema_version=np.asarray(
                        ROUTE_FIRST_FEATURE_SCHEMA_VERSION
                    ),
                    feature_dimension=np.asarray(
                        ROUTE_FIRST_FEATURE_DIMENSION, dtype=np.int32
                    ),
                    feature_group_names=np.asarray(
                        tuple(ROUTE_FIRST_FEATURE_GROUPS), dtype=np.str_
                    ),
                    feature_group_widths=np.asarray(
                        tuple(ROUTE_FIRST_FEATURE_GROUPS.values()), dtype=np.int16
                    ),
                    control_influence=np.asarray(False, dtype=np.bool_),
                    payload_sha256=np.asarray(payload_sha256),
                    features=features,
                    teacher_layer=teacher,
                    teacher_fallback=fallback,
                    episode_id=episode_id,
                    task_id=task_id,
                    step_id=step_id,
                    call_ordinal=call_ordinal,
                )
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        result = self.summary()
        result.update(
            {
                "path": str(target),
                "payload_sha256": payload_sha256,
                "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        return result


__all__ = [
    "ROUTE_FIRST_COLLECTION_SCHEMA_VERSION",
    "RouteFirstCollectionError",
    "RouteFirstTeacherCollector",
]
