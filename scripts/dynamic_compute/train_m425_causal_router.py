"""Fit M4.25 causal routers using task-group OOF calibration only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.causal_route_router import (  # noqa: E402
    AffineBinaryHead,
    calibrate_zero_false_positive,
    fit_pca_logistic_affine,
    route_metrics,
    save_router_npz,
    sequential_routes,
)


EXPECTED_FEATURE_SCOPE = "m425_causal_route_feature_shard"
EXPECTED_FEATURE_SCHEMA = "phase-route-vla.m425-causal-route-features.v1"
EXPECTED_CACHE_INDEX_SHA256 = "6a89aafd930435c22ad2eac2665c2c88a304cc8b35843e394e198da4d5f5c3c8"
EXPECTED_RECORDS = 154
DEV_TASKS = tuple(range(8))
TEST_TASKS = (8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-result", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--layer-norm-eps", type=float, default=1e-6)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_descriptor_sha256(descriptors: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        sorted((dict(item) for item in descriptors), key=lambda item: int(item["task_id"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FeatureTable:
    layer11: np.ndarray
    layer13: np.ndarray
    proprio: np.ndarray
    task_id: np.ndarray
    step_id: np.ndarray
    teacher_route: np.ndarray
    identity_sha256: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    input_files: tuple[Mapping[str, Any], ...]
    cache_index_sha256: str


def load_feature_table(
    paths: Sequence[Path],
    *,
    checkpoint_sha256: str,
    expected_cache_index_sha256: str = EXPECTED_CACHE_INDEX_SHA256,
) -> FeatureTable:
    if len(paths) != 4:
        raise ValueError("M4.25 requires exactly four frozen feature shards")
    chunks: list[dict[str, np.ndarray]] = []
    rows: list[Mapping[str, Any]] = []
    descriptors: list[Mapping[str, Any]] = []
    inputs: list[Mapping[str, Any]] = []
    seen_tasks: set[int] = set()
    for source_path in paths:
        path = source_path.resolve()
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "PASS"
            or result.get("scope") != EXPECTED_FEATURE_SCOPE
            or result.get("schema_version") != EXPECTED_FEATURE_SCHEMA
        ):
            raise ValueError(f"invalid/non-PASS feature result: {path}")
        if result.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"feature/checkpoint SHA mismatch: {path}")
        if not all(bool(value) for value in result.get("local_checks", {}).values()):
            raise ValueError(f"feature shard local checks failed: {path}")
        arrays_path = Path(result["arrays_path"])
        if not arrays_path.is_file() or sha256_file(arrays_path) != result["arrays_sha256"]:
            raise ValueError(f"feature array SHA mismatch: {path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            chunk = {name: arrays[name].copy() for name in arrays.files}
        count = int(result["records"])
        required = {
            "layer11_hidden",
            "layer13_hidden",
            "normalized_proprio",
            "task_id",
            "step_id",
            "teacher_route",
            "identity_sha256",
        }
        if not required.issubset(chunk):
            raise KeyError(f"feature shard misses arrays: {sorted(required - set(chunk))}")
        if any(chunk[name].shape[0] != count for name in required):
            raise ValueError(f"feature shard arrays have inconsistent rows: {path}")
        shard_rows = result.get("rows", [])
        if len(shard_rows) != count:
            raise ValueError(f"feature shard row metadata differs: {path}")
        array_hashes = [value.decode("ascii") for value in chunk["identity_sha256"]]
        json_hashes = [str(row["identity_sha256"]) for row in shard_rows]
        if array_hashes != json_hashes:
            raise ValueError(f"feature identity metadata differs: {path}")
        shard_tasks = {int(value) for value in chunk["task_id"].tolist()}
        if shard_tasks & seen_tasks:
            raise ValueError("feature shards overlap task IDs")
        seen_tasks.update(shard_tasks)
        chunks.append(chunk)
        rows.extend(shard_rows)
        descriptors.extend(result["input_manifests"])
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "arrays_path": str(arrays_path.resolve()),
                "arrays_sha256": result["arrays_sha256"],
            }
        )

    if seen_tasks != set(range(10)):
        raise ValueError(f"feature task grid differs: {sorted(seen_tasks)}")
    descriptor_tasks = [int(item["task_id"]) for item in descriptors]
    if len(descriptor_tasks) != 10 or len(set(descriptor_tasks)) != 10:
        raise ValueError("input manifest descriptors are incomplete or duplicated")
    cache_sha = canonical_descriptor_sha256(descriptors)
    if cache_sha != expected_cache_index_sha256:
        raise ValueError(f"teacher-cache index SHA differs: {cache_sha}")

    names = (
        "layer11_hidden",
        "layer13_hidden",
        "normalized_proprio",
        "task_id",
        "step_id",
        "teacher_route",
        "identity_sha256",
    )
    merged = {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in names}
    if merged["task_id"].shape[0] != EXPECTED_RECORDS or len(rows) != EXPECTED_RECORDS:
        raise ValueError("merged feature table does not have 154 rows")
    identities = [value.decode("ascii") for value in merged["identity_sha256"]]
    if len(set(identities)) != EXPECTED_RECORDS:
        raise ValueError("merged feature identities are not unique")
    if not set(np.unique(merged["teacher_route"]).tolist()).issubset({11, 13, 27}):
        raise ValueError("feature table contains an unsupported teacher route")
    for name in ("layer11_hidden", "layer13_hidden", "normalized_proprio"):
        if not np.isfinite(merged[name]).all():
            raise ValueError(f"merged {name} contains non-finite values")

    order = np.lexsort((merged["step_id"], merged["task_id"]))
    ordered_rows = tuple(rows[int(index)] for index in order)
    return FeatureTable(
        layer11=merged["layer11_hidden"][order].astype(np.float64),
        layer13=merged["layer13_hidden"][order].astype(np.float64),
        proprio=merged["normalized_proprio"][order].astype(np.float64),
        task_id=merged["task_id"][order].astype(np.int64),
        step_id=merged["step_id"][order].astype(np.int64),
        teacher_route=merged["teacher_route"][order].astype(np.int64),
        identity_sha256=merged["identity_sha256"][order],
        rows=ordered_rows,
        input_files=tuple(inputs),
        cache_index_sha256=cache_sha,
    )


def low_cost_features(table: FeatureTable) -> np.ndarray:
    step = np.minimum(table.step_id.astype(np.float64), 250.0)[:, None] / 250.0
    return np.concatenate([table.proprio, step], axis=1)


def _fit_heads(
    feature11: np.ndarray,
    feature13: np.ndarray,
    teacher_route: np.ndarray,
    train_mask: np.ndarray,
    *,
    pca_rank: int,
    l2: float,
    max_iter: int,
    eps: float,
) -> tuple[AffineBinaryHead, AffineBinaryHead]:
    head11 = fit_pca_logistic_affine(
        feature11[train_mask],
        (teacher_route[train_mask] == 11).astype(np.int64),
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        eps=eps,
    )
    train13 = train_mask & (teacher_route >= 13)
    head13 = fit_pca_logistic_affine(
        feature13[train13],
        (teacher_route[train13] == 13).astype(np.int64),
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        eps=eps,
    )
    return head11, head13


def fit_grouped_router(
    feature11: np.ndarray,
    feature13: np.ndarray,
    teacher_route: np.ndarray,
    task_id: np.ndarray,
    *,
    dev_tasks: Sequence[int] = DEV_TASKS,
    pca_rank: int = 32,
    l2: float = 1.0,
    max_iter: int = 100,
    eps: float = 1e-6,
) -> dict[str, Any]:
    dev_tasks = tuple(int(task) for task in dev_tasks)
    dev_mask = np.isin(task_id, dev_tasks)
    if set(np.unique(task_id[dev_mask]).tolist()) != set(dev_tasks):
        raise ValueError("development task grid is incomplete")
    p11 = np.full(task_id.shape, np.nan, dtype=np.float64)
    p13 = np.full(task_id.shape, np.nan, dtype=np.float64)
    folds = []
    for held_task in dev_tasks:
        held = dev_mask & (task_id == held_task)
        train = dev_mask & (task_id != held_task)
        head11, head13 = _fit_heads(
            feature11,
            feature13,
            teacher_route,
            train,
            pca_rank=pca_rank,
            l2=l2,
            max_iter=max_iter,
            eps=eps,
        )
        p11[held] = head11.probabilities(feature11[held], eps=eps)
        p13[held] = head13.probabilities(feature13[held], eps=eps)
        folds.append(
            {
                "held_task": held_task,
                "train_rows": int(train.sum()),
                "held_rows": int(held.sum()),
                "head11_rank": head11.pca_rank,
                "head13_rank": head13.pca_rank,
            }
        )
    if not np.isfinite(p11[dev_mask]).all() or not np.isfinite(p13[dev_mask]).all():
        raise RuntimeError("OOF prediction grid is incomplete")
    label11 = (teacher_route[dev_mask] == 11).astype(np.int64)
    threshold11 = calibrate_zero_false_positive(p11[dev_mask], label11)
    eligible13 = dev_mask & (teacher_route >= 13)
    label13 = (teacher_route[eligible13] == 13).astype(np.int64)
    threshold13 = calibrate_zero_false_positive(p13[eligible13], label13)
    oof_routes = sequential_routes(
        p11[dev_mask],
        p13[dev_mask],
        threshold11=threshold11,
        threshold13=threshold13,
    )
    oof_metrics = route_metrics(oof_routes, teacher_route[dev_mask])
    final11, final13 = _fit_heads(
        feature11,
        feature13,
        teacher_route,
        dev_mask,
        pca_rank=pca_rank,
        l2=l2,
        max_iter=max_iter,
        eps=eps,
    )
    return {
        "head11": final11,
        "head13": final13,
        "threshold11": threshold11,
        "threshold13": threshold13,
        "oof_probability11": p11[dev_mask],
        "oof_probability13": p13[dev_mask],
        "oof_routes": oof_routes,
        "oof_teacher": teacher_route[dev_mask].copy(),
        "oof_metrics": oof_metrics,
        "folds": folds,
        "development_rows": int(dev_mask.sum()),
    }


def router_probabilities_from_npz(
    checkpoint_path: Path,
    feature11: np.ndarray,
    feature13: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    from a1.vla.dynamic_compute.causal_route_router import CausalRouteRouter

    router = CausalRouteRouter.from_npz(checkpoint_path)
    p11 = (
        router.probability(11, __import__("torch").from_numpy(feature11.astype(np.float32)))
        .detach()
        .cpu()
        .numpy()
    )
    p13 = (
        router.probability(13, __import__("torch").from_numpy(feature13.astype(np.float32)))
        .detach()
        .cpu()
        .numpy()
    )
    return p11, p13, router.config.threshold11, router.config.threshold13


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.seed != 20260825 or args.pca_rank != 32 or args.l2 != 1.0:
        raise ValueError("M4.25 v1 freezes seed=20260825, PCA rank=32 and L2=1.0")
    table = load_feature_table(
        args.feature_result,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    if int(np.isin(table.task_id, DEV_TASKS).sum()) != 113:
        raise ValueError("development split must contain 113 rows")
    if int(np.isin(table.task_id, TEST_TASKS).sum()) != 41:
        raise ValueError("sealed test split must contain 41 rows")

    hidden_fit = fit_grouped_router(
        table.layer11,
        table.layer13,
        table.teacher_route,
        table.task_id,
        pca_rank=args.pca_rank,
        l2=args.l2,
        max_iter=args.max_iter,
        eps=args.layer_norm_eps,
    )
    cheap = low_cost_features(table)
    lowcost_fit = fit_grouped_router(
        cheap,
        cheap,
        table.teacher_route,
        table.task_id,
        pca_rank=args.pca_rank,
        l2=args.l2,
        max_iter=args.max_iter,
        eps=args.layer_norm_eps,
    )
    if int(hidden_fit["oof_metrics"]["false_shallow"]) != 0:
        raise RuntimeError("calibrated hidden router is not fail-closed on OOF")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    hidden_path = args.output_dir / "hidden_router.npz"
    lowcost_path = args.output_dir / "step_proprio_router.npz"
    common_extra = {
        "seed": np.asarray(args.seed, dtype=np.int64),
        "checkpoint_sha256": args.checkpoint_sha256,
        "cache_index_sha256": table.cache_index_sha256,
        "development_tasks": np.asarray(DEV_TASKS, dtype=np.int16),
        "sealed_test_tasks": np.asarray(TEST_TASKS, dtype=np.int16),
    }
    save_router_npz(
        hidden_path,
        head11=hidden_fit["head11"],
        head13=hidden_fit["head13"],
        threshold11=hidden_fit["threshold11"],
        threshold13=hidden_fit["threshold13"],
        layer_norm_eps=args.layer_norm_eps,
        extra_arrays={**common_extra, "feature_kind": "layer_hidden"},
    )
    save_router_npz(
        lowcost_path,
        head11=lowcost_fit["head11"],
        head13=lowcost_fit["head13"],
        threshold11=lowcost_fit["threshold11"],
        threshold13=lowcost_fit["threshold13"],
        layer_norm_eps=args.layer_norm_eps,
        extra_arrays={**common_extra, "feature_kind": "step_proprio"},
    )
    hidden_sha = sha256_file(hidden_path)
    lowcost_sha = sha256_file(lowcost_path)

    # Reload from disk and require bitwise-identical route decisions before reporting fit.
    dev = np.isin(table.task_id, DEV_TASKS)
    reloaded_p11, reloaded_p13, t11, t13 = router_probabilities_from_npz(
        hidden_path, table.layer11[dev], table.layer13[dev]
    )
    reloaded_routes = sequential_routes(
        reloaded_p11, reloaded_p13, threshold11=t11, threshold13=t13
    )
    direct_p11 = hidden_fit["head11"].probabilities(table.layer11[dev], eps=args.layer_norm_eps)
    direct_p13 = hidden_fit["head13"].probabilities(table.layer13[dev], eps=args.layer_norm_eps)
    direct_routes = sequential_routes(
        direct_p11,
        direct_p13,
        threshold11=hidden_fit["threshold11"],
        threshold13=hidden_fit["threshold13"],
    )
    reload_exact = bool(np.array_equal(reloaded_routes, direct_routes))

    result = {
        "status": "PASS" if reload_exact else "FAIL",
        "scope": "m425_causal_router_grouped_fit",
        "checkpoint_sha256": args.checkpoint_sha256,
        "cache_index_sha256": table.cache_index_sha256,
        "seed": args.seed,
        "fit_config": {
            "pca_rank": args.pca_rank,
            "l2": args.l2,
            "max_iter": args.max_iter,
            "layer_norm_eps": args.layer_norm_eps,
            "development_tasks": list(DEV_TASKS),
            "sealed_test_tasks": list(TEST_TASKS),
        },
        "records": EXPECTED_RECORDS,
        "development_rows": 113,
        "sealed_test_rows": 41,
        "hidden_router": {
            "path": str(hidden_path.resolve()),
            "sha256": hidden_sha,
            "threshold11": hidden_fit["threshold11"],
            "threshold13": hidden_fit["threshold13"],
            "oof_metrics": hidden_fit["oof_metrics"],
            "folds": hidden_fit["folds"],
        },
        "step_proprio_router": {
            "path": str(lowcost_path.resolve()),
            "sha256": lowcost_sha,
            "threshold11": lowcost_fit["threshold11"],
            "threshold13": lowcost_fit["threshold13"],
            "oof_metrics": lowcost_fit["oof_metrics"],
            "folds": lowcost_fit["folds"],
        },
        "engineering_checks": {
            "complete_feature_table": table.task_id.size == EXPECTED_RECORDS,
            "development_rows": int(dev.sum()) == 113,
            "sealed_test_rows": int((~dev).sum()) == 41,
            "oof_hidden_finite": bool(
                np.isfinite(hidden_fit["oof_probability11"]).all()
                and np.isfinite(hidden_fit["oof_probability13"]).all()
            ),
            "oof_hidden_zero_false_shallow": int(hidden_fit["oof_metrics"]["false_shallow"]) == 0,
            "checkpoint_reload_exact": reload_exact,
        },
        "feature_inputs": list(table.input_files),
    }
    if not all(result["engineering_checks"].values()):
        result["status"] = "FAIL"
    result_path = args.output_dir / "fit_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
