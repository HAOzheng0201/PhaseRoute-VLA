#!/usr/bin/env python3
"""Aggregate an exact grid of route-first teacher shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.route_first_dataset import (  # noqa: E402
    aggregate_route_first_teacher_shards,
    save_route_first_teacher_aggregate,
)


def _indices(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must be unique non-negative integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_ids = _indices(args.task_ids, "task-ids")
    episode_indices = _indices(args.episode_indices, "episode-indices")
    output = args.output.resolve()
    summary_path = args.summary.resolve()
    if output.parent != summary_path.parent:
        raise ValueError("aggregate NPZ and summary JSON must share one directory")
    if summary_path.exists() or summary_path.with_name(
        summary_path.name + ".incomplete"
    ).exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    arrays, summary = aggregate_route_first_teacher_shards(
        args.input,
        expected_task_ids=task_ids,
        expected_episode_indices=episode_indices,
    )
    result = save_route_first_teacher_aggregate(output, arrays, summary)
    temporary = summary_path.with_name(summary_path.name + ".incomplete")
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("x", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2, allow_nan=False)
            output_file.write("\n")
        temporary.replace(summary_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
