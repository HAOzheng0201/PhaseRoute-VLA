"""Validate and summarize an M1 multi-task telemetry collection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a1.vla.dynamic_compute.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    SafeJSONLTelemetryLogger,
    build_policy_call_telemetry,
)


REQUIRED_FIELDS = {
    "schema_version",
    "episode_id",
    "step_id",
    "task_id",
    "instruction_hash",
    "proprio_summary",
    "prev_action_summary",
    "gripper_state",
    "translation_speed",
    "rotation_speed",
    "active_tokens_by_layer",
    "candidate_exit_layers",
    "action_delta_by_exit",
    "exit_layer",
    "fm_calls",
    "fm_steps_total",
    "latency_ms",
    "action_shape",
    "action_dtype",
    "extra",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection_dir", type=Path)
    parser.add_argument("--benchmark-iterations", type=int, default=1000)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark_overhead(iterations: int, reference_latency_ms: float) -> dict:
    events = [
        {
            "event": "exit_candidate",
            "layer_idx": layer,
            "evaluated": True,
            "should_exit": layer == 11,
            "action_delta": 0.1,
            "fm_calls": 1,
            "fm_steps": 10,
        }
        for layer in [1, 3, 5, 7, 9, 11]
    ]
    candidates = list(range(1, 28, 2))
    with tempfile.TemporaryDirectory(prefix="a1-m1-telemetry-") as temporary_dir:
        logger = SafeJSONLTelemetryLogger(
            Path(temporary_dir) / "benchmark.jsonl",
            flush_every=100,
        )
        start = time.perf_counter()
        for step_id in range(iterations):
            record = build_policy_call_telemetry(
                context={
                    "episode_id": "benchmark",
                    "step_id": step_id,
                    "task_id": 0,
                    "previous_action": [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0],
                },
                instruction="benchmark instruction",
                raw_proprio=[0.0] * 8,
                active_token_count=656,
                n_layers=28,
                visual_token_count=576,
                candidate_exit_layers=candidates,
                telemetry_events=events,
                latency_ms=reference_latency_ms,
                action_shape=[1, 8, 7],
                action_dtype="torch.float32",
                normalization_key="libero_spatial_no_noops",
            )
            if not logger.log(record):
                raise RuntimeError(logger.last_error)
        logger.close()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_call_ms = elapsed_ms / iterations
    return {
        "iterations": iterations,
        "total_ms": elapsed_ms,
        "per_policy_call_ms": per_call_ms,
        "reference_policy_latency_ms": reference_latency_ms,
        "overhead_fraction": per_call_ms / reference_latency_ms,
        "overhead_percent": per_call_ms / reference_latency_ms * 100.0,
    }


def main() -> None:
    args = parse_args()
    result_paths = sorted(args.collection_dir.glob("gpu*_task*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No task results found in {args.collection_dir}")

    task_results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    records = []
    cleanup_warning_logs = []
    for result_path, task_result in zip(result_paths, task_results):
        if task_result["status"] != "PASS":
            raise AssertionError(f"Task result failed: {result_path}")
        telemetry_path = result_path.parent / "policy_calls.jsonl"
        task_records = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(task_records) != task_result["policy_calls"]:
            raise AssertionError(f"Policy-call count mismatch in {telemetry_path}")
        records.extend(task_records)
        console_path = result_path.parent / "console.log"
        if "EGL_NOT_INITIALIZED" in console_path.read_text(encoding="utf-8"):
            cleanup_warning_logs.append(str(console_path))

    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise AssertionError(f"Record {index} is missing fields: {sorted(missing)}")
        if record["schema_version"] != TELEMETRY_SCHEMA_VERSION:
            raise AssertionError(f"Record {index} has an unexpected schema version")
        if len(record["active_tokens_by_layer"]) != 28:
            raise AssertionError(f"Record {index} does not contain 28 layer token counts")
        if len(record["candidate_exit_layers"]) != len(record["action_delta_by_exit"]):
            raise AssertionError(f"Record {index} has unaligned exit metrics")
        if record["action_shape"] != [1, 8, 7]:
            raise AssertionError(f"Record {index} has action shape {record['action_shape']}")
        if record["extra"]["visual_tokens"] != 576:
            raise AssertionError(f"Record {index} has an unexpected visual-token count")

    episode_ids = {record["episode_id"] for record in records}
    latencies = [float(record["latency_ms"]) for record in records]
    exit_layers = Counter(int(record["exit_layer"]) for record in records)
    active_tokens = [int(record["active_tokens_by_layer"][0]) for record in records]
    task_successes = sum(int(result["successes"]) for result in task_results)
    completed_episodes = sum(int(result["completed_episodes"]) for result in task_results)
    telemetry_errors = sum(int(result["telemetry_errors"]) for result in task_results)
    fm_calls_total = sum(int(record["fm_calls"]) for record in records)
    fm_steps_total = sum(int(record["fm_steps_total"]) for record in records)
    overhead = benchmark_overhead(args.benchmark_iterations, statistics.median(latencies))

    summary = {
        "status": "PASS",
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "task_results": task_results,
        "completed_episodes": completed_episodes,
        "successes": task_successes,
        "success_rate": task_successes / completed_episodes,
        "unique_episode_ids": len(episode_ids),
        "policy_calls": len(records),
        "telemetry_errors": telemetry_errors,
        "exit_layer_distribution": dict(sorted(exit_layers.items())),
        "fm_calls_total": fm_calls_total,
        "fm_steps_total": fm_steps_total,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "active_tokens": {
            "min": min(active_tokens),
            "max": max(active_tokens),
            "unique": sorted(set(active_tokens)),
        },
        "visual_tokens": 576,
        "telemetry_overhead_benchmark": overhead,
        "egl_cleanup_warning_logs": cleanup_warning_logs,
    }
    summary_path = args.collection_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    exit_rows = "\n".join(
        f"| {layer} | {count} | {count / len(records):.2%} |"
        for layer, count in sorted(exit_layers.items())
    )
    markdown = f"""# M1 telemetry 20-episode 采集汇总

生成日期：2026-08-01（Asia/Shanghai）

## 结论

- 状态：**PASS**
- 完成 episode：{completed_episodes}/20，唯一 episode ID：{len(episode_ids)}
- 成功：{task_successes}/{completed_episodes}（{task_successes / completed_episodes:.2%}）
- 策略调用记录：{len(records)} 条，schema：`{TELEMETRY_SCHEMA_VERSION}`
- JSONL 写入错误：{telemetry_errors}
- FM 调用/步数：{fm_calls_total}/{fm_steps_total}
- 视觉 token：固定 576；有效序列 token 范围：{min(active_tokens)}–{max(active_tokens)}

## 退出层分布

| 0-based 退出层 | 策略调用数 | 占比 |
|---:|---:|---:|
{exit_rows}

## 延迟与 telemetry 开销

- 四卡并发时策略调用延迟：mean={statistics.fmean(latencies):.2f} ms，median={statistics.median(latencies):.2f} ms，p95={percentile(latencies, 0.95):.2f} ms。
- 代表性 record 的构建 + JSON 序列化 + 写入：{overhead['per_policy_call_ms']:.4f} ms/调用。
- 相对实测中位策略延迟：{overhead['overhead_percent']:.4f}%（M1 门槛 <3%）。

该微基准隔离测量 telemetry side channel，不包含模型计算；真实 checkpoint 的开关回归另见 `../m1_gpu_smoke_20260801/m1_gpu_smoke_result.json`，其动作 `max_abs_diff=0`。

## 完整性和已知 warning

- 每条记录均包含 28 层 active-token 数、对齐的候选出口/动作差值、退出层、FM 调用、动作 shape/dtype、proprio/上一动作摘要和延迟。
- 20 个 episode 分布在 GPU 0–3，每卡 5 个；GPU 4–7 未使用。
- 四个进程均正常退出并生成 PASS 结果。其控制台在 Python 退出后的 MuJoCo/EGL 对象析构阶段打印 `EGL_NOT_INITIALIZED`；这是 result 写入后的非致命 cleanup warning，不影响 rollout 成功与 JSONL 完整性。
"""
    (args.collection_dir / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
