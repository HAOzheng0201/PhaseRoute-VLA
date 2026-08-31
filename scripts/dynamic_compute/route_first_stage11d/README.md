# Route-first Stage 11D state pipeline

This directory contains the CPU-only generated-state boundary for Stage 11D.
It does not collect policy observations, replay action candidates, train a
router, or control an environment with the new method.

The execution order is:

```bash
# 1. Machine-readable static readiness (published by the repository workflow).
python scripts/validate_route_first_stage11d_state_runner.py \
  --output results/route_first/route_first_stage11d_state_runner_readiness.json

# 2. Non-mutating preflight after the readiness artifact is committed.
python scripts/dynamic_compute/route_first_stage11d/generate_states.py \
  --max-workers 8 --preflight-only

# 3. CPU/OSMesa generation: 200 schedule rows × two isolated processes.
python scripts/dynamic_compute/route_first_stage11d/generate_states.py \
  --max-workers 8

# 4. Byte-exact two-pass validation and immutable payload publication.
python scripts/dynamic_compute/route_first_stage11d/aggregate_states.py
```

Generation is fail-closed and non-resumable by design. If an initially solved,
invalid, duplicated, or nondeterministic state is observed, retain the
`.incomplete` evidence and investigate it. Do not replace its state seed or
delete the evidence to obtain a more favorable sample.

Only after `state_attestation.json` and `fresh_states.pt` are hash-bound in a
clean commit may the original-A1 observation-only collection runner be added.
