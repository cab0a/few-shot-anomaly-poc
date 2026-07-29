# Pre-Evaluation Freeze

## Status

The v0.1 evaluation definition is frozen before any final-test scoring, label
join, metric calculation, or project decision.

The controlling machine-readable record is
[`artifacts/v0.1/freeze/pre-evaluation-freeze.json`](../artifacts/v0.1/freeze/pre-evaluation-freeze.json).

## Frozen Source and CI

- Evaluation source commit:
  `fd9857acb29903fadb570680ecb5d4d8ebf5a5aa`
- GitHub Actions run:
  [CI #21](https://github.com/cab0a/few-shot-anomaly-poc/actions/runs/30434900673)
- CI conclusion: `success`
- Frozen tree SHA-256:
  `cf9460eb919025417c988771926e00d06641ea63b242c397be466dd7823970f9`
- Freeze date: `2026-07-29`

The checkpoint commit itself adds only the checkpoint generator, verifier,
record, tests, and public status documentation. The evaluation source paths in
the record match the CI-verified source commit.

## Frozen Scope

The record fixes the SHA-256 identity of:

- `configs/v0.1.yaml`
- `pyproject.toml` and `uv.lock`
- CI workflow and license boundary files
- dataset acquisition record and normal partition manifest
- problem, method, dependency, evaluation, and artifact-contract documents
- the machine-readable artifact schema
- shared preprocessing and both method implementations
- calibration, classification, label reveal, metrics, latency, failure
  selection, hard-gate decision, and artifact serialization code

It also records the exact runtime and development dependency pins.

## Fixed Data Selection

- Dataset scope: VisA `pcb1`
- Seed: `42`
- Selection procedure: `sha256-path-ranking-v1`
- Reference count: `20`
- Calibration count: `884`
- Final-test source: the pinned official one-class test split
- Official split revision:
  `2a692ab575001cbde74d402d897a7286086c6199`
- Official split SHA-256:
  `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`
- Normal partition manifest SHA-256:
  `953478e04c20d74cc1994022e4a757388123b4db4020a10898fcee74b1a192a7`

All 20 relative reference IDs are embedded in the machine-readable record.
Partition overlap remains forbidden.

## Fixed Decision Rules

The checkpoint preserves the calibration quantile and strict comparison rule,
CPU timing boundary and summary rules, deterministic failure-case selection,
and the six hard gates in their application order.

The fixed hard gates remain:

1. Final-test normal FPR `<= 0.05`
2. Final-test anomaly recall `>= 0.90`
3. CPU p95 scoring latency `<= 1.0` second per image
4. Normal reference count `<= 20`
5. Anomaly training labels used equals `false`
6. Reproducibility verified equals `true`

A weighted aggregate score and a hard-gate waiver remain disallowed.

## Boundary State at Freeze

At this checkpoint:

- final-test scoring has not started
- final-test labels have not been joined to classifications
- final-test metrics have not been calculated
- no final-test decision has been recorded
- no `first-fixed-final-test` artifact bundle exists

The earlier `synthetic-e2e` bundle is excluded from these claims because it
uses generated records and is explicitly marked `run_kind=synthetic`.

## Change Policy

Any change to a frozen file invalidates this checkpoint. An implementation
defect would require a documented replacement checkpoint before another
eligible final-test run; it cannot silently alter this record or overwrite a
result.

Post-freeze additions may verify local data integrity, orchestrate the already
frozen primitives without changing them, write non-overwritable result
artifacts, and document results. Final-test evidence cannot change the
threshold rule, hard gates, failure policy, or artifact contract.

Run:

```bash
uv run --locked --no-sync pytest tests/test_freeze_checkpoint.py
```

The test recomputes every frozen file digest and the aggregate tree digest.
