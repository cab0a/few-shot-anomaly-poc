# Synthetic End-to-End Evaluation

## 日本語概要

本書は、画像やVisAの評価値を使わず、生成した記録だけで校正からスコア、分類、ラベル結合、指標、失敗例、採否判定、成果物出力までを接続する統合試験を説明します。意図した合否結果、完全性、再生成方法、性能を主張できない境界の詳細は以下の英語本文を参照してください。

---

## English Summary

This document explains the deterministic synthetic fixture that connects calibration, scoring, classification, label reveal, metrics, failure selection, gate decisions, and immutable artifact output. It is plumbing evidence and makes no VisA performance claim.

## Status

This is a deterministic integration fixture made only from generated records.
It is not a VisA experiment, CPU benchmark, method comparison, or project
decision.

The committed bundle is
[`artifacts/v0.1/evaluation/synthetic-e2e`](../artifacts/v0.1/evaluation/synthetic-e2e).
Its generator source commit is
`7193a89e0cff8d543c0f7274e834d902026752d5`.

## Purpose

The fixture proves that the already fixed primitives connect without changing
their boundaries:

```text
synthetic normal calibration scores
    -> normal-only threshold calibration
    -> synthetic label-free final scores
    -> fixed-threshold batch classification
    -> exact-path label reveal
    -> image-level metrics
    -> mechanical failure-case selection
    -> hard-gate decision
    -> versioned JSON/CSV artifacts
```

The ECC and Patch HOG score records use their real v0.1 dataclass contracts,
but their values are constructed directly. No method is fitted, no image is
opened, and no dataset path or label is read.

## Deliberate Fixture Outcomes

Each method receives:

- 20 synthetic normal calibration scores
- 20 synthetic final normal records
- 20 synthetic final anomaly records
- one deliberate false positive
- two deliberate false negatives
- no score-generation failure
- a synthetic p95 latency record below one second

This produces exactly 5% normal FPR and 90% anomaly recall. The boundary values
are intentional test inputs, not observed performance. The resulting
method-level `ADOPT` labels prove only that passing inputs produce the
preregistered output label.

AUROC and AUPRC values in the bundle are also fixture outputs. They must not be
quoted as evidence for either method.

## Artifact Integrity

The bundle contains label-free calibration and final scores, label-free
classifications, separately revealed labels, metrics, synthetic latency
observations, selected failures, and hard-gate decisions for both method
identifiers.

`artifact-manifest.json` records:

- `run_kind=synthetic`
- `dataset=synthetic-records`
- `category=not-applicable`
- the generator source commit
- fixed configuration and synthetic partition digests
- every non-manifest file path, record count, and SHA-256

Tests regenerate the complete bundle in a temporary directory and require
byte-for-byte equality with the committed files. The writer refuses to
overwrite an existing bundle and rejects a tampered intermediate result before
creating output.

## Reproduction

From the repository root:

```bash
uv sync --locked
uv run --locked --no-sync python scripts/run_synthetic_evaluation.py \
  --output-root /tmp/few-shot-anomaly-poc-synthetic-reproduction \
  --source-commit 7193a89e0cff8d543c0f7274e834d902026752d5
```

Compare the generated directory with the committed bundle. Do not target the
committed artifact parent because overwrite is intentionally rejected.

## Claim Boundary

The fixture does not establish:

- VisA `pcb1` performance
- actual CPU latency
- behavior of the fitted ECC or Patch HOG models
- robustness to image variation
- suitability for another validation phase
- the final v0.1 decision

Those claims require the frozen real-data workflow and remain pending.
