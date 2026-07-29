# Local VisA `pcb1` Integrity Verification

## 日本語概要

本書は、評価に使用したローカルのVisA `pcb1`が、固定したアーカイブ、公式分割、ファイル構成、件数、チェックサムと一致することを確認した記録です。正常参照の学習や最終評価より前に行った検証手順と、入力確認に限定される主張範囲の詳細は以下の英語本文を参照してください。

---

## English Summary

This checkpoint verifies that the local VisA `pcb1` asset matches the fixed archive observation, official split, structure, counts, manifests, and checksums before fitting or scoring. It is input-integrity evidence, not performance evidence.

## Status

The fixed local VisA `pcb1` asset passed this checkpoint before
normal-reference fitting, threshold calibration, or final-test scoring.

This is an input-integrity result, not an anomaly-detection result.

## Fixed identities

| Item | Required identity |
| --- | --- |
| Archive | `VisA_20220922.tar` |
| Archive byte count | `1,929,840,640` |
| Previously observed archive SHA-256 | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| Pinned split revision | `2a692ab575001cbde74d402d897a7286086c6199` |
| Pinned split SHA-256 | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |
| Dataset category | `pcb1` |
| Dataset license | CC BY 4.0 |

The archive SHA-256 is an observation recorded during the earlier acquisition
checkpoint. The official project and AWS Registry of Open Data do not publish
an independent SHA-256 for this object. Matching it establishes identity with
the previously acquired object; it is not an upstream checksum assertion.

## Verification procedure

The verifier performs the following checks in order:

1. Read the committed dataset record and require the fixed category, license,
   byte count, digests, member count, and category counts.
2. Hash the local archive and pinned official split.
3. Apply the safe tar member rules to the complete archive, including members
   outside `pcb1`.
4. Build the exact regular-file inventory for the selected `pcb1` subtree.
5. Reject missing, extra, duplicate, linked, or unsupported extracted entries.
6. Compare every selected extracted file with the corresponding archive member
   by byte count and SHA-256.
7. Require the extracted image and mask path sets to equal the pinned official
   split path sets.
8. Verify aggregate train/test counts and the fixed `image_anno.csv` digest.
9. Write one deterministic aggregate JSON record only after every check passes.

The completed check compared all `1,205` selected files, totaling
`296,989,825` bytes. The resulting path-size-content tree has SHA-256
`ce716f7ad476efb5b5aea630773330fc591f520db517b267509c766f10485b7a`.
There were zero missing or extra image paths, zero missing or extra mask paths,
and zero archive/extraction content mismatches.

## Reproduction

After the standard download and extraction workflow in
[`data/README.md`](../data/README.md), run:

```bash
uv run --locked --no-sync python scripts/verify_visa_pcb1_integrity.py
```

To verify an already available local copy without placing raw data inside the
repository:

```bash
uv run --locked --no-sync python scripts/verify_visa_pcb1_integrity.py \
  --archive /path/to/VisA_20220922.tar \
  --dataset-root /path/to/extracted/VisA_20220922
```

The command refuses to overwrite
`artifacts/v0.1/data/pcb1-local-integrity.json`. Reproduction should therefore
write to a deliberate alternate path or compare against the committed record:

```bash
uv run --locked --no-sync python scripts/verify_visa_pcb1_integrity.py \
  --output /tmp/pcb1-local-integrity.json
cmp artifacts/v0.1/data/pcb1-local-integrity.json \
  /tmp/pcb1-local-integrity.json
```

## Evaluation boundary

This checkpoint:

- does not decode or display an image
- does not export per-path final-test labels
- does not join labels to scores
- does not fit either anomaly-detection method
- does not calculate an anomaly score, threshold, metric, or latency
- does not alter the preregistered acceptance gates
- does not run a final-test evaluation

Aggregate label counts were already part of the fixed public dataset record and
are checked only for structural consistency. They are never joined to an
image-level score in this checkpoint.

## License and repository boundary

VisA remains separately licensed under CC BY 4.0. The repository's PolyForm
Noncommercial License 1.0.0 does not apply to VisA and does not impose
additional restrictions on it.

The archive, extracted images, masks, pinned split CSV, and local provenance
files remain outside Git. The committed
[`pcb1-local-integrity.json`](../artifacts/v0.1/data/pcb1-local-integrity.json)
contains aggregate facts and hashes only.
