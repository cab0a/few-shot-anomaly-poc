# Data Preparation

## 日本語概要

本書は、VisAの取得、検証、安全な展開、正常参照・校正・最終評価への決定的な分割手順を記録します。生データをGit管理外に保ち、チェックサム、出所、重複なしの分割、最終評価ラベルの境界を確認するコマンドと制約の詳細は以下の英語本文を参照してください。

---

This directory stores only documentation and, after an authorized local run,
small provenance and manifest records. Raw VisA files stay outside Git under
`data/external/`, which is ignored.

VisA is a third-party dataset released under CC BY 4.0. The repository's
PolyForm Noncommercial License does not apply to VisA and does not add
restrictions to it. See [`../NOTICE.md`](../NOTICE.md) for the license boundary.

## Fixed v0.1 inputs

| Item | Fixed value |
| --- | --- |
| Dataset archive | `VisA_20220922.tar` |
| Archive source | `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar` |
| Official split repository | `https://github.com/amazon-science/spot-diff` |
| Official split revision | `2a692ab575001cbde74d402d897a7286086c6199` |
| Official split path | `split_csv/1cls.csv` |
| Official split SHA-256 | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |
| Category | `pcb1` |
| Reference count | `20` |
| Selection seed | `42` |
| Selection procedure | `sha256-path-ranking-v1` |

The archive checksum is intentionally not prefilled in the download
configuration because the official sources do not publish an independently
trusted SHA-256. The acquired object and its observed SHA-256 are fixed in
[`../docs/data-acquisition-record.md`](../docs/data-acquisition-record.md).
The download command always computes and records its own observation. If a
digest is later obtained through an independent trusted channel, pass it with
`--expected-sha256`; the provenance status will then be `verified` rather than
`observed_only`.

## WSL workflow

Run the commands inside Ubuntu 24.04 on WSL, from the repository root:

```bash
uv sync --locked
uv run --locked --no-sync few-shot-data fetch-split
uv run --locked --no-sync few-shot-data download-archive
uv run --locked --no-sync few-shot-data extract-archive
uv run --locked --no-sync few-shot-data build-manifests
uv run --locked --no-sync few-shot-data validate-manifests
```

To verify a separately obtained archive digest during download:

```bash
uv run --locked --no-sync few-shot-data download-archive \
  --expected-sha256 <64-lowercase-hex-characters>
```

Every command refuses to overwrite its destination. A retry after an
interrupted or rejected operation therefore requires deliberate review of the
existing local file first.

## Download and extraction boundary

`fetch-split` downloads only the fixed CSV and rejects any content whose hash
does not match the pinned value.

`download-archive` streams the official archive to a temporary sibling file,
calculates SHA-256, and moves it into place only after the download completes.
It records the requested and effective URLs, timestamp, byte count, SHA-256,
optional trusted SHA-256, checksum status, ETag, and Last-Modified header in
`data/provenance/visa-archive.json`.

`extract-archive` first verifies the archive against that provenance record. It
then validates every tar member before extracting anything. Absolute paths,
parent traversal, links, devices, FIFOs, duplicate targets, and configured size
limit violations are rejected. After whole-archive validation, only the
`pcb1/` subtree is extracted. Extraction copies opaque bytes only; it does not
decode or inspect images.

## Partition manifests

The official one-class split supplies the train/test boundary. For `pcb1`, the
generator:

1. normalizes every official normal training path to POSIX form;
2. hashes `few-shot-anomaly-poc:v0.1:42:<relative-path>`;
3. sorts by the digest and then the path;
4. assigns the first 20 paths to `reference`;
5. assigns all remaining official normal training paths to `calibration`;
6. copies official test paths and source-row IDs to `final-test` without opening
   a dataset file or exposing class labels.

It creates:

- `data/manifests/v0.1/reference.jsonl`
- `data/manifests/v0.1/calibration.jsonl`
- `data/manifests/v0.1/final-test.jsonl`
- `data/manifests/v0.1/manifest-set.json`

The set record pins the official split, archive checksum, selection procedure,
manifest counts, and manifest checksums. The validator checks required fields,
IDs, counts, checksums, source split rules, unique paths, zero overlap, and an
exact reconstruction from the pinned split CSV.

## Final-test protection

During this milestone, final-test access is metadata-only:

- manifest generation accepts the official split CSV, not a dataset root;
- manifest validation opens only CSV, JSON, and JSONL metadata files;
- neither command checks image existence or decodes image bytes;
- final-test records omit class labels and pixel-mask paths;
- no anomaly score, threshold, image display, label aggregation, or parameter
  selection is implemented.

The `final_test_access_policy` in `manifest-set.json` records this boundary.
