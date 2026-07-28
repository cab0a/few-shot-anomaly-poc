# VisA `pcb1` Acquisition and Partition Record

## Status

The v0.1 VisA acquisition and normal-only partition milestone was completed on 2026-07-28.

This milestone downloaded the official archive outside Git, recorded its observed identity, validated the `pcb1` file structure against the pinned official one-class split, and fixed the 20-image reference partition and 884-image threshold-calibration partition. It did not implement an anomaly method, decode or display image content, calculate an anomaly score, or run a final evaluation.

## Official Source and License

| Item | Recorded value |
| --- | --- |
| Dataset | Visual Anomaly (VisA) |
| Category | `pcb1` |
| Official project | <https://github.com/amazon-science/spot-diff> |
| AWS Open Data entry | <https://registry.opendata.aws/visa/> |
| Archive | `VisA_20220922.tar` |
| Archive URL | <https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar> |
| License | [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) |

VisA is not included in this repository and is not governed by the repository's PolyForm Noncommercial License 1.0.0. The normal-partition manifest contains metadata derived from VisA and the official split; the repository does not impose additional restrictions on that third-party material.

The dataset citation is:

> Zou, Yang, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation." 2022.

## Archive Identity

| Field | Recorded value |
| --- | --- |
| Content length | `1,929,840,640 bytes` |
| Observed SHA-256 | `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| S3 ETag | `05c830591a1172938cb714895c9e0cfb-113` |
| Last-Modified | `2022-09-22T19:23:39Z` |
| Tar entries | `12,122` |

The SHA-256 is an observed digest of the acquired official object. Neither the AWS registry entry nor the official project publishes an independent SHA-256 for this archive. The digest therefore identifies the exact acquired artifact but is not presented as an upstream checksum assertion.

The hyphenated S3 ETag identifies a multipart object and is recorded only as HTTP metadata. It is not treated as an MD5 or SHA-256 verification value.

The transfer was accepted only after its byte count matched the official `Content-Length`. The archive was then copied to the external dataset location and hashed again; both copies produced the recorded SHA-256. The temporary transfer copy was removed after verification.

## Official Split Identity

The one-class split is pinned independently of the moving repository branch:

| Field | Recorded value |
| --- | --- |
| Repository | <https://github.com/amazon-science/spot-diff> |
| Revision | `2a692ab575001cbde74d402d897a7286086c6199` |
| Path | `split_csv/1cls.csv` |
| SHA-256 | `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995` |

For `pcb1`, the pinned split contains:

| Split | Label | Count |
| --- | --- | ---: |
| Train | Normal | 904 |
| Test | Normal | 100 |
| Test | Anomaly | 100 |

No duplicate image path exists in the 1,104 `pcb1` split records.

## `pcb1` Structure Validation

The extracted category contains:

| Content | Count |
| --- | ---: |
| Normal images | 1,004 |
| Anomaly images | 100 |
| Anomaly masks | 100 |
| `image_anno.csv` | 1 |

`image_anno.csv` has SHA-256 `e41a1595f07a33fcd97e978e991d324e4ff7e0c201f965368d2c2d7860a0cdf4`.

Path-level validation found:

- zero split images missing from the extracted category
- zero anomaly masks missing for the official anomaly records
- zero extracted images absent from the pinned split
- zero overlap between the reference and calibration partitions

This was a metadata and filesystem-structure check. Image pixels were not decoded, displayed, scored, or summarized.

## Fixed Normal-Only Partitions

The partition procedure is the preregistered `sha256-path-ranking-v1` rule:

1. Select the 904 official `pcb1` normal training paths.
2. Normalize every relative path to POSIX form.
3. Encode `few-shot-anomaly-poc:v0.1:42:<relative-path>` as UTF-8.
4. Calculate SHA-256.
5. Sort by digest and then relative path.
6. Assign ranks 1 through 20 to `reference`.
7. Assign ranks 21 through 904 to `calibration`.

The resulting manifest is [`pcb1-normal-partitions.csv`](../artifacts/v0.1/data/pcb1-normal-partitions.csv), with SHA-256 `953478e04c20d74cc1994022e4a757388123b4db4020a10898fcee74b1a192a7`.

| Partition | Count |
| --- | ---: |
| Reference | 20 |
| Threshold calibration | 884 |

The first five fixed reference paths are:

1. `pcb1/Data/Images/Normal/0246.JPG`
2. `pcb1/Data/Images/Normal/0704.JPG`
3. `pcb1/Data/Images/Normal/0440.JPG`
4. `pcb1/Data/Images/Normal/0927.JPG`
5. `pcb1/Data/Images/Normal/0546.JPG`

All 20 reference paths and all 884 calibration paths are recorded in the manifest. The manifest contains no final-test record. The final test remains defined by the pinned official split and is not used for normal representation fitting, threshold calibration, or parameter selection.

## Machine-Readable Record

[`dataset-record.json`](../artifacts/v0.1/data/dataset-record.json) records the source, license, archive identity, official split identity, structure counts, partition procedure, manifest checksum, and verification boundary without a machine-specific absolute path.

Raw VisA files, extracted images, masks, and the official split CSV remain outside Git.

## Claim Boundary

This milestone establishes provenance, structural consistency, and fixed normal-only partitions for one acquired archive. It does not establish:

- image-content integrity beyond the archive and path-level checks
- anomaly-detection accuracy
- CPU latency
- suitability for a real inspection process
- equivalence to another archive that has a different digest
- any result for a category other than `pcb1`

The final `ADOPT`, `ADOPT WITH CONDITIONS`, or `REJECT` decision remains unavailable until the preregistered methods and evaluation are completed.
