# DINOv2 Adoption Research

## 日本語概要

本調査は、DINOv2のpatch特徴量と正常参照画像のnearest-neighbor距離を用いる方式を、v0.1の次候補として検証する価値があるかを判定した記録です。公式モデル情報、AnomalyDINO論文と公開実装、ライセンス、CPU負荷、モデル資産、依存関係、評価リークのリスクを確認しました。

結論は`PROCEED WITH CONDITIONS`です。独立したv0.2のCPU実行可能性検証には進めますが、手法そのものは未採用です。現在の1秒以内というCPU条件、再現可能なモデル取得、分離した依存環境、未使用の評価境界を満たせない場合は実評価へ進みません。次段階のモデル、計測、依存環境、資産検証、停止条件、評価境界は別の事前登録文書へ固定しました。DINOv2のコード、重み、依存関係、データ、実験結果は本調査では追加していません。根拠、条件、停止規則の詳細は以下の英語本文を参照してください。

---

## English Summary

This record asks whether a frozen DINOv2 patch-nearest-neighbor method deserves a separately preregistered v0.2 study after both v0.1 methods were rejected. The decision is `PROCEED WITH CONDITIONS`: proceed only to an isolated CPU and asset feasibility spike, not to method adoption or final evaluation. The unresolved conditions are CPU latency, reproducible model acquisition, dependency isolation, and a genuinely untouched evaluation boundary.

## Decision

| Item | Decision |
| --- | --- |
| Candidate | Frozen DINOv2 ViT-S/14 patch features with normal-only nearest-neighbor scoring |
| Research decision | `PROCEED WITH CONDITIONS` |
| Authorized next step | A separately preregistered CPU, dependency, and model-asset feasibility spike |
| Not authorized | A claim that DINOv2 is adopted, superior, production-ready, or able to pass the project gates |
| Evidence date | 2026-07-30 |

`PROCEED WITH CONDITIONS` means that the evidence is strong enough to justify a bounded experiment but insufficient to select the method. No local DINOv2 timing, anomaly score, VisA metric, or comparison result exists at this stage.

The v0.1 `REJECT` decision remains unchanged. DINOv2 is a new research candidate, not a retrospective exception to a failed v0.1 gate.

The authorized experiment is now fixed in the [v0.2 preflight preregistration](v0.2-preflight-preregistration.md). That document does not constitute implementation or model adoption.

## Question

Can a frozen, pretrained patch representation improve the fixed-threshold behavior that limited the two v0.1 methods while preserving the hypothetical operating constraints?

The relevant constraints remain:

- no more than 20 normal reference images;
- no anomalous training labels;
- normal-only threshold calibration;
- image-level scoring on a general CPU;
- CPU p95 latency no greater than one second per image;
- reproducible source, model, configuration, partitions, and artifacts; and
- a hard-gate decision without an aggregate score.

This investigation does not assume that results reported on another dataset, category, split, reference set, resolution, or machine will transfer to VisA `pcb1` or to a real inspection system.

## Evidence Reviewed

### DINOv2

The official DINOv2 model card describes Vision Transformer backbones that return a class token and patch tokens. The standard backbones use a patch size of 14. ViT-S has a 384-dimensional embedding; a `224 x 224` input produces 256 patch tokens. Larger dimensions are accepted when they are multiples of 14, while other dimensions are cropped to the nearest smaller multiple.

The official repository lists the non-register ViT-S/14 backbone at 21 million parameters. It states that the DINOv2 code and standard model weights are released under Apache License 2.0. The repository also contains newer domain-specific models with different terms; those assets are outside this decision.

The DINOv2 paper supports the general claim that self-supervised features can serve downstream vision tasks without task-specific fine-tuning. It does not establish anomaly-detection performance or CPU latency for this repository.

### AnomalyDINO

The peer-reviewed WACV 2025 paper presents a training-free, patch-level deep-nearest-neighbor method over DINOv2 features for few-shot anomaly detection. It reports image-level detection and pixel-level segmentation results on MVTec-AD and VisA.

The official AnomalyDINO runner defaults to:

- `dinov2_vits14`;
- resolution `448`;
- L2-normalized nearest-neighbor matching;
- one nearest neighbor; and
- optional CPU FAISS search.

The same implementation also exposes optional rotation augmentation, masking, segmentation evaluation, multiple seeds, and other behavior. Those choices make the complete AnomalyDINO pipeline broader than the minimum experiment proposed here.

Published AnomalyDINO metrics are evidence that the representation family is worth testing. They are not evidence that the candidate will pass this repository's gates. No published score is copied into the project results.

### Dependency and execution boundary

The DINOv2 loading path requires PyTorch. PyTorch and its binary distributions are governed by their own licenses and notices. The AnomalyDINO requirements file is not a suitable lock for this repository because it is a broad, mostly unpinned research environment and includes packages not needed for the minimum image-level experiment.

The root v0.1 `pyproject.toml`, `uv.lock`, runtime versions, and evaluation files are covered by the committed v0.1 freeze. A v0.2 experiment must therefore use a separate, locked environment rather than modifying the historical v0.1 environment in place.

## Candidate Method Boundary

The first candidate is deliberately narrower than a full AnomalyDINO reproduction:

1. Use the standard non-register `dinov2_vits14` backbone as the only backbone candidate.
2. Freeze all model parameters; perform no fine-tuning or linear probing.
3. Extract final-layer patch tokens from RGB inputs.
4. L2-normalize each patch embedding.
5. Build the memory bank from no more than 20 fixed normal reference images.
6. Use exact one-nearest-neighbor distance to normal reference patches.
7. Aggregate patch distances into one image score using a rule fixed before label reveal.
8. Calibrate the classification threshold from normal calibration images only.
9. Produce image-level metrics and the existing ordered hard-gate decision.

The exact resize rule, input resolution, color normalization, interpolation, token selection, distance implementation, score aggregation, batch size, thread count, and failure behavior are intentionally not selected by this research note. They must be preregistered after the label-free feasibility spike and before any evaluation scoring.

The feasibility spike may compare `224 x 224` and `448 x 448` only for CPU latency, memory, and deterministic execution. It may not use anomaly labels or performance metrics to choose the resolution.

## Why This Candidate Is Worth a Bounded Study

| Evidence | Expected value | Claim boundary |
| --- | --- | --- |
| Frozen pretrained patch features | Tests a representation less tied to direct pixel equality or handcrafted gradients | Robustness to `pcb1` variation is unverified |
| Normal-only memory bank | Fits the 20-reference, no-anomaly-training-label constraint | Pretraining used external images and is not "trained only on 20 images" |
| Training-free nearest-neighbor scoring | Avoids model fine-tuning and a learned anomaly classifier | Model loading and feature extraction are still computationally substantial |
| Patch distances | Provide a mechanical basis for selecting high-scoring regions or records | Pixel-level localization is outside v0.2's initial scope |
| Published AnomalyDINO study | Supplies peer-reviewed evidence for the method family | The proposed minimum method is not a full AnomalyDINO reproduction |
| Apache-2.0 standard DINOv2 assets | Permits a reviewable third-party boundary | Exact source, checkpoint, hashes, and notices still need to be pinned |

The main technical reason to investigate DINOv2 is representation quality, not novelty or deep learning by itself.

## CPU and Memory Risk

No official source establishes one-second p95 latency on the recorded v0.1 CPU. CPU feasibility is therefore the first stop gate, not an assumption.

Planning estimates from the official ViT-S/14 shape are:

| Input | Patch tokens per image | 20-reference float32 patch bank | Query-to-bank patch pairs |
| --- | ---: | ---: | ---: |
| `224 x 224` | 256 | about 7.5 MiB | about 1.31 million |
| `448 x 448` | 1,024 | about 30 MiB | about 20.97 million |

These are arithmetic estimates for stored embeddings and pair counts. They are not measurements and exclude model parameters, activations, framework overhead, image buffers, temporary distance blocks, and allocator behavior. The 21 million float32 model parameters alone are roughly 80 MiB before runtime overhead.

The v0.2 latency boundary must include:

- deterministic RGB preprocessing from an already decoded image;
- DINOv2 feature extraction;
- patch normalization;
- nearest-neighbor scoring; and
- image-score aggregation.

It must exclude one-time model loading, weight download, memory-bank fitting, and file decoding, and it must separately report any scoring failure. Warm-up count, timed passes, thread settings, batching, timer, quantile rule, target machine, and failure timing must be fixed before measurement.

## Expected Failure Conditions

- CPU feature extraction or exact nearest-neighbor search exceeds the one-second p95 gate.
- Coarse patch resolution suppresses defects smaller than, or poorly aligned with, the effective patch grid.
- Resizing removes fine PCB evidence or changes aspect-sensitive geometry.
- Legitimate position, illumination, texture, or component variation produces large feature distances.
- Pretrained semantic invariance suppresses appearance changes that matter for inspection.
- The fixed normal-only threshold fails to transfer from calibration normals to final-test normals.
- The 20-reference memory bank does not cover normal appearance variation.
- Model or framework behavior changes because source, checkpoint, preprocessing, or binary distributions are not fully pinned.
- A mutable network loading path prevents offline reproduction.
- v0.1 final-test labels influence parameter or method choices.
- A selected PyTorch CPU wheel or bundled component has unresolved compatibility or notice requirements.

## License and Asset Separation

The proposed boundaries are:

| Material | Governing terms | Repository treatment |
| --- | --- | --- |
| Original v0.2 code and documentation | PolyForm Noncommercial License 1.0.0 | Source-available repository content |
| Standard DINOv2 code and backbone weight | Apache License 2.0 | Third-party material; PolyForm must not replace or narrow its terms |
| AnomalyDINO code | Apache License 2.0 | Research reference only unless exact reused code is later identified |
| PyTorch and other distributions | Their respective licenses and bundled notices | Separately locked and inventoried |
| VisA | CC BY 4.0 | External data; not included in Git |

Before any model download, the implementation milestone must record:

- the exact standard DINOv2 model identifier;
- the exact DINOv2 source commit;
- the official checkpoint URL;
- checkpoint byte count and SHA-256;
- the controlling license and required notices;
- the local cache boundary; and
- confirmation that the checkpoint is not committed to Git.

Loading mutable `main` through an unpinned `torch.hub.load` call is not an acceptable final reproduction path. Domain-specific DINO derivatives in the evolving upstream repository must not be substituted for the standard Apache-2.0 ViT-S/14 asset.

The v0.1 `NOTICE.md` and dependency inventory are freeze-listed historical files. If DINOv2 assets or dependencies are introduced, their exact terms and notices must be recorded in a new v0.2-specific inventory and linked from the README without rewriting the v0.1 records. This research-only milestone introduces neither.

## Conditions to Proceed

Every condition below must be satisfied in order. A later success cannot waive an earlier failure.

1. **New preregistration:** define the v0.2 problem, method boundary, normal-only calibration, latency protocol, failure rules, artifacts, and hard gates before evaluation.
2. **Untouched evaluation boundary:** do not tune against the already revealed v0.1 final-test labels. Create a fixed, opaque-ID final-test boundary that has not informed method or parameter selection.
3. **Isolated locked environment:** keep the frozen v0.1 root environment unchanged; pin a separate CPU-only runtime and every resolved distribution for v0.2.
4. **License and asset record:** verify the exact standard checkpoint and source terms before download, record its SHA-256 and notices after download, and keep the weight outside Git.
5. **CPU preflight:** on the recorded target CPU, measure the full decoded-RGB-to-score boundary using label-free deterministic inputs. At least one preregistered resolution must meet p95 `<= 1.0 s/image`.
6. **Determinism and offline reuse:** the same pinned asset and configuration must reproduce scores within a preregistered numeric tolerance without fetching mutable code.
7. **Normal-only fitting:** limit the memory bank to at most 20 fixed normal references and use anomaly labels only at the final evaluation boundary.
8. **Fair comparison:** if a new final-test boundary is used to compare methods, rerun the frozen classical baselines on that same boundary rather than comparing unrelated dataset results.

## Stop Rules

Return `DO NOT PROCEED` before final evaluation if any of the following occurs:

- no candidate resolution meets the fixed CPU p95 gate;
- the exact model source, checkpoint identity, license, or required notices cannot be verified;
- an isolated reproducible CPU environment cannot be locked;
- deterministic scoring cannot be reproduced from the pinned asset;
- an untouched, opaque-ID evaluation boundary cannot be established; or
- completing the minimum method would require adding augmentation, masking, pixel-level evaluation, fine-tuning, GPU execution, or another unregistered scope expansion.

## Explicit Non-Goals

This decision does not authorize:

- DINOv2 implementation in the current milestone;
- model or dataset download;
- modification of v0.1 frozen files, artifacts, metrics, or decision;
- full AnomalyDINO reproduction;
- PCA masking or rotation augmentation;
- pixel-level anomaly localization or segmentation metrics;
- DINOv2 backbone sweeps;
- DINOv3 or another foundation-model comparison;
- fine-tuning, prompt learning, or anomalous-label training;
- GPU benchmarking;
- Web UI, API, Docker, or cloud infrastructure; or
- production-readiness, state-of-the-art, or real-world generalization claims.

## Final Interpretation

DINOv2 patch nearest neighbor is a credible research successor because it changes the representation assumption that limited both v0.1 methods while preserving normal-only fitting. Its published support and licensing boundary are sufficient for a feasibility study.

It is not yet an adopted anomaly detector. CPU latency on the target machine, the external model and dependency cost, and evaluation contamination are material unresolved risks. The next valid decision point is after the isolated, label-free CPU and asset preflight:

- `PROCEED` to a preregistered v0.2 evaluation if every preflight condition passes;
- `PROCEED WITH CONDITIONS` only if an explicitly preregistered non-performance condition remains and does not waive a hard gate; or
- `DO NOT PROCEED` if a stop rule is triggered.

## Official Sources

Research snapshots were taken from DINOv2 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` and AnomalyDINO commit `b9d1c2648e3a5247437d4d953d907a8f3d994457`.

- [DINOv2 paper](https://arxiv.org/abs/2304.07193)
- [DINOv2 official repository snapshot](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
- [DINOv2 model card snapshot](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/MODEL_CARD.md)
- [DINOv2 Apache License 2.0 text](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE)
- [AnomalyDINO WACV 2025 paper](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)
- [AnomalyDINO official repository snapshot](https://github.com/dammsi/AnomalyDINO/tree/b9d1c2648e3a5247437d4d953d907a8f3d994457)
- [AnomalyDINO runner snapshot](https://github.com/dammsi/AnomalyDINO/blob/b9d1c2648e3a5247437d4d953d907a8f3d994457/run_anomalydino.py)
- [AnomalyDINO requirements snapshot](https://github.com/dammsi/AnomalyDINO/blob/b9d1c2648e3a5247437d4d953d907a8f3d994457/requirements.txt)
- [AnomalyDINO Apache License 2.0 text](https://github.com/dammsi/AnomalyDINO/blob/b9d1c2648e3a5247437d4d953d907a8f3d994457/LICENSE)
- [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE)

This is a technical license inventory, not legal advice. The controlling terms are the licenses distributed with each exact asset and dependency.
