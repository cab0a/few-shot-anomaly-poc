# Runtime Dependencies and License Boundaries

## Status

This document preregisters the direct v0.1 runtime baseline before dependency installation or algorithm implementation.

The versions and license metadata below were reviewed from official release pages and distribution records on 2026-07-28. They have not yet been installed or smoke-tested in this repository. This document therefore records a selected baseline, not a compatibility or platform-support claim.

## Selected Runtime Baseline

The primary environment will use the standard GIL-enabled CPython build, not the experimental free-threaded build.

| Component | Fixed version | v0.1 purpose | Recorded licensing boundary |
| --- | --- | --- | --- |
| CPython | `3.13.14` | Runtime | Python Software Foundation License Version 2; incorporated components may carry separate terms |
| NumPy | `2.5.1` | Arrays, numeric operations, and shared dependency | Top-level NumPy code uses BSD-3-Clause terms; the published distribution metadata also identifies separately licensed bundled components |
| `opencv-python-headless` | `4.13.0.92` | Image preprocessing and ECC registration | Packaging scripts use MIT terms; OpenCV uses Apache License 2.0; wheels include FFmpeg under LGPL-2.1 and other separately noticed binaries |
| `scikit-image` | `0.26.0` | HOG feature extraction | Main project terms are BSD-3-Clause; its license file identifies BSD-2-Clause and MIT components |
| `scikit-learn` | `1.9.0` | Standard scaling and One-Class SVM | BSD-3-Clause |

`opencv-python-headless` is selected because v0.1 has no GUI requirement. Staying on OpenCV 4.x keeps the preregistered method aligned with the reviewed OpenCV 4.x ECC API and avoids an unneeded major-version change. NumPy 2.5.1 satisfies the selected OpenCV wheel's Python 3.9-and-later requirement of NumPy 2 or later.

CPython 3.13.14 is selected as a maintained, non-experimental runtime baseline for which the selected packages publish compatible wheel tags for common x86-64 environments. Actual installation and execution remain to be verified before implementation.

## Direct and Transitive Dependency Policy

Only the five components in the baseline table are selected directly for v0.1. No deep-learning framework, model downloader, plotting framework, web framework, or notebook runtime is part of this runtime baseline.

The selected packages resolve additional runtime dependencies. At minimum, the published metadata identifies:

- SciPy, joblib, Narwhals, and threadpoolctl through scikit-learn
- SciPy, NetworkX, Pillow, ImageIO, tifffile, packaging, and lazy-loader through scikit-image

Those transitive packages are not yet pinned by this document. Before the first implementation commit:

1. Resolve the environment for the recorded Python version and target platform.
2. Commit a machine-readable lock or constraints artifact with exact versions and cryptographic hashes.
3. Record the source and license of every resolved distribution.
4. Verify that installation does not introduce an unreviewed dependency outside the documented runtime scope.
5. Run an import and API smoke test without downloading VisA or executing the anomaly-detection experiment.

A top-level or transitive dependency change after that lock is committed requires a documented reason and a new environment record. No dependency may be changed in response to final-test performance.

## License Separation

The PolyForm Noncommercial License 1.0.0 applies only to original code and documentation in this repository. It does not replace, narrow, or modify the terms of Python, the selected packages, their transitive dependencies, or bundled binary components.

This repository does not vendor or redistribute Python, dependency source archives, wheels, shared libraries, or dependency license texts at the current stage. Future environment instructions may download packages from their official distribution channels, where each package remains governed by its own terms.

If a future release bundles or redistributes a dependency or binary artifact, its exact distribution file and all required copyright notices, license texts, and bundled-component notices must be reviewed and included as required by that distribution. The repository's noncommercial terms must not be presented as restrictions on third-party materials.

VisA is a dataset, not a runtime dependency. Its separate CC BY 4.0 boundary and attribution policy are recorded in the README, `NOTICE.md`, and the evaluation plan.

## Reproducibility and Claim Boundary

The eventual experiment record must include:

- exact Python and package versions from the resolved environment
- operating system, architecture, and CPU information
- installer and lock-file identity
- hashes for resolved distributions
- relevant numeric and thread environment variables
- a smoke-test result for the APIs used by both shortlisted methods

Official wheel availability and version metadata are selection evidence only. Until the locked environment is installed and the smoke test passes, the project makes no claim that the selected package combination executes correctly.

## Official Sources

- Python 3.13.14 release:
  <https://www.python.org/downloads/release/python-31314/>
- Python 3.13 license:
  <https://docs.python.org/3.13/license.html>
- NumPy 2.5.1 distribution:
  <https://pypi.org/project/numpy/2.5.1/>
- NumPy 2.5.1 license:
  <https://github.com/numpy/numpy/blob/v2.5.1/LICENSE.txt>
- `opencv-python-headless` 4.13.0.92 distribution and bundled-license summary:
  <https://pypi.org/project/opencv-python-headless/4.13.0.92/>
- `opencv-python` packaging license at release tag 92:
  <https://github.com/opencv/opencv-python/blob/92/LICENSE.txt>
- OpenCV 4.x license:
  <https://github.com/opencv/opencv/blob/4.x/LICENSE>
- scikit-image 0.26.0 distribution:
  <https://pypi.org/project/scikit-image/0.26.0/>
- scikit-image 0.26.0 license:
  <https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt>
- scikit-learn 1.9.0 distribution:
  <https://pypi.org/project/scikit-learn/1.9.0/>
- scikit-learn 1.9.0 license:
  <https://github.com/scikit-learn/scikit-learn/blob/1.9.0/COPYING>

This is a technical license inventory, not legal advice. The controlling texts are the licenses distributed with each exact component.
