# Runtime Dependencies and License Boundaries

## Status

This document records the locked v0.1 runtime baseline before algorithm implementation.

The versions and license metadata below were reviewed from official release pages, distribution records, and the license files installed from the lock on 2026-07-28. The locked environment passed the bounded API smoke test described below. No anomaly-detection implementation, dataset operation, or experiment was performed.

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

CPython 3.13.14 is selected as a maintained, non-experimental runtime baseline for which the selected packages publish compatible wheel tags for common x86-64 environments. The recorded smoke test covers one Linux x86-64 environment only and is not a general platform-support claim.

## Lock and Installation Policy

The environment is declared in `pyproject.toml`, the exact CPython patch version is recorded in `.python-version`, and all resolved distributions are recorded in `uv.lock`. The lock includes exact package versions, source URLs, and SHA-256 hashes for source archives and available wheels.

The lock was generated with `uv 0.11.32`. `uv` is an external environment-management tool, not a runtime dependency or vendored repository component. It is separately offered under MIT or Apache-2.0 terms at the user's option.

Create or synchronize the environment without changing the lock:

```bash
uv sync --locked --no-dev
```

Run the committed bounded verification without modifying or resynchronizing the environment:

```bash
uv run --locked --no-sync python scripts/verify_environment.py
```

Only the four packages in `pyproject.toml` are selected as direct package dependencies. No deep-learning framework, model downloader, plotting framework, web framework, or notebook runtime is part of this runtime baseline.

A top-level or transitive dependency change after that lock is committed requires a documented reason and a new environment record. No dependency may be changed in response to final-test performance.

## Resolved Distribution Inventory

The following table covers every package resolved by `uv.lock`. "Direct" means declared in `pyproject.toml`; "transitive" means introduced by a direct dependency.

| Distribution | Version | Role | Published or installed license evidence |
| --- | --- | --- | --- |
| [NumPy](https://pypi.org/project/numpy/2.5.1/) | `2.5.1` | Direct | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` |
| [`opencv-python-headless`](https://pypi.org/project/opencv-python-headless/4.13.0.92/) | `4.13.0.92` | Direct | Packaging scripts: MIT; OpenCV: Apache-2.0; installed wheel: separate third-party notices including FFmpeg under LGPL-2.1 |
| [scikit-image](https://pypi.org/project/scikit-image/0.26.0/) | `0.26.0` | Direct | BSD-3-Clause main terms with identified BSD-2-Clause and MIT components |
| [scikit-learn](https://pypi.org/project/scikit-learn/1.9.0/) | `1.9.0` | Direct | BSD-3-Clause |
| [ImageIO](https://pypi.org/project/imageio/2.37.4/) | `2.37.4` | Transitive | BSD-2-Clause |
| [joblib](https://pypi.org/project/joblib/1.5.3/) | `1.5.3` | Transitive | BSD-3-Clause |
| [lazy-loader](https://pypi.org/project/lazy-loader/0.5/) | `0.5` | Transitive | BSD-3-Clause |
| [Narwhals](https://pypi.org/project/narwhals/2.24.0/) | `2.24.0` | Transitive | MIT |
| [NetworkX](https://pypi.org/project/networkx/3.6.1/) | `3.6.1` | Transitive | BSD-3-Clause |
| [packaging](https://pypi.org/project/packaging/26.2/) | `26.2` | Transitive | Apache-2.0 OR BSD-2-Clause |
| [Pillow](https://pypi.org/project/pillow/12.3.0/) | `12.3.0` | Transitive | MIT-CMU |
| [SciPy](https://pypi.org/project/scipy/1.18.0/) | `1.18.0` | Transitive | BSD-3-Clause main terms; the installed Linux wheel also notices OpenBLAS under BSD-3-Clause, LAPACK under BSD-3-Clause-Open-MPI, GCC runtime libraries under GPL-3.0-or-later WITH GCC-exception-3.1, and libquadmath under LGPL-2.1-or-later |
| [threadpoolctl](https://pypi.org/project/threadpoolctl/3.6.0/) | `3.6.0` | Transitive | BSD-3-Clause |
| [tifffile](https://pypi.org/project/tifffile/2026.7.14/) | `2026.7.14` | Transitive | BSD-3-Clause |

The table is an inventory of distribution metadata and installed notices, not a replacement for the controlling license files. Binary contents and notices can differ by platform. Anyone redistributing a wheel must inspect the license files inside the exact selected wheel.

## License Separation

The PolyForm Noncommercial License 1.0.0 applies only to original code and documentation in this repository. It does not replace, narrow, or modify the terms of Python, the selected packages, their transitive dependencies, or bundled binary components.

This repository does not vendor or redistribute Python, dependency source archives, wheels, shared libraries, or dependency license texts. The environment command downloads packages from their recorded distribution channel, where each package remains governed by its own terms.

If a future release bundles or redistributes a dependency or binary artifact, its exact distribution file and all required copyright notices, license texts, and bundled-component notices must be reviewed and included as required by that distribution. The repository's noncommercial terms must not be presented as restrictions on third-party materials.

VisA is a dataset, not a runtime dependency. Its separate CC BY 4.0 boundary and attribution policy are recorded in the README, `NOTICE.md`, and the evaluation plan.

## Smoke-Test Record

The dependency-only smoke test was run on 2026-07-28 with:

- `uv 0.11.32`
- standard GIL-enabled CPython `3.13.14`
- Linux x86-64 under WSL2
- the committed `uv.lock`

The verification checked:

- exact Python and direct distribution versions
- OpenCV `findTransformECC` on a deterministic synthetic image
- the preregistered 324-element HOG descriptor shape for a `64 x 64` patch
- StandardScaler fitting and transformation
- One-Class SVM fitting and a finite decision score

The check passed. This establishes only that the selected environment imports and exposes the required API path on the recorded platform. It does not validate the anomaly methods, their accuracy, VisA compatibility, latency, or reproducibility of a future decision.

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
- uv installation and project-management documentation:
  <https://docs.astral.sh/uv/>
- uv license:
  <https://github.com/astral-sh/uv#license>

This is a technical license inventory, not legal advice. The controlling texts are the licenses distributed with each exact component.
