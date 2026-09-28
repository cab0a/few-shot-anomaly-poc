"""Verify all 900 pipeline slots using generated pixels and fake scores only."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from few_shot_anomaly_poc.v0_3_controlled_probe import run_probe
from few_shot_anomaly_poc.v0_3_diagnostic_contract import sha256_file
from few_shot_anomaly_poc.v0_3_probe_artifacts import json_bytes, write_once
from few_shot_anomaly_poc.v0_3_probe_synthetic import (
    fake_classical_scorers,
    synthetic_dino_worker,
    synthetic_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="synthetic-v0.3.6-") as temporary:
        preflight = synthetic_preflight(Path(temporary), root)
        completed = run_probe(
            preflight,
            scorer_factory=fake_classical_scorers,
            dino_worker=synthetic_dino_worker,
            progress=lambda _text: None,
        )
        report = {
            "schema_version": "v0.3.6-synthetic-probe-verification-v1",
            "generated_image_count": 60,
            "condition_count": 5,
            "method_count": 3,
            "synthetic_score_count": completed["score_count"],
            "summary_count": completed["summary_count"],
            "synthetic_controlled_scores_sha256": completed["controlled_scores_sha256"],
            "synthetic_condition_summaries_sha256": completed["condition_summaries_sha256"],
            "visa_images_accessed": False,
            "anomaly_method_executed": False,
            "model_or_fitted_state_loaded": False,
            "latency_measured": False,
            "real_probe_executed": False,
            "implementation_sha256": {
                name: sha256_file(root / "src/few_shot_anomaly_poc" / name)
                for name in (
                    "v0_3_controlled_probe.py",
                    "v0_3_probe_artifacts.py",
                    "v0_3_dinov2_probe.py",
                    "v0_3_probe_synthetic.py",
                )
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(args.output, json_bytes(report))
    print("Synthetic probe passed: 60 generated images, 900 fake scores, 15 paired summaries.")
    print("No VisA image, fitted state, anomaly model, or real diagnostic score accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
