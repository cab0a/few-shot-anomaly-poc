"""Render the v0.1 gate summary from committed numeric artifacts."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts/v0.1/evaluation/visa-pcb1-v0-1-final"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/assets/v0.1-gate-summary.svg"

METHODS = (
    ("ecc_residual", "ECC residual", "#2563eb"),
    ("patch_hog_one_class_svm", "Patch HOG + One-Class SVM", "#7c3aed"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _status_label(passed: bool) -> tuple[str, str]:
    return ("PASS", "#15803d") if passed else ("FAIL", "#b91c1c")


def _text(
    x: int,
    y: int,
    value: str,
    *,
    size: int = 20,
    weight: int = 400,
    fill: str = "#172033",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def _gate_rows(
    metrics: dict[str, Any],
    latency: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[tuple[str, str, str, bool], ...]:
    outcomes = {
        outcome["gate_id"]: outcome
        for outcome in decision["gate_outcomes"]
        if isinstance(outcome, dict)
    }
    return (
        (
            "Normal FPR",
            f"{metrics['normal_false_positive_rate']:.0%}",
            "≤ 5%",
            bool(outcomes["final_test_normal_fpr"]["passed"]),
        ),
        (
            "Anomaly recall",
            f"{metrics['anomaly_recall']:.0%}",
            "≥ 90%",
            bool(outcomes["final_test_anomaly_recall"]["passed"]),
        ),
        (
            "CPU p95",
            f"{latency['p95_latency_seconds']:.3f} s",
            "≤ 1.000 s",
            bool(outcomes["cpu_p95_scoring_latency"]["passed"]),
        ),
    )


def render_summary(*, artifact_root: Path, output: Path) -> None:
    """Render a deterministic SVG without reading source images."""
    cards: list[str] = []
    for index, (method_id, display_name, accent) in enumerate(METHODS):
        method_root = artifact_root / method_id
        metrics = _load_json(method_root / "metrics.json")
        latency = _load_json(method_root / "latency.json")
        decision = _load_json(method_root / "decision.json")
        if {
            metrics.get("method"),
            latency.get("method"),
            decision.get("method"),
        } != {method_id}:
            raise ValueError(f"method identity mismatch under {method_root}")

        x = 55 + (index * 565)
        cards.extend(
            [
                (
                    f'<rect x="{x}" y="145" width="530" height="350" rx="16" '
                    'fill="#ffffff" stroke="#d7deea" stroke-width="2"/>'
                ),
                (
                    f'<rect x="{x}" y="145" width="10" height="350" rx="5" '
                    f'fill="{accent}"/>'
                ),
                _text(x + 32, 190, display_name, size=23, weight=700),
                _text(
                    x + 498,
                    190,
                    str(decision["decision"]),
                    size=18,
                    weight=700,
                    fill="#b91c1c",
                    anchor="end",
                ),
                _text(
                    x + 32,
                    224,
                    (
                        f"AUROC {metrics['image_level_auroc']:.4f}  ·  "
                        f"AUPRC {metrics['image_level_auprc']:.4f}"
                    ),
                    size=17,
                    fill="#526077",
                ),
            ]
        )

        for row_index, (label, observed, gate, passed) in enumerate(
            _gate_rows(metrics, latency, decision)
        ):
            y = 286 + (row_index * 78)
            status, status_color = _status_label(passed)
            cards.extend(
                [
                    _text(x + 32, y, label, size=18, weight=600),
                    _text(x + 272, y, observed, size=20, weight=700, anchor="end"),
                    _text(
                        x + 292,
                        y,
                        f"gate {gate}",
                        size=16,
                        fill="#667085",
                    ),
                    (
                        f'<rect x="{x + 429}" y="{y - 25}" width="69" height="32" '
                        f'rx="16" fill="{status_color}"/>'
                    ),
                    _text(
                        x + 463,
                        y - 2,
                        status,
                        size=14,
                        weight=700,
                        fill="#ffffff",
                        anchor="middle",
                    ),
                    (
                        f'<line x1="{x + 32}" y1="{y + 24}" x2="{x + 498}" '
                        f'y2="{y + 24}" stroke="#e8ecf3"/>'
                    ),
                ]
            )

    svg = "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="550" '
            'viewBox="0 0 1200 550" role="img" '
            'aria-labelledby="title description">',
            '<title id="title">v0.1 hard-gate result</title>',
            (
                '<desc id="description">Numeric comparison of ECC residual and '
                "Patch HOG plus One-Class SVM against preregistered false-positive, "
                "recall, and CPU latency gates. Both methods were rejected.</desc>"
            ),
            '<rect width="1200" height="550" fill="#f5f7fb"/>',
            _text(55, 62, "v0.1 hard-gate result", size=32, weight=700),
            _text(
                55,
                99,
                (
                    "VisA pcb1 · 20 normal references · 200 final-test images · "
                    "CPU-only"
                ),
                size=19,
                fill="#526077",
            ),
            *cards,
            _text(
                600,
                526,
                "Generated only from committed metrics.json, latency.json, and decision.json",
                size=15,
                fill="#667085",
                anchor="middle",
            ),
            "</svg>",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the deterministic v0.1 numeric gate-summary SVG."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Final evaluation bundle containing per-method JSON artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="SVG output path.",
    )
    arguments = parser.parse_args()
    render_summary(
        artifact_root=arguments.artifact_root.resolve(),
        output=arguments.output.resolve(),
    )
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
