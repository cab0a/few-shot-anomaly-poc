"""Print the committed v0.2.5 label-free summary without writing files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SOURCE = PROJECT_ROOT / "src"
if str(PROJECT_SOURCE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE))

from few_shot_anomaly_poc.v0_2_evaluation_contract import (  # noqa: E402
    V0_2EvaluationContractError,
)
from few_shot_anomaly_poc.v0_2_label_free_summary import (  # noqa: E402
    V0_2LabelFreeSummaryError,
    build_v0_2_label_free_summary,
)
from few_shot_anomaly_poc.v0_2_scoring_artifacts import (  # noqa: E402
    V0_2ScoringArtifactError,
)


def main() -> int:
    try:
        summary = build_v0_2_label_free_summary(PROJECT_ROOT)
    except (
        OSError,
        V0_2EvaluationContractError,
        V0_2LabelFreeSummaryError,
        V0_2ScoringArtifactError,
    ) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
