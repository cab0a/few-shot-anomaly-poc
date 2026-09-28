"""Restore the pinned isolated environment and model files; do not score or load weights."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from few_shot_anomaly_poc.v0_3_diagnostic_contract import load_v0_3_config  # noqa: E402
from few_shot_anomaly_poc.v0_3_probe_runtime import (  # noqa: E402
    check_isolated_runtime,
    restore_model_assets,
    verify_file,
)


def main() -> int:
    try:
        config = load_v0_3_config(ROOT / "configs/v0.3.yaml")
        verify_file(
            ROOT / "environments/v0.2-preflight/uv.lock",
            config["dependencies"]["dinov2_lock_sha256"],
        )
        uv = shutil.which("uv") or str(ROOT / "work/bootstrap/bin/uv")
        subprocess.run(
            [uv, "sync", "--locked", "--project", str(ROOT / "environments/v0.2-preflight")],
            cwd=ROOT,
            check=True,
        )
        restore_model_assets(ROOT)
        check_isolated_runtime(ROOT)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    print("Pinned DINOv2 runtime restored and verified. No model loaded; no images accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
