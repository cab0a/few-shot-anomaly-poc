from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from few_shot_anomaly_poc.v0_3_inventory import (
    V0_3InventoryError,
    construct_inventory_bundle,
    write_inventory_bundle,
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the v0.3.2 metadata-only inventory and closed access checkpoint."
    )
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    try:
        if _git(repository_root, "status", "--porcelain"):
            raise V0_3InventoryError("worktree must be clean before inventory construction")
        source_commit = _git(repository_root, "rev-parse", "HEAD")
        origin_commit = _git(repository_root, "rev-parse", "origin/main")
        if source_commit != origin_commit:
            raise V0_3InventoryError("HEAD must equal pushed origin/main")
        files = construct_inventory_bundle(
            repository_root=repository_root,
            external_root=arguments.external_root.resolve(),
            source_commit=source_commit,
        )
        write_inventory_bundle(arguments.output_root.resolve(), files)
    except (OSError, subprocess.CalledProcessError, V0_3InventoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("v0.3.2 metadata-only inventory created")
    print(f"source_commit={source_commit}")
    print("diagnostic_image_accessed=false")
    print("diagnostic_score_computed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
