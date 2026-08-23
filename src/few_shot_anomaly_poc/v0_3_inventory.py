"""Metadata-only v0.3 diagnostic inventory construction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from few_shot_anomaly_poc.v0_3_diagnostic_contract import (
    ARTIFACT_CONTRACT_VERSION,
    DIAGNOSTIC_ID,
    EXPECTED_ASSET_IDS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_METHOD_CASES,
    EXPECTED_SCHEMA_SHA256,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_DOCUMENT_SHA256,
    V0_3DiagnosticContractError,
    load_v0_3_config,
    load_v0_3_schema,
    sha256_file,
    validate_checkpoint_record,
    validate_tabular_record,
)

FAILURE_HEADER = (
    "contract_version",
    "run_id",
    "run_kind",
    "method",
    "case_type",
    "rank",
    "asset_id",
    "source_path",
    "true_class",
    "predicted_class",
    "anomaly_score",
    "threshold",
    "score_margin",
    "score_status",
    "score_failure_code",
)
REVIEW_ASSET_FIELDS = (
    "contract_version",
    "diagnostic_id",
    "asset_id",
    "byte_count",
    "sha256",
)
REVIEW_CASE_FIELDS = (
    "contract_version",
    "diagnostic_id",
    "method",
    "case_type",
    "rank",
    "asset_id",
    "parent_failure_sha256",
)
NORMAL_PARTITION_FIELDS = (
    "contract_version",
    "diagnostic_id",
    "selection_rank",
    "selection_sha256",
    "relative_path",
    "byte_count",
    "sha256",
)
NORMAL_RECORD_KEYS = {
    "byte_count",
    "partition",
    "relative_path",
    "schema_version",
    "selection_rank",
    "selection_sha256",
    "sha256",
}


class V0_3InventoryError(Exception):
    """Reject an inventory input or output outside the fixed metadata boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V0_3InventoryError(message)


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> None:
    _require(path.is_file(), f"missing {label}")
    _require(sha256_file(path) == expected_sha256, f"{label} SHA-256 changed")


def _canonical_relative_path(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and value and "\\" not in value, f"invalid {label}")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"unsafe {label}",
    )
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_failure_case_identities(
    path: Path,
    *,
    method: str,
    expected_sha256: str,
) -> list[tuple[str, str, int, str]]:
    _verify_file(path, expected_sha256, label=f"{method} failure cases")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(tuple(reader.fieldnames or ()) == FAILURE_HEADER, "failure header changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise V0_3InventoryError(f"cannot read {method} failure cases") from error
    identities: list[tuple[str, str, int, str]] = []
    for row in rows:
        _require(set(row) == set(FAILURE_HEADER), "failure fields changed")
        _require(row["method"] == method, "failure method changed")
        _require(row["case_type"] in {"false_positive", "false_negative"}, "invalid case")
        try:
            rank = int(row["rank"])
        except ValueError as error:
            raise V0_3InventoryError("failure rank is not an integer") from error
        _require(rank >= 1, "failure rank is invalid")
        identities.append((method, row["case_type"], rank, row["asset_id"]))
    return identities


def load_scoring_manifest(path: Path, *, expected_sha256: str) -> dict[str, dict[str, Any]]:
    """Load only the fixed opaque manifest; do not inspect any referenced image file."""
    _verify_file(path, expected_sha256, label="opaque scoring manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_3InventoryError("cannot read opaque scoring manifest") from error
    _require(
        isinstance(value, dict) and set(value) == {"records", "schema_version"},
        "invalid scoring manifest",
    )
    _require(value["schema_version"] == "v0.2-opaque-scoring-manifest-v1", "scoring schema changed")
    records = value["records"]
    _require(isinstance(records, list) and len(records) == 200, "scoring record count changed")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, dict), "scoring record must be an object")
        _require(
            set(record) == {"asset_id", "byte_count", "relative_path", "sha256"},
            "scoring fields changed",
        )
        asset_id = f"asset-{index:06d}"
        _require(record["asset_id"] == asset_id, "scoring asset order changed")
        _require(record["relative_path"] == f"assets/{asset_id}.jpg", "opaque path changed")
        _require(
            isinstance(record["byte_count"], int) and record["byte_count"] > 0, "invalid byte count"
        )
        _require(
            isinstance(record["sha256"], str)
            and len(record["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in record["sha256"]),
            "invalid image identity",
        )
        _require(asset_id not in result, "duplicate scoring asset")
        result[asset_id] = dict(record)
    return result


def load_normal_manifest(
    path: Path, *, expected_sha256: str, partition: str
) -> list[dict[str, Any]]:
    """Load a normal-only JSONL manifest without resolving its image paths."""
    _verify_file(path, expected_sha256, label=f"normal {partition} manifest")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                _require(line.endswith("\n"), "normal manifest line ending changed")
                record = json.loads(line)
                _require(
                    isinstance(record, dict) and set(record) == NORMAL_RECORD_KEYS,
                    "normal fields changed",
                )
                _require(
                    record["schema_version"] == "v0.2-normal-partition-manifest-v1",
                    "normal schema changed",
                )
                _require(record["partition"] == partition, "normal partition changed")
                relative_path = _canonical_relative_path(
                    record["relative_path"], label="normal path"
                )
                _require(
                    relative_path.startswith("pcb2/Data/Images/Normal/"),
                    "normal path prefix changed",
                )
                _require(
                    isinstance(record["byte_count"], int) and record["byte_count"] > 0,
                    "invalid normal byte count",
                )
                _require(
                    isinstance(record["selection_rank"], int)
                    and record["selection_rank"]
                    == line_number + (20 if partition == "calibration" else 0),
                    "normal parent rank changed",
                )
                for field in ("selection_sha256", "sha256"):
                    _require(
                        isinstance(record[field], str)
                        and len(record[field]) == 64
                        and all(character in "0123456789abcdef" for character in record[field]),
                        f"invalid normal {field}",
                    )
                records.append(record)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V0_3InventoryError(f"cannot read normal {partition} manifest") from error
    expected_count = 881 if partition == "calibration" else 20
    _require(len(records) == expected_count, f"normal {partition} count changed")
    paths = [record["relative_path"] for record in records]
    _require(len(paths) == len(set(paths)), f"duplicate normal {partition} path")
    return records


def select_normal_partition(
    records: Sequence[Mapping[str, Any]],
    *,
    selection_prefix: str,
    selected_count: int,
) -> list[dict[str, Any]]:
    """Apply the preregistered SHA-256 path ranking to metadata records."""
    _require(len(records) >= selected_count > 0, "normal selection count is invalid")
    ranked: list[tuple[str, str, Mapping[str, Any]]] = []
    for record in records:
        path = _canonical_relative_path(record.get("relative_path"), label="normal path")
        digest = _sha256_text(f"{selection_prefix}{path}")
        ranked.append((digest, path, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: list[dict[str, Any]] = []
    for rank, (selection_sha256, relative_path, source) in enumerate(
        ranked[:selected_count], start=1
    ):
        selected.append(
            {
                "contract_version": ARTIFACT_CONTRACT_VERSION,
                "diagnostic_id": DIAGNOSTIC_ID,
                "selection_rank": rank,
                "selection_sha256": selection_sha256,
                "relative_path": relative_path,
                "byte_count": source["byte_count"],
                "sha256": source["sha256"],
            }
        )
    return selected


def build_review_assets(
    scoring_records: Mapping[str, Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project the fixed opaque identities onto the first-pass safe allowlist."""
    assets: list[dict[str, Any]] = []
    for asset_id in EXPECTED_ASSET_IDS:
        _require(asset_id in scoring_records, "selected asset is absent from scoring manifest")
        source = scoring_records[asset_id]
        record = {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "diagnostic_id": DIAGNOSTIC_ID,
            "asset_id": asset_id,
            "byte_count": source["byte_count"],
            "sha256": source["sha256"],
        }
        assets.append(validate_tabular_record("review_asset", record, schema=schema))
    return assets


def build_review_case_links(
    *,
    failure_hashes: Mapping[str, str],
    schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the protected later-join linkage separately from first-pass inputs."""
    links: list[dict[str, Any]] = []
    for method, case_type, rank, asset_id in EXPECTED_METHOD_CASES:
        record = {
            "contract_version": ARTIFACT_CONTRACT_VERSION,
            "diagnostic_id": DIAGNOSTIC_ID,
            "method": method,
            "case_type": case_type,
            "rank": rank,
            "asset_id": asset_id,
            "parent_failure_sha256": failure_hashes[method],
        }
        links.append(validate_tabular_record("review_case_link", record, schema=schema))
    return links


def serialize_csv(fieldnames: Sequence[str], records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize one artifact deterministically as UTF-8 RFC-style CSV with LF endings."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return stream.getvalue().encode("utf-8")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_parent_evidence(
    repository_root: Path, external_root: Path, config: Mapping[str, Any]
) -> None:
    run_root = repository_root / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final"
    parent = config["parent_evidence"]
    checks = (
        (
            run_root / "artifact-manifest.json",
            parent["artifact_manifest_sha256"],
            "artifact manifest",
        ),
        (run_root / "project-decision.json", parent["project_decision_sha256"], "project decision"),
        (
            run_root / "boundary/boundary-record.json",
            parent["boundary_record_sha256"],
            "boundary record",
        ),
        (repository_root / "configs/v0.2.yaml", parent["v0_2_config_sha256"], "v0.2 config"),
        (
            repository_root / "schemas/v0.2/evaluation-artifacts.json",
            parent["v0_2_schema_sha256"],
            "v0.2 schema",
        ),
        (
            repository_root / "uv.lock",
            config["dependencies"]["root_lock_sha256"],
            "root dependency lock",
        ),
        (
            repository_root / "environments/v0.2-preflight/uv.lock",
            config["dependencies"]["dinov2_lock_sha256"],
            "DINOv2 dependency lock",
        ),
        (
            external_root / "normal-manifests/manifest-set.json",
            parent["normal_manifest_set_sha256"],
            "normal manifest set",
        ),
    )
    for path, expected, label in checks:
        _verify_file(path, expected, label=label)


def construct_inventory_bundle(
    *,
    repository_root: Path,
    external_root: Path,
    source_commit: str,
) -> dict[str, bytes]:
    """Construct all v0.3.2 artifacts in memory from metadata-only inputs."""
    config_path = repository_root / "configs/v0.3.yaml"
    schema_path = repository_root / "schemas/v0.3/diagnostic-artifacts.json"
    config = load_v0_3_config(config_path)
    schema = load_v0_3_schema(schema_path)
    _require(
        len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit),
        "invalid source commit",
    )
    _verify_file(
        repository_root / config["preregistration"]["document"],
        PREREGISTRATION_DOCUMENT_SHA256,
        label="v0.3 preregistration",
    )
    _verify_parent_evidence(repository_root, external_root, config)

    failure_root = repository_root / "artifacts/v0.2/evaluation/visa-pcb2-v0-2-final"
    failure_hashes = config["parent_evidence"]["failure_case_sha256"]
    observed_cases: list[tuple[str, str, int, str]] = []
    for method in config["method_order"]:
        observed_cases.extend(
            _load_failure_case_identities(
                failure_root / method / "failure-cases.csv",
                method=method,
                expected_sha256=failure_hashes[method],
            )
        )
    _require(tuple(observed_cases) == EXPECTED_METHOD_CASES, "failure selections changed")
    duplicates = Counter(asset_id for _, _, _, asset_id in observed_cases)
    _require(
        {asset_id: count for asset_id, count in duplicates.items() if count > 1}
        == {"asset-000112": 2},
        "shared review asset changed",
    )

    scoring = load_scoring_manifest(
        external_root / "scorer/scoring-manifest.json",
        expected_sha256=config["parent_evidence"]["opaque_scoring_manifest_sha256"],
    )
    calibration = load_normal_manifest(
        external_root / "normal-manifests/calibration.jsonl",
        expected_sha256=config["parent_evidence"]["normal_calibration_manifest_sha256"],
        partition="calibration",
    )
    reference = load_normal_manifest(
        external_root / "normal-manifests/reference.jsonl",
        expected_sha256=config["parent_evidence"]["normal_reference_manifest_sha256"],
        partition="reference",
    )
    calibration_paths = {record["relative_path"] for record in calibration}
    reference_paths = {record["relative_path"] for record in reference}
    _require(not (calibration_paths & reference_paths), "normal partitions overlap")

    review_assets = build_review_assets(scoring, schema=schema)
    review_links = build_review_case_links(failure_hashes=failure_hashes, schema=schema)
    normal_partition = select_normal_partition(
        calibration,
        selection_prefix=config["normal_probe"]["selection_prefix"],
        selected_count=config["normal_probe"]["selected_count"],
    )
    for record in normal_partition:
        validate_tabular_record("normal_partition", record, schema=schema)
    _require(
        len({record["relative_path"] for record in normal_partition}) == 60,
        "normal selection duplicates",
    )
    _require(
        not ({record["relative_path"] for record in normal_partition} & reference_paths),
        "selected normals overlap reference",
    )

    review_bytes = serialize_csv(REVIEW_ASSET_FIELDS, review_assets)
    link_bytes = serialize_csv(REVIEW_CASE_FIELDS, review_links)
    normal_bytes = serialize_csv(NORMAL_PARTITION_FIELDS, normal_partition)
    checkpoint = {
        "contract_version": ARTIFACT_CONTRACT_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_commit": source_commit,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        "parent_artifact_manifest_sha256": config["parent_evidence"]["artifact_manifest_sha256"],
        "opaque_scoring_manifest_sha256": config["parent_evidence"][
            "opaque_scoring_manifest_sha256"
        ],
        "normal_calibration_manifest_sha256": config["parent_evidence"][
            "normal_calibration_manifest_sha256"
        ],
        "normal_reference_manifest_sha256": config["parent_evidence"][
            "normal_reference_manifest_sha256"
        ],
        "ecc_failure_cases_sha256": failure_hashes["ecc_residual"],
        "patch_hog_failure_cases_sha256": failure_hashes["patch_hog_ocsvm"],
        "dinov2_failure_cases_sha256": failure_hashes["dinov2_vits14_224_nn"],
        "review_assets_sha256": _sha256_bytes(review_bytes),
        "review_case_linkage_sha256": _sha256_bytes(link_bytes),
        "normal_partition_sha256": _sha256_bytes(normal_bytes),
        "review_asset_count": len(review_assets),
        "method_case_count": len(review_links),
        "normal_partition_count": len(normal_partition),
        "diagnostic_image_accessed": False,
        "diagnostic_image_decoded": False,
        "diagnostic_image_displayed": False,
        "diagnostic_score_computed": False,
        "v0_2_final_test_rescored": False,
        "unselected_final_test_image_accessed": False,
        "protected_metadata_exposed": False,
        "v0_2_artifacts_modified": False,
        "preexisting_output_detected": False,
        "output_overwrite_performed": False,
        "raw_data_in_git": False,
    }
    try:
        validate_checkpoint_record(checkpoint)
    except V0_3DiagnosticContractError as error:
        raise V0_3InventoryError("constructed checkpoint violates the contract") from error
    return {
        "review-assets.csv": review_bytes,
        "review-case-linkage.csv": link_bytes,
        "normal-diagnostic-partition.csv": normal_bytes,
        "pre-access-checkpoint.json": _canonical_json_bytes(checkpoint),
    }


def write_inventory_bundle(output_root: Path, files: Mapping[str, bytes]) -> None:
    """Write a complete metadata bundle once, without overwriting prior evidence."""
    _require(not output_root.exists(), "v0.3 diagnostic output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        for relative_name in (
            "review-assets.csv",
            "review-case-linkage.csv",
            "normal-diagnostic-partition.csv",
            "pre-access-checkpoint.json",
        ):
            _require(relative_name in files, "inventory bundle is incomplete")
            path = temporary / relative_name
            with path.open("xb") as stream:
                stream.write(files[relative_name])
        _require(
            set(files) == {path.name for path in temporary.iterdir()}, "unexpected inventory file"
        )
        temporary.rename(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
