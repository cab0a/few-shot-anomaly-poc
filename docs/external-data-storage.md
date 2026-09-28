# External Data Storage and Restoration

## 日本語概要

本書は、実行データをGoogle Driveへ保存し、別PCで復元する運用を定めます。GitHubにはコード、設定、評価記録、固定checksumと転送コマンドを置き、元画像・外部manifest・学習済み状態はDriveで共有します。元データのあるPCで一度backupを実行し、他のPCではrestoreで所定の場所へ戻します。初回保存はまだ必要で、保存先フォルダの作成だけではバックアップ完了になりません。詳細は以下の英語本文を参照してください。

---

## English Summary

GitHub stores the code, fixed identities, evaluation records, and transfer
commands. A Google Drive sync folder stores the external v0.2 inputs and
fitted state needed for v0.3.4. The original data PC publishes one immutable
archive and checksum receipt; another PC restores it after verifying the
archive, every file, and the preregistered required identities. Initial
backup still requires the original data. Creating the folder does not
create or recover that data.

## Storage Contract

The checked-in configuration is
[`external-data-store.json`](../configs/external-data-store.json).
The store location is:

```text
My Drive / inefficiencylab / few-shot-anomaly-poc / external-data
```

The Japanese desktop-client folder name is `マイドライブ`. The store holds:

| File | Purpose |
| --- | --- |
| `README.txt` | Entry point and operational commands |
| `visa-pcb2-v0-2-inputs.zip` | Immutable, byte-preserving external-data bundle |
| `visa-pcb2-v0-2-inputs.json` | Configuration identity, archive SHA-256, sizes, and counts |

The ZIP includes an internal per-file SHA-256 manifest and exactly the files
found under these two repository-relative roots:

```text
data/external/v0.2/evaluation/visa-pcb2-v0-2-final/
work/v0.2/evaluation/visa-pcb2-v0-2-final/fitted-state/
```

This covers the current diagnostic's external manifests, image assets,
source data, and existing fitted state. It does not claim to back up every
historical cache, v0.1 asset, environment, or model download in the project.
The fitted state is copied as opaque bytes and is never loaded by the
transfer tool. There is no refitting or reconstruction of missing assets.

The store is operator material: it can contain sealed mappings and semantic
source paths, and must not be given to the blinded reviewer. No public link
or sharing-permission change is part of this workflow.

## Prepare Drive Access Once Per PC

Sign in to the same Google Drive storage location with the desktop client.
In WSL, make the Windows drive available under `/mnt`. For drive `G:`:

```bash
sudo mkdir -p /mnt/g
mountpoint -q /mnt/g || sudo mount -t drvfs G: /mnt/g
```

Use the actual drive letter on that PC. The mount may need to be repeated
after WSL restarts. The transfer script detects one mounted `My Drive` or
`マイドライブ` folder. If the folder is elsewhere or more than one is found,
pass its full store path explicitly:

```bash
python3 scripts/external_data.py status \
  --store '/mnt/g/マイドライブ/inefficiencylab/few-shot-anomaly-poc/external-data'
```

`FEW_SHOT_DATA_STORE` can also specify this path. Windows Python supports a
Windows store path directly. The script uses only Python 3.11+ standard
library modules and does not require `uv`, NumPy, OpenCV, or PyTorch.

## Original Data PC: Initial Backup

From the repository containing the original, unchanged external inputs:

```bash
git pull --ff-only
python3 scripts/external_data.py backup
```

Before packaging, the script verifies 35 fixed identities from the committed
v0.3 contract: four external metadata files, three fitted states, and the
28 selected image files. It then inventories all files in the two allowed
roots, calculates their hashes, and checks bytes again while archiving.
No unrelated work directory, `.git`, credential file outside the allowed
roots, or environment is collected.

The archive is written to the Drive sync folder and verified there. The
receipt is written only afterward. Wait until the desktop client reports
that synchronization has finished before relying on the other PC. A local
file or receipt alone is not proof of completed cloud upload. Repeating
the command with identical inputs is safe; a different or incomplete
existing archive is preserved and causes a stop.

## Another PC: Restore and Continue

After the initial backup has synchronized:

```bash
git pull --ff-only
python3 scripts/external_data.py restore
uv sync --locked
uv run --locked --no-sync python scripts/run_v0_3_4_first_observation.py \
  --check-only
```

The transfer command restores the expected directory layout automatically.
It verifies the receipt, archive checksum, complete archive inventory,
individual file checksums, and all 35 fixed input identities before
installing any files. Existing identical files are retained; an existing
file with different bytes stops the entire installation before any new
file is installed. Symlinks, traversal paths, unexpected members, duplicate
paths, corrupt archives, and bundles exceeding 20 GiB are rejected.

Temporary archives and verification files stay in ignored `work/data-transfer/`.
Allow local disk space for the downloaded archive, extracted verification
copy, and restored files. An interrupted installation may leave newly
created files; they are never silently overwritten on a later attempt.

If `uv` is absent on the currently prepared PC, its project-local copy can
be used by adding it to the current shell's PATH:

```bash
export PATH="$PWD/work/bootstrap/bin:$PATH"
```

That ignored bootstrap executable is local to this PC, not part of the data
backup. Other PCs need the project's pinned `uv` version `0.11.32` if it is
not already installed.

## Check What Is Still Missing

```bash
python3 scripts/external_data.py status
```

`status` reports missing required local files and local receipt presence;
it does not claim a successful restore or a finished cloud upload.
`init` creates only the store directory and its instruction file.
When preparing this workflow, the store was initialized but the original
external data was unavailable on this PC. The first real `backup` remains
an action for the original data PC.

## Scientific and License Boundaries

This is a storage operation that reads and hashes opaque bytes. It does
not decode or display images, collect observations, run an anomaly scorer,
change fitted state, or rescore the completed final test. It does not create
a v0.3.4 review completion checkpoint. After restoration, the review runner
still requires its normal preflight and an independent blinded reviewer.

The original v0.2 evidence remains immutable in Git. VisA retains its
CC BY 4.0 license; repository code and documentation retain the PolyForm
Noncommercial License 1.0.0. Storage does not change these boundaries.
