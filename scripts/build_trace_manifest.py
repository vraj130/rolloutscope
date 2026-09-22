"""Regenerate the frozen TRACE benchmark manifest.

The manifest pins what a reported number was measured on: the dataset revision,
the exact rows, a content hash per row, and each row's split. It carries no row
content, because TRACE is gated and CC-BY-SA-4.0; anyone with dataset access
refetches at the pinned revision and the hashes confirm the bytes match.

Usage:

    uv run --extra benchmark python scripts/build_trace_manifest.py
    uv run --extra benchmark python scripts/build_trace_manifest.py --revision <sha>

Without ``--revision`` the script pins whatever the dataset's current revision
is, which is how a new pin is minted. Passing an explicit revision reproduces an
existing manifest. The dataset is gated: put ``HF_TOKEN`` in a repo-root ``.env``
or export it. Anything the network or the license refuses prints a reason and
exits 0, so this is safe to run anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rolloutscope.benchmark import (
    BenchmarkUnavailable,
    DatasetRef,
    build_manifest,
    current_revision,
    dataset_info,
    dataset_license,
    load_dotenv_token,
    load_rows,
    save_manifest,
)
from rolloutscope.benchmark.hf import DATASET

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "benchmarks" / "trace" / "manifest.json"
PARQUET_FILE = "data/train-00000-of-00001.parquet"

NOTES = [
    "Row content is not committed: TRACE is gated and CC-BY-SA-4.0. This manifest "
    "carries row identity, labels, and per-row content hashes so a fetch at the "
    "pinned revision can be verified byte for byte.",
    "Splits are assigned per scenario, never per row, so sibling trajectories of "
    "one task cannot straddle a split boundary. At this revision the 517 rows "
    "yield 516 distinct scenario keys, so the grouping is close to a no-op here; "
    "it is kept as the correct guard and it does catch the one duplicate pair.",
    "The holdout partition is sealed. It exists so an accuracy claim can be made "
    "once the Phase 2 detector repairs land. Thresholds are set on tuning only.",
]


def main() -> int:
    """Fetch, build, and write the manifest; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=None, help="dataset commit SHA to pin")
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH, help="manifest path")
    args = parser.parse_args()

    load_dotenv_token(REPO_ROOT)
    try:
        info = dataset_info(DATASET)
        license_value = dataset_license(info)
        revision = args.revision or current_revision(info)
        rows, source, file_digest = load_rows(revision=revision, parquet_file=PARQUET_FILE)
    except BenchmarkUnavailable as exc:
        print(f"SKIP: {exc}")
        return 0

    if source != "parquet":
        print(
            "SKIP: only the revision-pinned parquet path may mint a manifest, and it "
            f"was unavailable (fell back to {source}). Install the extra with "
            "`uv sync --extra benchmark` and retry."
        )
        return 0

    dataset = DatasetRef(
        name=DATASET,
        revision=revision,
        license=license_value,
        gated=bool(info.get("gated")),
        redistribution=("restricted: gated dataset under CC-BY-SA-4.0; identity and hashes only"),
        files={PARQUET_FILE: file_digest},
    )
    manifest = build_manifest(list(rows), dataset, notes=NOTES)
    save_manifest(manifest, args.out)

    print(f"wrote {args.out.relative_to(REPO_ROOT)}")
    print(f"  revision   {revision}")
    print(f"  license    {license_value}")
    print(f"  rows       {manifest.counts.rows}")
    print(f"  scenarios  {manifest.counts.scenarios}")
    print(f"  by split   {manifest.counts.by_split}")
    print(f"  positives  {manifest.counts.positives_by_split}")
    print(f"  negatives  {manifest.counts.negatives_by_split}")
    print(f"  digest     {manifest.digest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
