"""Identity, format dispatch, and exact ingestion accounting regressions.

All records created here are synthetic fixtures using the pinned raw contract
and RolloutScope's versioned normalized schema.
"""

from dataclasses import replace
from pathlib import Path

import orjson
import pytest

from rolloutscope.adapters import NORMALIZED, PRIME_RL_TRAIN, VERIFIERS_EVAL, resolve_adapter
from rolloutscope.analysis import aggregate_rollouts
from rolloutscope.detectors import DetectorConfig, builtin_detectors, execute_detector
from rolloutscope.schema import write_rollouts
from rolloutscope.schema.execution import IngestionSummary

RAW = {
    "example_id": 0,
    "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
    "completion": [{"role": "assistant", "content": "4"}],
    "reward": 1.0,
    "metrics": {"accuracy": 1.0},
    "is_completed": True,
    "is_truncated": False,
}


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write small synthetic JSON records as test inputs."""
    path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))


def test_train_conversion_preserves_every_identity_and_step(
    prime_rl_run_dir: Path, tmp_path: Path
) -> None:
    original = list(PRIME_RL_TRAIN.load(prime_rl_run_dir))
    converted = tmp_path / "converted.jsonl"
    write_rollouts(converted, original)
    assert resolve_adapter(converted) is NORMALIZED
    once = list(NORMALIZED.load(converted))
    twice = list(NORMALIZED.load(converted))
    assert once == twice == original
    assert [row.step_index for row in once] == [0, 0, 0, 1, 1]
    assert all(row.provenance.source_path.startswith("step_") for row in once)
    assert aggregate_rollouts(iter(once)) == aggregate_rollouts(iter(original))
    config = DetectorConfig()
    for detector in builtin_detectors().values():
        assert execute_detector(detector, once, config) == execute_detector(
            detector, original, config
        )


def test_normalized_namespaces_survive_relocation_and_repeated_reads(tmp_path: Path) -> None:
    run = tmp_path / "source"
    run.mkdir()
    write_rows(run / "results.jsonl", [RAW])
    (run / "metadata.json").write_text(
        '{"env_id":"stable-environment","task_namespace":"stable-task"}'
    )
    raw = next(VERIFIERS_EVAL.load(run))
    assert raw.environment_namespace == "stable-environment"
    assert raw.task_namespace == "stable-task"
    assert raw.provenance.namespace == raw.run_id

    relocated = tmp_path / "other-machine" / "normalized.jsonl"
    write_rollouts(relocated, [raw])
    once = next(NORMALIZED.load(relocated))
    twice = next(NORMALIZED.load(relocated))
    assert once == twice == raw


def test_probe_skips_invalid_leaders_but_load_counts_them(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    write_rows(raw_path, [RAW])
    normalized = next(VERIFIERS_EVAL.load(raw_path)).model_dump(mode="json")
    path = tmp_path / "normalized.jsonl"
    path.write_bytes(b"\nnot JSON\n[]\n{}\n" + orjson.dumps(normalized) + b"\n")
    assert resolve_adapter(path) is NORMALIZED
    counts = IngestionSummary()
    assert len(list(NORMALIZED.load(path, diagnostics=counts))) == 1
    assert (counts.observed, counts.accepted, counts.rejected, counts.blank) == (5, 1, 3, 1)
    assert counts.status == "partial"
    assert [issue.line for issue in counts.files[0].examples] == [2, 3, 4]


@pytest.mark.parametrize("version", ["999.0", "invalid", None, {}, 2.5])
def test_schema_marker_cannot_fall_back_to_raw(version: object, tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    write_rows(path, [{**RAW, "schema_version": version}])
    assert resolve_adapter(path) is NORMALIZED
    counts = IngestionSummary()
    assert list(resolve_adapter(path).load(path, diagnostics=counts)) == []
    assert counts.rejected == 1
    assert counts.accepted == 0
    assert counts.files[0].reason_counts == {"unsupported_schema": 1}
    assert counts.status == "unsupported"


@pytest.mark.parametrize("normalized_first", [True, False])
def test_mixed_rows_preserve_normalized_identity(normalized_first: bool, tmp_path: Path) -> None:
    source = tmp_path / "original.jsonl"
    write_rows(source, [RAW])
    original = next(VERIFIERS_EVAL.load(source))
    normalized = original.model_dump(mode="json")
    rows = [normalized, RAW] if normalized_first else [RAW, normalized]
    mixed = tmp_path / "mixed.jsonl"
    write_rows(mixed, rows)
    counts = IngestionSummary()
    loaded = list(resolve_adapter(mixed).load(mixed, diagnostics=counts))
    assert loaded[0 if normalized_first else 1] == original
    raw = loaded[1 if normalized_first else 0]
    assert raw.run_id != original.run_id
    assert raw.occurrence_id != original.occurrence_id
    assert counts.files[0].format == "mixed"
    assert counts.accepted == 2


def test_raw_occurrences_survive_append_and_rescore(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    write_rows(path, [RAW, RAW])
    initial = list(VERIFIERS_EVAL.load(path))
    assert initial[0].occurrence_id != initial[1].occurrence_id
    assert initial[0].content_fingerprint == initial[1].content_fingerprint
    write_rows(path, [{**RAW, "reward": 0.5, "metrics": {"accuracy": 0.5}}, RAW, RAW])
    updated = list(VERIFIERS_EVAL.load(path))
    assert [row.occurrence_id for row in initial] == [row.occurrence_id for row in updated[:2]]
    assert initial[0].content_fingerprint == updated[0].content_fingerprint
    assert initial[0].scoring_revision != updated[0].scoring_revision
    assert initial[0].run_id == updated[0].run_id


def test_normalized_duplicate_occurrences_retained_and_counted(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    write_rows(path, [RAW])
    row = next(VERIFIERS_EVAL.load(path))
    path = tmp_path / "normalized.jsonl"
    write_rollouts(path, [row, row, row])
    counts = IngestionSummary()
    assert list(NORMALIZED.load(path, diagnostics=counts)) == [row, row, row]
    assert counts.accepted == 3
    assert counts.duplicates == 2
    assert counts.rejected == 0


def test_duplicate_accounting_across_manifest_files(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    write_rows(raw_path, [RAW])
    row = next(VERIFIERS_EVAL.load(raw_path))
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_rollouts(first, [row])
    write_rollouts(second, [row])
    manifest = NORMALIZED.load_run(first)
    manifest = replace(manifest, files=(manifest.files[0], replace(manifest.files[0], path=second)))
    counts = IngestionSummary()
    assert len(list(NORMALIZED.load_manifest(manifest, diagnostics=counts))) == 2
    assert [item.duplicates for item in counts.files] == [0, 1]


def test_same_basename_runs_are_distinct_and_metadata_is_mutable(tmp_path: Path) -> None:
    runs = [tmp_path / "a" / "run", tmp_path / "b" / "run"]
    for run in runs:
        run.mkdir(parents=True)
        write_rows(run / "results.jsonl", [RAW])
        (run / "metadata.json").write_text('{"cost": 1}')
    first = next(VERIFIERS_EVAL.load(runs[0]))
    second = next(VERIFIERS_EVAL.load(runs[1]))
    assert first.run_id != second.run_id
    (runs[0] / "metadata.json").write_text('{"cost": 2}')
    assert next(VERIFIERS_EVAL.load(runs[0])) == first
    manifest = VERIFIERS_EVAL.load_run(runs[0])
    assert manifest.metadata_sources == (runs[0] / "metadata.json",)


def test_load_manifest_reuses_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "raw.jsonl"
    write_rows(path, [RAW])
    manifest = VERIFIERS_EVAL.load_run(path)

    def fail_discovery(path: Path) -> None:
        raise AssertionError("discovered a second time")

    monkeypatch.setattr(VERIFIERS_EVAL, "load_run", fail_discovery)
    assert len(list(VERIFIERS_EVAL.load_manifest(manifest))) == 1


def test_old_normalized_ids_and_extras_survive_migration(tmp_path: Path) -> None:
    path = tmp_path / "legacy.jsonl"
    write_rows(
        path,
        [
            {
                **RAW,
                "schema_version": "1.0",
                "run_id": "legacy-run",
                "rollout_id": "legacy-rollout",
                "group_id": "legacy-group",
                "step_index": 12,
                "custom": {"keep": [1, 2]},
            }
        ],
    )
    row = next(NORMALIZED.load(path))
    assert row.schema_version == "2.0"
    assert (row.run_id, row.rollout_id, row.group_id, row.step_index) == (
        "legacy-run",
        "legacy-rollout",
        "legacy-group",
        12,
    )
    assert row.model_dump()["custom"] == {"keep": [1, 2]}
    assert row.provenance.source_path == "legacy.jsonl"
    assert row.provenance.line == 1
    exported = tmp_path / "exported.jsonl"
    write_rollouts(exported, [row])
    assert next(NORMALIZED.load(exported)) == row


def test_exact_source_hash_and_byte_count(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "raw.jsonl"
    write_rows(path, [RAW, RAW])
    counts = IngestionSummary()
    list(VERIFIERS_EVAL.load(path, diagnostics=counts))
    source = counts.files[0]
    assert source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.size_bytes == path.stat().st_size
    assert (source.observed, source.accepted, source.duplicates) == (2, 2, 0)


def test_deleted_manifest_source_reports_failure(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    write_rows(path, [RAW])
    manifest = VERIFIERS_EVAL.load_run(path)
    path.unlink()
    counts = IngestionSummary()
    with pytest.raises(FileNotFoundError):
        list(VERIFIERS_EVAL.load_manifest(manifest, diagnostics=counts))
    assert counts.status == "failed"
    assert counts.files[0].error
