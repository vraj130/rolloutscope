"""Versioned analysis accounting shared by ingestion, detectors, and renderers."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, computed_field, model_validator

from rolloutscope.schema.models import JsonSafeModel

AnalysisStatus = Literal["complete", "partial", "empty", "unsupported", "failed"]


class IngestionIssue(JsonSafeModel):
    """One bounded diagnostic example, located in the source without copying its row."""

    line: int | None = None
    reason: str
    message: str


class FileIngestion(JsonSafeModel):
    """Exact line accounting for a source; accepted duplicates are retained."""

    path: str
    format: str = "unknown"
    adapter_version: str = "1"
    sha256: str | None = None
    size_bytes: int = 0
    observed: int = 0
    accepted: int = 0
    rejected: int = 0
    blank: int = 0
    duplicates: int = 0
    reason_counts: dict[str, int] = Field(default_factory=dict)
    examples: list[IngestionIssue] = Field(default_factory=list)
    error: str | None = None

    def reject(self, line: int | None, reason: str, message: str) -> None:
        """Count a rejected record, keeping at most five bounded diagnostic examples."""
        self.rejected += 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        if len(self.examples) < 5:
            self.examples.append(IngestionIssue(line=line, reason=reason, message=message[:400]))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> AnalysisStatus:
        """Distinguish empty, rejected, partially usable, and complete input files."""
        if self.error:
            return "failed"
        if not self.accepted:
            return "unsupported" if self.rejected else "empty"
        return "partial" if self.rejected else "complete"


class IngestionSummary(JsonSafeModel):
    """Run-wide accounting; records = accepted + rejected + blank, including duplicates."""

    files: list[FileIngestion] = Field(default_factory=list)
    duplicate_policy: str = "retain occurrences; count repeated occurrence IDs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observed(self) -> int:
        """Return the number of physical source lines examined."""
        return sum(item.observed for item in self.files)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accepted(self) -> int:
        """Return the number of validated records delivered for analysis."""
        return sum(item.accepted for item in self.files)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rejected(self) -> int:
        """Return the number of malformed or unsupported records."""
        return sum(item.rejected for item in self.files)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blank(self) -> int:
        """Return the number of empty physical source lines."""
        return sum(item.blank for item in self.files)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duplicates(self) -> int:
        """Return retained records that repeated an occurrence identity."""
        return sum(item.duplicates for item in self.files)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> AnalysisStatus:
        """Summarize ingestion without interpreting detector findings."""
        if any(item.error for item in self.files):
            return "failed"
        if not self.accepted:
            return "unsupported" if self.rejected else "empty"
        return "partial" if self.rejected else "complete"


class UnitCounts(JsonSafeModel):
    """One detector mode's disjoint candidate outcomes and underlying measurements."""

    model_config = ConfigDict(allow_inf_nan=False)

    unit: Literal["rollout", "group", "step", "run"] = "rollout"
    mode: str = "snapshot"
    candidate: int = Field(default=0, ge=0)
    eligible: int = Field(default=0, ge=0)
    fired: int = Field(default=0, ge=0)
    clean: int = Field(default=0, ge=0)
    insufficient_data: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    reason_counts: dict[str, int] = Field(default_factory=dict)
    measurements: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _accounting(self) -> UnitCounts:
        if self.eligible != self.fired + self.clean:
            raise ValueError("eligible must equal fired + clean")
        if self.candidate != self.eligible + self.insufficient_data + self.skipped + self.errors:
            raise ValueError(
                "candidate counts must partition into eligible, insufficient, skipped, errors"
            )
        return self


class DetectorExecution(JsonSafeModel):
    """Execution and coverage of one selected detector, even if it emits no verdicts."""

    detector: str
    category: str = "unknown"
    version: str = "unknown"
    status: Literal["complete", "insufficient_data", "skipped", "failed", "unsupported"] = (
        "complete"
    )
    config: dict[str, JsonValue] = Field(default_factory=dict)
    units: list[UnitCounts] = Field(default_factory=list)
    verdict_count: int = 0
    errors: list[str] = Field(default_factory=list)


class OutputArtifact(JsonSafeModel):
    """Output reference; digests are populated in the final external manifest ledger."""

    name: str
    role: str
    sha256: str | None = None
    size_bytes: int | None = None


class ExecutionManifest(JsonSafeModel):
    """Deterministic execution inputs and status, independent of findings."""

    manifest_version: str = "1.0"
    tool_version: str
    schema_version: str
    source_revision: str | None = None
    implementation_sha256: str | None = None
    input_format: str
    adapter_version: str = "1"
    effective_config: dict[str, JsonValue] = Field(default_factory=dict)
    selected_detectors: list[str] = Field(default_factory=list)
    detector_versions: dict[str, str] = Field(default_factory=dict)
    run_ids: list[str] = Field(default_factory=list)
    environment_namespaces: list[str] = Field(default_factory=list)
    task_namespaces: list[str] = Field(default_factory=list)
    ingestion: IngestionSummary = Field(default_factory=IngestionSummary)
    status: AnalysisStatus = "complete"
    errors: list[str] = Field(default_factory=list)
    artifacts: list[OutputArtifact] = Field(default_factory=list)
    artifact_manifest: str | None = None
