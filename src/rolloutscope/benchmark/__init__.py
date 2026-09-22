"""One reproducible detector benchmark, shared by the script and the test.

This package exists to remove a specific defect: the repository had two TRACE
paths that mapped rows differently, sampled different row counts, and reported
different numbers for the same dataset. Both callers now go through
:func:`evaluate_rows`, and everything that could move a result is pinned in a
manifest and echoed on the report.

Nothing here is imported by the analysis path. Fetching touches the network and
belongs to opt-in tooling only.
"""

from rolloutscope.benchmark.applicability import (
    DETECTOR_REQUIREMENTS,
    Applicability,
    Requirement,
    applicability,
    available_signals,
)
from rolloutscope.benchmark.evaluate import (
    evaluate_rows,
    population_signals,
    score_detector,
)
from rolloutscope.benchmark.hf import (
    DATASET,
    BenchmarkUnavailable,
    current_revision,
    dataset_info,
    dataset_license,
    load_dotenv_token,
    load_rows,
)
from rolloutscope.benchmark.manifest import (
    MANIFEST_VERSION,
    BenchmarkManifest,
    DatasetRef,
    RowRef,
    Verification,
    build_manifest,
    load_manifest,
    row_digest,
    save_manifest,
    verify,
)
from rolloutscope.benchmark.metrics import (
    Counts,
    DetectorMetrics,
    Interval,
    Outcome,
    Rate,
    aggregate,
    wilson_interval,
)
from rolloutscope.benchmark.report import (
    REPORT_VERSION,
    BenchmarkReport,
    DetectorPin,
    ExampleCase,
    detector_fingerprint,
    render_text,
)
from rolloutscope.benchmark.report import to_json as report_to_json
from rolloutscope.benchmark.splits import (
    SPLIT_VERSION,
    SPLIT_WEIGHTS,
    SPLITS,
    Split,
    assign_split,
    normalize_scenario_text,
    scenario_key,
)
from rolloutscope.benchmark.taxonomy import (
    FAMILIES,
    SUBCATEGORIES,
    families_of,
    is_hacked,
    name_of,
    parse_label,
)
from rolloutscope.benchmark.trace_map import (
    TRACE_MAPPING_VERSION,
    TraceRowError,
    map_row,
    provenance_of,
)

__all__ = [
    "DATASET",
    "DETECTOR_REQUIREMENTS",
    "FAMILIES",
    "MANIFEST_VERSION",
    "REPORT_VERSION",
    "SPLITS",
    "SPLIT_VERSION",
    "SPLIT_WEIGHTS",
    "SUBCATEGORIES",
    "TRACE_MAPPING_VERSION",
    "Applicability",
    "BenchmarkManifest",
    "BenchmarkReport",
    "BenchmarkUnavailable",
    "Counts",
    "DatasetRef",
    "DetectorMetrics",
    "DetectorPin",
    "ExampleCase",
    "Interval",
    "Outcome",
    "Rate",
    "Requirement",
    "RowRef",
    "Split",
    "TraceRowError",
    "Verification",
    "aggregate",
    "applicability",
    "assign_split",
    "available_signals",
    "build_manifest",
    "current_revision",
    "dataset_info",
    "dataset_license",
    "detector_fingerprint",
    "evaluate_rows",
    "families_of",
    "is_hacked",
    "load_dotenv_token",
    "load_manifest",
    "load_rows",
    "map_row",
    "name_of",
    "normalize_scenario_text",
    "parse_label",
    "population_signals",
    "provenance_of",
    "render_text",
    "report_to_json",
    "row_digest",
    "save_manifest",
    "scenario_key",
    "score_detector",
    "verify",
    "wilson_interval",
]
