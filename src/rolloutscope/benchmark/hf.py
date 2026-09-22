"""Fetching TRACE rows, the one way, for both the script and the integration test.

Two access paths exist and they are not equivalent.

The parquet path downloads ``data/train-*.parquet`` at the exact repository
revision the manifest pins and reads it with pyarrow. It is the only path that
honours a revision pin, so it is preferred whenever pyarrow is importable
(``uv sync --extra benchmark``).

The datasets-server path uses the public rows API with stdlib urllib only. That
API takes no revision argument, so it returns whatever the dataset's default
branch currently holds. It is kept because it needs no extra dependency, and it
is safe only because every row is checked against the manifest's per-row content
hash afterwards: drift shows up as a changed row rather than as a quietly
different number.

Nothing here runs during analysis. This module is reached from the validation
script and from the integration test, both of which are opt in.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "PatronusAI/trace-dataset"
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"
SERVER_BASE = "https://datasets-server.huggingface.co"
TIMEOUT_SECONDS = 60
SERVER_PAGE = 100
"""Hard per-request cap on the datasets-server rows endpoint."""

USER_AGENT = "rolloutscope-benchmark"

PERMISSIVE_LICENSES = frozenset(
    {
        "mit",
        "apache-2.0",
        "bsd-3-clause",
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-sa-4.0",
        "odc-by",
        "cdla-permissive-2.0",
        "openrail",
    }
)
"""Licenses that permit fetching a slice for local automated validation.

An unrecognized or unverifiable license stops the fetch rather than proceeding on
an assumption. TRACE is cc-by-sa-4.0, which is in this set.
"""


class BenchmarkUnavailable(RuntimeError):
    """The benchmark data could not be reached, with the reason attached.

    Callers turn this into a skip, never a failure: no offline environment
    should break because a gated dataset was not reachable.
    """


def _token() -> str | None:
    """Return ``HF_TOKEN`` from the environment, or None."""
    return os.environ.get("HF_TOKEN") or None


def load_dotenv_token(repo_root: Path) -> None:
    """Load ``HF_TOKEN`` from a repo-root ``.env`` when python-dotenv is present.

    Input: the repository root. Output: nothing; the environment may gain
    ``HF_TOKEN``. python-dotenv is a dev dependency, so its absence is not an
    error: whatever is already exported is used instead.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(repo_root / ".env")


def _open(url: str) -> bytes:
    """GET a URL with optional bearer auth; any problem raises BenchmarkUnavailable."""
    headers = {"User-Agent": USER_AGENT}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data: bytes = response.read()
            return data
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise BenchmarkUnavailable(
                f"{url} returned {exc.code}: the dataset is gated. "
                "Set HF_TOKEN in .env or the environment and accept the dataset terms."
            ) from exc
        raise BenchmarkUnavailable(f"HTTP {exc.code} from {url}: {exc}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BenchmarkUnavailable(f"network unavailable for {url}: {exc}") from exc


def _get_json(url: str) -> Any:
    """GET and decode a JSON document, or raise BenchmarkUnavailable."""
    try:
        return json.loads(_open(url).decode("utf-8"))
    except ValueError as exc:
        raise BenchmarkUnavailable(f"non-JSON response from {url}: {exc}") from exc


def dataset_info(dataset: str = DATASET) -> dict[str, Any]:
    """Fetch the Hub metadata document for a dataset repository."""
    info = _get_json(f"{HF_API_BASE}/{dataset}")
    if not isinstance(info, dict):
        raise BenchmarkUnavailable(f"unexpected dataset info shape for {dataset}")
    return info


def dataset_license(info: dict[str, Any]) -> str:
    """Return the declared license, or raise if it is absent or not permissive.

    Input: a dataset info document. Output: the license string. Raises
    :class:`BenchmarkUnavailable` when the license cannot be read or is outside
    :data:`PERMISSIVE_LICENSES`, so an unverifiable license never leads to a
    fetch.
    """
    card = info.get("cardData") or {}
    value = card.get("license") or info.get("license")
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        raise BenchmarkUnavailable("could not verify a license for the dataset")
    if value.lower() not in PERMISSIVE_LICENSES:
        raise BenchmarkUnavailable(f"license {value!r} is not in the recognized permissive set")
    return value


def current_revision(info: dict[str, Any]) -> str:
    """Return the dataset repository's current commit SHA."""
    sha = info.get("sha")
    if not isinstance(sha, str) or not sha:
        raise BenchmarkUnavailable("dataset info carries no revision SHA")
    return sha


def download_parquet(revision: str, filename: str, destination: Path) -> str:
    """Download one parquet file at a pinned revision and return its SHA-256.

    Input: the repository revision, the path of the file inside the dataset repo,
    and where to write it. Output: the hex SHA-256 of the bytes written. The file
    is written whole before the hash is returned, so a truncated download hashes
    differently rather than being mistaken for the real thing.
    """
    import hashlib

    url = f"{HF_RESOLVE_BASE}/{DATASET}/resolve/{revision}/{filename}"
    payload = _open(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read TRACE rows from a local parquet file.

    Input: the parquet path. Output: one dict per row. Raises
    :class:`BenchmarkUnavailable` when pyarrow is not installed, which is the
    normal state of the default dev environment; install the ``benchmark``
    extra to enable the revision-pinned path.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BenchmarkUnavailable(
            "pyarrow is required to read the revision-pinned parquet; "
            "install it with `uv sync --extra benchmark`"
        ) from exc
    table = pq.read_table(path)
    columns = table.to_pydict()
    length = table.num_rows
    return [{name: columns[name][i] for name in table.column_names} for i in range(length)]


def fetch_rows_via_server(
    limit: int | None = None,
    *,
    dataset: str = DATASET,
    config: str = "default",
    split: str = "train",
) -> list[dict[str, Any]]:
    """Page the datasets-server rows endpoint.

    Input: an optional row cap (None fetches every row), and the dataset
    coordinates. Output: raw row dicts in server order. This path is not
    revision pinned; verify the result against a manifest before reporting any
    number from it.
    """
    encoded = urllib.parse.quote(dataset, safe="")
    rows: list[dict[str, Any]] = []
    offset = 0
    while limit is None or len(rows) < limit:
        want = SERVER_PAGE if limit is None else min(SERVER_PAGE, limit - len(rows))
        url = (
            f"{SERVER_BASE}/rows?dataset={encoded}"
            f"&config={urllib.parse.quote(config)}"
            f"&split={urllib.parse.quote(split)}"
            f"&offset={offset}&length={want}"
        )
        payload = _get_json(url)
        batch = payload.get("rows") if isinstance(payload, dict) else None
        if not batch:
            break
        rows.extend(entry.get("row", {}) for entry in batch)
        offset += len(batch)
    if not rows:
        raise BenchmarkUnavailable(f"datasets-server returned no rows for {dataset}")
    return rows


def load_rows(
    *,
    revision: str,
    parquet_file: str = "data/train-00000-of-00001.parquet",
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Load TRACE rows by the best path available.

    Input: the pinned revision, the parquet path inside the dataset repo, an
    optional cache directory for the downloaded parquet, and an optional row cap
    (applied only on the fallback path). Output: ``(rows, source, file_sha256)``,
    where ``source`` is ``"parquet"`` or ``"datasets-server"`` and the digest is
    empty on the fallback path.

    The parquet path is tried first because it is the only one that honours the
    revision pin. It falls back to the rows API when pyarrow is missing or the
    download fails, and raises :class:`BenchmarkUnavailable` when neither works.
    """
    cache = cache_dir or Path.home() / ".cache" / "rolloutscope" / "trace"
    local = cache / revision / Path(parquet_file).name
    try:
        if local.is_file():
            import hashlib

            digest = hashlib.sha256(local.read_bytes()).hexdigest()
        else:
            digest = download_parquet(revision, parquet_file, local)
        return read_parquet_rows(local), "parquet", digest
    except BenchmarkUnavailable:
        return fetch_rows_via_server(limit), "datasets-server", ""
