"""Schema version migration chain.

Policy (rollout-schema-design skill, versioning-and-streaming.md): within a major
version changes are additive only; a breaking change bumps the major and ships a
migration function keyed by the major it leaves. Readers upgrade rows on load;
writers only ever write the current version.

The registry holds one function per major boundary plus an identity entry for the
current major, so every row passes through the same mechanism whether it needs
upgrading or not. The 0.x entry is a worked example proving the chain; no real
0.x data exists.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rolloutscope.schema.models import SCHEMA_VERSION

CURRENT_MAJOR = 1

Migration = Callable[[dict[str, Any]], dict[str, Any]]

MIGRATIONS: dict[int, Migration] = {}


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a row's schema_version cannot be migrated to the current one."""


def register_migration(from_major: int) -> Callable[[Migration], Migration]:
    """Register a migration function for rows leaving ``from_major``.

    The function for major N takes a dict-shaped row at version N.x and returns a
    row at version N+1 (the entry for CURRENT_MAJOR is identity and returns the
    row at the current version unchanged).
    """

    def decorator(func: Migration) -> Migration:
        MIGRATIONS[from_major] = func
        return func

    return decorator


def _major_of(version: str) -> int:
    try:
        return int(str(version).split(".")[0])
    except ValueError as exc:
        raise UnsupportedSchemaVersionError(f"unparseable schema_version {version!r}") from exc


@register_migration(0)
def migrate_v0_to_v1(row: dict[str, Any]) -> dict[str, Any]:
    """Worked example: migrate the fake 0.x schema forward to 1.0.

    The fake 0.x schema named the grouping key ``episode_id`` and the scalar
    reward ``score``; 1.0 uses the verifiers-aligned ``example_id`` and
    ``reward``. Unknown keys pass through untouched.
    """
    migrated = dict(row)
    if "example_id" not in migrated and "episode_id" in migrated:
        migrated["example_id"] = migrated.pop("episode_id")
    if "reward" not in migrated and "score" in migrated:
        migrated["reward"] = migrated.pop("score")
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


@register_migration(CURRENT_MAJOR)
def migrate_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Identity entry for the current major: rows already at 1.x pass unchanged."""
    return row


def migrate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Migrate a raw row dict forward to the current schema version.

    Input: a JSON-decoded row dict. Rows without a ``schema_version`` (raw
    verifiers output that predates normalization) are treated as current. Output:
    a row at CURRENT_MAJOR, having passed through every migration between its
    major and now, ending with the identity entry. Raises
    UnsupportedSchemaVersionError for majors newer than this reader or with no
    registered migration path.
    """
    major = _major_of(str(row.get("schema_version", SCHEMA_VERSION)))
    if major > CURRENT_MAJOR:
        raise UnsupportedSchemaVersionError(
            f"row schema major {major} is newer than supported major {CURRENT_MAJOR}"
        )
    while major < CURRENT_MAJOR:
        migration = MIGRATIONS.get(major)
        if migration is None:
            raise UnsupportedSchemaVersionError(f"no migration registered from major {major}")
        row = migration(row)
        major += 1
    return MIGRATIONS[CURRENT_MAJOR](row)
