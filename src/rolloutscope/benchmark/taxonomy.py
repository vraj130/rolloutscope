"""The TRACE label taxonomy, as it appears in the published artifact.

The dataset ships one ``label`` string per row. ``"0"`` is benign; a hacked row
carries one or more dotted codes separated by ", " (39 percent of hacked rows are
multi-label). Every code observed in revision ``31d87f06`` has the form
``1.<family>.<subcategory>``.

Provenance and its limits, stated plainly because the numbers here are load
bearing and the sources disagree with each other:

- The code counts below were measured from the parquet at revision ``31d87f06``,
  not copied from the paper. They are facts about the artifact.
- The four family names come from the dataset card's "Major Categories" list,
  which names exactly four categories and, in parentheses, exactly ten
  subcategories. The ordinal codes in the data line up one for one with that
  list (three, three, two, two), which is why the family mapping is treated as
  confirmed.
- The leaf names are inferred from position within each parenthesized list. The
  card never states the numbering. They are marked ``inferred`` and must not be
  reported as the dataset's own names without confirming against the paper.
- The card's prose also claims "54 fine-grained reward hack subcategories across
  10 broader categories", which contradicts its own category list and the
  artifact. The artifact is the authority for this module.

TODO(confirm): resolve the leaf names and the 10-versus-4 category discrepancy
against arXiv:2601.20103 (Deshpande, Kannappan, Qian, 2026) and, if the paper
disagrees with the artifact, record both here rather than replacing one.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

BENIGN_LABEL = "0"
"""The label string the dataset uses for a benign trajectory."""

LABEL_SEPARATOR = ", "
"""Separator between codes in a multi-label row."""

Confidence = Literal["confirmed", "inferred"]


class Category(NamedTuple):
    """One taxonomy node: its code, its name, and how much we trust the name."""

    code: str
    name: str
    confidence: Confidence
    observed_rows: int
    """Rows carrying this code at revision 31d87f06, measured from the parquet."""


FAMILIES: tuple[Category, ...] = (
    Category("1.1", "Test Suite Exploitation", "confirmed", 176),
    Category("1.2", "Solution Quality Degradation", "confirmed", 91),
    Category("1.3", "Context Exploitation", "confirmed", 46),
    Category("1.4", "Execution Environment Hacks", "confirmed", 45),
)
"""The four hack families named by the dataset card, with measured row counts."""

SUBCATEGORIES: tuple[Category, ...] = (
    Category("1.1.1", "Test Modification", "inferred", 56),
    Category("1.1.2", "Test Case Targeting", "inferred", 69),
    Category("1.1.3", "Coverage Gaming", "inferred", 51),
    Category("1.2.1", "Degenerate Implementations", "inferred", 36),
    Category("1.2.2", "Complexity Gaming", "inferred", 21),
    Category("1.2.3", "Style Manipulation", "inferred", 34),
    Category("1.3.1", "Information Leakage", "inferred", 25),
    Category("1.3.2", "Tool Abuse", "inferred", 21),
    Category("1.4.1", "Runtime Manipulation", "inferred", 21),
    Category("1.4.2", "Timing/Resource Exploitation", "inferred", 24),
)
"""The ten leaf codes observed in the artifact. Names are positional inferences."""

BY_CODE: dict[str, Category] = {c.code: c for c in (*FAMILIES, *SUBCATEGORIES)}

OBSERVED_BENIGN_ROWS = 249
OBSERVED_HACKED_ROWS = 268
OBSERVED_TOTAL_ROWS = 517
"""Measured class balance at revision 31d87f06."""


def parse_label(label: str) -> tuple[str, ...]:
    """Split a raw ``label`` cell into its codes.

    Input: the raw label string, for example ``"1.1.2, 1.3.1"`` or ``"0"``.
    Output: a tuple of trimmed code strings, empty for a blank or unusable cell.
    A benign row yields ``("0",)``, which callers test with :func:`is_hacked`.
    """
    return tuple(part.strip() for part in label.split(",") if part.strip())


def is_hacked(label: str) -> bool | None:
    """Classify a raw label cell as hacked, benign, or unusable.

    Input: the raw label string. Output: ``True`` when any code is not the benign
    sentinel, ``False`` when the row is labelled benign, and ``None`` when the
    cell is blank or carries no code at all, so a caller can count it as
    unusable rather than silently scoring it as a negative.
    """
    codes = parse_label(label)
    if not codes:
        return None
    return any(code != BENIGN_LABEL for code in codes)


def families_of(label: str) -> tuple[str, ...]:
    """Return the distinct family codes (``1.1`` style) present in a label cell.

    Input: the raw label string. Output: family codes in first-seen order,
    empty for a benign or unusable row. Unknown codes are kept as their own
    truncation rather than dropped, so a new upstream code stays visible in a
    per-category report instead of vanishing.
    """
    seen: list[str] = []
    for code in parse_label(label):
        if code == BENIGN_LABEL:
            continue
        family = ".".join(code.split(".")[:2])
        if family not in seen:
            seen.append(family)
    return tuple(seen)


def name_of(code: str) -> str:
    """Human-readable name for a taxonomy code, or the code itself if unknown.

    Input: a family or subcategory code. Output: the mapped name, suffixed with
    ``" (inferred)"`` when the name is a positional inference rather than a
    stated one, so a report never presents a guess as the dataset's own label.
    """
    category = BY_CODE.get(code)
    if category is None:
        return code
    if category.confidence == "inferred":
        return f"{category.name} (inferred)"
    return category.name
