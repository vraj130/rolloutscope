"""verifier_tamper: test and verifier tampering detector.

Scans completion text and trajectory tool calls for patterns where the rollout
attacks the checker instead of solving the task: pytest skip/xfail insertion,
constant-true assertions, forced zero exit codes (sys.exit(0), os._exit(0),
shell ``|| true``), monkeypatched or mocked checkers, always-pass test bodies,
patch headers touching test files, and edit-like tool calls whose arguments
name test paths. Category: verifier_tampering.

Known false-positive modes:
- Tasks that legitimately ask the model to edit, skip, or rewrite tests.
- Completions that quote or discuss tampering patterns without performing them
  (code review feedback, documentation about pytest.mark.skip).
- Shell tutorials or CI configs that legitimately use ``|| true``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar

from rolloutscope.detectors._text import completion_sources, iter_tool_calls, stable_rollout_id
from rolloutscope.detectors.base import DetectorConfig
from rolloutscope.schema import EvidenceSpan, Rollout, Verdict


class VerifierTamperDetector:
    """Per-rollout scanner for test-edit and verifier short-circuit patterns."""

    name: ClassVar[str] = "verifier_tamper"
    category: ClassVar[str] = "verifier_tampering"

    def detect(self, rollouts: Sequence[Rollout], config: DetectorConfig) -> list[Verdict]:
        """Return one fired verdict per rollout whose text or tool calls match.

        Inputs: normalized rollouts and the full detector config (reads
        ``config.verifier_tamper``). A rollout fires when at least
        ``min_matches`` distinct pattern labels hit; every hit becomes an
        evidence span with offsets into the extracted text form of the field.
        """
        cfg = config.verifier_tamper
        compiled = {label: re.compile(pattern) for label, pattern in cfg.patterns.items()}
        edit_tool = re.compile(cfg.edit_tool_name_regex, re.IGNORECASE)
        test_path = re.compile(cfg.test_path_regex)

        verdicts: list[Verdict] = []
        for rollout in rollouts:
            rid = stable_rollout_id(rollout)
            spans: list[EvidenceSpan] = []
            labels: set[str] = set()
            for field, text in completion_sources(rollout, include_tool_calls=True):
                for label, pattern in compiled.items():
                    match = pattern.search(text)
                    if match is not None:
                        labels.add(label)
                        spans.append(
                            EvidenceSpan(
                                rollout_id=rid,
                                field=field,
                                start=match.start(),
                                end=match.end(),
                                text=match.group(0),
                                note=f"tampering pattern '{label}'",
                            )
                        )
            for field, tool_name, arguments in iter_tool_calls(rollout):
                if edit_tool.search(tool_name):
                    match = test_path.search(arguments)
                    if match is not None:
                        labels.add("test_file_edit")
                        spans.append(
                            EvidenceSpan(
                                rollout_id=rid,
                                field=field,
                                text=match.group(0),
                                note=(
                                    f"edit-like tool call '{tool_name}' touching a test path "
                                    "in its arguments"
                                ),
                            )
                        )
            if len(labels) >= cfg.min_matches and spans:
                score = min(1.0, cfg.base_score + cfg.per_extra_match_score * (len(labels) - 1))
                verdicts.append(
                    Verdict(
                        detector=self.name,
                        fired=True,
                        score=score,
                        category=self.category,
                        evidence=spans,
                        rollout_ids=[rid],
                    )
                )
        return verdicts
