"""Generate representative synthetic rollout datasets for performance measurement.

Writes a prime-rl step layout (``step_<n>/train_rollouts.jsonl``) that the
current adapter can load. Sizes are documented below. Generated data is not
committed; this script is the artifact.

Usage:

    uv run python scripts/perf/generate.py --size small --out /tmp/rs-small
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SizeSpec:
    """Named generator size: step count times groups times rollouts per group."""

    name: str
    steps: int
    groups: int
    rollouts_per_group: int
    multi_turn_fraction: float
    long_completion_fraction: float
    turns: int
    seed: int

    def record_count(self) -> int:
        """Return the exact number of JSONL rows this spec emits."""
        return self.steps * self.groups * self.rollouts_per_group


@dataclass(frozen=True)
class DatasetStats:
    """Summaries recorded alongside a generated dataset."""

    record_count: int
    step_count: int
    group_cardinality: int
    input_bytes: int
    kind_mix: dict[str, int]
    mean_trajectory_turns: float


SIZES: dict[str, SizeSpec] = {
    "small": SizeSpec(
        name="small",
        steps=2,
        groups=20,
        rollouts_per_group=4,
        multi_turn_fraction=0.25,
        long_completion_fraction=0.2,
        turns=3,
        seed=1,
    ),
    "medium": SizeSpec(
        name="medium",
        steps=5,
        groups=200,
        rollouts_per_group=5,
        multi_turn_fraction=0.25,
        long_completion_fraction=0.2,
        turns=4,
        seed=1,
    ),
    "large": SizeSpec(
        name="large",
        steps=20,
        groups=500,
        rollouts_per_group=5,
        multi_turn_fraction=0.25,
        long_completion_fraction=0.2,
        turns=4,
        seed=1,
    ),
}

_LONG_SENTENCE = (
    "The assistant restates the same intermediate calculation with extra padding "
    "so the completion is long enough to exercise length-sensitive detectors. "
)


def _completion_text(long: bool, group: int, step: int, replica: int) -> str:
    base = f"Result for task {group} step {step} replica {replica} is {group + step}."
    if long:
        return base + " " + (_LONG_SENTENCE * 20)
    return base


def _row(
    rng: random.Random,
    spec: SizeSpec,
    step: int,
    group: int,
    replica: int,
) -> dict[str, Any]:
    prompt_text = f"Solve task {group} at step {step}, replica {replica}."
    prompt = [{"role": "user", "content": prompt_text}]
    long = rng.random() < spec.long_completion_fraction
    multi = rng.random() < spec.multi_turn_fraction
    if group % 5 == 0:
        reward = 1.0
        correct = 1.0
    else:
        reward = round(rng.random(), 4)
        correct = 1.0 if reward >= 0.5 else 0.0
    format_reward = 1.0 if replica % 2 == 0 else 0.0
    completion_text = _completion_text(long, group, step, replica)
    completion = [{"role": "assistant", "content": completion_text}]
    row: dict[str, Any] = {
        "answer": f"answer-{group}",
        "completion": completion,
        "example_id": group,
        "info": {"step": step, "task": "synthetic-perf"},
        "is_completed": True,
        "is_truncated": False,
        "metrics": {"correct_answer": correct, "format_reward": format_reward},
        "prompt": prompt,
        "reward": reward,
    }
    if multi:
        trajectory_id = f"t-{group}-{replica}-{step}"
        row["trajectory"] = [
            {
                "completion": [{"role": "assistant", "content": f"{completion_text} turn {turn}"}],
                "is_truncated": False,
                "prompt": prompt,
                "trajectory_id": trajectory_id,
            }
            for turn in range(spec.turns)
        ]
    return row


def generate_dataset(out_dir: Path, spec: SizeSpec) -> DatasetStats:
    """Write a prime-rl-shaped run under ``out_dir`` and return input summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(spec.seed)
    kind_mix = {"single_turn": 0, "multi_turn": 0}
    turns_total = 0
    input_bytes = 0
    for step in range(spec.steps):
        step_dir = out_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        path = step_dir / "train_rollouts.jsonl"
        lines: list[str] = []
        for group in range(spec.groups):
            for replica in range(spec.rollouts_per_group):
                row = _row(rng, spec, step, group, replica)
                if row.get("trajectory"):
                    kind_mix["multi_turn"] += 1
                    turns_total += len(row["trajectory"])
                else:
                    kind_mix["single_turn"] += 1
                lines.append(json.dumps(row, separators=(",", ":"), sort_keys=True))
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        path.write_bytes(payload)
        input_bytes += len(payload)
    record_count = spec.record_count()
    mean_turns = turns_total / record_count if record_count else 0.0
    (out_dir / "generator.json").write_text(
        json.dumps({"seed": spec.seed, "size": spec.name}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DatasetStats(
        record_count=record_count,
        step_count=spec.steps,
        group_cardinality=spec.groups,
        input_bytes=input_bytes,
        kind_mix=kind_mix,
        mean_trajectory_turns=mean_turns,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: write a named-size synthetic run to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(SIZES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override the size's default seed.")
    args = parser.parse_args(argv)
    spec = SIZES[args.size]
    if args.seed is not None:
        spec = SizeSpec(
            name=spec.name,
            steps=spec.steps,
            groups=spec.groups,
            rollouts_per_group=spec.rollouts_per_group,
            multi_turn_fraction=spec.multi_turn_fraction,
            long_completion_fraction=spec.long_completion_fraction,
            turns=spec.turns,
            seed=args.seed,
        )
    stats = generate_dataset(args.out, spec)
    print(
        json.dumps(
            {
                "size": spec.name,
                "seed": spec.seed,
                "record_count": stats.record_count,
                "step_count": stats.step_count,
                "group_cardinality": stats.group_cardinality,
                "input_bytes": stats.input_bytes,
                "kind_mix": stats.kind_mix,
                "mean_trajectory_turns": stats.mean_trajectory_turns,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
