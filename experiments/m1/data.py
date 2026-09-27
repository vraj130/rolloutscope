"""RubricHub medical prompts with weighted criteria, split by prompt.

Source: sojuL/RubricHub_v1 (Apache-2.0), file RuRL/rurbichub_v1_Medical.parquet.
Fields used, checked against the raw rows (29,681 rows, all single-turn user prompts):

- ``prompt``: list of {role, content} messages.
- ``Rubrics``: list of {criterion, points}. Identical to ``reward_model.rubrics`` and
  ``extra_info.reward_model.rubrics`` on every row. All points are positive (no pitfall
  criteria), all criteria carry verifier tag "llm", and ``ground_truth`` is always empty.

The split is a seeded shuffle of the whole medical file. Train and eval never share a
prompt, and duplicate prompt texts are collapsed before sampling.

Usage (prints one raw row, then split sizes):

    uv run python m1/data.py --config m1/config.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any

REPO_ID = "sojuL/RubricHub_v1"
MEDICAL_FILE = "RuRL/rurbichub_v1_Medical.parquet"


@dataclass(frozen=True)
class Criterion:
    """One rubric criterion and its weight (RubricHub ``points``)."""

    text: str
    weight: float


@dataclass(frozen=True)
class Example:
    """A prompt with its rubric. ``source_index`` is the row index in the parquet file."""

    source_index: int
    prompt: list[dict[str, str]]
    criteria: list[Criterion]


def _read_rows() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, MEDICAL_FILE, repo_type="dataset")
    return pq.read_table(path, columns=["prompt", "Rubrics", "data_source"]).to_pylist()


def load_medical() -> list[Example]:
    """Load every medical row as an Example, keeping the first copy of duplicate prompts."""
    seen: set[str] = set()
    out: list[Example] = []
    for i, row in enumerate(_read_rows()):
        if row["data_source"] != "Medical":
            continue
        key = json.dumps(row["prompt"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        criteria = [Criterion(c["criterion"], float(c["points"])) for c in row["Rubrics"]]
        out.append(Example(source_index=i, prompt=row["prompt"], criteria=criteria))
    return out


def split(n_train: int, n_eval: int, seed: int) -> tuple[list[Example], list[Example]]:
    """Return disjoint (train, eval) prompt sets from a seeded shuffle."""
    examples = load_medical()
    if n_train + n_eval > len(examples):
        raise ValueError(f"asked for {n_train + n_eval} prompts, only {len(examples)} exist")
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    train = [examples[i] for i in order[:n_train]]
    eval_ = [examples[i] for i in order[n_train : n_train + n_eval]]
    return train, eval_


def criteria_to_json(criteria: list[Criterion]) -> list[dict[str, Any]]:
    """Serialize criteria for datasets columns and JSONL output."""
    return [{"criterion": c.text, "weight": c.weight} for c in criteria]


def criteria_from_json(items: list[dict[str, Any]]) -> list[Criterion]:
    """Inverse of ``criteria_to_json``."""
    return [Criterion(d["criterion"], float(d["weight"])) for d in items]


def rubric_prompt(prompt: list[dict[str, str]], criteria: list[Criterion]) -> list[dict[str, str]]:
    """The RGSD teacher prompt (arXiv 2606.12507, Figure 10): question, rubric, instruction.

    The rubric goes into the single user turn after the question, numbered, without weights.
    """
    (turn,) = prompt  # RubricHub medical prompts are one user message
    lines = "\n".join(f"{i}. {c.text}" for i, c in enumerate(criteria, 1))
    content = (
        f"{turn['content']}\n\n"
        "Hidden evaluation criteria that a good response should satisfy:\n"
        f"{lines}\n\n"
        "After understanding these evaluation criteria, provide your own thorough response to "
        "the problem. Address the criteria naturally without explicitly referencing them."
    )
    return [{"role": "user", "content": content}]


def main() -> None:
    """Print one raw row and the split sizes for a config."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    with open(parser.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raw = _read_rows()[0]
    print(json.dumps(raw, indent=1, ensure_ascii=False)[:3000])
    train, eval_ = split(cfg["n_train"], cfg["n_eval"], cfg["seed"])
    n_crit = [len(e.criteria) for e in train + eval_]
    print(
        f"train={len(train)} eval={len(eval_)} criteria/prompt min={min(n_crit)} max={max(n_crit)}"
    )


if __name__ == "__main__":
    main()
