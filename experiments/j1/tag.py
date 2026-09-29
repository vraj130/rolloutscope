"""J1 step a: tag every eval criterion by type (PLAN.md J1).

Types: structural (format, organization, tone, style), presence (mentions or includes a topic),
numeric (specific numbers, doses, ranges, durations), negation (the answer must NOT do or say
something), factual (a specific correct claim or explanation), compound (two or more distinct
requirements). One type per criterion.

High-precision regex rules tag what they can; one pass of the local Llama-3.1-8B-Instruct
server (serve_proxy.sh, temperature 0, schema-constrained to the six types) tags the rest.
Writes $ROLLOUTSCOPE_DATA/j1/tags.jsonl and tag_spotcheck.tsv (100 random tags, seed 0).

    uv run python j1/tag.py
"""

# ruff: noqa: E501  (the prompt strings are kept verbatim, as run)

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from j1data import TYPES, criteria_table, out_dir

UNITS = r"(mg|mcg|µg|g|kg|ml|mL|l|L|%|mmHg|mmol|mEq|IU|units?|days?|weeks?|months?|years?|hours?|minutes?|ppm|cm|mm|bpm)"
RULES = [
    (
        "negation",
        re.compile(
            r"^\s*(the\s+)?(response|answer)?\s*(does not|doesn't|do not|should not|must not|"
            r"avoids|never|refrains)\b",
            re.I,
        ),
    ),
    (
        "structural",
        re.compile(
            r"\b(headings?|bullet(ed)? (points?|lists?)|numbered list|tone|concise|readab\w*|"
            r"well[- ]organi[sz]ed|clearly organi[sz]ed|formatted)\b",
            re.I,
        ),
    ),
    ("numeric", re.compile(r"\d+(\.\d+)?\s*" + UNITS + r"\b")),
    ("compound", re.compile(r"\(1\).*\(2\)")),
]

PROMPT = """Classify one rubric criterion used to grade a medical answer. Pick exactly one type:
- structural: about how the answer is written (organization, headings, lists, tone, clarity, length, style)
- presence: requires the answer to mention, include, or address a topic or item, without a specific claim to verify
- numeric: requires specific numbers, doses, ranges, thresholds, or durations
- negation: requires the answer NOT to do or say something (avoid, do not)
- factual: requires a specific correct medical claim or explanation
- compound: requires two or more distinct things at once, each of which could be met separately
Criterion: {text}
Return JSON {{"type": "<one type>"}}."""


def regex_tag(text: str) -> str | None:
    """First matching rule, or None."""
    for name, rx in RULES:
        if rx.search(text):
            return name
    return None


async def llm_tags(texts: list[str]) -> list[str | None]:
    """One schema-constrained call per criterion to the local server."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=os.environ.get("PROXY_URL", "http://127.0.0.1:8001/v1"),
        api_key="EMPTY",
        timeout=300,
    )
    schema = {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": TYPES}},
        "required": ["type"],
        "additionalProperties": False,
    }
    sem = asyncio.Semaphore(24)

    async def one(t: str) -> str | None:
        async with sem:
            for _ in range(3):
                try:
                    out = await client.chat.completions.create(
                        model="proxy-judge",
                        messages=[{"role": "user", "content": PROMPT.format(text=t)}],
                        temperature=0.0,
                        max_tokens=20,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "tag", "schema": schema, "strict": True},
                        },
                    )
                    return json.loads(out.choices[0].message.content or "{}")["type"]
                except Exception:
                    await asyncio.sleep(2)
            return None

    return list(await asyncio.gather(*(one(t) for t in texts)))


def main() -> None:
    rows = criteria_table()
    for r in rows:
        r["type"] = regex_tag(r["text"])
        r["tag_source"] = "regex" if r["type"] else None
    todo = [r for r in rows if r["type"] is None]
    for r, t in zip(todo, asyncio.run(llm_tags([r["text"] for r in todo])), strict=True):
        r["type"], r["tag_source"] = t, "llm" if t else "failed"
    out = out_dir()
    with (out / "tags.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sample = random.Random(0).sample(rows, 100)
    with (out / "tag_spotcheck.tsv").open("w") as f:
        f.write("id\ttype\ttag_source\tyour_type\tcriterion\n")
        for r in sample:
            f.write(f"{r['id']}\t{r['type']}\t{r['tag_source']}\t\t{r['text']}\n")
    counts = {t: sum(r["type"] == t for r in rows) for t in TYPES}
    by_source = {s: sum(r["tag_source"] == s for r in rows) for s in ("regex", "llm", "failed")}
    print(json.dumps({"n": len(rows), "types": counts, "sources": by_source}, indent=2))


if __name__ == "__main__":
    main()
