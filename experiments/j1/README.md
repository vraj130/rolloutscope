# J1: proxy-judge audit versus overclaim growth

Scripts for PLAN.md J1. They read the six M2 runs under `$ROLLOUTSCOPE_DATA/m1/`
(`m2-{a,b,c}-qwen1.5b-s{1,2}`) and make no training runs and no gold calls. The local
Llama-3.1-8B-Instruct server (`bash m1/serve_proxy.sh <name>`, GPU 1) serves the tagger, the
editor and the proxy judge.

| File | Role |
|---|---|
| `j1data.py` | eval criteria and the logged proxy and gold verdicts |
| `tag.py` | criterion type (structural, presence, numeric, negation, factual, compound): regex, then one LLM pass |
| `audit.py` | synthetic failing edits of gold-yes answers, graded by the proxy |
| `analyze_j1.py` | audit false-positive rates, overclaim growth, prediction tests, gold-yes by type |

Run from `experiments/`, in order: `tag.py`, `audit.py`, `analyze_j1.py` (each `uv run python
j1/<file>`). Outputs go to `$ROLLOUTSCOPE_DATA/j1/`: `tags.jsonl`, `tag_spotcheck.tsv` (100
random tags, with an empty column for a manual type), `audit_edits.jsonl`, `summary.json`,
`summary.md`.
