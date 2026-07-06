# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Opt-in reproducible validation script (scripts/trace_validation.py) that maps
  Patronus TRACE rows into the schema and reports verifier_tamper and
  answer_leakage_echo fire rates on hacked versus clean coding trajectories.
- HF_TOKEN authentication for the TRACE integration test, loaded from a gitignored
  .env through python-dotenv, so the optional network validation can reach the gated
  dataset.

### Fixed

- degenerate_repetition no longer fires on concatenated multi-turn transcripts (a
  multi_turn rollout, or a completion holding more than one message); repetition is
  measured within a single completion only.

## [0.1.0] - 2026-07-05

### Added

- Project scaffold: uv-managed environment, src layout, CI matrix (3.11 to 3.13),
  placeholder CLI with --version.
- Normalized rollout schema (frozen contract): discriminated union on kind, per-row
  schema_version, extra-key preservation, Verdict and Finding models, content-derived
  IDs with the v1 join contract, streaming JSONL IO with skip-and-log, and a tested
  migration chain.
- ADR-0001 recording the normalized-schema decision and its alternatives.
- Adapters for verifiers eval output (results.jsonl plus metadata.json) and prime-rl
  training rollouts (train_rollouts.jsonl across step directories), with
  auto-detection, content-derived IDs, unknown-key preservation, and step_index
  attached from on-disk layout only.
- Six reward-hacking detectors (verifier_tamper, reward_saturation_group_collapse,
  length_inflation, format_only_wins, degenerate_repetition, answer_leakage_echo)
  with evidence spans, documented false-positive modes, configurable heuristic
  thresholds, entry-point discovery, and labeled hacked/clean fixture pairs.
- Streaming aggregates, verdict-to-finding assembly, and three renderers: rich
  terminal summary, deterministic JSON, and a single-file self-contained HTML report
  with server-side SVG charts.
- Command-line interface (typer): analyze (terminal summary, JSON sidecar,
  self-contained HTML report, --fail-on exit codes), convert (raw artifacts to
  normalized JSONL), detectors list, and schema export, plus TOML config loading over
  the detector, aggregation, and severity settings.
- Synthetic demo run fixture (tests/fixtures/demo) and an end-to-end test suite
  covering the full adapter-to-report pipeline.
- README quickstart, CONTRIBUTING guide with a third-party detector plugin
  walkthrough, and this changelog.
