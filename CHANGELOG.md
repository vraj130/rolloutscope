# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: uv-managed environment, src layout, CI matrix (3.11 to 3.13),
  placeholder CLI with --version.
- Normalized rollout schema (frozen contract): discriminated union on kind, per-row
  schema_version, extra-key preservation, Verdict and Finding models, content-derived
  IDs with the v1 join contract, streaming JSONL IO with skip-and-log, and a tested
  migration chain.
- ADR-0001 recording the normalized-schema decision and its alternatives.
