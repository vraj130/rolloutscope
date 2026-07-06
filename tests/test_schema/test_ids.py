"""ID scheme: content-derived, deterministic, sensitive to content changes."""

from rolloutscope.schema import group_id, rollout_id, run_id_from_manifest, run_id_from_name

PROMPT = [{"role": "user", "content": "What is 17 + 25?"}]
COMPLETION = [{"role": "assistant", "content": "Final answer: 42"}]


def test_rollout_id_deterministic() -> None:
    a = rollout_id(0, PROMPT, COMPLETION, 1.0)
    b = rollout_id(0, PROMPT, COMPLETION, 1.0)
    assert a == b
    assert a.startswith("r")
    assert len(a) == 17


def test_rollout_id_ignores_reward_int_float_encoding() -> None:
    assert rollout_id(0, PROMPT, COMPLETION, 1) == rollout_id(0, PROMPT, COMPLETION, 1.0)


def test_rollout_id_ignores_key_order() -> None:
    reordered = [{"content": "What is 17 + 25?", "role": "user"}]
    assert rollout_id(0, PROMPT, COMPLETION, 1.0) == rollout_id(0, reordered, COMPLETION, 1.0)


def test_rollout_id_changes_with_content() -> None:
    base = rollout_id(0, PROMPT, COMPLETION, 1.0)
    assert rollout_id(1, PROMPT, COMPLETION, 1.0) != base
    assert rollout_id(0, PROMPT, COMPLETION, 0.5) != base
    other = [{"role": "assistant", "content": "Final answer: 41"}]
    assert rollout_id(0, PROMPT, other, 1.0) != base


def test_group_id_derives_from_example_id() -> None:
    assert group_id(0) == "grp-0"
    assert group_id(17) == "grp-17"


def test_run_ids_deterministic_and_distinct() -> None:
    manifest = {"model": "synthetic/test-model", "env_id": "synthetic-arith"}
    a = run_id_from_manifest(manifest)
    assert a == run_id_from_manifest(dict(manifest))
    assert a.startswith("run-")
    assert run_id_from_manifest({"model": "other"}) != a
    b = run_id_from_name("run-directory-name")
    assert b == run_id_from_name("run-directory-name")
    assert b != run_id_from_name("another-name")
