"""Placeholder smoke test: the installed package imports and reports a version."""

import rolloutscope


def test_package_imports() -> None:
    assert rolloutscope.__version__ == "0.1.0"
