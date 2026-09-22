"""Load repository scripts by path so tests can exercise them without installing them."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(relative: str) -> ModuleType:
    """Import a script file relative to the repository root.

    Input: a path such as ``scripts/check_packaging.py``. Output: the loaded
    module. Raises FileNotFoundError when the script is missing.
    """
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"rolloutscope_script_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses with from __future__ import annotations need the module in
    # sys.modules while the class body is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
