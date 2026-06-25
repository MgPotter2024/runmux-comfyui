"""Structural smoke test for the RunMux ComfyUI node pack.

This does NOT hit the network. It loads the pack's ``__init__.py`` exactly the
way ComfyUI would (by importing the package), then asserts the node mappings and
each node's ComfyUI contract (INPUT_TYPES / RETURN_TYPES / FUNCTION / CATEGORY)
are well formed.

Run with the SDK installed:
    pip install -e ../../sdk/python
    pytest tests/test_nodes.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys


PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pack():
    """Import the pack's __init__.py as ComfyUI would (folder on sys.path)."""
    parent = os.path.dirname(PACK_DIR)
    pkg_name = os.path.basename(PACK_DIR)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    init_path = os.path.join(PACK_DIR, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_path, submodule_search_locations=[PACK_DIR]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)
    return module


def test_node_mappings_are_valid():
    pack = _load_pack()

    mappings = pack.NODE_CLASS_MAPPINGS
    display = pack.NODE_DISPLAY_NAME_MAPPINGS

    assert isinstance(mappings, dict) and mappings, "NODE_CLASS_MAPPINGS must be a non-empty dict"
    assert isinstance(display, dict) and display, "NODE_DISPLAY_NAME_MAPPINGS must be a non-empty dict"
    assert set(display) == set(mappings), "display names must cover exactly the registered nodes"

    for name, cls in mappings.items():
        # Required ComfyUI contract.
        assert hasattr(cls, "INPUT_TYPES"), f"{name}: missing INPUT_TYPES"
        assert hasattr(cls, "RETURN_TYPES"), f"{name}: missing RETURN_TYPES"
        assert hasattr(cls, "FUNCTION"), f"{name}: missing FUNCTION"
        assert hasattr(cls, "CATEGORY"), f"{name}: missing CATEGORY"

        it = cls.INPUT_TYPES()
        assert isinstance(it, dict), f"{name}: INPUT_TYPES() must return a dict"
        assert "required" in it, f"{name}: INPUT_TYPES needs a 'required' section"

        assert isinstance(cls.RETURN_TYPES, tuple) and cls.RETURN_TYPES, f"{name}: RETURN_TYPES must be a non-empty tuple"

        fn = cls.FUNCTION
        assert isinstance(fn, str) and hasattr(cls, fn), f"{name}: FUNCTION '{fn}' must name a method on the class"

        assert cls.CATEGORY == "RunMux", f"{name}: CATEGORY should be 'RunMux'"


if __name__ == "__main__":
    test_node_mappings_are_valid()
    print("OK: RunMux ComfyUI node pack structure is valid.")
