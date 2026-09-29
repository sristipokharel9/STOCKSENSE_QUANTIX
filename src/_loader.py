"""
Robust module loader so any src/*.py file can be imported by full path.
Usage:
    from _loader import load_sibling
    cfg = load_sibling("config")
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_sibling(module_name: str):
    """Load a sibling module in the same src/ folder by name."""
    path = HERE / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod