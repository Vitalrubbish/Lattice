"""KCMM Python integration layer.

Phase I.C is observer-only: create a KCMM pool beside vLLM, run a small
allocation probe, inspect future patch seams, and leave vLLM behavior unchanged.
"""

from .bindings import KcmmError, KcmmLibrary, KcmmPool, probe_once
from .config import ObserverConfig

__all__ = [
    "KcmmError",
    "KcmmLibrary",
    "KcmmPool",
    "ObserverConfig",
    "probe_once",
]
