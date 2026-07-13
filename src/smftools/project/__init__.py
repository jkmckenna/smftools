"""Project-level layer: register experiments by pointer and query/analyze across them.

A project directory accumulates experiments over time without merging or copying data.
Experiments are registered as pointers (:mod:`registry`); references are harmonized
across experiments by sequence identity (:mod:`reference_registry`); and cross-experiment
selection runs over a DuckDB catalog of views (:mod:`catalog`).
"""

from __future__ import annotations
