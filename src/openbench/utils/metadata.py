"""
Metadata utilities for openbench benchmarks.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkMetadata:
    """Minimal metadata for a benchmark - only what can't be extracted."""

    name: str  # Human-readable display name
    description: str  # Human-written description
    category: str  # Category for grouping
    tags: List[str]  # Tags for searchability

    # Registry info - still needed
    module_path: str
    function_name: str

    # Alpha/experimental flag
    is_alpha: bool = False  # Whether this benchmark is experimental/alpha
