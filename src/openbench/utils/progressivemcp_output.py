"""
Helpers for parsing ProgressiveMCPBench model outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def parse_progressivemcp_output(raw: str | None) -> Dict[str, Any] | None:
    """Parse a ProgressiveMCPBench completion JSON block."""
    if not raw:
        return None

    text = raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None
    return obj
