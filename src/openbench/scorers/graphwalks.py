# src/openbench/scorers/graphwalks.py
from __future__ import annotations

import re
from typing import Set

from inspect_ai.scorer import scorer, Score, Target, mean, stderr

# Parse ONLY the very last line, which must look like:
#   Final Answer: [a, b, c]
_FINAL_LINE_RE = re.compile(r"Final Answer:\s*\[(.*)\]\s*$", re.IGNORECASE)


def _parse_nodes(text: str) -> tuple[list[str], bool]:
    """Return (nodes, parse_error_flag). Dedup while preserving order."""
    if not text:
        return [], True
    last_line = text.strip().splitlines()[-1]
    m = _FINAL_LINE_RE.search(last_line)
    if not m:
        return [], True
    inner = m.group(1)
    # split by commas only; trim; drop empties; dedup preserving order
    raw = [t.strip() for t in inner.split(",")]
    seen: Set[str] = set()
    out: list[str] = []
    for t in raw:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out, False


def _prf1(pred: list[str], gold: list[str]) -> tuple[float, float, float]:
    sp, sg = set(pred), set(gold)
    inter = len(sp & sg)
    p = inter / len(sp) if sp else 0.0
    r = inter / len(sg) if sg else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


@scorer(metrics=[mean(), stderr()])  # UI will show Mean (and stderr) of F1
def graphwalks_scorer():
    async def score(state, target: Target) -> Score:
        # Inspect model output: prefer .completion, fall back to .text if needed
        out = ""
        if getattr(state, "output", None) is not None:
            out = (
                getattr(state.output, "completion", None)
                or getattr(state.output, "text", "")
                or ""
            )

        pred, parse_err = _parse_nodes(out)
        gold = list(target)  # Target is a sequence of gold node strings

        p, r, f1 = _prf1(pred, gold)
        return Score(
            value=f1,  # Mean in the UI = mean F1
            answer=str(pred),
            metadata={
                "precision": p,
                "recall": r,
                "f1": f1,
                "parsed_ok": (not parse_err),
                "pred": pred,
                "gold": gold,
            },
        )

    return score
