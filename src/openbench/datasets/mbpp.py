# src/openbench/datasets/mbpp.py
"""
MBPP dataset adapter (sanitized/test) for OpenBench (Inspect).

- Loads: Muennighoff/mbpp (config="sanitized", split="test")
- Builds chat-style Samples with the *paper-style* prompt:
    "You are an expert Python programmer, and here is your task: {prompt}
     Your code should pass these tests:

     {tests}
     [BEGIN]
     {code}
     [DONE]"

No entry-point inference is performed; the tests reveal the function name.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from inspect_ai.dataset import Dataset, Sample, FieldSpec, hf_dataset
from inspect_ai.model import ChatMessageUser


def _get_first(rec: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return default


def _normalize_tests(tests: Any) -> List[str]:
    if tests is None:
        return []
    if isinstance(tests, list):
        return [str(t).rstrip() for t in tests if str(t).strip()]
    if isinstance(tests, str):
        return [ln.rstrip() for ln in tests.splitlines() if ln.strip()]
    return []


PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {prompt} "
    "Your code should pass these tests:\n\n"
    "{tests}\n"
    "[BEGIN]\n"
    "{code}\n"
    "[DONE]"
)


def record_to_sample() -> FieldSpec | Callable[[Dict[str, Any]], Sample]:
    """
    Map a raw MBPP record to an Inspect Sample using the paper-style prompt.
    """

    def _map(record: Dict[str, Any]) -> Sample:
        # primary fields (sanitized schema)
        task_id = str(record.get("task_id", "") or "").strip() or None
        prompt_text = str(record["prompt"]).strip()

        # optional fields
        tests_list = _normalize_tests(record.get("test_list", []))
        reference = record.get("code") or None

        # normalize setup
        raw_setup = record.get("test_imports")
        if isinstance(raw_setup, list):
            setup = (
                "\n".join(s for s in (str(x).rstrip() for x in raw_setup) if s) or None
            )
        else:
            setup = (str(raw_setup).strip() or None) if raw_setup is not None else None

        # build user message
        tests_block = "\n".join(tests_list)
        user_text = PROMPT_TEMPLATE.format(
            prompt=prompt_text,
            tests=tests_block,
            code="{code}",  # literal placeholder per published paper format
        )

        metadata = {
            "task_id": task_id,
            "prompt": prompt_text,
            "tests": tests_list,  # list[str] of assert lines
            "setup": setup,  # optional imports/helpers (str or None)
            "reference_code": reference,  # optional
        }

        return Sample(
            id=task_id,
            input=[ChatMessageUser(content=user_text)],
            target="",
            metadata=metadata,
        )

    return _map


def get_dataset(*, limit: Optional[int] = None, name: Optional[str] = None) -> Dataset:
    """
    Load MBPP (sanitized) test split from Hugging Face and adapt to Inspect Samples.
    """
    return hf_dataset(
        path="Muennighoff/mbpp",
        name="sanitized",  # config, limited to sanitized
        split="test",  # only split we use
        sample_fields=record_to_sample(),
        limit=limit,
    )
