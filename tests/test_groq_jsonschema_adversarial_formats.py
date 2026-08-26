"""Accept/reject regressions for the adversarial runner's format checkers.

The runner overrides python-jsonschema's "time" checker to mirror the API's
validator (santhosh-tekuri isTime): without this, valid RFC 3339 full-time
values like "10:00:00Z" were flagged as schema leaks (2026-08-26 production
run), and a naive regex accepted invalid leap seconds like "12:30:60Z".
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_RUNNER = Path(__file__).parent.parent / "scripts" / "run_groq_jsonschema_adversarial.py"
_spec = importlib.util.spec_from_file_location("groq_jsonschema_adversarial", _RUNNER)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

FORMAT_CHECKER = _module.FORMAT_CHECKER


@pytest.mark.parametrize(
    "value",
    [
        "10:00:00Z",
        "21:00:00.000Z",
        "10:00:00z",
        "10:00:00+00:00",
        "00:00:00.5-07:00",
        "23:59:60Z",  # leap second at UTC midnight boundary
        "23:59:60+00:00",
        "00:29:60+00:30",  # 23:59 UTC after offset normalization
    ],
)
def test_time_checker_accepts_valid_rfc3339_full_time(value):
    assert FORMAT_CHECKER.conforms(value, "time"), value


@pytest.mark.parametrize(
    "value",
    [
        "12:30:60Z",  # leap second outside 23:59 UTC
        "23:59:60+01:00",  # 22:59 UTC after offset normalization
        "24:00:00Z",
        "10:60:00Z",
        "10:00:61Z",
        "10:00:00",  # missing offset
        "10:00Z",  # missing seconds
        "10:00:00+24:00",
        "not-a-time",
    ],
)
def test_time_checker_rejects_invalid_times(value):
    assert not FORMAT_CHECKER.conforms(value, "time"), value


def test_other_formats_remain_registered():
    assert FORMAT_CHECKER.conforms("user@example.com", "email")
    assert FORMAT_CHECKER.conforms("2026-08-26", "date")
