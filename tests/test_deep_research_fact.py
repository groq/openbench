"""FACT judgements must cover each citation exactly once before scoring."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState

pytest.importorskip("google.genai")

from openbench.solvers.deep_research_bench import deep_research_bench_fact as fact


def _groups():
    return {
        "https://example.org/report": {
            "facts": ["The budget is 10.", "The budget is 20."],
            "url_content": "The budget is 10.",
        }
    }


@pytest.fixture
def judge(monkeypatch):
    generate = AsyncMock()
    monkeypatch.setattr(fact, "FACTEvaluatorLLM", lambda: Mock(generate=generate))
    monkeypatch.setattr(fact.asyncio, "sleep", AsyncMock())
    return generate


@pytest.mark.parametrize(
    "reply",
    [
        [{"idx": 1, "result": "supported"}, {"idx": 1, "result": "supported"}],
        [{"idx": 0, "result": "supported"}, {"idx": 2, "result": "supported"}],
        [{"idx": 1, "result": "supported"}, {"idx": 3, "result": "supported"}],
        [{"idx": True, "result": "supported"}, {"idx": 2, "result": "supported"}],
        [{"idx": 1.0, "result": "supported"}, {"idx": 2, "result": "supported"}],
        [{"idx": "1", "result": "supported"}, {"idx": 2, "result": "supported"}],
        [{"idx": 1, "result": "supported"}, {"idx": 2, "result": "yes"}],
        [{"idx": 1, "result": "supported"}, {"idx": 2}],
        [{"idx": 1, "result": "supported"}, {"idx": 2, "result": None}],
        [{"idx": 1, "result": "supported"}, {"idx": 2, "result": ["supported"]}],
        [{"idx": 1, "result": "supported"}],
        {"idx": 1, "result": "supported"},
        ["supported", "unsupported"],
    ],
)
async def test_invalid_judgements_retry_and_do_not_enter_statistics(judge, reply):
    judge.return_value = json.dumps(reply)
    groups = await fact.validate_citations(_groups(), "en")
    assert judge.await_count == 3
    group = next(iter(groups.values()))
    assert group["validate_error"]
    assert group["validate_res"] == []
    assert fact.calculate_fact_statistics(groups) == {
        "total_citations": 0,
        "valid_citations": 0,
        "valid_rate": 0.0,
    }


async def test_valid_reordered_judgements_keep_unknown_and_extra_fields(judge):
    groups = _groups()
    next(iter(groups.values()))["facts"].append("An unavailable claim.")
    judge.return_value = json.dumps(
        [
            {"idx": 3, "result": "unknown", "reason": "No information"},
            {"idx": 1, "result": "supported"},
            {"idx": 2, "result": "unsupported"},
        ]
    )
    validated = await fact.validate_citations(groups, "en")
    group = next(iter(validated.values()))
    assert group["validate_error"] is None
    assert [item["idx"] for item in group["validate_res"]] == [2, 0, 1]
    assert group["validate_res"][0]["reason"] == "No information"
    assert judge.await_count == 1
    assert fact.calculate_fact_statistics(validated) == {
        "total_citations": 2,
        "valid_citations": 1,
        "valid_rate": 0.5,
    }


async def test_fact_pipeline_retries_duplicate_verdict_instead_of_counting_it_twice(
    judge, monkeypatch
):
    url = "https://example.org/report"
    citations = [
        {"fact": "The budget is 10.", "ref_idx": 1, "url": url},
        {"fact": "The budget is 20.", "ref_idx": 1, "url": url},
    ]
    judge.side_effect = [
        json.dumps(citations),
        "[1, 2]",
        '[{"idx": 1, "result": "supported"}, {"idx": 1, "result": "supported"}]',
        '[{"idx": 2, "result": "unsupported"}, {"idx": 1, "result": "supported"}]',
    ]
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    response = Mock(status_code=200)
    response.json.return_value = {
        "data": {
            "url": url,
            "title": "Budget",
            "description": "",
            "content": "The budget is 10.",
        }
    }
    monkeypatch.setattr(fact.requests, "get", Mock(return_value=response))
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="report",
        epoch=1,
        input="Research the budget",
        messages=[],
        metadata={"task_id": "report", "language": "en"},
    )
    result = await fact.run_fact_evaluation(
        "A report with two claims.", state, AsyncMock()
    )
    assert "fact_error" not in result
    assert result["total_citations"] == 2
    assert result["valid_citations"] == 1
    assert result["valid_rate"] == 0.5
    assert judge.await_count == 4
    assert result["citation_groups"][url]["validate_error"] is None
