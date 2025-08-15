"""Unit tests for JSON schema scorer."""

import pytest
from unittest.mock import Mock
from inspect_ai.scorer import Target, Score, SampleScore, CORRECT, INCORRECT

from openbench.scorers.json_schema import (
    json_schema_scorer,
    json_validity,
    schema_compliance,
    overall_success,
)


def create_mock_state(completion: str, metadata: dict | None = None) -> Mock:
    """Create a mock TaskState for testing."""
    mock_state = Mock()
    mock_state.output.completion = completion
    mock_state.metadata = metadata or {}
    return mock_state


# Target typically contains expected answer for comparison, but json_schema_scorer
# only validates JSON structure against schema, so target is unused
TEST_TARGET = "test_target"


class TestJSONSchemaScorer:
    """Test the JSON schema scorer function."""

    @pytest.mark.asyncio
    async def test_valid_json_and_schema(self):
        """Test with valid JSON that conforms to schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }

        state = create_mock_state(
            completion='{"name": "John", "age": 25}', metadata={"schema": schema}
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == CORRECT
        assert result.answer == '{"name": "John", "age": 25}'
        assert result.metadata["json_valid"]
        assert result.metadata["schema_compliant"]
        assert result.metadata["error"] is None

    @pytest.mark.asyncio
    async def test_valid_json_invalid_schema(self):
        """Test with valid JSON that doesn't conform to schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }

        state = create_mock_state(
            completion='{"name": "John"}',  # Missing required "age"
            metadata={"schema": schema},
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == INCORRECT
        assert result.answer == '{"name": "John"}'
        assert result.metadata["json_valid"]
        assert not result.metadata["schema_compliant"]
        assert "schema_validation_error" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test with invalid JSON."""
        schema = {"type": "object"}

        state = create_mock_state(
            completion='{"name": "John", invalid}', metadata={"schema": schema}
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == INCORRECT
        assert result.answer == '{"name": "John", invalid}'
        assert not result.metadata["json_valid"]
        assert not result.metadata["schema_compliant"]
        assert "json_decode_error" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_no_schema_in_metadata(self):
        """Test when no schema is provided in metadata."""
        state = create_mock_state(
            completion='{"name": "John"}',
            metadata={},  # No schema
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == INCORRECT
        assert result.answer == '{"name": "John"}'
        assert not result.metadata["json_valid"]
        assert not result.metadata["schema_compliant"]
        assert result.metadata["error"] == "no_schema"

    @pytest.mark.asyncio
    async def test_none_metadata(self):
        """Test when metadata is None."""
        state = create_mock_state(completion='{"name": "John"}', metadata=None)
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == INCORRECT
        assert result.answer == '{"name": "John"}'
        assert not result.metadata["json_valid"]
        assert not result.metadata["schema_compliant"]
        assert result.metadata["error"] == "no_schema"

    @pytest.mark.asyncio
    async def test_empty_completion(self):
        """Test with empty completion."""
        schema = {"type": "object"}

        state = create_mock_state(completion="", metadata={"schema": schema})
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == INCORRECT
        assert result.answer == ""
        assert not result.metadata["json_valid"]
        assert not result.metadata["schema_compliant"]
        assert "json_decode_error" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_whitespace_handling(self):
        """Test that whitespace is properly stripped for JSON parsing."""
        schema = {"type": "object", "properties": {"test": {"type": "boolean"}}}

        state = create_mock_state(
            completion='  {"test": true}  \n',  # Leading/trailing whitespace
            metadata={"schema": schema},
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == CORRECT
        assert result.answer == '  {"test": true}  \n'  # Raw output preserved
        assert result.metadata["json_valid"]
        assert result.metadata["schema_compliant"]

    @pytest.mark.asyncio
    async def test_complex_schema(self):
        """Test with a more complex JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string", "format": "email"},
                        },
                        "required": ["name", "email"],
                    },
                }
            },
            "required": ["users"],
        }

        state = create_mock_state(
            completion='{"users": [{"name": "John", "email": "john@example.com"}]}',
            metadata={"schema": schema},
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == CORRECT
        assert result.metadata["json_valid"]
        assert result.metadata["schema_compliant"]


class TestJSONValidityMetric:
    """Test the JSON validity metric."""

    def test_all_valid_json(self):
        """Test metric with all valid JSON scores."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": True, "schema_compliant": False},
                ),
            ),
        ]

        metric_fn = json_validity()
        result = metric_fn(scores)

        assert result == 1.0  # 2/2 valid JSON

    def test_mixed_json_validity(self):
        """Test metric with mixed JSON validity."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": False, "schema_compliant": False},
                ),
            ),
        ]

        metric_fn = json_validity()
        result = metric_fn(scores)

        assert result == 0.5  # 1/2 valid JSON

    def test_no_metadata_scores(self):
        """Test metric with scores that have no metadata."""
        scores = [
            SampleScore(sample_id="1", score=Score(value=CORRECT)),  # No metadata
            SampleScore(
                sample_id="2", score=Score(value=INCORRECT, metadata=None)
            ),  # None metadata
        ]

        metric_fn = json_validity()
        result = metric_fn(scores)

        assert result == 0.0  # 0/2 valid JSON

    def test_empty_scores(self):
        """Test metric with empty scores list."""
        metric_fn = json_validity()
        result = metric_fn([])

        assert result == 0.0


class TestSchemaComplianceMetric:
    """Test the schema compliance metric."""

    def test_all_compliant(self):
        """Test metric with all schema compliant JSON."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
        ]

        metric_fn = schema_compliance()
        result = metric_fn(scores)

        assert result == 1.0  # 2/2 compliant among valid JSON

    def test_mixed_compliance(self):
        """Test metric with mixed schema compliance."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": True, "schema_compliant": False},
                ),
            ),
        ]

        metric_fn = schema_compliance()
        result = metric_fn(scores)

        assert result == 0.5  # 1/2 compliant among valid JSON

    def test_no_valid_json(self):
        """Test metric when no JSON is valid."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": False, "schema_compliant": False},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": False, "schema_compliant": False},
                ),
            ),
        ]

        metric_fn = schema_compliance()
        result = metric_fn(scores)

        assert result == 0.0  # No valid JSON to check compliance


class TestOverallSuccessMetric:
    """Test the overall success metric."""

    def test_all_successful(self):
        """Test metric with all successful (valid JSON + compliant) scores."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
        ]

        metric_fn = overall_success()
        result = metric_fn(scores)

        assert result == 1.0  # 2/2 successful

    def test_mixed_success(self):
        """Test metric with mixed success rates."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={"json_valid": True, "schema_compliant": True},
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": True, "schema_compliant": False},
                ),
            ),
            SampleScore(
                sample_id="3",
                score=Score(
                    value=INCORRECT,
                    metadata={"json_valid": False, "schema_compliant": False},
                ),
            ),
        ]

        metric_fn = overall_success()
        result = metric_fn(scores)

        assert result == 1.0 / 3.0  # 1/3 successful
