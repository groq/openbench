"""Unit tests for JSON schema scorer."""

from unittest.mock import Mock
from inspect_ai.scorer import Target, Score, SampleScore, CORRECT, INCORRECT

from openbench.scorers.json_schema import (
    json_schema_scorer,
    json_validity,
    schema_compliance,
    api_success_rate,
)


def create_mock_state(
    completion: str, metadata: dict | None = None, error: str | None = None
) -> Mock:
    """Create a mock TaskState for testing."""
    mock_state = Mock()
    mock_state.output.completion = completion
    mock_state.output.error = error  # Add error attribute for API error testing
    mock_state.metadata = metadata or {}
    return mock_state


# Target typically contains expected answer for comparison, but json_schema_scorer
# only validates JSON structure against schema, so target is unused
TEST_TARGET = "test_target"


class TestJSONSchemaScorer:
    """Test the JSON schema scorer function."""

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
        assert not result.metadata["api_error"]
        assert result.metadata["error"] is None

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
        assert not result.metadata["api_error"]
        assert "schema_validation_error" in result.metadata["error"]

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
        assert not result.metadata["api_error"]
        assert "json_decode_error" in result.metadata["error"]

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
        assert not result.metadata["api_error"]
        assert result.metadata["error"] == "no_schema"

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
        assert not result.metadata["api_error"]
        assert result.metadata["error"] == "no_schema"

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
        assert not result.metadata["api_error"]
        assert "json_decode_error" in result.metadata["error"]

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
        assert not result.metadata["api_error"]

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
        assert not result.metadata["api_error"]

    async def test_api_error_handling(self):
        """Test scorer handles API errors correctly."""
        schema = {"type": "object"}

        # Create state that simulates an API error
        state = create_mock_state(
            completion="",
            metadata={"schema": schema},
            error="API timeout error",  # Simulate API error
        )
        target = Target(TEST_TARGET)

        scorer_fn = json_schema_scorer()
        result = await scorer_fn(state, target)

        assert result.value == INCORRECT
        assert result.answer == ""
        assert not result.metadata["json_valid"]
        assert not result.metadata["schema_compliant"]
        assert result.metadata["api_error"]
        assert "api_error: API timeout error" in result.metadata["error"]


class TestJSONValidityMetric:
    """Test the JSON validity metric."""

    def test_all_valid_json(self):
        """Test metric with all valid JSON scores from successful API calls."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": True,
                        "api_error": False,
                    },
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": False,
                        "api_error": False,
                    },
                ),
            ),
        ]

        metric_fn = json_validity()
        result = metric_fn(scores)

        assert result == 1.0  # 2/2 successful API calls produced valid JSON

    def test_mixed_json_validity(self):
        """Test metric with mixed JSON validity from successful API calls."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": True,
                        "api_error": False,
                    },
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": False,
                        "schema_compliant": False,
                        "api_error": False,
                    },
                ),
            ),
        ]

        metric_fn = json_validity()
        result = metric_fn(scores)

        assert result == 0.5  # 1/2 successful API calls produced valid JSON

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

        assert result == 0.0  # 0/0 successful API calls (no valid denominators)

    def test_with_api_errors(self):
        """Test metric excludes API errors from denominator (empirical coverage formula)."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": False,
                        "schema_compliant": False,
                        "api_error": True,
                    },
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=CORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": True,
                        "api_error": False,
                    },
                ),
            ),
            SampleScore(
                sample_id="3",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": False,
                        "schema_compliant": False,
                        "api_error": False,
                    },
                ),
            ),
        ]

        metric_fn = json_validity()
        result = metric_fn(scores)

        assert (
            result == 0.5
        )  # 1/2 successful API calls produced valid JSON (API error excluded)

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


class TestAPISuccessRateMetric:
    """Test the API success rate metric."""

    def test_all_api_success(self):
        """Test metric with all successful API calls (no API errors)."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=CORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": True,
                        "api_error": False,
                    },
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": False,
                        "api_error": False,
                    },
                ),
            ),
        ]

        metric_fn = api_success_rate()
        result = metric_fn(scores)

        assert result == 1.0  # 2/2 successful API calls

    def test_mixed_api_success(self):
        """Test metric with mixed API success rates."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": False,
                        "schema_compliant": False,
                        "api_error": True,
                    },
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=CORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": True,
                        "api_error": False,
                    },
                ),
            ),
            SampleScore(
                sample_id="3",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": True,
                        "schema_compliant": False,
                        "api_error": False,
                    },
                ),
            ),
        ]

        metric_fn = api_success_rate()
        result = metric_fn(scores)

        assert result == 2.0 / 3.0  # 2/3 successful API calls

    def test_all_api_errors(self):
        """Test metric when all API calls fail."""
        scores = [
            SampleScore(
                sample_id="1",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": False,
                        "schema_compliant": False,
                        "api_error": True,
                    },
                ),
            ),
            SampleScore(
                sample_id="2",
                score=Score(
                    value=INCORRECT,
                    metadata={
                        "json_valid": False,
                        "schema_compliant": False,
                        "api_error": True,
                    },
                ),
            ),
        ]

        metric_fn = api_success_rate()
        result = metric_fn(scores)

        assert result == 0.0  # 0/2 successful API calls
