"""Unit tests for open answer scorer."""

import pytest
from unittest.mock import Mock
from inspect_ai.scorer import Target

from openbench.scorers.open_answer import (
    create_open_answer_scorer,
    simple_open_answer_scorer,
    grouped_open_answer_scorer,
    _parse_open_response,
    _eval_open,
    _normalize_str,
    _extract_numbers,
)


def create_mock_state(completion: str, metadata: dict | None = None) -> Mock:
    """Create a mock TaskState for testing."""
    mock_state = Mock()
    mock_state.output.completion = completion
    mock_state.metadata = metadata or {}
    return mock_state


# Target typically contains expected answer for comparison
TEST_TARGET = "test_target"


class TestExtractNumbers:
    """Test the number extraction function."""

    def test_extract_integers(self):
        """Test extraction of integer values."""
        numbers = _extract_numbers("The answer is 42")
        assert numbers == [42.0]

    def test_extract_negative_numbers(self):
        """Test extraction of negative numbers."""
        numbers = _extract_numbers("The result is -15")
        assert numbers == [-15.0]

    def test_extract_decimals(self):
        """Test extraction of decimal numbers."""
        numbers = _extract_numbers("The value is 3.14159")
        assert numbers == [3.14159]

    def test_extract_multiple_numbers(self):
        """Test extraction of multiple numbers."""
        numbers = _extract_numbers("First 10, then 20.5, finally -3")
        assert numbers == [10.0, 20.5, -3.0]

    def test_extract_with_commas(self):
        """Test extraction of numbers with comma separators."""
        numbers = _extract_numbers("The population is 1,234,567")
        assert numbers == [1234567.0]

    def test_no_numbers(self):
        """Test when no numbers are present."""
        numbers = _extract_numbers("No numbers here!")
        assert numbers == []

    def test_empty_string(self):
        """Test with empty string."""
        numbers = _extract_numbers("")
        assert numbers == []


class TestNormalizeStr:
    """Test the string normalization function."""

    def test_normalize_numeric_values(self):
        """Test normalization of numeric inputs."""
        result = _normalize_str(42)
        assert result == [42.0]

        result = _normalize_str(3.14)
        assert result == [3.14]

    def test_normalize_numeric_strings(self):
        """Test normalization of strings that contain numbers."""
        result = _normalize_str("42")
        assert 42.0 in result
        assert "42" in result

        result = _normalize_str("3.14")
        assert 3.14 in result
        assert "3.14" in result

    def test_normalize_text_strings(self):
        """Test normalization of text strings."""
        result = _normalize_str("Hello World")
        assert "hello world" in result
        assert len(result) == 1  # Only string, no numeric interpretation

    def test_normalize_with_whitespace(self):
        """Test normalization handles whitespace."""
        result = _normalize_str("  Hello   World  ")
        assert "hello world" in result

    def test_normalize_with_punctuation(self):
        """Test normalization handles punctuation."""
        result = _normalize_str("Hello, World!")
        assert "hello, world!" in result

    def test_normalize_comma_separated_numbers(self):
        """Test normalization of comma-separated numbers."""
        result = _normalize_str("1,234")
        assert 1234.0 in result
        assert "1,234" in result

    def test_normalize_empty_string(self):
        """Test normalization of empty/whitespace strings."""
        result = _normalize_str("")
        assert result == []

        result = _normalize_str("   ")
        assert result == []


class TestParseOpenResponse:
    """Test the open response parsing function."""

    def test_simple_answer(self):
        """Test parsing simple direct answers."""
        result = _parse_open_response("The answer is 42")
        assert 42.0 in result

    def test_multiple_indicators(self):
        """Test parsing responses with multiple answer indicators."""
        result = _parse_open_response(
            "First, we calculate. Then the result is 25. So the final answer is 30."
        )
        # The parser extracts the final answer based on key indicators
        assert 30.0 in result

    def test_complex_reasoning(self):
        """Test parsing complex reasoning with final answer."""
        response = """
        Let me think about this step by step.
        The initial value is 10.
        We need to multiply by 3.
        So the final answer is 30.
        """
        result = _parse_open_response(response)
        assert 30.0 in result

    def test_text_only_response(self):
        """Test parsing text-only responses."""
        result = _parse_open_response("The answer is blue")
        assert any("blue" in str(item) for item in result)

    def test_empty_response(self):
        """Test parsing empty responses."""
        result = _parse_open_response("")
        assert result == []

    def test_no_clear_indicators(self):
        """Test parsing responses without clear answer indicators."""
        result = _parse_open_response("This is a complex problem with multiple steps")
        assert len(result) > 0

    def test_mathematical_expression(self):
        """Test parsing mathematical expressions."""
        result = _parse_open_response("The calculation gives us x = 15")
        assert 15.0 in result


class TestEvalOpen:
    """Test the open answer evaluation function."""

    def test_exact_string_match(self):
        """Test exact string matching."""
        assert _eval_open("blue", ["blue"]) is True
        assert _eval_open("blue", ["red"]) is False

    def test_numeric_match(self):
        """Test numeric matching."""
        assert _eval_open("42", [42.0]) is True
        assert _eval_open(42, [42.0]) is True
        assert _eval_open("42", [43.0]) is False

    def test_partial_string_match(self):
        """Test partial string matching (substring)."""
        assert _eval_open("blue", ["the color is blue"]) is True
        assert _eval_open("answer", ["the final answer"]) is True

    def test_multiple_gold_answers(self):
        """Test evaluation with multiple acceptable answers."""
        gold_answers = ["red", "blue", "green"]
        assert _eval_open(gold_answers, ["blue"]) is True
        assert _eval_open(gold_answers, ["yellow"]) is False

    def test_mixed_type_answers(self):
        """Test evaluation with mixed string and numeric answers."""
        assert _eval_open("42", ["the answer is 42", 42.0]) is True
        assert _eval_open(42, ["forty-two", 42.0]) is True

    def test_case_insensitive_matching(self):
        """Test case-insensitive string matching."""
        assert _eval_open("Blue", ["blue"]) is True
        assert _eval_open("ANSWER", ["the final answer"]) is True

    def test_no_match(self):
        """Test when no predictions match gold answers."""
        assert _eval_open("zebra", ["wrong", "elephant"]) is False

    def test_empty_predictions(self):
        """Test evaluation with empty predictions."""
        assert _eval_open("answer", []) is False


class TestOpenAnswerScorer:
    """Test the open answer scorer function."""

    @pytest.mark.asyncio
    async def test_correct_answer_from_target(self):
        """Test scoring correct answer using target text."""
        state = create_mock_state("The answer is 42")
        target = Target("42")

        scorer_fn = simple_open_answer_scorer()
        result = await scorer_fn(state, target)

        assert result.value == 1.0
        assert result.answer == "The answer is 42"
        assert result.metadata["open_eval"] is True
        assert 42.0 in result.metadata["pred_list"]

    @pytest.mark.asyncio
    async def test_correct_answer_from_metadata(self):
        """Test scoring correct answer using metadata."""
        state = create_mock_state("The result is 25", metadata={"answer": "25"})
        target = Target("unused")

        scorer_fn = simple_open_answer_scorer()
        result = await scorer_fn(state, target)

        assert result.value == 1.0
        assert 25.0 in result.metadata["pred_list"]

    @pytest.mark.asyncio
    async def test_incorrect_answer(self):
        """Test scoring incorrect answer."""
        state = create_mock_state("The answer is 41")
        target = Target("42")

        scorer_fn = simple_open_answer_scorer()
        result = await scorer_fn(state, target)

        assert result.value == 0.0
        assert result.answer == "The answer is 41"
        assert 41.0 in result.metadata["pred_list"]

    @pytest.mark.asyncio
    async def test_text_answer_matching(self):
        """Test scoring with text-based answers."""
        state = create_mock_state("The color is blue")
        target = Target("blue")

        scorer_fn = simple_open_answer_scorer()
        result = await scorer_fn(state, target)

        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_custom_answer_key(self):
        """Test scorer with custom answer key in metadata."""
        custom_scorer = create_open_answer_scorer(answer_key="gold_answer")()

        state = create_mock_state(
            "The answer is correct", metadata={"gold_answer": "correct"}
        )
        target = Target("unused")

        result = await custom_scorer(state, target)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_empty_completion(self):
        """Test scoring with empty completion."""
        state = create_mock_state("")
        target = Target("42")

        scorer_fn = simple_open_answer_scorer()
        result = await scorer_fn(state, target)

        assert result.value == 0.0
        assert result.answer == ""

    @pytest.mark.asyncio
    async def test_none_completion(self):
        """Test scoring with None completion."""
        state = create_mock_state(None)
        target = Target("42")

        scorer_fn = simple_open_answer_scorer()
        result = await scorer_fn(state, target)

        assert result.value == 0.0
        assert result.answer is None


class TestScorerFactories:
    """Test the scorer factory functions."""

    def test_simple_scorer_creation(self):
        """Test creation of simple scorer."""
        scorer = simple_open_answer_scorer()
        assert callable(scorer)

    def test_grouped_scorer_creation(self):
        """Test creation of grouped scorer."""
        scorer = grouped_open_answer_scorer("subject")
        assert callable(scorer)

    def test_create_scorer_with_group_keys(self):
        """Test creation of scorer with group keys."""
        scorer = create_open_answer_scorer(group_keys=["subject", "difficulty"])()
        assert callable(scorer)

    def test_create_scorer_with_custom_answer_key(self):
        """Test creation of scorer with custom answer key."""
        scorer = create_open_answer_scorer(answer_key="gold_standard")()
        assert callable(scorer)
