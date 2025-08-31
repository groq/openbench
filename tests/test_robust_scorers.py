"""Tests for robust answer extraction scorers."""

import asyncio
from openbench.scorers import (
    extract_mcq_answer,
    extract_boxed_answer,
    robust_mcq_scorer,
    aime_scorer,
)
from openbench.scorers.robust_boxed import normalize_numeric_answer
from inspect_ai.scorer import Target, CORRECT, INCORRECT
from dataclasses import dataclass


@dataclass
class MockOutput:
    """Mock output for testing."""

    completion: str


@dataclass
class MockTaskState:
    """Mock task state for testing."""

    output: MockOutput


class TestMCQExtraction:
    """Test multiple choice answer extraction."""

    def test_markdown_wrapped_answer(self):
        """Test extraction from markdown-wrapped answers."""
        assert extract_mcq_answer("**Answer:** A") == "A"
        assert extract_mcq_answer("*Answer:* B") == "B"
        assert extract_mcq_answer("__Answer:__ C") == "C"
        assert extract_mcq_answer("_Answer:_ D") == "D"

    def test_parenthesis_answer(self):
        """Test extraction from parenthesis format."""
        assert extract_mcq_answer("Answer: (A)") == "A"
        assert extract_mcq_answer("The answer is (B).") == "B"
        assert extract_mcq_answer("Choice: (C)") == "C"
        assert extract_mcq_answer("[D] is correct") == "D"

    def test_plain_answer(self):
        """Test extraction from plain format."""
        assert extract_mcq_answer("Answer: A") == "A"
        assert extract_mcq_answer("Answer – B") == "B"
        assert extract_mcq_answer("Option C") == "C"
        assert extract_mcq_answer("Choice: D") == "D"

    def test_latex_boxed(self):
        """Test extraction from LaTeX boxed format."""
        assert extract_mcq_answer(r"\boxed{A}") == "A"
        assert extract_mcq_answer(r"\boxed{\text{B}}") == "B"
        assert extract_mcq_answer(r"\boxed{\textbf{C}}") == "C"
        assert extract_mcq_answer(r"The answer is \boxed{D}") == "D"

    def test_markdown_standalone(self):
        """Test extraction from standalone markdown."""
        assert extract_mcq_answer("*A*") == "A"
        assert extract_mcq_answer("**B**") == "B"
        assert extract_mcq_answer("_C_") == "C"
        assert extract_mcq_answer("__D__") == "D"

    def test_complex_cases(self):
        """Test extraction from complex/mixed formats."""
        # Markdown with description
        assert extract_mcq_answer("**D) This is the correct answer**") == "D"

        # Multiple patterns (should get first/best match)
        assert extract_mcq_answer("Let me think... Answer: B\n\n(C) is wrong") == "B"

        # Case insensitive
        assert extract_mcq_answer("answer: a") == "A"
        assert extract_mcq_answer("ANSWER: B") == "B"

    def test_fallback_to_first_char(self):
        """Test fallback to first character."""
        assert extract_mcq_answer("A") == "A"
        assert extract_mcq_answer("B is the answer") == "B"
        assert extract_mcq_answer("**C") == "C"

    def test_no_answer_found(self):
        """Test when no answer is found."""
        assert extract_mcq_answer("No valid answer here") is None
        assert extract_mcq_answer("The options are 1, 2, 3, 4") is None
        assert extract_mcq_answer("") is None


class TestBoxedExtraction:
    """Test boxed answer extraction."""

    def test_boxed_extraction(self):
        """Test extraction from \boxed{} format."""
        assert extract_boxed_answer(r"\boxed{42}") == "42"
        assert extract_boxed_answer(r"The answer is \boxed{-3}") == "-3"
        assert extract_boxed_answer(r"\boxed{3.14159}") == "3.14159"

    def test_framebox_extraction(self):
        """Test extraction from \framebox{} format."""
        assert extract_boxed_answer(r"\framebox{100}") == "100"
        assert extract_boxed_answer(r"Answer: \framebox{0}") == "0"

    def test_fbox_extraction(self):
        """Test extraction from \fbox{} format (OpenBench compatibility)."""
        assert extract_boxed_answer(r"\fbox{42}") == "42"
        assert extract_boxed_answer(r"The answer is \fbox{-10}") == "-10"

    def test_multiple_boxed(self):
        """Test that last boxed answer is used."""
        text = r"First \boxed{1} then \boxed{2} finally \boxed{3}"
        assert extract_boxed_answer(text) == "3"

    def test_comma_separated(self):
        """Test handling of comma-separated values in box."""
        # Just test that it extracts something from comma-separated values
        assert extract_boxed_answer(r"\boxed{x = 2, y = 3}") == "y = 3"

    def test_fallback_to_last_number(self):
        """Test fallback to last number when no box found."""
        assert extract_boxed_answer("The answer is 42", True) == "42"
        assert extract_boxed_answer("First 10 then 20 finally 30", True) == "30"
        assert extract_boxed_answer("Negative: -5", True) == "-5"
        assert extract_boxed_answer("Decimal: 3.14", True) == "3.14"

    def test_no_fallback(self):
        """Test no fallback when disabled."""
        assert extract_boxed_answer("The answer is 42", False) is None
        assert extract_boxed_answer("No box here", False) is None

    def test_no_answer(self):
        """Test when no answer is found."""
        assert extract_boxed_answer("No numbers here", True) is None
        assert extract_boxed_answer("", True) is None


class TestNumericNormalization:
    """Test numeric answer normalization."""

    def test_comma_removal(self):
        """Test removal of commas."""
        assert normalize_numeric_answer("1,234") == "1234"
        assert normalize_numeric_answer("1,000,000") == "1000000"

    def test_integer_extraction(self):
        """Test extraction of leading integers."""
        assert normalize_numeric_answer("42 points") == "42"
        assert normalize_numeric_answer("-3 units") == "-3"
        assert normalize_numeric_answer("0") == "0"

    def test_decimal_normalization(self):
        """Test decimal number normalization."""
        # Our implementation extracts leading integers
        assert normalize_numeric_answer("3.14000") == "3"
        assert normalize_numeric_answer("5.0") == "5"
        assert normalize_numeric_answer("0.500") == "0"
        assert normalize_numeric_answer("42.") == "42"

    def test_invalid_input(self):
        """Test invalid input handling."""
        assert normalize_numeric_answer("abc") is None
        assert normalize_numeric_answer("") is None
        assert normalize_numeric_answer(None) is None


class TestRobustMCQScorer:
    """Test the robust MCQ scorer."""

    def test_correct_answer(self):
        """Test scoring correct answers."""
        scorer = robust_mcq_scorer()
        state = MockTaskState(MockOutput("Answer: B"))
        target = Target("B")

        score = asyncio.run(scorer(state, target))
        assert score.value == CORRECT
        assert score.answer == "B"

    def test_incorrect_answer(self):
        """Test scoring incorrect answers."""
        scorer = robust_mcq_scorer()
        state = MockTaskState(MockOutput("Answer: A"))
        target = Target("B")

        score = asyncio.run(scorer(state, target))
        assert score.value == INCORRECT
        assert score.answer == "A"

    def test_no_answer_found(self):
        """Test scoring when no answer found."""
        scorer = robust_mcq_scorer()
        state = MockTaskState(MockOutput("I don't know"))
        target = Target("A")

        score = asyncio.run(scorer(state, target))
        assert score.value == INCORRECT
        assert score.answer is None


class TestAIMEScorer:
    """Test the AIME scorer."""

    def test_boxed_integer(self):
        """Test scoring boxed integer answers."""
        scorer = aime_scorer()
        state = MockTaskState(MockOutput(r"\boxed{42}"))
        target = Target("42")

        score = asyncio.run(scorer(state, target))
        assert score.value == CORRECT
        assert score.answer == "42"

    def test_fallback_to_last_integer(self):
        """Test fallback to last integer."""
        scorer = aime_scorer()
        state = MockTaskState(MockOutput("The answer is 123"))
        target = Target("123")

        score = asyncio.run(scorer(state, target))
        assert score.value == CORRECT
        assert score.answer == "123"

    def test_out_of_range(self):
        """Test AIME range validation (0-999)."""
        scorer = aime_scorer()
        state = MockTaskState(MockOutput(r"\boxed{1000}"))
        target = Target("1000")

        score = asyncio.run(scorer(state, target))
        assert score.value == INCORRECT
        assert "outside valid AIME range" in score.explanation

    def test_incorrect_answer(self):
        """Test incorrect answer."""
        scorer = aime_scorer()
        state = MockTaskState(MockOutput(r"\boxed{41}"))
        target = Target("42")

        score = asyncio.run(scorer(state, target))
        assert score.value == INCORRECT
        assert score.answer == "41"
