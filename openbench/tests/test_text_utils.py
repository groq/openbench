"""Unit tests for text utility functions."""

from openbench.utils.text import (
    create_dynamic_multiple_choice_prompt,
    strip_md_latex,
    normalize_mcq_answer,
    parse_json_from_response,
    parse_numeric_answer,
    normalize_number,
    extract_confidence_score,
)


class TestCreateDynamicMultipleChoicePrompt:
    """Test the dynamic multiple choice prompt creation function."""

    def test_basic_two_options(self):
        """Test creating prompt with 2 options."""
        question = "What is the capital of France?"
        options = ["London", "Paris"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        expected_lines = [
            "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B:",
            "",
            "What is the capital of France?",
            "",
            "A) London",
            "B) Paris",
        ]
        expected = "\n".join(expected_lines)

        assert result == expected

    def test_basic_four_options(self):
        """Test creating prompt with 4 options (standard multiple choice)."""
        question = "Which planet is closest to the sun?"
        options = ["Earth", "Venus", "Mercury", "Mars"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert "LETTER is one of A, B, C, D:" in result
        assert "A) Earth" in result
        assert "B) Venus" in result
        assert "C) Mercury" in result
        assert "D) Mars" in result
        assert "Which planet is closest to the sun?" in result

    def test_many_options(self):
        """Test creating prompt with many options (more than 4)."""
        question = "Which is a programming language?"
        options = ["Python", "Java", "C++", "JavaScript", "Ruby", "Go", "Swift"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert "LETTER is one of A, B, C, D, E, F, G:" in result
        assert "A) Python" in result
        assert "B) Java" in result
        assert "C) C++" in result
        assert "D) JavaScript" in result
        assert "E) Ruby" in result
        assert "F) Go" in result
        assert "G) Swift" in result

    def test_single_option(self):
        """Test creating prompt with single option."""
        question = "Is this a test?"
        options = ["Yes"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert "LETTER is one of A:" in result
        assert "A) Yes" in result

    def test_empty_options(self):
        """Test creating prompt with empty options list."""
        question = "What is this?"
        options = []

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert result == question

    def test_options_with_special_characters(self):
        """Test creating prompt with options containing special characters."""
        question = "Which is a mathematical expression?"
        options = ["x + y = z", "∫ f(x) dx", "∂f/∂x", "√(x² + y²)"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert "A) x + y = z" in result
        assert "B) ∫ f(x) dx" in result
        assert "C) ∂f/∂x" in result
        assert "D) √(x² + y²)" in result

    def test_multiline_question(self):
        """Test creating prompt with multiline question."""
        question = "Consider the following code:\n\nfor i in range(10):\n    print(i)\n\nWhat does this do?"
        options = ["Prints numbers 0-9", "Prints numbers 1-10", "Creates an error"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert question in result
        assert "A) Prints numbers 0-9" in result
        assert "B) Prints numbers 1-10" in result
        assert "C) Creates an error" in result

    def test_options_with_unicode(self):
        """Test creating prompt with Unicode characters in options."""
        question = "Which is a greeting in Japanese?"
        options = ["こんにちは", "Hello", "Bonjour", "Hola"]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert "A) こんにちは" in result
        assert "B) Hello" in result
        assert "C) Bonjour" in result
        assert "D) Hola" in result

    def test_long_options(self):
        """Test creating prompt with very long option text."""
        question = "Which statement is true?"
        options = [
            "This is a very long option that contains multiple sentences and goes on for quite some time to test how the function handles lengthy text.",
            "Short option",
            "Another somewhat longer option that also tests the formatting but is not as extremely long as the first one.",
        ]

        result = create_dynamic_multiple_choice_prompt(question, options)

        assert "A) This is a very long option" in result
        assert "B) Short option" in result
        assert "C) Another somewhat longer option" in result

    def test_alphabetical_limits(self):
        """Test that options beyond Z would work (though impractical)."""
        question = "Test question"
        options = [f"Option {i}" for i in range(30)]  # More than 26 options

        result = create_dynamic_multiple_choice_prompt(question, options)

        # Should handle A-Z, then continue with extended ASCII
        assert "A) Option 0" in result
        assert "Z) Option 25" in result
        # The function uses chr(65 + i), so index 26 would be chr(91) = '['
        # This tests the edge case behavior
        lines = result.split("\n")
        option_lines = [line for line in lines if line and line[1:3] == ") "]
        assert len(option_lines) == 30


class TestStripMdLatex:
    """Test the Markdown/LaTeX stripping function."""

    def test_bold_markdown(self):
        """Test removing bold markdown."""
        text = "This is **bold** text"
        result = strip_md_latex(text)
        assert result == "This is bold text"

    def test_latex_boxed(self):
        """Test removing LaTeX boxed expressions."""
        text = "The answer is $\\boxed{42}$"
        result = strip_md_latex(text)
        assert result == "The answer is 42"

    def test_complex_latex(self):
        """Test removing various LaTeX expressions."""
        text = "$\\text{The result is}$ $\\mathrm{value}$ $\\mathbf{42}$"
        result = strip_md_latex(text)
        assert result == "The result is value 42"

    def test_mixed_formatting(self):
        """Test removing mixed Markdown and LaTeX formatting."""
        text = "**Answer:** $\\boxed{\\text{solution}}$"
        result = strip_md_latex(text)
        assert result == "Answer: solution}"

    def test_no_formatting(self):
        """Test text with no special formatting."""
        text = "This is plain text"
        result = strip_md_latex(text)
        assert result == text

    def test_empty_string(self):
        """Test with empty string."""
        result = strip_md_latex("")
        assert result == ""


class TestNormalizeMcqAnswer:
    """Test the MCQ answer normalization function."""

    def test_arabic_letters(self):
        """Test normalizing Arabic MCQ letters."""
        assert normalize_mcq_answer("أ") == "A"
        assert normalize_mcq_answer("ب") == "B"
        assert normalize_mcq_answer("ج") == "C"
        assert normalize_mcq_answer("د") == "D"

    def test_bengali_letters(self):
        """Test normalizing Bengali MCQ letters."""
        assert normalize_mcq_answer("অ") == "A"
        assert normalize_mcq_answer("ব") == "B"
        assert normalize_mcq_answer("ড") == "C"
        assert normalize_mcq_answer("ঢ") == "D"

    def test_japanese_letters(self):
        """Test normalizing Japanese MCQ letters."""
        assert normalize_mcq_answer("Ａ") == "A"
        assert normalize_mcq_answer("Ｂ") == "B"
        assert normalize_mcq_answer("Ｃ") == "C"
        assert normalize_mcq_answer("Ｄ") == "D"

    def test_latin_letters_unchanged(self):
        """Test that Latin letters remain unchanged."""
        assert normalize_mcq_answer("A").strip() == "A"
        assert normalize_mcq_answer("B").strip() == "B"

    def test_mixed_text(self):
        """Test normalization in mixed text."""
        text = "الإجابة هي أ"
        result = normalize_mcq_answer(text)
        assert "A" in result

    def test_empty_string(self):
        """Test with empty string."""
        result = normalize_mcq_answer("")
        assert result == ""


class TestParseJsonFromResponse:
    """Test the JSON parsing function."""

    def test_clean_json(self):
        """Test parsing clean JSON."""
        text = '{"key": "value", "number": 42}'
        result = parse_json_from_response(text)
        assert result == {"key": "value", "number": 42}

    def test_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        text = """Here is the JSON:
```json
{"answer": "A", "confidence": 0.95}
```
That's the result."""
        result = parse_json_from_response(text)
        assert result == {"answer": "A", "confidence": 0.95}

    def test_json_in_code_block(self):
        """Test parsing JSON in generic code block."""
        text = """```
{"result": "success"}
```"""
        result = parse_json_from_response(text)
        assert result == {"result": "success"}

    def test_embedded_json(self):
        """Test finding JSON embedded in text."""
        text = 'The analysis shows {"score": 85} as the result.'
        result = parse_json_from_response(text)
        assert result == {"score": 85}

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        text = "This is not JSON at all"
        result = parse_json_from_response(text)
        assert result == {}

    def test_malformed_json(self):
        """Test handling malformed JSON."""
        text = '{"incomplete": json'
        result = parse_json_from_response(text)
        assert result == {}

    def test_empty_string(self):
        """Test with empty string."""
        result = parse_json_from_response("")
        assert result == {}


class TestParseNumericAnswer:
    """Test the numeric answer parsing function."""

    def test_basic_answer(self):
        """Test parsing basic numeric answer."""
        response = "The calculation gives us Answer: 42"
        result = parse_numeric_answer(response)
        assert result == "42"

    def test_decimal_answer(self):
        """Test parsing decimal answer."""
        response = "Answer: 3.14159"
        result = parse_numeric_answer(response)
        assert result == "3.14159"

    def test_answer_with_commas(self):
        """Test parsing answer with comma separators."""
        response = "Answer: 1,234,567"
        result = parse_numeric_answer(response)
        assert result == "1234567"

    def test_custom_prefix(self):
        """Test parsing with custom answer prefix."""
        response = "計算結果: 42"
        result = parse_numeric_answer(response, "計算結果")
        assert result == "42"

    def test_multiple_numbers(self):
        """Test parsing when multiple numbers present after prefix."""
        response = "Answer: First we get 10, then 20, finally 30"
        result = parse_numeric_answer(response)
        assert result == "30"  # Should get the last number

    def test_no_answer_prefix(self):
        """Test when answer prefix is not found."""
        response = "This doesn't contain the expected prefix"
        result = parse_numeric_answer(response)
        assert result == ""

    def test_prefix_no_numbers(self):
        """Test when prefix found but no numbers after it."""
        response = "Answer: No numbers here"
        result = parse_numeric_answer(response)
        assert result == ""

    def test_colon_handling(self):
        """Test proper handling of colon after prefix."""
        response = "Answer:42"
        result = parse_numeric_answer(response)
        assert result == "42"


class TestNormalizeNumber:
    """Test the number normalization function."""

    def test_remove_commas(self):
        """Test removing commas from numbers."""
        assert normalize_number("1,234") == "1234"
        assert normalize_number("1,000,000") == "1000000"

    def test_remove_trailing_zeros(self):
        """Test removing trailing zeros after decimal."""
        assert normalize_number("3.1400") == "3.14"
        assert normalize_number("5.00") == "5"

    def test_remove_trailing_decimal(self):
        """Test removing trailing decimal point."""
        assert normalize_number("42.") == "42"

    def test_preserve_significant_decimals(self):
        """Test preserving significant decimal places."""
        assert normalize_number("3.14") == "3.14"
        assert normalize_number("0.5") == "0.5"

    def test_integer_unchanged(self):
        """Test that plain integers remain unchanged."""
        assert normalize_number("42") == "42"
        assert normalize_number("0") == "0"

    def test_complex_formatting(self):
        """Test normalizing complex formatted numbers."""
        assert normalize_number("1,234.5600") == "1234.56"


class TestExtractConfidenceScore:
    """Test the confidence score extraction function."""

    def test_percentage_format(self):
        """Test extracting percentage format confidence."""
        response = "Answer: A\nConfidence: 85%"
        result = extract_confidence_score(response)
        assert result == 85

    def test_decimal_format(self):
        """Test extracting decimal format confidence."""
        response = "I am 0.95 confident in this answer"
        result = extract_confidence_score(response)
        assert result == 95

    def test_confidence_colon_format(self):
        """Test extracting with colon format."""
        response = "Confidence: 73"
        result = extract_confidence_score(response)
        assert result == 73

    def test_confident_suffix(self):
        """Test extracting with 'confident' suffix."""
        response = "I am 90% confident in this solution"
        result = extract_confidence_score(response)
        assert result == 90

    def test_decimal_confident(self):
        """Test extracting decimal with 'confident' suffix."""
        response = "I am 0.8 confident"
        result = extract_confidence_score(response)
        assert result == 80

    def test_no_confidence_mentioned(self):
        """Test default when no confidence mentioned."""
        response = "This is just a regular answer without confidence"
        result = extract_confidence_score(response)
        assert result == 100

    def test_custom_default(self):
        """Test custom default value."""
        response = "No confidence here"
        result = extract_confidence_score(response, default=50)
        assert result == 50

    def test_out_of_range_clamping(self):
        """Test clamping values to valid range."""
        response = "Confidence: 150%"
        result = extract_confidence_score(response)
        assert result == 100

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        # The pattern [Cc]onfidence matches Confidence and confidence but not CONFIDENCE
        response = "Confidence: 85%"
        result = extract_confidence_score(response)
        assert result == 85

        response = "confidence: 75%"
        result = extract_confidence_score(response)
        assert result == 75
