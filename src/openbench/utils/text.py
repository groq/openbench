import json
import tiktoken
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessage,
)

"""Text processing utilities for openbench.

This module contains helper functions for processing and normalizing text in various
benchmarking contexts, such as cleaning model outputs and standardizing answer formats.
"""


# Adapted from https://github.com/openai/simple-evals
def strip_md_latex(response: str) -> str:
    """
    Strip Markdown and LaTeX formatting artifacts from a model response.

    This is useful when evaluating generated text where visual formatting
    may interfere with exact string matching or scoring logic.

    Parameters:
        response (str): The raw response string potentially containing Markdown or LaTeX syntax.

    Returns:
        str: A cleaned string with Markdown and LaTeX formatting removed.
    """
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


# Adapted from https://github.com/openai/simple-evals
def normalize_mcq_answer(extracted_answer: str) -> str:
    """
    Normalize multiple-choice answer letters to standard Latin A-D format.

    Converts commonly used localized characters (Arabic, Bengali, Japanese)
    representing multiple-choice options to their A-D equivalents. Useful for
    consistent scoring across multilingual datasets.

    Parameters:
        extracted_answer (str): A raw answer string with potential localized MCQ letters.

    Returns:
        str: A normalized answer string using A, B, C, or D.
    """
    return (
        # In Arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


# Adapted from https://github.com/openai/simple-evals
SIMPLE_EVALS_SYSTEM_MESSAGE = "You are a helpful assistant."

# Adapted from https://github.com/openai/simple-evals. Removed the "Think step by step before answering." to make it faster and less leading.
MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{prompt}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}
""".strip()

# Adapted from https://github.com/openai/simple-evals
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)

# All the different ways "Answer" is written in different languages.
# Adapted from https://github.com/openai/simple-evals
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]

# Adapted from https://github.com/openai/simple-evals
ANSWER_PATTERN_MULTIPLE_CHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def parse_json_from_response(text: str) -> dict:
    """
    Extract and parse JSON from a model response that may contain markdown formatting.

    This function handles common patterns where models wrap JSON in markdown code blocks
    or include extra text around the JSON object.

    Parameters:
        text (str): The model response potentially containing JSON

    Returns:
        dict: Parsed JSON as a dictionary, or empty dict if parsing fails
    """
    import json
    import re

    # First try to extract from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        json_pattern = r"\{[^{}]*\}"
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return {}


def format_chat_messages(messages: list) -> str:
    """
    Format a list of chat messages into a readable conversation string.

    Handles both dictionary-style messages and ChatMessage objects from Inspect AI.

    Parameters:
        messages (list): List of messages (dicts or ChatMessage objects)

    Returns:
        str: Formatted conversation with role labels
    """
    formatted = []
    for msg in messages:
        # Handle both dict messages and ChatMessage objects
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            # ChatMessage object
            role = getattr(msg, "role", "")
            content = getattr(msg, "text", getattr(msg, "content", ""))

        if role and content:
            formatted.append(f"{role}: {content}")

    return "\n\n".join(formatted)


def parse_numeric_answer(response: str, answer_prefix: str = "Answer") -> str:
    """
    Extract a numerical answer from model response after a given prefix.

    Useful for math problems where the answer follows a pattern like "Answer: 42"
    or in other languages like "答え: 42". Extracts the last number found after
    the prefix, handling commas and decimal points.

    Parameters:
        response (str): Model's complete response
        answer_prefix (str): Prefix that precedes the answer (default: "Answer")

    Returns:
        str: Extracted numerical answer, or empty string if not found

    Examples:
        >>> parse_numeric_answer("The calculation gives us Answer: 42")
        '42'
        >>> parse_numeric_answer("答え: 3.14", "答え")
        '3.14'
        >>> parse_numeric_answer("Answer: 1,234.5")
        '1234.5'
    """
    import re

    if answer_prefix not in response:
        return ""

    # Get text after the answer prefix
    answer_text = response.split(answer_prefix)[-1].strip()

    # Remove colon if present
    if answer_text.startswith(":"):
        answer_text = answer_text[1:].strip()

    # Find all numbers (including decimals) in the string
    # Remove commas first, then extract numbers
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # Return the last number (removing trailing decimal point if present)
    return numbers[-1].rstrip(".") if numbers else ""


def normalize_number(value: str) -> str:
    """
    Normalize a numerical string for comparison.

    Removes commas, trailing zeros after decimal points, and trailing decimal
    points. Useful for comparing numerical answers where formatting may vary.

    Parameters:
        value (str): String representation of a number

    Returns:
        str: Normalized number string

    Examples:
        >>> normalize_number("1,234")
        '1234'
        >>> normalize_number("3.1400")
        '3.14'
        >>> normalize_number("5.0")
        '5'
        >>> normalize_number("42.")
        '42'
    """
    # Remove commas
    value = value.replace(",", "")

    # If it has a decimal point, remove trailing zeros and the decimal point if needed
    if "." in value:
        value = value.rstrip("0").rstrip(".")

    return value


def extract_confidence_score(response: str, default: int = 100) -> int:
    """
    Extract a confidence score from model response.

    Looks for patterns like "Confidence: 85%", "confidence: 0.85", etc.
    Handles both percentage (0-100) and decimal (0-1) formats.

    Parameters:
        response (str): Model response potentially containing confidence score
        default (int): Default confidence to return if none found (default: 100)

    Returns:
        int: Confidence score between 0 and 100

    Examples:
        >>> extract_confidence_score("Answer: A\\nConfidence: 85%")
        85
        >>> extract_confidence_score("I am 0.95 confident in this answer")
        95
        >>> extract_confidence_score("No confidence mentioned")
        100
    """
    import re

    patterns = [
        r"[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%",  # Confidence: 85%
        r"[Cc]onfidence:\s*(\d+)",  # Confidence: 85
        r"[Cc]onfidence:\s*(0?\.\d+)",  # Confidence: 0.85
        r"(\d+(?:\.\d+)?)\s*%\s*[Cc]onfident",  # 85% confident
        r"(0?\.\d+)\s*[Cc]onfident",  # 0.85 confident
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            value = float(match.group(1))
            # Convert to percentage if it's a decimal
            if value <= 1:
                return int(value * 100)
            # Clamp to valid range
            return min(100, max(0, int(value)))

    return default


def str_to_chat_messages(messages_str: str) -> list[ChatMessage]:
    """
    Convert a string to a list of chat messages.

    Parameters:
        messages_str (str): The string to convert

    Returns:
        list[ChatMessage]: The list of chat messages
    """
    message_mapping = {
        "system": ChatMessageSystem,
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
    }
    messages = json.loads(messages_str)
    return [
        message_mapping[message["role"]](content=message["content"])
        for message in messages
    ]


def get_token_count(text: str, model: str = "gpt-4o") -> int:
    """
    Get the token count of a text.
    """
    return len(tiktoken.encoding_for_model(model).encode(text))


def get_chatml_tok_cnt(chat_messages_str: str) -> int:
    """
    Get the token count of a string in chatml format.
    """
    messages = json.loads(chat_messages_str)
    total = 3
    for message in messages:
        total += 3
        for key, value in message.items():
            total += get_token_count(value)
            if key == "name":
                total += 1
    return total
