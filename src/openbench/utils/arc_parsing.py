import json
import re
from typing import List, Optional


def backscan_json_parser(log_str: str) -> Optional[List[List[int]]]:
    """
    Extract the last valid JSON substring that matches the List[List] structure
    from the given log string by scanning backwards from the end.

    Parameters:
        log_str (str): The full log output text.

    Returns:
        The parsed List[List] object if found and valid, otherwise None.
    """
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in ("]", "}"):
            last_bracket_idx = i
            closing_bracket = char
            break

    if last_bracket_idx == -1:
        return None

    opening_bracket = "[" if closing_bracket == "]" else "{"

    bracket_counter = 1  # Start at 1 to account for the found closing bracket
    start_idx = -1

    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break

    if start_idx == -1:
        return None

    json_candidate = log_str[start_idx : last_bracket_idx + 1]

    try:
        parsed_json = json.loads(json_candidate)

        # Validate the structure: must be a non-empty list of lists.
        if (
            isinstance(parsed_json, list)
            and parsed_json
            and all(isinstance(row, list) for row in parsed_json)
        ):
            return parsed_json
        else:
            return None

    except json.JSONDecodeError:
        return None


def extract_from_boxed(log_str: str) -> Optional[List[List[int]]]:
    """
    Extracts JSON from a LaTeX-style \\boxed{} command in a string.
    """
    match = re.search(r"\\boxed\{(.*?)\}", log_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            # The content inside boxed is often a list of lists
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all(
                isinstance(i, list) for i in parsed_json
            ):
                return parsed_json
        except json.JSONDecodeError:
            # If json.loads fails, it's not the JSON we're looking for
            pass
    return None


def parse_arc_response(response: str) -> Optional[List[List[int]]]:
    """
    Parse an ARC-AGI model response to extract the predicted grid.

    Tries multiple parsing strategies in order:
    1. Extract from LaTeX \\boxed{} format
    2. Backscan for the last valid JSON array

    Args:
        response: Raw model response text

    Returns:
        Parsed grid as List[List[int]] or None if parsing fails
    """
    parsing_attempts = [extract_from_boxed, backscan_json_parser]

    for parser in parsing_attempts:
        result = parser(response)
        if result is not None:
            # Validate the structure: must be list of lists
            if isinstance(result, list) and all(
                isinstance(row, list) for row in result
            ):
                return result  # Return immediately on first success and validation

    return None
