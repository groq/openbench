"""
RACE evaluation module for DeepResearch Bench.
Handles article quality evaluation, evaluated on comprehensiveness, insight, instruction-following, and readability.
"""

from __future__ import annotations

import json
import asyncio
import re
import os
from typing import Any, Dict, Optional
from inspect_ai.solver import TaskState, Generate
from inspect_ai.model import ChatMessageUser
from google import genai  # type: ignore[import-not-found]
from google.genai import types  # type: ignore[import-not-found]

from openbench.datasets.deep_research_bench import get_criteria, get_reference_clean
from openbench.utils.deep_research_bench_prompts import (
    get_clean_article_prompt,
    get_merged_score_prompt,
)


class RACEEvaluatorLLM:
    """Gemini evaluator LLM for RACE evaluation."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided! Please set GEMINI_API_KEY environment variable."
            )

        # Configure client (following original api.py)
        self.client = genai.Client(
            api_key=self.api_key, http_options={"timeout": 600000}
        )
        self.model = "gemini-2.5-pro-preview-06-05"  # Same as original model

    async def generate(self, user_prompt: str, system_prompt: str = "") -> str:
        """Generate text response using Gemini model."""
        # Build request content
        contents = []

        # Add system prompt if provided
        if system_prompt:
            contents.append({"role": "system", "parts": [{"text": system_prompt}]})

        # Add user prompt
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=16000)
                ),
            )

            return response.text

        except Exception as e:
            raise Exception(f"Failed to generate content with evaluator: {str(e)}")


def extract_json_from_markdown(text):
    """
    Extract JSON from a markdown text that may contain ```json ... ``` blocks.

    Implements 6 robust fallback methods for maximum reliability:
    - Method 0: Direct JSON parsing (clean responses)
    - Method 1: String-based code block extraction
    - Method 2: Regex-based code block extraction
    - Method 3: Entire text as JSON
    - Method 4: Brace-balanced extraction
    - Method 5: Simple start/end brace matching
    - Method 6: Keyword pattern matching (emergency fallback)

    Args:
        text (str): The input text which might contain JSON blocks

    Returns:
        str or None: The extracted JSON string or None if not found/not valid
    """
    if not isinstance(text, str) or not text or not text.strip():
        return None

    # Normalize text to handle various whitespace issues
    text = text.strip()

    # Method 0: Try to parse the complete text directly
    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            pass

    # Method 1: Extract JSON from code blocks
    if "```json" in text and "```" in text[text.find("```json") + 7 :]:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass

    # Method 2: Regex matching for code blocks
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    # Method 3: Try parsing entire text as JSON
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass

    # Method 4: Extract content within outermost curly braces
    start = text.find("{")
    if start != -1:
        level = 0
        for i, char in enumerate(text[start:]):
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    end = start + i + 1
                    potential_json = text[start:end]
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        pass
                    break

    # Method 5: Final fallback method, try simple start and end brace matching
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        potential_json = text[start : end + 1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # All methods failed, try final fallback method
            pass

    # Method 6: Special fallback method - build simplified JSON object by keyword pattern matching
    if (
        "comprehensiveness" in text
        and "article_1_score" in text
        and "article_2_score" in text
    ):
        try:
            dimensions = [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ]
            result = {}

            for dim in dimensions:
                if dim in text:
                    result[dim] = []

                    # Search for all scoring entries under this dimension
                    # First find the dimension starting position
                    dim_start = text.find(f'"{dim}"')
                    if dim_start == -1:
                        dim_start = text.find(f"'{dim}'")
                    if dim_start == -1:
                        dim_start = text.find(dim)

                    if dim_start != -1:
                        # Determine dimension end position (next dimension start or end of text)
                        next_dim_start = len(text)
                        for next_dim in dimensions:
                            if next_dim != dim:
                                pos = text.find(f'"{next_dim}"', dim_start)
                                if pos == -1:
                                    pos = text.find(f"'{next_dim}'", dim_start)
                                if pos == -1:
                                    pos = text.find(next_dim, dim_start + len(dim))
                                if pos != -1 and pos < next_dim_start:
                                    next_dim_start = pos

                        # Extract content for this dimension
                        dim_content = text[dim_start:next_dim_start]

                        # Use regex to find all "criterion", "article_1_score" and "article_2_score"
                        criterion_matches = re.finditer(
                            r'"criterion"\s*:\s*"([^"]+)"', dim_content
                        )
                        score1_matches = re.finditer(
                            r'"article_1_score"\s*:\s*(\d+\.?\d*)', dim_content
                        )
                        score2_matches = re.finditer(
                            r'"article_2_score"\s*:\s*(\d+\.?\d*)', dim_content
                        )

                        # Convert to lists for multiple access
                        criteria = [m.group(1) for m in criterion_matches]
                        scores1 = [float(m.group(1)) for m in score1_matches]
                        scores2 = [float(m.group(1)) for m in score2_matches]

                        # Combine into scoring entries
                        for i in range(min(len(criteria), len(scores1), len(scores2))):
                            result[dim].append(
                                {
                                    "criterion": criteria[i],
                                    "article_1_score": scores1[i],
                                    "article_2_score": scores2[i],
                                }
                            )

            # Validate if we successfully extracted scoring data
            if any(len(scores) > 0 for scores in result.values()):
                return json.dumps(result)
        except Exception:
            # Fallback extraction method failed, log error but continue trying
            pass

    # All methods failed, return None
    return None


def calculate_weighted_scores(llm_output_json, criteria_data, language="en"):
    """
    Calculates weighted scores based on LLM output and criteria weights.

    Args:
        llm_output_json: JSON output from LLM with scoring data
        criteria_data: Criteria configuration with dimension weights
        language: Language of the evaluation (default: "en")

    Returns:
        Dictionary with weighted scores for target and reference models
    """
    results = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0},
    }
    total_target_score = 0.0
    total_reference_score = 0.0

    # Get dimension weights
    dimension_weights = criteria_data.get("dimension_weight", {})
    task_id = criteria_data.get("id", "Unknown")

    # Check if criterions exist
    if "criterions" not in criteria_data or not criteria_data["criterions"]:
        raise ValueError(
            f"ID: {task_id} - Missing required criterions data, cannot calculate weighted scores"
        )

    # Create a mapping from criterion text to weight for easier lookup
    criterion_weights = {}
    for dim, criterions in criteria_data.get("criterions", {}).items():
        criterion_weights[dim] = {
            crit["criterion"]: crit["weight"] for crit in criterions
        }

    # Record all unmatched criteria for warnings
    unmatched_criteria = set()

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            continue

        if dim not in dimension_weights:
            continue

        if dim not in criterion_weights:
            continue

        dim_target_weighted_sum = 0.0
        dim_reference_weighted_sum = 0.0
        dim_total_weight = 0.0

        dim_criteria_map = criterion_weights.get(dim, {})

        # Skip dimension if no criteria mapping exists
        if not dim_criteria_map:
            continue

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue

            criterion_text_raw = score_item.get("criterion")
            criterion_text = (
                criterion_text_raw.strip()
                if isinstance(criterion_text_raw, str)
                else None
            )

            # Check different score field formats
            article_1_score_raw = score_item.get("article_1_score")
            article_2_score_raw = score_item.get("article_2_score")
            target_score_raw = score_item.get("target_score")  # Single scoring mode

            # If target_score exists but article_1_score doesn't, assign target_score to article_1_score
            if target_score_raw is not None and article_1_score_raw is None:
                article_1_score_raw = target_score_raw

            try:
                article_1_score = (
                    float(article_1_score_raw)
                    if article_1_score_raw is not None
                    else None
                )
                article_2_score = (
                    float(article_2_score_raw)
                    if article_2_score_raw is not None
                    else None
                )
            except (ValueError, TypeError):
                continue

            # If criterion_text exists and article_1_score is not None
            if criterion_text and article_1_score is not None:
                # Check for exact match
                weight = dim_criteria_map.get(criterion_text)

                # If exact match not found, try fuzzy matching
                if weight is None:
                    # First try case-insensitive matching
                    criterion_lower = criterion_text.lower()
                    for key, val in dim_criteria_map.items():
                        if key.lower() == criterion_lower:
                            weight = val
                            break

                    # If still not found, try substring matching
                    if weight is None:
                        for key, val in dim_criteria_map.items():
                            # Check if criterion text contains criteria or criteria contains criterion text
                            if (
                                criterion_lower in key.lower()
                                or key.lower() in criterion_lower
                            ):
                                weight = val
                                break

                # If still no match found, record and use average weight
                if weight is None:
                    unmatched_criteria.add(f"{dim}:{criterion_text}")
                    # Calculate average weight for this dimension's criteria
                    weight = sum(dim_criteria_map.values()) / len(dim_criteria_map)

                dim_target_weighted_sum += article_1_score * weight
                dim_total_weight += weight

                # Only calculate reference part if article_2_score exists
                if article_2_score is not None:
                    dim_reference_weighted_sum += article_2_score * weight

        if dim_total_weight > 0:
            dim_target_avg = dim_target_weighted_sum / dim_total_weight
            # Only calculate reference average if reference scores exist
            dim_reference_avg = (
                dim_reference_weighted_sum / dim_total_weight
                if article_2_score is not None
                else 0
            )
        else:
            dim_target_avg = 0
            dim_reference_avg = 0

        results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim, 0)
        total_target_score += dim_target_avg * dim_weight
        total_reference_score += dim_reference_avg * dim_weight

    results["target"]["total"] = total_target_score
    results["reference"]["total"] = total_reference_score

    return results


def format_criteria_list(criteria_data):
    """Format evaluation criteria list as JSON string, without weight information"""
    criteria_for_prompt = {}
    criterions_dict = criteria_data.get("criterions", {})

    for dim, criterions_list in criterions_dict.items():
        if not isinstance(criterions_list, list):
            continue

        criteria_for_prompt[dim] = []
        for crit_item in criterions_list:
            if (
                isinstance(crit_item, dict)
                and "criterion" in crit_item
                and "explanation" in crit_item
            ):
                criteria_for_prompt[dim].append(
                    {
                        "criterion": crit_item["criterion"],
                        "explanation": crit_item["explanation"],
                    }
                )

    try:
        return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)
    except TypeError as e:
        raise ValueError(f"Failed to serialize criteria to JSON: {e}")


async def clean_article(
    raw_article: str, language: str, generate: Generate, state: TaskState
) -> Optional[str]:
    """Clean article by removing citations using Gemini evaluator LLM, following DeepResearch Bench logic."""

    user_prompt = get_clean_article_prompt(language).format(article=raw_article)

    # Use Gemini evaluator LLM for cleaning (not the user's model)
    try:
        evaluator = RACEEvaluatorLLM()

        # Try to clean the article with retries
        max_retries = 5
        min_valid_length = 100

        for retry in range(max_retries):
            try:
                cleaned_text = await evaluator.generate(
                    user_prompt=user_prompt, system_prompt=""
                )

                # Validate result
                if is_valid_cleaning_result(cleaned_text, min_valid_length):
                    return cleaned_text.strip()

            except Exception as e:
                # Check if token limit error
                if is_token_limit_error(e):
                    return await chunk_clean_article(
                        raw_article, language, generate, state
                    )

        # All retries failed
        return None

    except Exception:
        return None


def is_valid_cleaning_result(text: str, min_length: int = 100) -> bool:
    """Check if cleaning result is valid."""
    return bool(text and len(text.strip()) >= min_length)


def is_token_limit_error(error: Exception) -> bool:
    """Check if error is related to token limit."""
    error_str = str(error).lower()
    return "tokens" in error_str and "less than" in error_str


async def chunk_clean_article(
    raw_article: str, language: str, generate: Generate, state: TaskState
) -> Optional[str]:
    """
    Split long article into two chunks for processing, then combine results.
    """
    print("Attempting to process article in 2 chunks")

    # Split article into 2 chunks
    chunks = []
    chunk_size = len(raw_article) // 2

    for i in range(2):
        start = i * chunk_size
        # Last chunk goes to end of article
        end = len(raw_article) if i == 1 else chunk_size

        # Split at sentence boundaries (first chunk only)
        if i == 0:
            # Look for sentence boundaries near chunk_size
            search_start = max(0, end - 200)

            for j in range(end, search_start, -1):
                if j < len(raw_article) and raw_article[j] in [
                    ".",
                    "?",
                    "!",
                    "",
                    "",
                    "",
                    "\n",
                ]:
                    end = j + 1
                    break

        chunk = raw_article[start:end]
        chunks.append(chunk)

    # Clean each chunk
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            # Use regular clean_article function but prevent infinite recursion
            clean_result = await clean_article_single_attempt(
                chunk, language, generate, state
            )

            # If returns None and chunk is too large, indicates token limit error
            if clean_result is None and len(chunk) > 200000:
                print(f"Chunk {i + 1} too large, cannot process")
                return None

            cleaned_chunks.append(clean_result if clean_result else "")

        except Exception as e:
            print(f"Failed to clean chunk {i + 1}/2: {e}")
            return None

    # Merge results
    print("All chunks processed, merging results")
    merged_article = "".join(cleaned_chunks)
    return merged_article


async def clean_article_single_attempt(
    raw_article: str, language: str, generate: Generate, state: TaskState
) -> Optional[str]:
    """Clean article with single attempt (no recursion) for chunking."""
    user_prompt = get_clean_article_prompt(language).format(article=raw_article)

    try:
        # Save original messages
        original_messages = state.messages.copy() if state.messages else []

        # Add cleaning prompt as new user message
        cleaning_message = ChatMessageUser(content=user_prompt)
        state.messages = [cleaning_message]

        result = await generate(state)

        # Restore original messages
        state.messages = original_messages
        cleaned_text = result.output.completion

        if is_valid_cleaning_result(cleaned_text):
            return cleaned_text.strip()
    except Exception:
        # Restore messages in case of error
        try:
            state.messages = original_messages
        except Exception:
            pass

    return None


async def run_race_evaluation(
    raw_article: str, state: TaskState, generate: Generate
) -> Dict[str, Any]:
    """Run the RACE evaluation pipeline following DeepResearch Bench."""
    # Get metadata
    task_id = state.metadata.get("task_id")
    prompt = state.input_text
    language = state.metadata.get("language", "en")

    # Initialize evaluator LLM
    try:
        race_evaluator = RACEEvaluatorLLM()
    except Exception as e:
        return {"error": f"Failed to initialize evaluator: {str(e)}"}

    # Step 1: Clean article by removing citations (needed for RACE evaluation)
    cleaned_article = await clean_article(raw_article, language, generate, state)

    if cleaned_article is None:
        return {"error": "Failed to clean article"}

    # Step 2A: Load and match criteria/reference data by task ID
    # Load datasets
    reference_dataset = get_reference_clean()
    criteria_dataset = get_criteria()

    # Create lookup maps by task_id (more reliable than prompt text matching)
    reference_articles_map = {}
    for sample in reference_dataset:
        if sample.metadata:
            task_id_key = sample.metadata.get("task_id")
            reference_articles_map[task_id_key] = {
                "article": sample.target
                if hasattr(sample, "target")
                else sample.metadata.get("article", "")
            }

    criteria_map = {}
    for sample in criteria_dataset:
        if sample.metadata:
            task_id_key = sample.metadata.get("task_id")
            criteria_map[task_id_key] = sample.metadata  # Criteria data is in metadata

    # Data retrieval and validation using task_id (more reliable)
    if task_id not in reference_articles_map:
        return {"id": task_id, "prompt": prompt, "error": "Reference article not found"}

    if task_id not in criteria_map:
        return {
            "id": task_id,
            "prompt": prompt,
            "error": "Evaluation criteria not found",
        }

    # Get the matched data using task_id
    reference_article_data = reference_articles_map[task_id]
    criteria_data = criteria_map[task_id]

    reference_article = reference_article_data.get("article", "")

    # Phase 2B: Format evaluation criteria list in JSON
    try:
        criteria_list_str = format_criteria_list(criteria_data)
    except ValueError as e:
        return {
            "id": task_id,
            "prompt": prompt,
            "error": f"Failed to format criteria: {str(e)}",
        }

    # Phase 2C: Generate scoring prompt with target/reference articles
    user_prompt = get_merged_score_prompt(language).format(
        task_prompt=prompt,
        article_1=cleaned_article,
        article_2=reference_article,
        criteria_list=criteria_list_str,
    )

    # Phase 2D: Call LLM evaluator with retry logic
    llm_response_str = None
    llm_output_json = None
    success = False
    retry_count = 0
    max_retries = 10

    while retry_count < max_retries and not success:
        try:
            # Call evaluator LLM
            llm_response_str = await race_evaluator.generate(user_prompt)

            # Phase 2E: Parse JSON response and validate dimensions
            # Extract JSON from response
            json_str_extracted = extract_json_from_markdown(llm_response_str)
            if not json_str_extracted:
                raise ValueError("Failed to extract JSON from LLM response")

            llm_output_json = json.loads(json_str_extracted)

            # Check if all required dimensions exist
            expected_dims = [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ]
            if not all(dim in llm_output_json for dim in expected_dims):
                missing_dims = [
                    dim for dim in expected_dims if dim not in llm_output_json
                ]
                raise ValueError(f"Missing expected dimensions: {missing_dims}")

            # All checks passed
            success = True

        except Exception:
            retry_count += 1
            if retry_count < max_retries:
                # Exponential backoff
                await asyncio.sleep(1.5**retry_count)
            else:
                # Failed after all retries
                return {
                    "id": task_id,
                    "prompt": prompt,
                    "error": f"Failed to get valid response after {max_retries} retries",
                    "model_output": llm_response_str[:500]
                    if llm_response_str
                    else "No response",
                }

    if not success:
        return {
            "id": task_id,
            "prompt": prompt,
            "error": f"Failed to get valid response after {max_retries} retries",
            "model_output": llm_response_str[:500]
            if llm_response_str
            else "No response",
        }

    # Phase 2F: Calculate weighted scores using score_calculator
    try:
        scores = calculate_weighted_scores(llm_output_json, criteria_data, language)

        # Phase 2G: Normalize scores (target/(target+reference))
        # Calculate overall score = target / (target + reference)
        target_total = scores["target"]["total"]
        reference_total = scores["reference"]["total"]
        overall_score = 0
        if target_total + reference_total > 0:
            overall_score = target_total / (target_total + reference_total)

        # Calculate normalized dimension scores
        normalized_dims = {}
        for dim in [
            "comprehensiveness",
            "insight",
            "instruction_following",
            "readability",
        ]:
            dim_key = f"{dim}_weighted_avg"
            if dim_key in scores["target"]["dims"]:
                target_score = scores["target"]["dims"][dim_key]
                reference_score = scores["reference"]["dims"][dim_key]
                if target_score + reference_score > 0:
                    normalized_dims[dim] = target_score / (
                        target_score + reference_score
                    )
                else:
                    normalized_dims[dim] = 0
            else:
                normalized_dims[dim] = 0

    except Exception as e:
        return {
            "id": task_id,
            "prompt": prompt,
            "error": f"Error calculating scores: {str(e)}",
        }

    # Prepare final result with simplified format
    final_result = {
        "id": task_id,
        "prompt": prompt,
        "comprehensiveness": normalized_dims.get("comprehensiveness", 0),
        "insight": normalized_dims.get("insight", 0),
        "instruction_following": normalized_dims.get("instruction_following", 0),
        "readability": normalized_dims.get("readability", 0),
        "overall_score": overall_score,
    }

    return final_result
