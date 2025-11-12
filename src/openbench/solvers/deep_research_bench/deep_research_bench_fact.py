"""
FACT evaluation module for DeepResearch Bench.
Handles citation extraction, deduplication, validation, and fact-checking pipeline. Evaluated on total citations, valid citations, and valid rate.
"""

from __future__ import annotations

import json
import asyncio
import re
import os
from typing import Any, Dict, List
from inspect_ai.solver import TaskState, Generate
from google import genai  # type: ignore[import-not-found]
from google.genai import types  # type: ignore[import-not-found]
import requests

from openbench.utils.deep_research_bench_prompts import (
    get_extract_citations_prompt,
    get_deduplicate_citations_prompt,
    get_validate_citations_prompt,
)


class FACTEvaluatorLLM:
    """Gemini evaluator LLM for FACT evaluation."""

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
        self.model = "gemini-2.5-flash-preview-05-20"  # Same as original model

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


class JinaWebScraper:
    """Web scraper using Jina API for URL content extraction."""

    def __init__(self):
        self.api_key = os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key not provided! Please set JINA_API_KEY environment variable."
            )

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape URL content using Jina API."""
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "Accept": "application/json",
                "Authorization": self.api_key,
                "X-Timeout": "60000",
                "X-With-Generated-Alt": "true",
            }
            response = requests.get(jina_url, headers=headers)

            if response.status_code != 200:
                raise Exception(
                    f"Jina AI Reader Failed for {url}: {response.status_code}"
                )

            response_dict = response.json()

            return {
                "url": response_dict["data"]["url"],
                "title": response_dict["data"]["title"],
                "description": response_dict["data"]["description"],
                "content": response_dict["data"]["content"],
                "publish_time": response_dict["data"].get("publishedTime", "unknown"),
            }

        except Exception as e:
            return {"url": url, "content": "", "error": str(e)}


def clean_urls(input_text: str) -> str:
    """Clean URLs by removing #:~:text= fragments."""
    # match [title](url) format
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def repl(match):
        title = match.group(1)
        url = match.group(2)
        # truncate #:~:text= and its content
        cut_idx = url.find("#:~:text=")
        if cut_idx != -1:
            url = url[:cut_idx]
        return f"[{title}]({url})"

    return pattern.sub(repl, input_text)


def remove_urls(input_text: str) -> str:
    """Remove URLs from [title](url) format, keep [title]."""
    # match [title](url) format, only remove the content in the parentheses, keep [title]
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    # replace [title](url) with [title]
    return pattern.sub(r"[\1]", input_text)


def clean_escape(input_text: str) -> str:
    """Clean illegal escape characters."""
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    return input_text


async def extract_citations(
    raw_article: str, language: str
) -> List[Dict[str, Any]] | Dict[str, str]:
    """
    Extract citations from research article using FACT evaluator LLM.

    Args:
        raw_article: The generated research article
        language: Language code ("zh" or "en")

    Returns:
        List of citation dictionaries with 'fact', 'ref_idx', 'url' fields
    """

    # Initialize FACT evaluator
    try:
        fact_evaluator = FACTEvaluatorLLM()
    except Exception as e:
        return {"error": f"Failed to initialize evaluator: {str(e)}"}

    user_prompt = get_extract_citations_prompt(language).format(report_text=raw_article)

    # Extract citations with retry logic
    max_retries = 3

    for retry in range(max_retries):
        try:
            response = await fact_evaluator.generate(
                user_prompt=user_prompt, system_prompt=""
            )

            if not response:
                continue

            # Clean response and parse JSON
            response = response.replace("```json", "").replace("```", "")
            response = clean_escape(response)

            citations = json.loads(response)

            # Clean URLs from facts
            for citation in citations:
                if "fact" in citation:
                    citation["fact"] = remove_urls(citation["fact"])

            return citations

        except Exception as e:
            if retry == max_retries - 1:
                return {
                    "error": f"Citation extraction failed after {max_retries} retries: {str(e)}"
                }

            sleep_time = 1.5**retry
            await asyncio.sleep(sleep_time)  # Exponential backoff

    return {"error": "Citation extraction failed"}


async def deduplicate_citations(
    citations: List[Dict[str, Any]], language: str
) -> Dict[str, Dict[str, Any]] | Dict[str, str]:
    """
    Deduplicate citations by URL using FACT evaluator LLM.

    Args:
        citations: List of citation dictionaries from extract_citations
        language: Language code ("zh" or "en")

    Returns:
        Dictionary with URLs as keys and citation groups as values
    """
    if not citations or "error" in citations:
        return {}

    # Initialize FACT evaluator
    try:
        fact_evaluator = FACTEvaluatorLLM()
    except Exception as e:
        return {"error": f"Failed to initialize evaluator for deduplication: {str(e)}"}

    # Group citations by URL
    citation_groups: Dict[str, List[Dict[str, Any]]] = {}
    for citation in citations:
        url = citation.get("url")
        if not url:
            continue
        if url not in citation_groups:
            citation_groups[url] = []
        citation_groups[url].append(citation)

    citations_groups_deduped = {}

    # Process each URL group
    for url, group in citation_groups.items():
        if len(group) == 1:
            # Single citation, no deduplication needed
            citations_groups_deduped[url] = {
                "facts": [group[0]["fact"]],
                "url_content": None,
            }
            continue

        # Multiple citations for same URL, need to deduplicate
        statements = "\n".join(
            [f"{i + 1}. {citation['fact']}" for i, citation in enumerate(group)]
        )

        user_prompt = get_deduplicate_citations_prompt(language).format(
            statements=statements
        )

        # Deduplicate with retry logic
        max_retries = 3
        deduped_idx = []

        for retry in range(max_retries):
            try:
                response = await fact_evaluator.generate(
                    user_prompt=user_prompt, system_prompt=""
                )
                deduped_idx = json.loads(
                    response.replace("```json", "").replace("```", "")
                )
                break
            except Exception:
                if retry == max_retries - 1:
                    # Failed, use default (keep all)
                    deduped_idx = [i + 1 for i in range(len(group))]
                await asyncio.sleep(1)

        # Validate deduped_idx
        if not deduped_idx or 0 in deduped_idx or len(deduped_idx) > len(group):
            deduped_idx = [i + 1 for i in range(len(group))]

        # Create deduplicated citation group
        citations_groups_deduped[url] = {
            "facts": [
                group[i - 1]["fact"] for i in deduped_idx if 1 <= i <= len(group)
            ],
            "url_content": None,
        }

    return citations_groups_deduped


async def scrape_citations(
    citations_groups_deduped: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Scrape URL content for citation groups using Jina web scraper.

    Args:
        citations_groups_deduped: Dictionary from deduplicate_citations

    Returns:
        Updated dictionary with 'url_content' fields populated
    """
    if not citations_groups_deduped:
        return citations_groups_deduped

    # Initialize web scraper
    try:
        scraper = JinaWebScraper()
    except Exception as e:
        # Mark all URLs as failed but return the structure
        for url_data in citations_groups_deduped.values():
            url_data["url_content"] = f"scraper initialization failed: {str(e)}"
        return citations_groups_deduped

    # Process each URL that needs scraping
    for url, citation_data in citations_groups_deduped.items():
        if citation_data.get("url_content") is not None:
            continue  # Already scraped

        # Scrape with retry logic
        max_retries = 3
        for retry in range(max_retries):
            try:
                result = await scraper.scrape_url(url)

                if "error" in result:
                    if retry < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        # Failed after all retries
                        citation_data["url_content"] = (
                            f"scrape failed: {result.get('error', 'unknown error')}"
                        )
                else:
                    # Success - combine title, description, and content
                    title = result.get("title", "")
                    description = result.get("description", "")
                    content = result.get("content", "")

                    citation_data["url_content"] = (
                        f"{title}\n\n{description}\n\n{content}"
                    )

                break

            except Exception as e:
                if retry < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    citation_data["url_content"] = f"scrape failed: {str(e)}"

    return citations_groups_deduped


async def validate_citations(
    citations_groups_scraped: Dict[str, Dict[str, Any]], language: str
) -> Dict[str, Dict[str, Any]]:
    """
    Validate citations against their scraped content using FACT evaluator LLM.

    Args:
        citations_groups_scraped: Dictionary from scrape_citations with url_content populated
        language: Language code ("zh" or "en")

    Returns:
        Updated dictionary with 'validate_res' and 'validate_error' fields
    """
    if not citations_groups_scraped:
        return citations_groups_scraped

    # Initialize FACT evaluator
    try:
        fact_evaluator = FACTEvaluatorLLM()
    except Exception as e:
        # Mark all validations as failed but preserve structure
        for citation_data in citations_groups_scraped.values():
            citation_data["validate_res"] = []
            citation_data["validate_error"] = (
                f"Failed to initialize evaluator: {str(e)}"
            )
        return citations_groups_scraped

    # Process each URL group
    for url, citation_data in citations_groups_scraped.items():
        reference_content = citation_data.get("url_content")
        facts = citation_data.get("facts", [])

        # Skip if no reference content or facts
        if not reference_content or not facts:
            citation_data["validate_res"] = []
            citation_data["validate_error"] = "no reference content or facts"
            continue

        # Format facts for validation prompt
        facts_str = "\n".join([f"{i + 1}. {fact}" for i, fact in enumerate(facts)])

        user_prompt = get_validate_citations_prompt(language).format(
            reference=reference_content, statements=facts_str
        )

        # Validate with retry logic
        max_retries = 3
        validation_error = None

        for retry in range(max_retries):
            try:
                response = await fact_evaluator.generate(
                    user_prompt=user_prompt, system_prompt=""
                )

                # Parse JSON response
                validate_res = json.loads(
                    response.replace("```json", "").replace("```", "")
                )

                # Adjust indices to be 0-based
                for val_item in validate_res:
                    val_item["idx"] -= 1

                # Validate response structure
                if len(validate_res) != len(facts):
                    if retry < max_retries - 1:
                        await asyncio.sleep(3)
                        continue
                    else:
                        raise ValueError(
                            f"Response length {len(validate_res)} doesn't match facts length {len(facts)}"
                        )

                # Success
                citation_data["validate_res"] = validate_res
                citation_data["validate_error"] = None
                break

            except Exception as e:
                validation_error = str(e)
                if retry < max_retries - 1:
                    await asyncio.sleep(3)
                else:
                    # All retries failed
                    citation_data["validate_res"] = []
                    citation_data["validate_error"] = validation_error

    return citations_groups_scraped


def calculate_fact_statistics(
    citations_groups_validated: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate FACT statistics from validated citation groups.

    Args:
        citations_groups_validated: Dictionary from validate_citations with validation results

    Returns:
        Dictionary with 'total_citations', 'valid_citations', and 'valid_rate' metrics
    """
    total_citations = 0
    total_valid_citations = 0

    if not citations_groups_validated:
        return {"total_citations": 0, "valid_citations": 0, "valid_rate": 0.0}

    # Count citations and valid citations
    for citation_data in citations_groups_validated.values():
        validate_error = citation_data.get("validate_error")
        validate_res = citation_data.get("validate_res", [])

        # Skip if validation failed
        if validate_error is not None:
            continue

        # Count each validation result
        for val_item in validate_res:
            result = val_item.get("result")
            if result != "unknown":  # Only count non-unknown results
                total_citations += 1
                if result == "supported":
                    total_valid_citations += 1

    # Calculate validation rate
    if total_citations > 0:
        valid_rate = total_valid_citations / total_citations
    else:
        valid_rate = 0.0

    return {
        "total_citations": total_citations,
        "valid_citations": total_valid_citations,
        "valid_rate": valid_rate,
    }


async def run_fact_evaluation(
    raw_article: str, state: TaskState, generate: Generate
) -> Dict[str, Any]:
    """
    Run the complete FACT evaluation pipeline.

    Pipeline: extract citations -> deduplicate -> scrape URLs -> validate -> calculate stats

    Args:
        raw_article: The generated research article
        state: TaskState containing metadata
        generate: Generate function (not used in FACT but kept for API consistency)

    Returns:
        Dictionary with FACT evaluation results and statistics
    """
    task_id = state.metadata.get("task_id")
    language = state.metadata.get("language", "en")

    try:
        # Phase 1: Extract citations
        citations = await extract_citations(raw_article, language)

        if isinstance(citations, dict) and "error" in citations:
            return {
                "id": task_id,
                "fact_error": citations["error"],
                "total_citations": 0,
                "valid_citations": 0,
                "valid_rate": 0.0,
            }

        # Check if no citations found
        if not citations:
            return {
                "id": task_id,
                "total_citations": 0,
                "valid_citations": 0,
                "valid_rate": 0.0,
            }

        # Phase 2: Deduplicate citations by URL
        # At this point, citations is guaranteed to be List[Dict[str, Any]] due to error check above
        citations_deduped = await deduplicate_citations(citations, language)  # type: ignore[arg-type]

        if isinstance(citations_deduped, dict) and "error" in citations_deduped:
            return {
                "id": task_id,
                "fact_error": citations_deduped["error"],
                "total_citations": 0,
                "valid_citations": 0,
                "valid_rate": 0.0,
            }

        # Phase 3: Scrape URLs
        citations_scraped = await scrape_citations(citations_deduped)  # type: ignore[arg-type]

        # Phase 4: Validate citations against scraped content
        citations_validated = await validate_citations(citations_scraped, language)

        # Phase 5: Calculate statistics
        fact_stats = calculate_fact_statistics(citations_validated)

        # Return results in format expected by scorer
        final_result = {
            "id": task_id,
            "total_citations": fact_stats.get("total_citations", 0),
            "valid_citations": fact_stats.get("valid_citations", 0),
            "valid_rate": fact_stats.get("valid_rate", 0.0),
            "citation_groups": citations_validated,
        }

        return final_result

    except Exception as e:
        return {
            "id": task_id,
            "fact_error": f"FACT evaluation failed: {str(e)}",
            "total_citations": 0,
            "valid_citations": 0,
            "valid_rate": 0.0,
        }
