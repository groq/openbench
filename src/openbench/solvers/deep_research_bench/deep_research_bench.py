"""
DeepResearch Bench solver.
Adapted from the original DeepResearch Bench: https://github.com/Ayanami0730/deep_research_bench
The core logic and code is preserved, but the architecture is slightly adapted to the openbench framework. This solver is
equivalent to run_benchmark.sh in the original DeepResearch Bench implementation.
"""

from __future__ import annotations

import re
from inspect_ai.solver import solver, TaskState, Generate

from .deep_research_bench_fact import run_fact_evaluation
from .deep_research_bench_race import run_race_evaluation


def extract_citations_from_state(
    state: TaskState, research_article: str
) -> tuple[list[str], list[dict]]:
    """
    Extract citations from TaskState using multiple approaches.

    Returns:
        tuple: (citations_list, annotations_list)
    """
    citations = []
    annotations = []

    # Method 1: Check state.output.metadata
    if hasattr(state.output, "metadata") and state.output.metadata:
        metadata = state.output.metadata

        # Check for citations field
        if "citations" in metadata and isinstance(metadata["citations"], list):
            citations.extend(metadata["citations"])

        # Check for search_results (from Perplexity provider)
        if "search_results" in metadata and isinstance(
            metadata["search_results"], list
        ):
            for result in metadata["search_results"]:
                if isinstance(result, dict) and "url" in result:
                    citations.append(result["url"])
                    if "title" in result:
                        annotations.append(
                            {
                                "url": result["url"],
                                "title": result.get("title", "Unknown Title"),
                                "start_index": 0,
                                "end_index": 0,
                            }
                        )

    # Method 2: Check state.messages for assistant message with metadata
    if state.messages:
        for message in reversed(state.messages):
            if hasattr(message, "role") and message.role == "assistant":
                # Check if message has metadata or other fields
                if hasattr(message, "metadata") and message.metadata:
                    msg_metadata = message.metadata
                    if "citations" in msg_metadata:
                        citations.extend(msg_metadata.get("citations", []))
                    if "annotations" in msg_metadata:
                        annotations.extend(msg_metadata.get("annotations", []))
                break

    # Method 3: Check state.output.choices for content with citations
    if hasattr(state.output, "choices") and state.output.choices:
        for choice in state.output.choices:
            if hasattr(choice, "message") and choice.message:
                message = choice.message

                # Check if content has citations (from Perplexity provider)
                if hasattr(message, "content") and isinstance(message.content, list):
                    for content_item in message.content:
                        if (
                            hasattr(content_item, "citations")
                            and content_item.citations
                        ):
                            for citation in content_item.citations:
                                if hasattr(citation, "url") and citation.url:
                                    citations.append(citation.url)
                                    if hasattr(citation, "title") and citation.title:
                                        annotations.append(
                                            {
                                                "url": citation.url,
                                                "title": citation.title,
                                                "start_index": 0,
                                                "end_index": 0,
                                            }
                                        )

    # Method 4: Check state.output.metadata for executed_tools (Groq format)
    if hasattr(state.output, "metadata") and state.output.metadata:
        metadata = state.output.metadata

        # Check for executed_tools in metadata (Groq format)
        if "executed_tools" in metadata and isinstance(
            metadata["executed_tools"], list
        ):
            for i, tool in enumerate(metadata["executed_tools"]):
                # Create a truncated version for logging
                tool_for_logging = tool.copy() if isinstance(tool, dict) else tool
                if (
                    isinstance(tool_for_logging, dict)
                    and "search_results" in tool_for_logging
                ):
                    sr_copy = []
                    for result in tool_for_logging["search_results"]:
                        if isinstance(result, dict):
                            result_copy = result.copy()
                            if "content" in result_copy:
                                result_copy["content"] = "..."  # Truncate content field
                            sr_copy.append(result_copy)
                        else:
                            sr_copy.append(result)
                    tool_for_logging["search_results"] = sr_copy

                if isinstance(tool, dict) and "search_results" in tool:
                    search_results = tool["search_results"]
                    # Handle Groq's search_results structure: {'results': [...], 'images': ...}
                    results_list = None
                    if isinstance(search_results, dict) and "results" in search_results:
                        results_list = search_results["results"]
                    elif isinstance(search_results, list):
                        # Fallback for direct list format
                        results_list = search_results

                    if results_list and isinstance(results_list, list):
                        for result in results_list:
                            if isinstance(result, dict) and "url" in result:
                                citations.append(result["url"])
                                if "title" in result:
                                    annotations.append(
                                        {
                                            "url": result["url"],
                                            "title": result.get(
                                                "title", "Unknown Title"
                                            ),
                                            "start_index": 0,
                                            "end_index": 0,
                                        }
                                    )

    # Method 5: Parse embedded citations from text (markdown format)
    embedded_citations = extract_embedded_citations(research_article)
    citations.extend(embedded_citations)

    # Remove duplicates while preserving order
    unique_citations = []
    seen_citations = set()
    for citation in citations:
        if citation not in seen_citations:
            unique_citations.append(citation)
            seen_citations.add(citation)

    # Remove duplicate annotations
    unique_annotations = []
    seen_urls = set()
    for annotation in annotations:
        url = annotation.get("url", "")
        if url and url not in seen_urls:
            unique_annotations.append(annotation)
            seen_urls.add(url)

    return unique_citations, unique_annotations


def extract_embedded_citations(text: str) -> list[str]:
    """Extract citations from embedded markdown links in text."""
    citations = []

    # Match markdown links [title](url)
    markdown_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
    for title, url in markdown_links:
        # Only include URLs that look like web URLs
        if url.startswith(("http://", "https://")):
            citations.append(url)

    # Match bare URLs
    url_pattern = re.compile(r"https?://[^\s\)]+")
    urls = url_pattern.findall(text)
    citations.extend(urls)

    return citations


def format_citation_section(citations: list[str], annotations: list[dict]) -> str:
    """Format citations and annotations into a readable section."""
    additional_content = "\n\n"

    # Add citations from root-level citations field
    if citations:
        additional_content += "## Citations\n\n"
        for i, citation_url in enumerate(citations, 1):
            additional_content += f"{i}. {citation_url}\n"
        additional_content += "\n"

    # Add detailed annotations with titles
    if annotations:
        additional_content += "## Sources with Titles\n\n"
        for i, annotation in enumerate(annotations, 1):
            title = annotation.get("title", "Unknown Title")
            url = annotation.get("url", "")
            additional_content += f"{i}. **{title}**\n"
            additional_content += f"   - URL: {url}\n\n"

    return additional_content


@solver
def deep_research_solver():
    """
    Multi-phase solver for DeepResearch Bench evaluation.

    Orchestrates the multi-phase pipeline:
    1. Generate research article from query
    2. RACE evaluation (article quality)
    3. FACT evaluation (citation accuracy)
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            # Phase 0: Generate research article from query
            result = await generate(state)
            research_article = result.output.completion

            # Extract citations using multiple approaches
            # Note: Ensure that citations are extracted from wherever the deep research agent returns them (if it isn't simply in the output completion)
            citations, annotations = extract_citations_from_state(
                state, research_article
            )

            # Append citations to research article if found
            if citations or annotations:
                citation_section = format_citation_section(citations, annotations)
                research_article += citation_section

            # Validate article generation
            if not research_article:
                state.metadata["evaluation_error"] = (
                    "Failed to generate research article"
                )
                state.metadata["overall_score"] = 0
                state.completed = True
                return state

            # Phase 1: RACE evaluation pipeline
            race_results = await run_race_evaluation(research_article, state, generate)

            # Store RACE results in TaskState metadata
            if "error" in race_results:
                state.metadata["evaluation_error"] = race_results["error"]
                state.metadata["overall_score"] = 0
                # Set default RACE scores for error case
                state.metadata["comprehensiveness"] = 0
                state.metadata["insight"] = 0
                state.metadata["instruction_following"] = 0
                state.metadata["readability"] = 0
            else:
                state.metadata["comprehensiveness"] = race_results.get(
                    "comprehensiveness", 0
                )
                state.metadata["insight"] = race_results.get("insight", 0)
                state.metadata["instruction_following"] = race_results.get(
                    "instruction_following", 0
                )
                state.metadata["readability"] = race_results.get("readability", 0)
                state.metadata["overall_score"] = race_results.get("overall_score", 0)

            # Phase 2: FACT evaluation pipeline
            fact_results = await run_fact_evaluation(research_article, state, generate)

            # Store FACT results in TaskState metadata
            if "fact_error" in fact_results:
                # FACT failed, but continue with RACE
                state.metadata["fact_error"] = fact_results["fact_error"]
                state.metadata["total_citations"] = 0
                state.metadata["valid_citations"] = 0
                state.metadata["valid_rate"] = 0.0
            else:
                state.metadata["total_citations"] = fact_results.get(
                    "total_citations", 0
                )
                state.metadata["valid_citations"] = fact_results.get(
                    "valid_citations", 0
                )
                state.metadata["valid_rate"] = fact_results.get("valid_rate", 0.0)

            state.completed = True

        except Exception as e:
            state.metadata["evaluation_error"] = f"Solver error: {str(e)}"
            state.metadata["overall_score"] = 0
            # Set default values for all metrics
            state.metadata["comprehensiveness"] = 0
            state.metadata["insight"] = 0
            state.metadata["instruction_following"] = 0
            state.metadata["readability"] = 0
            state.metadata["total_citations"] = 0
            state.metadata["valid_citations"] = 0
            state.metadata["valid_rate"] = 0.0
            state.completed = True

        return state

    return solve
