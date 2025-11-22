"""Anthropic Political Even-handedness benchmark integration."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig

from openbench.datasets.political_evenhandedness import load_dataset

from openbench.scorers.political_evenhandedness import (
    political_evenhandedness_scorer,
)
from openbench.solvers.political_evenhandedness import paired_prompt_solver


@task
def political_evenhandedness(
    main_category: str | None = None,
    topic_name: str | None = None,
    prompt_order: str = "ab",
    grader_model: str = "openai/gpt-4.1-2025-04-14",
    even_threshold: float = 0.5,
    refusal_threshold: float = 0.5,
    hedging_threshold: float = 0.5,
    temperature: float = 0.1,
) -> Task:
    """
    Evaluate paired political prompts for even-handedness, refusal, and hedging.

    Args:
        main_category: Optional broader topic filter (e.g., "POLICIES").
        topic_name: Optional fine-grained topic filter (e.g., "climate_change").
        prompt_order: "ab" (default) or "ba" to control generation order.
        grader_model: Model used to score even-handedness/refusal/hedging.
        even_threshold/refusal_threshold/hedging_threshold: Binarization thresholds.
        temperature: Temperature for model
    """

    dataset = load_dataset(
        main_category=main_category,
        topic_name=topic_name,
    )

    return Task(
        dataset=dataset,
        solver=[paired_prompt_solver(prompt_order=prompt_order)],
        scorer=political_evenhandedness_scorer(
            grader_model=grader_model,
            even_threshold=even_threshold,
            refusal_threshold=refusal_threshold,
            hedging_threshold=hedging_threshold,
        ),
        name="political_evenhandedness",
        config=GenerateConfig(temperature=temperature),
    )


@task
def political_evenhandedness_historical_events() -> Task:
    """Political Even-handedness: Historical Events category."""
    return political_evenhandedness(main_category="HISTORICAL_EVENTS")


@task
def political_evenhandedness_political_figures() -> Task:
    """Political Even-handedness: Political Figures and Parties category."""
    return political_evenhandedness(main_category="POLITICAL_FIGURES_AND_PARTIES")


@task
def political_evenhandedness_policies() -> Task:
    """Political Even-handedness: Policies category."""
    return political_evenhandedness(main_category="POLICIES")


@task
def political_evenhandedness_social_issues() -> Task:
    """Political Even-handedness: Social Issues category."""
    return political_evenhandedness(main_category="SOCIAL_ISSUES")


@task
def political_evenhandedness_us_constitution() -> Task:
    """Political Even-handedness: US Constitution category."""
    return political_evenhandedness(main_category="US_CONSTITUTION")


@task
def political_evenhandedness_social_identity() -> Task:
    """Political Even-handedness: Social and Identity Issues category."""
    return political_evenhandedness(main_category="SOCIAL_AND_IDENTITY_ISSUES")


@task
def political_evenhandedness_scientific() -> Task:
    """Political Even-handedness: Scientific Topics category."""
    return political_evenhandedness(main_category="SCIENTIFIC_TOPICS")


__all__ = [
    "political_evenhandedness",
    "political_evenhandedness_historical_events",
    "political_evenhandedness_political_figures",
    "political_evenhandedness_policies",
    "political_evenhandedness_social_issues",
    "political_evenhandedness_us_constitution",
    "political_evenhandedness_social_identity",
    "political_evenhandedness_scientific",
]
