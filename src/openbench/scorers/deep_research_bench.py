"""
DeepResearch Bench scorer.

Adapted from the original DeepResearch Bench: https://github.com/Ayanami0730/deep_research_bench
"""

from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    metric,
    Metric,
    SampleScore,
    Value,
)
from inspect_ai.solver import TaskState
from typing import List


# FACT Metrics
@metric
def total_citations() -> Metric:
    """Metric that calculates average total citations per sample."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        citation_counts = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                citation_counts.append(
                    sample_score.score.metadata.get("total_citations", 0)
                )

        return sum(citation_counts) / len(citation_counts) if citation_counts else 0.0

    return metric_fn


@metric
def total_valid_citations() -> Metric:
    """Metric that calculates average valid citations per sample."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        valid_counts = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                valid_counts.append(
                    sample_score.score.metadata.get("valid_citations", 0)
                )

        return sum(valid_counts) / len(valid_counts) if valid_counts else 0.0

    return metric_fn


@metric
def valid_rate() -> Metric:
    """Metric that calculates citation validation rate."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        total_citations_sum = 0
        total_valid_sum = 0

        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                total_citations_sum += sample_score.score.metadata.get(
                    "total_citations", 0
                )
                total_valid_sum += sample_score.score.metadata.get("valid_citations", 0)

        return total_valid_sum / total_citations_sum if total_citations_sum > 0 else 0.0

    return metric_fn


# RACE Metrics
@metric
def comprehensiveness() -> Metric:
    """Metric that calculates average comprehensiveness score."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        comprehensiveness_scores = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                # Results are stored directly in metadata by our solver
                comprehensiveness_scores.append(
                    sample_score.score.metadata.get("comprehensiveness", 0.0)
                )

        return (
            sum(comprehensiveness_scores) / len(comprehensiveness_scores)
            if comprehensiveness_scores
            else 0.0
        )

    return metric_fn


@metric
def insight() -> Metric:
    """Metric that calculates average insight score."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        insight_scores = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                insight_scores.append(sample_score.score.metadata.get("insight", 0.0))

        return sum(insight_scores) / len(insight_scores) if insight_scores else 0.0

    return metric_fn


@metric
def instruction_following() -> Metric:
    """Metric that calculates average instruction following score."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        instruction_scores = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                instruction_scores.append(
                    sample_score.score.metadata.get("instruction_following", 0.0)
                )

        return (
            sum(instruction_scores) / len(instruction_scores)
            if instruction_scores
            else 0.0
        )

    return metric_fn


@metric
def readability() -> Metric:
    """Metric that calculates average readability score."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        readability_scores = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                readability_scores.append(
                    sample_score.score.metadata.get("readability", 0.0)
                )

        return (
            sum(readability_scores) / len(readability_scores)
            if readability_scores
            else 0.0
        )

    return metric_fn


@metric
def overall_score() -> Metric:
    """Metric that calculates average overall RACE score."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        overall_scores = []
        for sample_score in scores:
            if (
                hasattr(sample_score, "score")
                and hasattr(sample_score.score, "metadata")
                and sample_score.score.metadata
            ):
                overall_scores.append(
                    sample_score.score.metadata.get("overall_score", 0.0)
                )

        return sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    return metric_fn


@scorer(
    metrics=[
        # RACE metrics
        comprehensiveness(),
        insight(),
        instruction_following(),
        readability(),
        overall_score(),
        # FACT metrics
        total_citations(),
        total_valid_citations(),
        valid_rate(),
    ]
)
def deep_research_scorer():
    """
    Scorer for DeepResearch Bench tasks.

    Since the solver handles all evaluation logic, this scorer primarily:
    1. Extracts the overall_score from solver metadata
    2. Provides custom metrics for RACE components
    3. Stores detailed evaluation results in metadata
    """

    async def score(state: TaskState, target: Target) -> Score:
        del target  # Unused parameter
        """
        Score a DeepResearch Bench sample.
        
        Args:
            state: TaskState containing solver results with evaluation scores
            target: Not used (no ground truth target for generative evaluation)
            
        Returns:
            Score with overall_score as value and detailed results in metadata
        """
        try:
            # Extract scores from metadata (where our solver stores them)
            overall_score_val = state.metadata.get("overall_score", 0.0)
            evaluation_error = state.metadata.get("evaluation_error")

            if evaluation_error:
                # Handle evaluation errors
                return Score(
                    value=0.0,
                    answer=state.output.completion or "No completion",
                    explanation=f"Evaluation error: {evaluation_error}",
                    metadata={
                        # RACE metrics
                        "overall_score": 0.0,
                        "comprehensiveness": 0.0,
                        "insight": 0.0,
                        "instruction_following": 0.0,
                        "readability": 0.0,
                        # FACT metrics
                        "total_citations": 0,
                        "valid_citations": 0,
                        "valid_rate": 0.0,
                        # Error info
                        "evaluation_error": evaluation_error,
                        "is_correct": False,
                    },
                )

            # Create score with detailed metadata including FACT metrics
            fact_explanation = ""
            if state.metadata.get("total_citations", 0) > 0:
                fact_explanation = f" | FACT scores: {state.metadata.get('valid_citations', 0)}/{state.metadata.get('total_citations', 0)} citations valid ({state.metadata.get('valid_rate', 0):.3f} citation accuracy)"

            return Score(
                value=float(overall_score_val),
                answer=state.output.completion
                or "Research article evaluation completed",
                explanation=f"RACE scores: {overall_score_val:.3f} "
                f"(Comprehensiveness: {state.metadata.get('comprehensiveness', 0):.3f}, "
                f"Insight: {state.metadata.get('insight', 0):.3f}, "
                f"Instruction Following: {state.metadata.get('instruction_following', 0):.3f}, "
                f"Readability: {state.metadata.get('readability', 0):.3f})"
                f"{fact_explanation}",
                metadata={
                    # RACE metrics
                    "overall_score": overall_score_val,
                    "comprehensiveness": state.metadata.get("comprehensiveness", 0.0),
                    "insight": state.metadata.get("insight", 0.0),
                    "instruction_following": state.metadata.get(
                        "instruction_following", 0.0
                    ),
                    "readability": state.metadata.get("readability", 0.0),
                    # FACT metrics
                    "total_citations": state.metadata.get("total_citations", 0),
                    "valid_citations": state.metadata.get("valid_citations", 0),
                    "valid_rate": state.metadata.get("valid_rate", 0.0),
                    # General metadata
                    "task_id": state.metadata.get("task_id"),
                },
            )

        except Exception as e:
            # Fallback for unexpected errors
            return Score(
                value=0.0,
                answer=state.output.completion or "No completion",
                explanation=f"Scorer error: {str(e)}",
                metadata={
                    # Default RACE metrics
                    "overall_score": 0.0,
                    "comprehensiveness": 0.0,
                    "insight": 0.0,
                    "instruction_following": 0.0,
                    "readability": 0.0,
                    # Default FACT metrics
                    "total_citations": 0,
                    "valid_citations": 0,
                    "valid_rate": 0.0,
                    # Error info
                    "scorer_error": str(e),
                },
            )

    return score
