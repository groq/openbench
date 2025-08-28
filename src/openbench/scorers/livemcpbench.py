"""LiveMCPBench scorer.

Provides scoring functionality for LiveMCPBench tasks which involve
evaluating LLM agents on real-world tasks using Model Context Protocol (MCP).
"""

import re
from typing import Callable
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
    metric,
    Metric,
    Value,
    SampleScore,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser, Model
from openbench.metrics.grouped import grouped


LIVEMCPBENCH_GRADER_TEMPLATE = """
You are an expert in evaluating the performance of a tool-use agent. The agent is designed to help a human user use multi-tools to complete a task. Given the user's task, the agent's final response, key points for task completion, and tool call history, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!

*Important Evaluation Criteria*:
1. You must carefully check whether the information (e.g. the coordinates of the addresses) comes from the tool call, if the agent get it from the internal knowledge, it should be considered failed.
2: Some tasks require to create files to be considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"

Key points of the task:
{metadata_context}

User Task: 
{task}

Agent Response: 
{response}

Tool call metadata: 
{tool_call_metadata}

Return only the letter (A, B, or C) with no additional text.
""".strip()


@metric
def livemcpbench_metrics() -> Metric:
    """Custom metrics for LiveMCPBench including category breakdown."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        # Calculate overall accuracy
        correct_count = sum(
            1 for sample_score in scores if sample_score.score.value == 1.0
        )
        total_count = len(scores)

        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Calculate category-wise metrics
        category_stats = {}
        for sample_score in scores:
            # Get category from score metadata
            category = (
                sample_score.score.metadata.get("category", "unknown")
                if sample_score.score.metadata
                else "unknown"
            )
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "partial": 0, "total": 0}

            category_stats[category]["total"] += 1
            if sample_score.score.value == 1.0:
                category_stats[category]["correct"] += 1
            elif sample_score.score.value == 0.5:
                category_stats[category]["partial"] += 1

        # Calculate category accuracies
        category_accuracies = {}
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                category_accuracies[f"{category}_accuracy"] = (
                    stats["correct"] / stats["total"]
                )
                category_accuracies[f"{category}_partial_accuracy"] = (
                    stats["correct"] + stats["partial"]
                ) / stats["total"]

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            **category_accuracies,
        }

    return metric_calculator


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        livemcpbench_metrics(),
        grouped(group_key="category", metric=[accuracy(), stderr()]),
    ]
)
def livemcpbench_scorer(model: str = "groq/llama-3.3-70b-versatile") -> Callable:
    """LiveMCPBench scorer using model-based grading.

    Args:
        model: The model to use for grading responses (defaults to llama 70b just for testing puproses)

    Returns:
        Scorer function for LiveMCPBench tasks
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        # Get the task from input
        task = state.input_text

        # Get the response from model output
        response = state.output.completion

        # Get expected answer from target
        expected_answer = target.text

        # Get tool call metadata from the model output
        tool_calls = []
        try:
            # Try to get tool calls from the message
            if hasattr(state.output, "message") and hasattr(
                state.output.message, "tool_calls"
            ):
                tool_calls = state.output.message.tool_calls or []
            elif hasattr(state.output, "choices") and state.output.choices:
                # Check if tool calls are in the choices
                for choice in state.output.choices:
                    if hasattr(choice, "message") and hasattr(
                        choice.message, "tool_calls"
                    ):
                        tool_calls.extend(choice.message.tool_calls or [])
            else:
                tool_calls = []
        except AttributeError:
            # If tool_calls attribute doesn't exist, set empty list
            tool_calls = []

        # Get annotator metadata from state metadata
        annotator_metadata = (
            state.metadata.get("annotator_metadata", {}) if state.metadata else {}
        )

        # Format metadata context for the grader
        metadata_context = ""
        if annotator_metadata:
            metadata_lines = []
            for key, value in annotator_metadata.items():
                metadata_lines.append(f"- {key}: {value}")
            metadata_context = "\n".join(metadata_lines)
        else:
            metadata_context = "No specific task metadata available."

        # Format the grading prompt
        grader_prompt = LIVEMCPBENCH_GRADER_TEMPLATE.format(
            task=task,
            response=response,
            tool_call_metadata=tool_calls,
            metadata_context=metadata_context,
        )

        # Create message for grading
        message = ChatMessageUser(content=grader_prompt)

        # Get grading response
        grading_response = await grader_model.generate([message])
        grading_text = grading_response.completion

        # Extract grade
        match = re.search(r"(success|failure)", grading_text)
        grade_letter = match.group(0) if match else "failure"

        # Map letter to grade and score
        grade_map = {
            "success": ("correct", 1.0),
            "failure": ("incorrect", 0.0),
        }

        grade_name, score_value = grade_map.get(grade_letter, ("incorrect", 0.0))

        # Get category from metadata
        category = (
            state.metadata.get("category", "unknown") if state.metadata else "unknown"
        )

        return Score(
            value=score_value,
            answer=response,
            metadata={
                "grade": grade_name,
                "grade_letter": grade_letter,
                "grading_response": grading_text,
                "category": category,
                "expected_answer": expected_answer,
                "metadata_context": metadata_context,
            },
        )

    return score
