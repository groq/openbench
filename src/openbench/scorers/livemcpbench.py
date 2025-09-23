"""LiveMCPBench scorer (LLM-as-Judge, baseline-aligned).

Implements the same judging approach as LiveMCPBench's baseline evaluator.
"""

import json
from typing import Callable, Any, Dict, List, Tuple, Sequence
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
from inspect_ai.model import (
    get_model,
    ChatMessageUser,
    Model,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
)
from openbench.metrics.grouped import grouped
from openbench.tools.livemcpbench.copilot.upstream_cache import get_tools_json_cached
from openbench.utils.text import (
    LIVEMCPBENCH_GRADER_USER_PROMPT,
    LIVEMCPBENCH_GRADER_SYSTEM_MSG,
    LIVEMCPBENCH_KEY_POINTS_SYSTEM_MSG,
    LIVEMCPBENCH_VERDICT_PATTERN,
)


async def identify_key_points(task: str, model: Model) -> str:
    """Identify key points from a task using a model call.

    Args:
        task: The task description to analyze
        model: The Model to use for identification

    Returns:
        String containing the identified key points
    """

    messages: list[
        ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool
    ] = [
        ChatMessageUser(
            content=f"System: {LIVEMCPBENCH_KEY_POINTS_SYSTEM_MSG}\n\nTask: {task}"
        )
    ]

    response = await model.generate(messages)
    return response.completion


_TOOL_MAP: Dict[str, Dict[str, Dict[str, Any]]] | None = None


def _load_tool_map() -> Dict[str, Dict[str, Dict[str, Any]]]:
    global _TOOL_MAP
    if _TOOL_MAP is not None:
        return _TOOL_MAP

    tools_list, _ = get_tools_json_cached()
    tool_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tool_server in tools_list:
        tools_dict = tool_server.get("tools", {})
        if not isinstance(tools_dict, dict):
            continue
        for server_name, server_tools in tools_dict.items():
            if not isinstance(server_tools, dict):
                continue
            tl_list = server_tools.get("tools", [])
            if not isinstance(tl_list, list):
                continue
            for tl in tl_list:
                if not isinstance(tl, dict):
                    continue
                name = tl.get("name")
                desc = tl.get("description", "")
                schema = tl.get("inputSchema", {})
                if not name:
                    continue
                tool_map.setdefault(server_name, {})[name] = {
                    "description": desc,
                    "inputSchema": schema,
                }

    _TOOL_MAP = tool_map
    return tool_map


def _format_tool_description(
    tool_map: Dict[str, Dict[str, Dict[str, Any]]], server_name: str, tool_name: str
) -> str:
    if server_name not in tool_map or tool_name not in tool_map[server_name]:
        return f"Tool {tool_name} not found in server {server_name}."
    info = tool_map[server_name][tool_name]
    return (
        f"Server: {server_name}\n"
        f"Tool: {tool_name}\n"
        f"Description: {info.get('description', '')}"
    )


def _extract_tool_calls_and_descriptions(
    messages: Sequence[Any],
) -> Tuple[List[str], str]:
    tool_map = _load_tool_map()
    call_lines: List[str] = []
    descriptions: List[str] = []
    unique_descriptions: set[str] = set()  # Track unique descriptions

    for message in messages or []:
        # OpenAI-style tool_calls
        if (
            isinstance(message, ChatMessageAssistant)
            and hasattr(message, "tool_calls")
            and message.tool_calls
        ):
            for tool_call in message.tool_calls:
                func_name = None
                args = None
                # function may be a string or an object with name/arguments
                if hasattr(tool_call, "function") and tool_call.function is not None:
                    func = tool_call.function
                    if isinstance(func, str):
                        func_name = func
                        args = getattr(tool_call, "arguments", None)
                    else:
                        func_name = getattr(func, "name", None) or getattr(
                            tool_call, "name", None
                        )
                        args = getattr(func, "arguments", None) or getattr(
                            tool_call, "arguments", None
                        )
                else:
                    func_name = getattr(tool_call, "name", None)
                    args = getattr(tool_call, "arguments", None)

                if func_name == "execute-tool":
                    try:
                        if isinstance(args, str):
                            call_lines.append(args)
                            parsed = json.loads(args)
                        else:
                            parsed = args if isinstance(args, dict) else {}
                            call_lines.append(json.dumps(parsed, ensure_ascii=False))
                    except Exception:
                        parsed = {}
                        call_lines.append(str(args))

                    server_name = str(parsed.get("server_name", "not_given"))
                    tool_name = str(parsed.get("tool_name", "not_given"))
                    description = _format_tool_description(
                        tool_map, server_name, tool_name
                    )
                    if description not in unique_descriptions:
                        unique_descriptions.add(description)
                        descriptions.append(description)

        # Anthropic-style content blocks with type: tool_use
        content = None
        if hasattr(message, "content"):
            content = getattr(message, "content")
        elif isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, list):
            for part in content:
                # Detect part type and fields in a robust way
                ptype = (
                    part.get("type")
                    if isinstance(part, dict)
                    else getattr(part, "type", None)
                )
                if ptype in {"tool_use", "tool", "tool_call"}:
                    name = (
                        part.get("name")
                        if isinstance(part, dict)
                        else getattr(part, "name", None)
                    )
                    # Anthropic uses 'input'; fallbacks for other shapes
                    args = (
                        part.get("input")
                        if isinstance(part, dict)
                        else getattr(part, "input", None)
                    )
                    if name == "execute-tool":
                        try:
                            if isinstance(args, str):
                                call_lines.append(args)
                                parsed = json.loads(args)
                            else:
                                parsed = args if isinstance(args, dict) else {}
                                call_lines.append(
                                    json.dumps(parsed, ensure_ascii=False)
                                )
                        except Exception:
                            parsed = {}
                            call_lines.append(str(args))

                        server_name = str(parsed.get("server_name", "not_given"))
                        tool_name = str(parsed.get("tool_name", "not_given"))
                        description = _format_tool_description(
                            tool_map, server_name, tool_name
                        )
                        if description not in unique_descriptions:
                            unique_descriptions.add(description)
                            descriptions.append(description)

    desc_text = "\n\n".join(descriptions).strip() if descriptions else ""
    return call_lines, desc_text


@metric
def livemcpbench_metrics() -> Metric:
    """Custom metrics for LiveMCPBench including category breakdown."""

    def metric_calculator(scores: list[SampleScore]) -> Value:
        # Calculate overall counts (leave overall accuracy to built-in metric)
        correct_count = sum(
            1 for sample_score in scores if sample_score.score.value == 1.0
        )
        total_count = len(scores)

        # Calculate category-wise metrics (avoid duplicating grouped accuracy)
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

        # Calculate category partial accuracies only (grouped() reports accuracy/stderr)
        category_accuracies = {}
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                category_accuracies[f"{category}_partial_accuracy"] = (
                    stats["correct"] + stats["partial"]
                ) / stats["total"]

        return {
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
        grouped(group_key="category", metric=[accuracy(), stderr()], all=False),
    ]
)
def livemcpbench_scorer(model: str = "openai/gpt-4.1-mini-2025-04-14") -> Callable:
    """LiveMCPBench scorer using model-based grading.

    Args:
        model: The model to use for grading responses (defaults to llama 70b just for testing puproses)

    Returns:
        Scorer function for LiveMCPBench tasks
    """
    grader_model: Model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        task = state.input_text
        response = (
            state.output.completion
            if state.output and state.output.completion
            else "No response generated"
        )
        expected_answer = target.text if target else "No expected answer provided"

        # Execution error passthrough
        execution_error = None
        error_message = None
        if state.metadata:
            execution_error = state.metadata.get("execution_error")
            error_message = state.metadata.get("error_message")
        try:
            tool_calls, tool_descs = _extract_tool_calls_and_descriptions(
                state.messages or []
            )
        except Exception:
            tool_calls, tool_descs = [], ""
        tool_call_history_str = (
            "\n".join(f"{i + 1}. {call}" for i, call in enumerate(tool_calls))
            if tool_calls
            else "No tool calls made"
        )

        annotator_metadata = (
            state.metadata.get("annotator_metadata", {}) if state.metadata else {}
        )

        # Use model-based identification, fallback to annotator metadata
        try:
            key_points = await identify_key_points(task, grader_model)
        except Exception:
            # Fallback to annotator metadata if model call fails
            key_points_obj = annotator_metadata.get("Steps", "")
            if isinstance(key_points_obj, list):
                key_points = "\n".join(str(s) for s in key_points_obj)
            else:
                key_points = str(key_points_obj)

        score_value = 0.0
        grading_text = ""

        if execution_error:
            score_value = 0.0
            grading_text = f"Task failed due to execution error ({execution_error}): {error_message}"

            response_with_error = (
                f"{response}\n\n[EXECUTION ERROR: {execution_error} - {error_message}]"
            )
        else:
            response_with_error = response

            try:
                # Build baseline prompt
                prompt_text = (
                    LIVEMCPBENCH_GRADER_SYSTEM_MSG
                    + "\n\n"
                    + LIVEMCPBENCH_GRADER_USER_PROMPT.format(
                        task=task,
                        key_points=key_points,
                        response=response_with_error,
                        tool_calls=tool_call_history_str,
                        tool_descriptions=tool_descs,
                    )
                )
                grading_messages: list[
                    ChatMessageSystem
                    | ChatMessageUser
                    | ChatMessageAssistant
                    | ChatMessageTool
                ] = [ChatMessageUser(content=prompt_text)]
                grading_response = await grader_model.generate(grading_messages)
                grading_text = grading_response.completion

                m = LIVEMCPBENCH_VERDICT_PATTERN.search(grading_text)
                judge_status = m.group(2).strip() if m else grading_text
                score_value = 1.0 if "success" in judge_status.lower() else 0.0

            except Exception as e:
                # Fall back to automatic failure scoring
                score_value = 0.0
                grading_text = f"Grading failed due to error: {str(e)}"

        # Get category from metadata
        category = (
            state.metadata.get("category", "unknown") if state.metadata else "unknown"
        )

        # Set grade based on score
        if score_value >= 1.0:
            grade_name = "success"
            grade_letter = "A"
        else:
            grade_name = "failure"
            grade_letter = "F"

        return Score(
            value=score_value,
            answer=response,
            metadata={
                "grade": grade_name,
                "grade_letter": grade_letter,
                "grading_response": grading_text,
                "category": category,
                "expected_answer": expected_answer,
                "key_points": key_points,
                "tool_calls": tool_call_history_str,
                "tool_descriptions": tool_descs,
                "execution_error": execution_error,
                "error_message": error_message,
            },
        )

    return score
