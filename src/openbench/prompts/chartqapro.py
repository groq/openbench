"""
ChartQAPro prompt templates from the official paper (Tables 6, 7, 8).

Three prompting strategies:
- Direct: Concise answer only
- Chain-of-Thought (CoT): Step-by-step reasoning with "The answer is X"
- Program-of-Thought (PoT): Executable Python code

Reference: https://arxiv.org/abs/2504.05506
"""

from __future__ import annotations
from typing import Literal

PromptStrategy = Literal["direct", "cot", "pot"]

# ============================================================================
# Direct Prompting (Table 6)
# ============================================================================

DIRECT_PROMPTS: dict[str, str] = {
    "Factoid": """You are given a factoid question that you need to answer based on the provided image.
Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!""",
    "Multi Choice": """You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
Your answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!""",
    "Hypothetical": """You are given a hypothetical question that you need to answer based on the provided image.
Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!""",
    "Fact Checking": """You are given a fact statement that you need to assess based on the provided image.
Your answer should be either true or false (without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!""",
    "Conversational": """You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
Your answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to generate the final answer only without any additional text!""",
}

# ============================================================================
# Chain-of-Thought Prompting (Table 7)
# ============================================================================

COT_PROMPTS: dict[str, str] = {
    "Factoid": """You are given a factoid question that you need to answer based on the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to think step-by-step and format the final answer in a separate sentence like "The answer is X" """,
    "Multi Choice": """You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
You need to think step-by-step, but your final answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to think step-by-step and format the final answer in a separate sentence like "The answer is X" """,
    "Hypothetical": """You are given a hypothetical question that you need to answer based on the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to think step-by-step and format the final answer in a separate sentence like "The answer is X" """,
    "Fact Checking": """You are given a fact statement that you need to assess based on the information in the provided image.
You need to think step-by-step, but your final answer should be either true or false (without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to think step-by-step and format the final answer in a separate sentence like "The answer is X" """,
    "Conversational": """You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
You need to think step-by-step, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to think step-by-step and format the final answer in a separate sentence like "The answer is X" """,
}

# ============================================================================
# Program-of-Thought Prompting (Table 8)
# ============================================================================

POT_PROMPTS: dict[str, str] = {
    "Factoid": """You are given a factoid question that you need to answer based on the provided image.
You need to write an executable python code that calculates and prints the final answer, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to return a python code only without any additional text.""",
    "Multi Choice": """You are given a question along with different possible answers. You need to select the correct answer from them based on the provided image.
You need to write an executable python code that calculates and prints the final answer, but your final answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to return a python code only without any additional text.""",
    "Hypothetical": """You are given a hypothetical question that you need to answer based on the provided image.
You need to write an executable python code that calculates and prints the final answer, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to return a python code only without any additional text.""",
    "Fact Checking": """You are given a fact statement that you need to assess based on the information in the provided image.
You need to write an executable python code that calculates and prints the final answer, but your final answer should be either true or false (without any additional text). If the question is unanswerable based on the information in the provided image, your answer should be unanswerable.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to return a python code only without any additional text.""",
    "Conversational": """You are given a multi-turn conversation, and your job is to answer the final question based on the conversation history and the information in the provided image.
You need to write an executable python code that calculates and prints the final answer, but your final answer should be a single word, number, or phrase. If the question is unanswerable based on the information in the provided image, your answer should be unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the chart.
If there are multiple final answers, put them in brackets using this format ['Answer1', 'Answer2'].
Remember to return a python code only without any additional text.""",
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_system_prompt(question_type: str, strategy: PromptStrategy = "direct") -> str:
    """
    Get the category-specific system prompt for a given question type and strategy.

    Args:
        question_type: One of ["Factoid", "Multi Choice", "Hypothetical",
                               "Fact Checking", "Conversational"]
        strategy: One of ["direct", "cot", "pot"]

    Returns:
        System prompt string for the category and strategy

    Raises:
        ValueError: If question_type or strategy is not recognized
    """
    if strategy == "direct":
        prompts = DIRECT_PROMPTS
    elif strategy == "cot":
        prompts = COT_PROMPTS
    elif strategy == "pot":
        prompts = POT_PROMPTS
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Must be one of: direct, cot, pot"
        )

    if question_type not in prompts:
        raise ValueError(
            f"Unknown question type: {question_type}. "
            f"Must be one of: {list(prompts.keys())}"
        )

    return prompts[question_type]


def format_question_with_paragraph(question: str, paragraph: str | None) -> str:
    """
    Format a question with optional context paragraph.

    Args:
        question: The question text
        paragraph: Optional context paragraph from the dataset

    Returns:
        Formatted question string with context if available
    """
    if paragraph and paragraph.strip():
        return f"Context: {paragraph.strip()}\n\nQuestion: {question}"
    else:
        return f"Question: {question}"
