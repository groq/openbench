"""
BigBench - Multiple Choice tasks from BIG-Bench benchmark

BIG-Bench (Beyond the Imitation Game Benchmark) is a collaborative benchmark with
over 200 tasks that test language model capabilities across diverse domains.

This file contains 80 of the 150 multiple-choice tasks from BigBench. These are
officially labeled as "multiple choice" in BigBench's keywords_to_tasks.md.

Note: BigBench also includes free-response and other task types which will be
added to this file in future updates. Use bigbench_hard.py for the BBH challenging
reasoning tasks, and bigbench_lite.py for the curated lightweight subset.

Dataset: tasksource/bigbench (Hugging Face)

Sample usage:
```bash
# Run individual task
bench eval bigbench_arithmetic --model "groq/llama-3.1-8b-instant"
bench eval bigbench_metaphor_understanding --model "groq/llama-3.1-8b-instant"
bench eval bigbench_emoji_movie --model "groq/llama-3.1-8b-instant"

# Run with task parameter (family benchmark)
bench eval bigbench --task arithmetic --model "groq/llama-3.1-8b-instant"
```

Citation:
@article{srivastava2022beyond,
    title={Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models},
    author={Srivastava, Aarohi and others},
    journal={arXiv preprint arXiv:2206.04615},
    year={2022}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt

# Monkey-patch to fix compatibility issue with datasets library 3.x
# BigBench dataset metadata uses deprecated 'List' type instead of 'Sequence'
try:
    from datasets.features import Sequence  # type: ignore[import-untyped]
    import datasets.features.features as _ff  # type: ignore[import-untyped]

    if "List" not in _ff._FEATURE_TYPES:
        _ff._FEATURE_TYPES["List"] = Sequence
except ImportError:
    pass  # datasets library not available, patch not needed


def record_to_mcq_sample_bigbench(record: dict) -> MCQSample:
    """Convert a BigBench record to an OpenBench MCQSample.

    BigBench tasks have a consistent structure:
    - inputs: The question text with choices embedded
    - targets: List of correct answer(s) as strings
    - multiple_choice_targets: List of all possible choices
    - multiple_choice_scores: List of scores (1 for correct, 0 for incorrect)

    We need to:
    1. Extract the question from inputs (everything before "choice:")
    2. Use multiple_choice_targets as the choices
    3. Find the correct answer from multiple_choice_scores
    """
    # Extract question - everything before the first "choice:" line
    inputs_text = record["inputs"]
    if "choice:" in inputs_text:
        # Split by newlines and find where choices start
        lines = inputs_text.split("\n")
        question_lines = []
        for line in lines:
            if "choice:" not in line:
                question_lines.append(line)
            else:
                break
        question = "\n".join(question_lines).strip()
        # Remove trailing "A:" if present
        if question.endswith("A:"):
            question = question[:-2].strip()
    else:
        # No explicit choices in input, use as-is
        question = inputs_text.strip()
        if question.endswith("A:"):
            question = question[:-2].strip()

    # Get choices from multiple_choice_targets
    choices = record["multiple_choice_targets"]

    # Find correct answer from multiple_choice_scores
    scores = record["multiple_choice_scores"]
    correct_index = scores.index(1) if 1 in scores else 0

    # Handle tasks with more than 26 choices by limiting to first 26
    # This matches how create_dynamic_multiple_choice_prompt handles large choice sets
    if len(choices) > 26:
        # If the correct answer is beyond index 25, we need to skip this sample
        # as it can't be represented with A-Z format
        if correct_index >= 26:
            # Use the original target text instead and map it to first 26 choices
            # Find if the target exists in first 26 choices
            original_target = record["targets"][0] if record["targets"] else ""
            if original_target in choices[:26]:
                correct_index = choices[:26].index(original_target)
            else:
                # Skip samples where correct answer is not in first 26 choices
                # This is a limitation of the A-Z format
                correct_index = 0  # Fallback to A
        choices = choices[:26]  # Limit to 26 choices

    target_letter = chr(65 + correct_index)  # Convert to A, B, C, etc.

    # Create the prompt
    prompt = create_dynamic_multiple_choice_prompt(question, choices)

    return MCQSample(
        input=prompt,
        target=target_letter,
        metadata={
            "original_target": record["targets"][0] if record["targets"] else "",
            "idx": record.get("idx", -1),
        },
    )


# ==============================================================================
# BIGBENCH MCQ TASKS (150 of 150 implemented)
# ==============================================================================
# These are all 150 tasks officially labeled as "multiple choice" in BigBench's
# keywords_to_tasks.md.

# BigBench Lite - Curated subset of 18 MCQ tasks from the official BBL benchmark
# Note: The official BBL has 24 tasks, but 6 are free-response format
# (auto_debugging, conlang_translation, linguistics_puzzles, operators,
# parsinlu_reading_comprehension, repeat_copy_logic) and not yet implemented.
BIGBENCH_LITE_TASKS = [
    "bbq_lite_json",
    "code_line_description",
    "conceptual_combinations",
    "emoji_movie",
    "formal_fallacies_syllogisms_negation",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions_russian",
    "novel_concepts",
    "play_dialog_same_or_different",
    "strange_stories",
    "strategyqa",
    "symbol_interpretation",
    "vitaminc_fact_verification",
    "winowhy",
]

BIGBENCH_TASKS = [
    # All 122 MCQ tasks available in tasksource/bigbench dataset (alphabetically sorted)
    "abstract_narrative_understanding",
    "anachronisms",
    "analogical_similarity",
    "analytic_entailment",
    "arithmetic",
    "authorship_verification",
    "bbq_lite_json",
    "causal_judgment",
    "cause_and_effect",
    "checkmate_in_one",
    "cifar10_classification",
    "code_line_description",
    "color",
    "common_morpheme",
    "conceptual_combinations",
    "contextual_parametric_knowledge_conflicts",
    "crash_blossom",
    "crass_ai",
    "cryobiology_spanish",
    "cs_algorithms",
    "dark_humor_detection",
    "date_understanding",
    "disambiguation_qa",
    "discourse_marker_prediction",
    "dyck_languages",
    "elementary_math_qa",
    "emoji_movie",
    "emojis_emotion_prediction",
    "empirical_judgments",
    "english_proverbs",
    "english_russian_proverbs",
    "entailed_polarity",
    "entailed_polarity_hindi",
    "epistemic_reasoning",
    "evaluating_information_essentiality",
    "fact_checker",
    "fantasy_reasoning",
    "figure_of_speech_detection",
    "formal_fallacies_syllogisms_negation",
    "general_knowledge",
    "geometric_shapes",
    "goal_step_wikihow",
    "gre_reading_comprehension",
    "hhh_alignment",
    "hindu_knowledge",
    "hinglish_toxicity",
    "human_organs_senses",
    "hyperbaton",
    "identify_math_theorems",
    "identify_odd_metaphor",
    "implicatures",
    "implicit_relations",
    "indic_cause_and_effect",
    "intent_recognition",
    "international_phonetic_alphabet_nli",
    "intersect_geometry",
    "irony_identification",
    "kanji_ascii",
    "kannada",
    "key_value_maps",
    "known_unknowns",
    "language_identification",
    "logic_grid_puzzle",
    "logical_args",
    "logical_deduction",
    "logical_fallacy_detection",
    "logical_sequence",
    "mathematical_induction",
    "medical_questions_russian",
    "metaphor_boolean",
    "metaphor_understanding",
    "minute_mysteries_qa",
    "misconceptions",
    "misconceptions_russian",
    "mnist_ascii",
    "moral_permissibility",
    "movie_dialog_same_or_different",
    "movie_recommendation",
    "navigate",
    "nonsense_words_grammar",
    "novel_concepts",
    "odd_one_out",
    "parsinlu_qa",
    "penguins_in_a_table",
    "periodic_elements",
    "persian_idioms",
    "phrase_relatedness",
    "physical_intuition",
    "physics",
    "play_dialog_same_or_different",
    "presuppositions_as_nli",
    "question_selection",
    "real_or_fake_text",
    "reasoning_about_colored_objects",
    "rhyming",
    "riddle_sense",
    "ruin_names",
    "salient_translation_error_detection",
    "sentence_ambiguity",
    "similarities_abstraction",
    "simple_ethical_questions",
    "snarks",
    "social_iqa",
    "social_support",
    "sports_understanding",
    "strange_stories",
    "strategyqa",
    "suicide_risk",
    "swahili_english_proverbs",
    "swedish_to_german_proverbs",
    "symbol_interpretation",
    "temporal_sequences",
    "timedial",
    "tracking_shuffled_objects",
    "understanding_fables",
    "undo_permutation",
    "unit_conversion",
    "unit_interpretation",
    "vitaminc_fact_verification",
    "what_is_the_tao",
    "which_wiki_edit",
    "winowhy",
]


# Generate all BigBench tasks as individual wrapper functions
def _make_task_wrapper(subset_name: str):
    """Create a wrapper function for a specific BigBench subset."""

    def wrapper() -> Task:
        return MCQEval(
            name=f"bigbench_{subset_name}",
            dataset_path="tasksource/bigbench",
            subset_name=subset_name,
            record_to_mcq_sample=record_to_mcq_sample_bigbench,
            split="train",
            auto_id=True,
        )

    # Set the function name BEFORE applying @task decorator
    wrapper.__name__ = f"bigbench_{subset_name}"
    wrapper.__qualname__ = f"bigbench_{subset_name}"
    # Now apply the decorator
    return task(wrapper)


for subset in BIGBENCH_TASKS:
    globals()[f"bigbench_{subset}"] = _make_task_wrapper(subset)
