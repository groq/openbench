"""
BigBench Hard (BBH) - Challenging BIG-Bench tasks from Suzgun et al.

Big-Bench Hard (BBH) is a suite of 23 challenging BIG-Bench tasks that language
models prior to GPT-4 and PaLM-2 struggled with. These tasks require multi-step
reasoning and are designed to be particularly difficult for models.

Dataset: lukaemon/bbh (Hugging Face)

This implementation includes the 18 core BBH tasks that are commonly evaluated:
- causal_judgment: Causal attribution questions
- date_understanding: Inferring dates from context
- disambiguation_qa: Disambiguating sentences with ambiguous pronouns
- geometric_shapes: Naming geometric shapes from SVG paths
- logical_deduction_*: Deducing object order (3, 5, 7 objects)
- movie_recommendation: Recommending similar movies
- navigate: Determining if navigation returns to start
- reasoning_about_colored_objects: Simple color reasoning
- ruin_names: Selecting humorous edits to movie/artist names
- salient_translation_error_detection: Detecting translation errors
- snarks: Identifying sarcastic sentences
- sports_understanding: Plausibility of sports-related sentences
- temporal_sequences: Answering questions about event timing
- tracking_shuffled_objects_*: Tracking objects through swaps (3, 5, 7 objects)

Sample usage:
```bash
bench eval bbh_causal_judgment --model "groq/llama-3.1-70b"
bench eval bbh_date_understanding --model "groq/llama-3.1-70b"
bench eval bbh_logical_deduction_five_objects --model "groq/llama-3.1-70b"
```

Citation:
@article{suzgun2022challenging,
    title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
    author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V. and Chi, Ed H. and Zhou, Denny and Wei, Jason},
    journal={arXiv preprint arXiv:2210.09261},
    year={2022}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt


def record_to_mcq_sample(record: dict) -> MCQSample:
    """
    Convert a BBH record to an OpenBench MCQSample.

    BBH tasks have a simple structure with 'input' (question text) and 'target' (answer).
    The input often contains the question and options formatted as text.
    """
    # The input contains the full question with options
    input_text = record["input"]
    target = record["target"]

    # Use input as-is - prompt_template will be prepended by MCQEval
    prompt = input_text

    # For BBH, the target is already the answer text
    # We'll format it as a letter-based choice for MCQEval
    # Extract choices from the input if they're formatted as "- Option"
    lines = input_text.split("\n")
    choices = []
    for line in lines:
        if line.strip().startswith("- "):
            choice = line.strip()[2:]  # Remove "- " prefix
            choices.append(choice)

    # If we found choices, create a proper MCQ prompt
    if choices:
        # Find which choice matches the target
        try:
            target_index = choices.index(target)
            target_letter = chr(65 + target_index)  # Convert to A, B, C, etc.
        except ValueError:
            # If exact match fails, use the target as-is
            target_letter = target

        # Extract the question (everything before "Options:")
        if "Options:" in input_text:
            question = input_text.split("Options:")[0].strip()
        else:
            # Use everything before the first "- " if no "Options:" marker
            question = (
                input_text.split("\n- ")[0].strip()
                if "\n- " in input_text
                else input_text
            )

        prompt = create_dynamic_multiple_choice_prompt(question, choices)

        return MCQSample(
            input=prompt,
            target=target_letter,
            metadata={"original_target": target},
        )
    else:
        # No clear choices found, use input as-is
        # Clean target - remove parentheses and take only first character if valid letter
        clean_target = target.strip().upper().replace("(", "").replace(")", "").strip()

        # Extract only first character if it's a letter, otherwise use first letter
        if clean_target and clean_target[0].isalpha():
            clean_target = clean_target[0]
        elif clean_target:
            # Find first alphabetic character
            for char in clean_target:
                if char.isalpha():
                    clean_target = char
                    break

        return MCQSample(
            input=prompt,
            target=clean_target,
            metadata={"original_target": target},
        )


@task
def bbh_causal_judgment(split: str = "test") -> Task:
    """
    Evaluate BBH causal judgment task.

    Tests the ability to answer questions about causal attribution.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH causal judgment evaluation
    """
    instruction = "Answer questions about causal attribution.\n\n"

    return MCQEval(
        name="bbh_causal_judgement",
        dataset_path="lukaemon/bbh",
        subset_name="causal_judgement",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_date_understanding(split: str = "test") -> Task:
    """
    Evaluate BBH date understanding task.

    Tests the ability to infer dates from context.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH date understanding evaluation
    """
    instruction = "Infer the date from context.\n\n"

    return MCQEval(
        name="bbh_date_understanding",
        dataset_path="lukaemon/bbh",
        subset_name="date_understanding",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_disambiguation_qa(split: str = "test") -> Task:
    """
    Evaluate BBH disambiguation QA task.

    Tests the ability to clarify the meaning of sentences with ambiguous pronouns.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH disambiguation QA evaluation
    """
    instruction = "Clarify the meaning of sentences with ambiguous pronouns.\n\n"

    return MCQEval(
        name="bbh_disambiguation_qa",
        dataset_path="lukaemon/bbh",
        subset_name="disambiguation_qa",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_geometric_shapes(split: str = "test") -> Task:
    """
    Evaluate BBH geometric shapes task.

    Tests the ability to name geometric shapes from their SVG paths.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH geometric shapes evaluation
    """
    instruction = "Name geometric shapes from their SVG paths.\n\n"

    return MCQEval(
        name="bbh_geometric_shapes",
        dataset_path="lukaemon/bbh",
        subset_name="geometric_shapes",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_logical_deduction_five_objects(split: str = "test") -> Task:
    """
    Evaluate BBH logical deduction task with five objects.

    Tests the ability to deduce the order of a sequence of five objects.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH logical deduction (5 objects) evaluation
    """
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"

    return MCQEval(
        name="bbh_logical_deduction_five_objects",
        dataset_path="lukaemon/bbh",
        subset_name="logical_deduction_five_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_logical_deduction_seven_objects(split: str = "test") -> Task:
    """
    Evaluate BBH logical deduction task with seven objects.

    Tests the ability to deduce the order of a sequence of seven objects.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH logical deduction (7 objects) evaluation
    """
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"

    return MCQEval(
        name="bbh_logical_deduction_seven_objects",
        dataset_path="lukaemon/bbh",
        subset_name="logical_deduction_seven_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_logical_deduction_three_objects(split: str = "test") -> Task:
    """
    Evaluate BBH logical deduction task with three objects.

    Tests the ability to deduce the order of a sequence of three objects.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH logical deduction (3 objects) evaluation
    """
    instruction = "A logical deduction task which requires deducing the order of a sequence of objects.\n\n"

    return MCQEval(
        name="bbh_logical_deduction_three_objects",
        dataset_path="lukaemon/bbh",
        subset_name="logical_deduction_three_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_movie_recommendation(split: str = "test") -> Task:
    """
    Evaluate BBH movie recommendation task.

    Tests the ability to recommend movies similar to a given list.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH movie recommendation evaluation
    """
    instruction = "Recommend movies similar to the given list of movies.\n\n"

    return MCQEval(
        name="bbh_movie_recommendation",
        dataset_path="lukaemon/bbh",
        subset_name="movie_recommendation",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_navigate(split: str = "test") -> Task:
    """
    Evaluate BBH navigate task.

    Tests the ability to determine whether navigation instructions return to start.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH navigate evaluation
    """
    instruction = "Given a series of navigation instructions, determine whether one would end up back at the starting point.\n\n"

    return MCQEval(
        name="bbh_navigate",
        dataset_path="lukaemon/bbh",
        subset_name="navigate",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_reasoning_about_colored_objects(split: str = "test") -> Task:
    """
    Evaluate BBH reasoning about colored objects task.

    Tests the ability to answer simple questions about colors of objects on a surface.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH colored objects reasoning evaluation
    """
    instruction = "Answer extremely simple questions about the colors of objects on a surface.\n\n"

    return MCQEval(
        name="bbh_reasoning_about_colored_objects",
        dataset_path="lukaemon/bbh",
        subset_name="reasoning_about_colored_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_ruin_names(split: str = "test") -> Task:
    """
    Evaluate BBH ruin names task.

    Tests the ability to select humorous edits that 'ruin' movie or artist names.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH ruin names evaluation
    """
    instruction = "Select the humorous edit that 'ruins' the input movie or musical artist name.\n\n"

    return MCQEval(
        name="bbh_ruin_names",
        dataset_path="lukaemon/bbh",
        subset_name="ruin_names",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_salient_translation_error_detection(split: str = "test") -> Task:
    """
    Evaluate BBH salient translation error detection task.

    Tests the ability to detect types of errors in English translations of German sentences.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH translation error detection evaluation
    """
    instruction = "Detect the type of error in an English translation of a German source sentence.\n\n"

    return MCQEval(
        name="bbh_salient_translation_error_detection",
        dataset_path="lukaemon/bbh",
        subset_name="salient_translation_error_detection",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_snarks(split: str = "test") -> Task:
    """
    Evaluate BBH snarks task.

    Tests the ability to determine which of two sentences is sarcastic.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH snarks evaluation
    """
    instruction = 'Determine which of two sentences is sarcastic.\n\nAccording to Cambridge University Dictionary, sarcasm is "the use of remarks that clearly mean the opposite of what they say, made in order to hurt someone\'s feelings or to criticize something in a humorous way." Sarcastic sentences often contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks.\n\n'

    return MCQEval(
        name="bbh_snarks",
        dataset_path="lukaemon/bbh",
        subset_name="snarks",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_sports_understanding(split: str = "test") -> Task:
    """
    Evaluate BBH sports understanding task.

    Tests the ability to determine whether artificially constructed sports sentences are plausible.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH sports understanding evaluation
    """
    instruction = "Determine whether an artificially constructed sentence relating to sports is plausible or not.\n\n"

    return MCQEval(
        name="bbh_sports_understanding",
        dataset_path="lukaemon/bbh",
        subset_name="sports_understanding",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_temporal_sequences(split: str = "test") -> Task:
    """
    Evaluate BBH temporal sequences task.

    Tests the ability to answer questions about which times certain events could have occurred.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH temporal sequences evaluation
    """
    instruction = "Task description: Answer questions about which times certain events could have occurred.\n\n"

    return MCQEval(
        name="bbh_temporal_sequences",
        dataset_path="lukaemon/bbh",
        subset_name="temporal_sequences",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_tracking_shuffled_objects_five_objects(split: str = "test") -> Task:
    """
    Evaluate BBH tracking shuffled objects task with five objects.

    Tests the ability to determine final positions of five objects after swaps.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH object tracking (5 objects) evaluation
    """
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n"

    return MCQEval(
        name="bbh_tracking_shuffled_objects_five_objects",
        dataset_path="lukaemon/bbh",
        subset_name="tracking_shuffled_objects_five_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_tracking_shuffled_objects_seven_objects(split: str = "test") -> Task:
    """
    Evaluate BBH tracking shuffled objects task with seven objects.

    Tests the ability to determine final positions of seven objects after swaps.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH object tracking (7 objects) evaluation
    """
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n"

    return MCQEval(
        name="bbh_tracking_shuffled_objects_seven_objects",
        dataset_path="lukaemon/bbh",
        subset_name="tracking_shuffled_objects_seven_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )


@task
def bbh_tracking_shuffled_objects_three_objects(split: str = "test") -> Task:
    """
    Evaluate BBH tracking shuffled objects task with three objects.

    Tests the ability to determine final positions of three objects after swaps.

    Args:
        split: Dataset split to use (only "test" available)

    Returns:
        Task: Inspect AI task for BBH object tracking (3 objects) evaluation
    """
    instruction = "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.\n\n"

    return MCQEval(
        name="bbh_tracking_shuffled_objects_three_objects",
        dataset_path="lukaemon/bbh",
        subset_name="tracking_shuffled_objects_three_objects",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        prompt_template=instruction,
    )
