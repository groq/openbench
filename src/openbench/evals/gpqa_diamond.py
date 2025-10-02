import random
from inspect_ai import Task, task, Epochs
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import SIMPLE_EVALS_SYSTEM_MESSAGE
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE


# There is one difference between this and the original gpqa simple eval - the prompts are not reshuffled for every epoch. Shouldn't be that big of a deal, but worth noting.
def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a GQPQA Diamond record to an OpenBench MCQSample."""
    random.seed(0)
    options = [
        record["Correct Answer"],
        record["Incorrect Answer 1"],
        record["Incorrect Answer 2"],
        record["Incorrect Answer 3"],
    ]
    random.shuffle(options)
    # Get index of correct answer and convert to A, B, C, D
    correct_index = options.index(record["Correct Answer"])
    correct_letter = "ABCD"[correct_index]
    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=record["Question"],
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        ),
        target=correct_letter,
    )


@task
def gpqa_diamond() -> Task:
    """Evaluate the GQPQA Diamond dataset (MCQ Abstracted)."""
    return MCQEval(
        name="gpqa_diamond",
        dataset_path="nmayorga7/gpqa_diamond",
        record_to_mcq_sample=record_to_mcq_sample,
        split="train",  # only train split available
        auto_id=True,
        prompt_template=SIMPLE_EVALS_SYSTEM_MESSAGE,
        config=GenerateConfig(temperature=0.5),
        epochs=Epochs(10),
    )
