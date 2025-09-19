"""Universal Multiple Choice Question Evaluation Task."""

from pydantic.dataclasses import dataclass
from inspect_ai import Task
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.dataset import Dataset, Sample, hf_dataset, csv_dataset, json_dataset
from typing import Any, Callable, Union, List, Optional, Mapping
import re


# -----------MCQ SCHEMA-----------
@dataclass
class MCQSchema:
    """Schema of fields needed to map raw records into MCQ samples."""

    question_field: str
    choices_field: Union[str, list[str]]
    answer_field: str
    context_field: Optional[str] = None
    metadata_field: Optional[Union[str, list[str]]] = None
    id_field: Optional[str] = None


# -----------HELPER FUNCTIONS-----------
def clean_choice(raw_choice: str) -> str:
    """Remove common letter prefixes like 'A)', 'A.', 'A:', 'A,' etc."""
    return re.sub(r"^[A-Z][\)\:\.,]?\s*", "", raw_choice.strip(), flags=re.IGNORECASE)


def parse_choices(
    record: Mapping[str, Any], choices_field: Union[str, List[str]]
) -> List[str]:
    """
    Parse choices into a list[str] without letter prefixes. Supports:
        - List of column names
        - Single string with delimiters or newlines
        - List of strings
        - Dict of letter: choice pairs
    """

    # case 1: list of fields — extract from multiple columns
    if isinstance(choices_field, list):
        raw = [record[field] for field in choices_field]

    # case 2: single field name
    elif isinstance(choices_field, str):
        raw = record[choices_field]

        # case 2a: it's a dict (ex {"A": "Red", "B": "Blue"})
        if isinstance(raw, dict):
            raw = list(raw.values())

        # case 2b: it's a newline-separated string
        elif isinstance(raw, str) and "\n" in raw:
            raw = raw.splitlines()

        # TODO: add support for more delimiters
        # case 2c: it's a delimited string (e.g. "A) Red; B) Blue")
        elif isinstance(raw, str) and ";" in raw:
            raw = raw.split(";")

        # case 2d: it's a comma-separated string (less preferred)
        elif isinstance(raw, str) and "," in raw:
            raw = raw.split(",")

        # case 2e: already a list of strings
        elif isinstance(raw, list):
            pass

        # case 2f: fallback to list with single string
        else:
            raw = [raw]

    else:
        raise ValueError(
            f"choices_field must be a str or list[str]. Got: {type(choices_field)}"
        )

    # clean, strip, and validate
    cleaned_choices = [clean_choice(c) for c in raw if isinstance(c, str) and c.strip()]

    if len(cleaned_choices) == 0:
        raise ValueError(f"Could not extract any valid choices from: {raw}")

    return cleaned_choices


# -----------RECORD TO SAMPLE FUNCTION-----------
def record_to_sample(
    schema: MCQSchema,
) -> Callable[[Mapping[str, Any]], Sample]:
    """Use MCQ Schema to generate Sample objects from records."""

    def _record_to_sample(record: Mapping[str, Any]) -> Sample:
        # extract question
        question = record[schema.question_field]

        # extract context, if given
        if schema.context_field:
            context = record.get(schema.context_field, "")
            question = context.strip() + "\n" + question.strip()

        # extract choices
        choices = parse_choices(record, schema.choices_field)

        # extract answer
        answer = record[schema.answer_field].strip().upper()
        if not re.match(r"^[A-Z]$", answer):
            raise ValueError(
                f"Answer must be a single capital letter A-Z. Got: '{answer}'"
            )

        # extract metadata
        metadata = {}
        if schema.metadata_field:
            if isinstance(schema.metadata_field, list):
                for field in schema.metadata_field:
                    metadata[field] = record.get(field)
            else:
                metadata = {"meta": record.get(schema.metadata_field)}

        # extract id
        id = record.get(schema.id_field) if schema.id_field else None

        return Sample(
            input=question,
            choices=choices,
            target=answer,
            metadata=metadata if metadata else None,
            id=id if id else None,
        )

    return _record_to_sample


# -----------MCQ EVAL TASK ABSTRACTION-----------
class MCQEval:
    """Universal MCQ eval task definition."""

    def __init__(
        self,
        dataset_type: str,
        path: str,
        split: str = "test",
        question_field: str = "question",
        choices_field: Union[str, list[str]] = "choices",
        answer_field: str = "answer",
        context_field: Optional[str] = None,
        metadata_field: Optional[Union[str, list[str]]] = None,
        id_field: Optional[str] = None,
    ):
        self.schema = MCQSchema(
            question_field=question_field,
            choices_field=choices_field,
            answer_field=answer_field,
            context_field=context_field,
            metadata_field=metadata_field,
            id_field=id_field,
        )
        self.dataset_type = dataset_type
        self.path = path
        self.split = split

    def get_dataset(self) -> Dataset:
        """Load a dataset using the MCQSchema."""
        auto_id = self.schema.id_field is None
        sample_fields = record_to_sample(self.schema)

        match self.dataset_type:
            case "hf":
                return hf_dataset(
                    self.path,
                    split=self.split,
                    sample_fields=sample_fields,
                    auto_id=auto_id,
                )
            case "csv":
                return csv_dataset(
                    self.path,
                    sample_fields=sample_fields,
                    auto_id=auto_id,
                )
            case "json":
                return json_dataset(
                    self.path,
                    sample_fields=sample_fields,
                    auto_id=auto_id,
                )
            case _:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    # treat MCQEval instance as a callable
    def __call__(self) -> Task:
        return Task(
            dataset=self.get_dataset(),
            solver=multiple_choice(),
            scorer=choice(),
        )


# -----------EXAMPLE USAGE-----------
# Example (for reference only):
# from inspect_ai import task
# @task
# def example_mcq_eval() -> Task:
#     return MCQEval(
#         dataset_type="hf",  # or "csv" / "json"
#         path="my/dataset",
#         split="test",
#         question_field="question",
#         choices_field=["A", "B", "C", "D"],
#         answer_field="answer",
#         context_field=None,
#         metadata_field=None,
#         id_field=None,
#     )
