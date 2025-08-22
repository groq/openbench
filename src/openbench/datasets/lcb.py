from inspect_ai.dataset import Sample
from typing import Any, Union
import json as jsonlib


def record_to_sample_for_code_generation(
    record: dict[str, Any],
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
) -> Sample:
    metadata = {
        "public_test_cases": record["public_test_cases"],
        "question_id": record["question_id"],
        "contest_id": record["contest_id"],
        "contest_date": record["contest_date"],
        "starter_code": record["starter_code"],
        "difficulty": record["difficulty"],
        "platform": record["platform"],
    }
    return Sample(input=record["question_content"], metadata=metadata)


def record_to_sample_for_code_execution(
    record: dict[str, Any],
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
) -> Sample:
    metadata = {
        "id": record["id"],
        "function_name": record["function_name"],
        "code": record["code"],
        "input": record["input"],
        "output": record["output"],
        "numsteps": record["numsteps"],
        "problem_id": record["problem_id"],
    }

    return Sample(input=record["code"], metadata=metadata)


def record_to_sample_for_test_output_prediction(
    record: dict[str, Any],
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
) -> Sample:
    metadata = {
        "question_title": record["question_title"],
        "question_content": record["question_content"],
        "question_id": record["question_id"],
        "contest_id": record["contest_id"],
        "test_id": record["test_id"],
        "contest_date": record["contest_date"],
        "starter_code": record["starter_code"],
        "function_name": record["function_name"],
        "difficulty": record["difficulty"],
        "test": jsonlib.loads(record["test"]),
    }
    return Sample(input=record["question_content"], metadata=metadata)
