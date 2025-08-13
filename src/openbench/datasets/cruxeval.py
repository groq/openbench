"""
CRUXEval dataset loader for OpenBench.

CRUXEval: Code Reasoning, Understanding, and eXecution Evaluation
A benchmark of Python functions and input-output pairs with two tasks:
1. CRUXEval-I (Input Prediction): Given a Python function and its output, predict the input
2. CRUXEval-O (Output Prediction): Given a Python function and its input, predict the output

Based on: https://github.com/facebookresearch/cruxeval
"""

from typing import Dict, Any
from inspect_ai.dataset import Dataset, Sample, hf_dataset


# Direct prompts (without chain-of-thought)
DIRECT_INPUT_PROMPT = """You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
"""

DIRECT_OUTPUT_PROMPT = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""

# Chain-of-thought prompts
COT_INPUT_PROMPT = """You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""

COT_OUTPUT_PROMPT = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""


def create_prompt(
    sample: Dict[str, Any], task_type: str = "output", use_cot: bool = False
) -> str:
    """
    Create a prompt for either CRUXEval-I (input prediction) or CRUXEval-O (output prediction) task.

    Args:
        sample: The dataset sample containing code, input, and output.
        task_type: The type of task, either "input" or "output". Defaults to "output".
        use_cot: Whether to use chain-of-thought prompting. Defaults to False.

    Returns:
        The formatted prompt string.
    """
    if task_type not in ["input", "output"]:
        raise ValueError('task_type must be either "input" or "output"')

    code = sample["code"]
    if task_type == "input":
        output = sample["output"]
        prompt_template = COT_INPUT_PROMPT if use_cot else DIRECT_INPUT_PROMPT
        return prompt_template.format(code=code, output=output)
    else:
        input_val = sample["input"]
        prompt_template = COT_OUTPUT_PROMPT if use_cot else DIRECT_OUTPUT_PROMPT
        return prompt_template.format(code=code, input=input_val)


def record_to_sample(
    record: Dict[str, Any], task_type: str = "output", use_cot: bool = False
) -> Sample:
    """Convert a dataset record to a Sample."""
    required_fields = ["id", "code", "input", "output"]
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    return Sample(
        id=str(record["id"]),
        input=create_prompt(record, task_type, use_cot),
        target=record["input"] if task_type == "input" else record["output"],
        metadata={
            "code": record["code"],
            "input": record["input"],
            "output": record["output"],
            "task_type": task_type,
            "use_cot": use_cot,
        },
    )


def get_dataset(task_type: str = "output", use_cot: bool = False) -> Dataset:
    """
    Get the CRUXEval dataset for the specified task type.

    Args:
        task_type: Either "input" for CRUXEval-I or "output" for CRUXEval-O
        use_cot: Whether to use chain-of-thought prompting

    Returns:
        Dataset configured for the specified task type
    """
    if task_type not in ["input", "output"]:
        raise ValueError('task_type must be either "input" or "output"')

    return hf_dataset(
        path="cruxeval-org/cruxeval",
        sample_fields=lambda record: record_to_sample(record, task_type, use_cot),
        split="test",
    )
