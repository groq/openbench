"""
LiveCodeBench

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan,
Tianju Zhang, Sida Wang, Armando Solar-Lezama,
Koushik Sen, Ion Stoica

Based on: https://livecodebench.github.io/

# run code generation
bench eval lcb --model "groq/llama-3.1-8b-instant" --T scenario=code_generation,release_version="release_version"

Please note: due to datasets dependency issues, pleaes keep datasets==4.0.0.
However, if you do not have the LiveCodeBench dataset cached through the HuggingFace dataset cache,
please temporarily downgrade to datasets==3.6.0, and revert it back to 4.0.0 after the dataset is cached.

Please refer to https://huggingface.co/datasets/livecodebench/code_generation_lite
for the release version to use.
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import solver, TaskState, Generate
import json as jsonlib
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import sandbox
from typing import Union, Any

from openbench.scorers.lcb import custom_scorer, test_output_prediction_scorer
from openbench.datasets.lcb import (
    record_to_sample_for_code_generation,
    record_to_sample_for_code_execution,
    record_to_sample_for_test_output_prediction,
)

CODE_EXECUTION_PROMPT_WITH_COT = """
You are given a Python function and an assertion containing an input to the function.
  → Complete the assertion with a literal (no unsimplified expressions, no function
  → calls) containing the output when executing the provided code on the given input
  → , even if the function is incorrect or incomplete. Do NOT output any extra
  → information. Execute the program step by step before arriving at an answer, and
  → provide the full assertion with the correct output in [ANSWER] and [/ANSWER]
  → tags, following the examples.

```python
def performOperation(s):
    s = s + s
    return "b" + s + "a"
assert performOperation(s = "hi") == ??
```
[THOUGHT]
Let’s execute the code step by step:
1. The function performOperation is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is
    → initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value
    → of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert performOperation(s = "hi") == "bhihia"
[/ANSWER]
[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[THOUGHT]
"""

TEST_OUTPUT_PREDICTION_PROMPT = """
### Instruction: You are a helpful programming assistant and an expert Python
,→ programmer. You are helping a user to write a test case to help to check the
,→ correctness of the function. The user has written a input for the testcase. You
,→ will calculate the output of the testcase and write the whole assertion
,→ statement in the markdown code block with the correct output.
Problem:
{problem_statement}
Function:
```
{function_signature}
```
Please complete the following test case:
```
assert {function_name}({testcase_input}) == # TODO
```
### Response:"""

CODE_GENERATION_PROMPT = """
You are an expert Python programmer. You will be given a question (problem
    → specification) and will generate a correct Python program that matches the
    → specification and passes all tests. You will NOT return anything except for the
    → program
### Question:\n{question.question_content}
{ if question.starter_code }
### Format: {PromptConstants.FORMATTING_MESSAGE}
```python
{question.starter_code}
```
{ else }
### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_MESSAGE}
```python
# YOUR CODE HERE
```
{ endif }
### Answer: (use the provided format with backticks)
"""

SELF_REPAIR_PROMPT = """
You are a helpful programming assistant and an expert Python programmer. You are
    → helping a user write a program to solve a problem. The user has written some
    → code, but it has some errors and is not passing the tests. You will help the
    → user by first giving a concise (at most 2-3 sentences) textual explanation of
    → what is wrong with the code. After you have pointed out what is wrong with the
    → code, you will then generate a fixed version of the program. You must put the
    → entired fixed program within code delimiters only for once.
### Question:\n{question.question_content}
### Answer: ```python
{code.code_to_be_corrected}
```
### Format: {PromptConstants.FORMATTING_CHECK_ERROR_MESSAGE}
### Answer: (use the provided format with backticks)
"""

CODE_EXECUTION_PROMPT = """
You are given a Python function and an assertion containing an input to the function.
    → Complete the assertion with a literal (no unsimplified expressions, no function
    → calls) containing the output when executing the provided code on the given input
    → , even if the function is incorrect or incomplete. Do NOT output any extra
    → information. Provide the full assertion with the correct output in [ANSWER] and
    → [/ANSWER] tags, following the examples.
[PYTHON]
def repeatNumber(number : int) -> int:
    return number
assert repeatNumber(number = 17) == ??
[/PYTHON]
[ANSWER]
assert repeatNumber(number = 17) == 17
[/ANSWER]
[PYTHON]
def addCharacterA(string : str) -> str:
    return string + "a"
assert addCharacterA(string = "x9j") == ??
[/PYTHON]
[ANSWER]
assert addCharacterA(string = "x9j") == "x9ja"
[/ANSWER]
[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[ANSWER]
"""

CODE_EXECUTION_PROMPT_WITH_COT = """
You are given a Python function and an assertion containing an input to the function.
  → Complete the assertion with a literal (no unsimplified expressions, no function
  → calls) containing the output when executing the provided code on the given input
  → , even if the function is incorrect or incomplete. Do NOT output any extra
  → information. Execute the program step by step before arriving at an answer, and
  → provide the full assertion with the correct output in [ANSWER] and [/ANSWER]
  → tags, following the examples.

```python
def performOperation(s):
    s = s + s
    return "b" + s + "a"
assert performOperation(s = "hi") == ??
```
[THOUGHT]
Let’s execute the code step by step:
1. The function performOperation is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is
    → initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value
    → of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert performOperation(s = "hi") == "bhihia"
[/ANSWER]
[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
[THOUGHT]
"""


VERIFY_TIMEOUT = 30

START_DATE: Union[str, None] = None
END_DATE: Union[str, None] = None
SCENARIO: Union[str, None] = None


def create_code_execution_prompt(code: str, input: str) -> str:
    CODE_EXECUTION_PROMPT = f"""
You are given a Python function and an assertion containing an input to the function.
Complete the assertion with a literal (no unsimplified expressions, no function
calls) containing the output when executing the provided code on the given input
, even if the function is incorrect or incomplete. Do NOT output any extra
information. Provide the full assertion with the correct output in [ANSWER] and
[/ANSWER] tags, following the examples.
[PYTHON]
def repeatNumber(number : int) -> int:
    return number
assert repeatNumber(number = 17) == ??
[/PYTHON]
[ANSWER]
assert repeatNumber(number = 17) == 17
[/ANSWER]
[PYTHON]
def addCharacterA(string : str) -> str:
    return string + "a"
assert addCharacterA(string = "x9j") == ??
[/PYTHON]
[ANSWER]
assert addCharacterA(string = "x9j") == "x9ja"
[/ANSWER]
[PYTHON]
{code}
assert {input} == ??
[/PYTHON]
"""

    return CODE_EXECUTION_PROMPT


def create_test_output_prediction_prompt(
    problem_statement: str,
    function_signature: str,
    function_name: str,
    testcase_input: str,
) -> str:
    TEST_OUTPUT_PREDICTION_PROMPT = f"""
### Instruction: You are a helpful programming assistant and an expert Python
programmer. You are helping a user to write a test case to help to check the
correctness of the function. The user has written a input for the testcase. You
will calculate the output of the testcase and write the whole assertion
statement in the markdown code block with the correct output.
Problem:
{problem_statement}
Function:
```
{function_signature}
```
Please complete the following test case:

```python
assert {function_name}({testcase_input}) == # TODO
```

### Response:"""

    return TEST_OUTPUT_PREDICTION_PROMPT


def create_self_repair_prompt(question_content: Any, code: str, error_message: str):
    FORMATTING_CHECK_ERROR_MESSAGE = f"Respond with the following format: ```python\n{error_message}\n```. The following code should fix the following error: {error_message}"
    SELF_REPAIR_PROMPT = f"""
You are a helpful programming assistant and an expert Python programmer. You are
helping a user write a program to solve a problem. The user has written some
code, but it has some errors and is not passing the tests. You will help the
user by first giving a concise (at most 2-3 sentences) textual explanation of
what is wrong with the code. After you have pointed out what is wrong with the
code, you will then generate a fixed version of the program. You must put the
entired fixed program within code delimiters only for once.
    ### Question:\n{question_content}
    ### Answer: ```python
    {code}
    ```
    ### Format: {FORMATTING_CHECK_ERROR_MESSAGE}
    ### Answer: (use the provided format with backticks)
    """

    return SELF_REPAIR_PROMPT


def create_code_generation_prompt(
    question_content: Any, starter_code: Union[str, None] = None
):
    FORMATTING_MESSAGE = ""
    FORMATTING_WITHOUT_STARTER_MESSAGE = """Only generate a function that accepts a single input parameter,
the raw input string.
Only generate the function body, do not write any function calls.
The function must be named foo, which will accept the raw input string as defined by the problem,
as a single input parameter to the function foo.
Instead of printing the output, return it in the same string representation.
"""

    CODE_GENERATION_PROMPT = f"""
You are an expert Python programmer. You will be given a question (problem
specification) and will generate a correct Python program that matches the
specification and passes all tests. You will NOT return anything except for the
program
### Question:\n{question_content}
"""
    if starter_code is not None:
        CODE_GENERATION_PROMPT += f"""
### Format: {FORMATTING_MESSAGE}
```python
{starter_code}
```
"""
    else:
        CODE_GENERATION_PROMPT += f"""
### Format: {FORMATTING_WITHOUT_STARTER_MESSAGE}
```python
# YOUR CODE HERE
```
"""

    CODE_GENERATION_PROMPT += """
### Answer: (use the provided format with backticks)
"""
    return CODE_GENERATION_PROMPT


@solver
def custom_solver(SCENARIO: str):
    # global SCENARIO

    if SCENARIO == "codegeneration":
        model = get_model()

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            resp = await model.generate(
                input=[
                    ChatMessageUser(
                        content=create_code_generation_prompt(
                            question_content=state.input
                        )
                    ),
                ],
            )

            model_str: Any = resp.choices[0].message.content

            try:
                model_str = model_str.split("```python")[1].split("```")[0]
            except IndexError:
                model_str = ""

            for test_case in jsonlib.loads(state.metadata["public_test_cases"]):
                model_str += f"\nassert str(foo('''{test_case['input'].strip()}''')) == '''{test_case['output'].strip()}'''"

            state.metadata["generated_code"] = model_str
            return state

        return solve
    elif SCENARIO == "selfrepair":
        model = get_model()

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            resp = await model.generate(
                input=[
                    ChatMessageUser(
                        content=create_code_generation_prompt(
                            question_content=state.input
                        )
                    ),
                ],
            )

            model_str_raw: Any = resp.choices[0].message.content

            try:
                model_str: Any = model_str_raw.split("```python")[1].split("```")[0]
                for test_case in jsonlib.loads(state.metadata["public_test_cases"]):
                    model_str += f"\nassert str(foo('''{test_case['input'].strip()}''')) == '''{test_case['output'].strip()}'''"
            except IndexError:
                model_str = ""

            state.metadata["generated_code"] = model_str

            result = await sandbox().exec(
                cmd=["python", "-c", "".join(state.metadata["generated_code"])],
                timeout=VERIFY_TIMEOUT,
            )
            if result.returncode == 0:
                return state

            else:
                resp = await model.generate(
                    input=[
                        ChatMessageUser(
                            content=create_self_repair_prompt(
                                question_content=state.input,
                                code=state.metadata["generated_code"],
                                error_message=result.stderr,
                            )
                        ),
                    ],
                )

                model_str_raw_2: Any = resp.choices[0].message.content

                try:
                    model_str_2: Any = model_str_raw_2.split("```python")[1].split(
                        "```"
                    )[0]
                    for test_case in jsonlib.loads(state.metadata["public_test_cases"]):
                        model_str_2 += f"\nassert str(foo('''{test_case['input'].strip()}''')) == '''{test_case['output'].strip()}'''"
                except IndexError:
                    model_str_2 = ""

                state.metadata["generated_code"] = model_str_2

                return state

        return solve
    else:
        return ValueError(f"Invalid scenario: {SCENARIO}")


@solver
def code_execution_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = await model.generate(
            input=[
                ChatMessageUser(
                    content=create_code_execution_prompt(
                        code=state.metadata["code"],
                        input=state.metadata["input"],
                    ),
                ),
            ],
        )

        code_to_run = "from typing import *\n"
        code_to_run += "import collections\n"

        code_to_run += state.metadata["code"]
        code_to_run += "\n"

        try:
            test_case = resp.choices[0].message.content
            code_to_run += test_case.split("[ANSWER]")[1].split("[/ANSWER]")[0]
        except IndexError:
            code_to_run += "assert False"

        state.metadata["generated_code"] = code_to_run

        return state

    return solve


@solver
def test_output_prediction_solver():
    model = get_model()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = await model.generate(
            input=[
                ChatMessageUser(
                    content=create_test_output_prediction_prompt(
                        problem_statement=state.metadata["question_content"],
                        function_signature=state.metadata["starter_code"],
                        function_name=state.metadata["function_name"],
                        testcase_input=state.metadata["test"][0]["input"],
                    ),
                ),
            ],
        )

        resp = resp.choices[0].message.content
        resp = resp.split("```python")[-1].split("```")[0]

        try:
            state.metadata["predicted_test_output"] = resp
        except IndexError:
            state.metadata["predicted_test_output"] = ""

        return state

    return solve


@task
def lcb(
    scenario: str = "codegeneration",
    release_version: str = "release_v6",
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
    sandbox: str = "local",
) -> Task:
    """
    Inspect Task implementation for LCB
    """

    # TODO: if needed in the future, make scenario required (not optional)

    global START_DATE
    global END_DATE
    global SCENARIO

    START_DATE = start_date
    END_DATE = end_date
    SCENARIO = scenario

    if scenario == "codegeneration" or scenario == "selfrepair":
        if release_version == "hf_groq_code_generation":
            code_generation_dataset = hf_dataset(
                "Groq/LiveCodeBench-CodeGeneration",
                split="test",
                sample_fields=record_to_sample_for_code_generation,
            )

        else:
            code_generation_dataset = hf_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                version_tag=release_version,
                sample_fields=record_to_sample_for_code_generation,
            )

        return Task(
            dataset=code_generation_dataset,
            solver=custom_solver(SCENARIO=scenario),
            scorer=custom_scorer(SCENARIO=scenario, VERIFY_TIMEOUT=VERIFY_TIMEOUT),
            config=GenerateConfig(),
            sandbox=sandbox,
        )
    elif scenario == "testoutputprediction":
        test_output_prediction_dataset = hf_dataset(
            "livecodebench/test_generation",
            split="test",
            sample_fields=record_to_sample_for_test_output_prediction,
        )

        return Task(
            dataset=test_output_prediction_dataset,
            solver=test_output_prediction_solver(),
            scorer=test_output_prediction_scorer(SCENARIO=scenario),
            config=GenerateConfig(),
            sandbox=sandbox,
        )
    elif scenario == "codeexecution":
        code_execution_dataset = hf_dataset(
            "livecodebench/execution",
            split="test",
            sample_fields=record_to_sample_for_code_execution,
        )

        return Task(
            dataset=code_execution_dataset,
            solver=code_execution_solver(),
            scorer=custom_scorer(SCENARIO=scenario),
            config=GenerateConfig(),
            sandbox=sandbox,
        )

    else:
        raise ValueError(f"Invalid scenario: {SCENARIO}")
