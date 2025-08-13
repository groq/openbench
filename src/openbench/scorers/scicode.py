from inspect_ai.scorer import Metric, Score, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target, metric
from pathlib import Path
import time
import shutil
import subprocess
from typing import Any


class ScicodeEvaluator:
    def __init__(
        self,
        h5py_file: str,
        code_dir: Path,
        log_dir: Path,
        with_background: bool,
    ):
        self.h5py_file = h5py_file
        self.code_dir = code_dir
        self.log_dir = log_dir
        self.with_background = with_background

    def _get_background_dir(self):
        return "with_background" if self.with_background else "without_background"

    def test_code(
        self,
        prob_data: dict,
    ):
        code_dir = Path(self.code_dir, "generated_code", self._get_background_dir())
        tmp_dir = Path(f"tmp_{time.time()}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        sub_steps = prob_data["sub_steps"]
        problem_id = prob_data["problem_id"]
        for idx in range(len(sub_steps)):
            if (
                (problem_id == "13" and idx == 5)
                or (problem_id == "62" and idx == 0)
                or (problem_id == "76" and idx == 2)
            ):
                continue
            step_id = sub_steps[idx]["step_number"]
            code_file_path = Path(code_dir, f"{step_id}.py")
            assert code_file_path.is_file(), f"Code file {code_file_path} not found."
            code_content = code_file_path.read_text(encoding="utf-8")
            test_lst = sub_steps[idx]["test_cases"]
            assert_file = Path(tmp_dir, f"{step_id}.py")
            with open(assert_file, "w", encoding="utf-8") as f:
                f.write(code_content)
                f.write("""

from scicode.parse.parse import process_hdf5_to_tuple

""")
                f.write(
                    f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)}, '{self.h5py_file}')"
                    + "\n"
                )
                for i in range(len(test_lst)):
                    f.write(f"target = targets[{i}]\n\n")
                    for line in test_lst[i].split("\n"):
                        f.write(line + "\n")

        def run_script(script_path):
            try:
                subprocess.run(
                    ["python", script_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                return 0
            except subprocess.CalledProcessError:
                return 1
            except subprocess.TimeoutExpired:
                return 2

        total_steps = len(sub_steps)
        total_correct = 0
        for idx in range(len(sub_steps)):
            if (
                (problem_id == "13" and idx == 5)
                or (problem_id == "62" and idx == 0)
                or (problem_id == "76" and idx == 2)
            ):
                continue
            step_id = sub_steps[idx]["step_number"]
            script_path = Path(tmp_dir, f"{step_id}.py")
            logs_dir = Path(self.log_dir, "evaluation_logs", self._get_background_dir())
            logs_dir.mkdir(parents=True, exist_ok=True)
            logs_file = Path(logs_dir, f"{step_id}.log")
            if logs_file.is_file():
                with open(logs_file, "r") as f:
                    content = f.read().splitlines()
                    if content[0] == "pass":
                        total_correct += 1
                continue
            ret = run_script(script_path)
            if ret == 0:
                with open(logs_file, "w") as f:
                    f.write("pass")
                total_correct += 1
            elif ret == 1:
                with open(logs_file, "w") as f:
                    f.write("fail")
            else:
                with open(logs_file, "w") as f:
                    f.write("time out")

        shutil.rmtree(tmp_dir)
        problem_correct = 1 if total_correct == total_steps else 0
        return problem_correct, total_correct, total_steps


@metric
def sub_problem_correctness() -> Metric:
    def metric(scores: list[Score]) -> int | float:
        total_correct = 0
        total_steps = 0
        for score in scores:
            total_correct += score.value["Total Correct"]  # type: ignore
            total_steps += score.value["Total Steps"]  # type: ignore
        return total_correct / total_steps

    return metric


@scorer(
    metrics=[
        {
            "Problem Correctness": [mean()],
        },
        sub_problem_correctness(),
    ]
)
def scicode_scorer(**params: dict[str, Any]):
    async def score(state: TaskState, target: Target):
        model_name = str(state.model).replace("/", "-")
        evaluator = ScicodeEvaluator(
            h5py_file=params["h5py_file"],  # type: ignore
            code_dir=Path(params["output_dir"], model_name),  # type: ignore
            log_dir=Path(params["output_dir"], model_name),  # type: ignore
            with_background=params["with_background"],  # type: ignore
        )
        problem_correct, total_correct, total_steps = evaluator.test_code(
            state.metadata
        )
        return Score(
            value={
                "Problem Correctness": problem_correct,
                "Total Correct": total_correct,
                "Total Steps": total_steps,
            }
        )

    return score
