"""
Scorer for Terraform generation: extract .tf code from model output, run terraform init + validate.
Pass = 1.0 if init and validate succeed; 0.0 otherwise. No LocalStack required.
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import TaskState


def _extract_tf_blocks(text: str) -> dict[str, str]:
    """Extract Terraform code blocks from model output (```filename.tf or ```terraform)."""
    files: dict[str, str] = {}
    # ```main.tf\n...\n``` or ```terraform\n...\n``` - capture optional name and content
    pattern = r"```(\w+\.tf|terraform|hcl)?\s*\n(.*?)```"
    for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        raw_name = (m.group(1) or "").strip().lower()
        block = (m.group(2) or "").strip()
        if not block or len(block) < 20:
            continue
        if raw_name and raw_name.endswith(".tf"):
            name = raw_name
        elif raw_name in ("terraform", "hcl") and "main.tf" not in files:
            name = "main.tf"
        else:
            name = (
                "main.tf"
                if "main.tf" not in files
                else "variables.tf"
                if "variables.tf" not in files
                else "outputs.tf"
            )
        if name not in files:
            files[name] = block
    required = ["main.tf", "variables.tf", "outputs.tf"]
    for f in required:
        if f not in files and files:
            files[f] = "# placeholder\n"
    return files


@scorer(metrics=[accuracy(), stderr()])
def terraform_generation_scorer() -> Callable:
    """Scorer for Terraform generation: init + validate only (no LocalStack)."""

    async def score(state: TaskState, target: Target) -> Score:
        # Last assistant message content
        output = ""
        for msg in reversed(state.messages):
            if getattr(msg, "role", None) != "assistant":
                continue
            content = getattr(msg, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                output = content
            elif hasattr(content, "text"):
                output = getattr(content, "text", "") or ""
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        output += part
                    elif getattr(part, "text", None):
                        output += part.text
            break

        if not output or not output.strip():
            return Score(value=0.0, explanation="No model output")

        files = _extract_tf_blocks(output)
        if not files:
            return Score(value=0.0, explanation="No Terraform code blocks found")

        with tempfile.TemporaryDirectory(prefix="tfgen_") as tmp:
            work = Path(tmp)
            for name, body in files.items():
                (work / name).write_text(body)

            try:
                subprocess.run(
                    ["terraform", "fmt", "-write"],
                    cwd=work,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                r = subprocess.run(
                    ["terraform", "init", "-backend=false"],
                    cwd=work,
                    capture_output=True,
                    timeout=60,
                    text=True,
                )
                if r.returncode != 0:
                    return Score(
                        value=0.0,
                        explanation=(
                            f"terraform init failed: "
                            f"{r.stderr[:200] if r.stderr else r.stdout or ''}"
                        ),
                    )
                r = subprocess.run(
                    ["terraform", "validate"],
                    cwd=work,
                    capture_output=True,
                    timeout=30,
                    text=True,
                )
                if r.returncode != 0:
                    return Score(
                        value=0.0,
                        explanation=(
                            f"terraform validate failed: "
                            f"{r.stderr[:200] if r.stderr else r.stdout or ''}"
                        ),
                    )
                return Score(
                    value=1.0, explanation="terraform init and validate passed"
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return Score(value=0.0, explanation=str(e))

    return score
