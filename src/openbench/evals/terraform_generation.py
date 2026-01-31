"""
Terraform Generation benchmark: prompt → Terraform code, scored by init + validate.

Authored for: sre-skills-bench / openbench
Based on: https://github.com/groq/openbench (same pattern as rootly_gmcq, PR #114)

Run:
  bench eval terraform_generation --model "groq/llama-3.1-8b-instant"

Requires: terraform CLI on PATH. No LocalStack required for this eval (validate only).
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from openbench.datasets.terraform_generation import load_dataset
from openbench.scorers.terraform_generation import terraform_generation_scorer


@task
def terraform_generation() -> Task:
    """Terraform Generation: generate Terraform from prompts, score by init + validate."""
    return Task(
        dataset=load_dataset(),
        solver=[generate()],
        scorer=terraform_generation_scorer(),
    )
