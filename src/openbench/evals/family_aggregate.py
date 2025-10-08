"""
Family benchmark aggregates.

This module provides entrypoint functions for family benchmarks.
These functions are placeholders - the actual expansion happens in the CLI
via expand_eval_groups() which detects the family name and expands it to all subtasks.
"""

from inspect_ai import Task, task


# This is a placeholder function that should never actually be called
# The CLI will intercept these family names and expand them via EVAL_GROUPS
def _family_placeholder() -> Task:
    """Placeholder for family benchmarks - should be intercepted by CLI."""
    raise NotImplementedError(
        "Family benchmarks should be expanded by CLI before task loading. "
        "This indicates a bug in the eval expansion logic."
    )


# Generate placeholder tasks for all families
bigbench = task(_family_placeholder)
bbh = task(_family_placeholder)
agieval = task(_family_placeholder)
ethics = task(_family_placeholder)
blimp = task(_family_placeholder)
glue = task(_family_placeholder)
superglue = task(_family_placeholder)
global_mmlu = task(_family_placeholder)
xcopa = task(_family_placeholder)
xstorycloze = task(_family_placeholder)
xwinograd = task(_family_placeholder)
mgsm = task(_family_placeholder)
headqa = task(_family_placeholder)
mmmu = task(_family_placeholder)
arabic_exams = task(_family_placeholder)
exercism = task(_family_placeholder)
anli = task(_family_placeholder)
healthbench = task(_family_placeholder)
openai_mrcr = task(_family_placeholder)
mmmu_pro = task(_family_placeholder)
arc = task(_family_placeholder)
arc_agi = task(_family_placeholder)
hle = task(_family_placeholder)
matharena = task(_family_placeholder)
qa4mre = task(_family_placeholder)
otis_mock_aime = task(_family_placeholder)
cti_bench = task(_family_placeholder)
bbl = task(_family_placeholder)
mmmlu = task(_family_placeholder)
