"""
LegalSupport: Legal Support Identification Benchmark

This benchmark evaluates models on their ability to identify the most supporting legal citation
given a context and two candidate citations. The task requires understanding legal language
and reasoning about which citation provides stronger support for a legal argument.

Dataset: lighteval/LegalSupport
Split: test (3047 samples)

Sample usage:
```bash
bench eval legalsupport --model groq/llama-3.1-70b-versatile
```
"""

from inspect_ai import Task, task
from inspect_ai.solver import Choices
from inspect_ai.solver._multiple_choice import prompt
from openbench.utils.mcq import MCQEval, MCQSample


LEGALSUPPORT_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a LegalSupport record to an OpenBench MCQSample.

    The dataset contains:
    - context: Legal text where citations appear
    - citation_a: First citation with signal, identifier, parenthetical, sentence
    - citation_b: Second citation with signal, identifier, parenthetical, sentence
    - label: Either 'a' or 'b' indicating which citation is more supportive
    """
    context = record["context"]

    # Format citation A
    citation_a = record["citation_a"]
    citation_a_text = f"Citation A: {citation_a['identifier']}"
    if citation_a.get("parenthetical"):
        citation_a_text += f" ({citation_a['parenthetical']})"

    # Format citation B
    citation_b = record["citation_b"]
    citation_b_text = f"Citation B: {citation_b['identifier']}"
    if citation_b.get("parenthetical"):
        citation_b_text += f" ({citation_b['parenthetical']})"

    question = f"Given the following legal context, which citation provides stronger support?\n\nContext: {context}\n\nWhich citation is more supportive?"

    # Create binary choice with both citations
    input_msg = prompt(
        question=question,
        choices=Choices([citation_a_text, citation_b_text]),
        template=str(LEGALSUPPORT_TEMPLATE),
    )

    # Convert 'a'/'b' label to 'A'/'B'
    target = record["label"].upper()

    return MCQSample(
        input=input_msg,
        target=target,
        metadata={
            "case_id": str(record.get("case_id", "")),
        },
    )


@task
def legalsupport(split: str = "test") -> Task:
    """Evaluate the LegalSupport benchmark for legal citation support identification."""
    return MCQEval(
        name="legalsupport",
        dataset_path="lighteval/LegalSupport",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )
