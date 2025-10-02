"""
AGIEval - A Human-Centric Benchmark for Evaluating Foundation Models

AGIEval is a benchmark designed to assess foundation models' general abilities in the context
of human cognition and problem-solving. It consists of 17 official qualifying exam questions,
including:
- AQUA-RAT: Algebraic question answering
- LogiQA (English and Chinese): Logical reasoning
- LSAT (AR, LR, RC): Law School Admission Test sections
- SAT (English, English without passage, Math): Scholastic Assessment Test
- Gaokao (Biology, Chemistry, Chinese, English, Geography, History, MathQA, Physics):
  Chinese national college entrance examination subjects

All tasks use multiple-choice format with 4-5 options.

Sample usage:
```bash
bench eval agieval_aqua_rat --model "groq/llama-3.1-70b"
bench eval agieval_lsat_lr --model "groq/llama-3.1-70b"
bench eval agieval_gaokao_english --model "groq/llama-3.1-70b"
```

Citation:
@article{zhong2023agieval,
    title={AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models},
    author={Zhong, Wanjun and Cui, Ruixiang and Guo, Yiduo and Liang, Yaobo and Lu, Shuai and Wang, Yanlin and Saied, Amin and Chen, Weizhu and Duan, Nan},
    journal={arXiv:2304.06364},
    year={2023}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample

# Mapping of subset names to HuggingFace dataset paths
AGIEVAL_DATASET_PATHS = {
    "aqua_rat": "dmayhem93/agieval-aqua-rat",
    "logiqa_en": "dmayhem93/agieval-logiqa-en",
    "logiqa_zh": "dmayhem93/agieval-logiqa-zh",
    "lsat_ar": "dmayhem93/agieval-lsat-ar",
    "lsat_lr": "dmayhem93/agieval-lsat-lr",
    "lsat_rc": "dmayhem93/agieval-lsat-rc",
    "sat_en": "dmayhem93/agieval-sat-en",
    "sat_en_without_passage": "dmayhem93/agieval-sat-en-without-passage",
    "sat_math": "dmayhem93/agieval-sat-math",
    "gaokao_biology": "dmayhem93/agieval-gaokao-biology",
    "gaokao_chemistry": "dmayhem93/agieval-gaokao-chemistry",
    "gaokao_chinese": "dmayhem93/agieval-gaokao-chinese",
    "gaokao_english": "dmayhem93/agieval-gaokao-english",
    "gaokao_geography": "dmayhem93/agieval-gaokao-geography",
    "gaokao_history": "dmayhem93/agieval-gaokao-history",
    "gaokao_mathqa": "dmayhem93/agieval-gaokao-mathqa",
    "gaokao_physics": "dmayhem93/agieval-gaokao-physics",
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """
    Convert an AGIEval record to an OpenBench MCQSample.

    AGIEval format:
    - query: Full question text with "Answer Choices" and "Among A through X, the answer is" prompt
    - choices: List of choice strings (e.g., ["(A)choice1", "(B)choice2", ...])
    - gold: List with single integer index (e.g., [0] for choice A)
    """
    query = record["query"]
    choices = record["choices"]
    gold_index = record["gold"][0]  # gold is a list with one element

    # Convert gold index to letter (0->A, 1->B, etc.)
    target = chr(65 + gold_index)

    # The query already includes formatting and prompting,
    # so we use it directly as input
    return MCQSample(
        input=query,
        target=target,
        metadata={
            "choices": choices,
        },
    )


# Main AGIEval function - all subset tasks call this
@task
def agieval(subset: str = "aqua_rat", split: str = "test") -> Task:
    """
    Family benchmark for AGIEval - run any AGIEval subset by name.

    Args:
        subset: AGIEval subset to evaluate. Available subsets:
                - aqua_rat: Algebraic reasoning
                - logiqa_en, logiqa_zh: Logical reasoning (English/Chinese)
                - lsat_ar, lsat_lr, lsat_rc: LSAT (Analytical/Logical/Reading)
                - sat_en, sat_en_without_passage, sat_math: SAT sections
                - gaokao_biology, gaokao_chemistry, gaokao_chinese, gaokao_english,
                  gaokao_geography, gaokao_history, gaokao_mathqa, gaokao_physics:
                  Chinese national college entrance exam subjects
        split: Dataset split to use (default: "test")

    Returns:
        Task: The specified AGIEval subset task
    """
    if subset not in AGIEVAL_DATASET_PATHS:
        available = ", ".join(AGIEVAL_DATASET_PATHS.keys())
        raise ValueError(f"Invalid AGIEval subset '{subset}'. Available: {available}")

    return MCQEval(
        name=f"agieval_{subset}",
        dataset_path=AGIEVAL_DATASET_PATHS[subset],
        subset_name="default",
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )


# Individual task functions - convenience wrappers that call agieval(subset=...)
@task
def agieval_aqua_rat(split: str = "test") -> Task:
    return agieval(subset="aqua_rat", split=split)


@task
def agieval_logiqa_en(split: str = "test") -> Task:
    return agieval(subset="logiqa_en", split=split)


@task
def agieval_logiqa_zh(split: str = "test") -> Task:
    return agieval(subset="logiqa_zh", split=split)


@task
def agieval_lsat_ar(split: str = "test") -> Task:
    return agieval(subset="lsat_ar", split=split)


@task
def agieval_lsat_lr(split: str = "test") -> Task:
    return agieval(subset="lsat_lr", split=split)


@task
def agieval_lsat_rc(split: str = "test") -> Task:
    return agieval(subset="lsat_rc", split=split)


@task
def agieval_sat_en(split: str = "test") -> Task:
    return agieval(subset="sat_en", split=split)


@task
def agieval_sat_en_without_passage(split: str = "test") -> Task:
    return agieval(subset="sat_en_without_passage", split=split)


@task
def agieval_sat_math(split: str = "test") -> Task:
    return agieval(subset="sat_math", split=split)


@task
def agieval_gaokao_biology(split: str = "test") -> Task:
    return agieval(subset="gaokao_biology", split=split)


@task
def agieval_gaokao_chemistry(split: str = "test") -> Task:
    return agieval(subset="gaokao_chemistry", split=split)


@task
def agieval_gaokao_chinese(split: str = "test") -> Task:
    return agieval(subset="gaokao_chinese", split=split)


@task
def agieval_gaokao_english(split: str = "test") -> Task:
    return agieval(subset="gaokao_english", split=split)


@task
def agieval_gaokao_geography(split: str = "test") -> Task:
    return agieval(subset="gaokao_geography", split=split)


@task
def agieval_gaokao_history(split: str = "test") -> Task:
    return agieval(subset="gaokao_history", split=split)


@task
def agieval_gaokao_mathqa(split: str = "test") -> Task:
    return agieval(subset="gaokao_mathqa", split=split)


@task
def agieval_gaokao_physics(split: str = "test") -> Task:
    return agieval(subset="gaokao_physics", split=split)
