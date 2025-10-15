"""
BLIMP: Benchmark of Linguistic Minimal Pairs

BLIMP tests language models' grammatical knowledge through minimal pair comparisons.
Each task presents two sentences that differ by one word, where one is grammatical
and one is ungrammatical. Models must identify which sentence is grammatical.

The benchmark includes 67 tasks covering syntactic and morphological phenomena:
- Island effects (adjunct_island, complex_NP_island, wh_island, etc.)
- Agreement phenomena (anaphor agreement, determiner-noun agreement, subject-verb agreement)
- Argument structure (drop_argument, causative, inchoative, passive, transitive, intransitive)
- NPI licensing (sentential_negation, matrix_question, only_npi, etc.)
- Binding principles (principle_A variants)
- And many more linguistic phenomena

Sample usage:
```bash
bench eval blimp_adjunct_island --model "openrouter/openai/gpt-oss-120b" -M only=groq --limit 10
bench eval blimp_determiner_noun_agreement_1 --model "openrouter/openai/gpt-oss-120b" -M only=groq --limit 10
bench eval blimp_passive_1 --model "openrouter/openai/gpt-oss-120b" -M only=groq --limit 10
```

Citation:
@article{warstadt2020blimp,
    title={BLiMP: The Benchmark of Linguistic Minimal Pairs for English},
    author={Warstadt, Alex and Parrish, Alicia and Liu, Haokun and Mohananey, Anhad and Peng, Wei and Wang, Sheng-Fu and Bowman, Samuel R},
    journal={Transactions of the Association for Computational Linguistics},
    volume={8},
    pages={377--392},
    year={2020}
}
"""

from inspect_ai import Task, task
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import create_dynamic_multiple_choice_prompt
import hashlib

# All 67 BLIMP tasks
BLIMP_TASKS = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "drop_argument",
    "ellipsis_n_bar_1",
    "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1",
    "existential_there_quantifiers_2",
    "existential_there_subject_raising",
    "expletive_it_object_raising",
    "inchoative",
    "intransitive",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "passive_1",
    "passive_2",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1",
    "superlative_quantifiers_2",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "transitive",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
]


def record_to_mcq_sample(record: dict) -> MCQSample:
    """
    Convert a BLIMP record to an OpenBench MCQSample.

    BLIMP format:
    - sentence_good: The grammatical sentence
    - sentence_bad: The ungrammatical sentence
    - field: Linguistic field (syntax, morphology, etc.)
    - linguistics_term: Specific linguistic phenomenon
    - pair_id: Unique identifier for the minimal pair

    We present both sentences as options A and B, with the task being to
    identify which sentence is grammatically correct.
    """
    sentence_good = record["sentence_good"]
    sentence_bad = record["sentence_bad"]

    # Create a binary choice prompt with deterministic A/B randomization to avoid positional bias
    question = "Which sentence is grammatically correct?"

    # Use pair_id (or sentences) to deterministically decide option order
    seed_input = str(record.get("pair_id", "")) or f"{sentence_good}||{sentence_bad}"
    seed_hash = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest(), 16)
    flip = seed_hash % 2 == 1

    if flip:
        options = [sentence_bad, sentence_good]
        target = "B"  # good sentence is second
    else:
        options = [sentence_good, sentence_bad]
        target = "A"  # good sentence is first

    prompt = create_dynamic_multiple_choice_prompt(question, options)

    return MCQSample(
        input=prompt,
        target=target,
        metadata={
            "field": record.get("field", ""),
            "linguistics_term": record.get("linguistics_term", ""),
            "pair_id": record.get("pair_id", ""),
        },
    )


# Main BLIMP function - all task functions call this
@task
def blimp(task: str = "adjunct_island", split: str = "train") -> Task:
    """
    Family benchmark for BLIMP - run any BLIMP task by name.

    Args:
        task: BLIMP task to evaluate (default: "adjunct_island")
        split: Dataset split to use (default: "train")

    Returns:
        Task: The specified BLIMP task

    Available tasks (67 total):
        adjunct_island, anaphor_gender_agreement, anaphor_number_agreement,
        animate_subject_passive, animate_subject_trans, causative,
        complex_NP_island, coordinate_structure_constraint_complex_left_branch,
        coordinate_structure_constraint_object_extraction,
        determiner_noun_agreement_1, determiner_noun_agreement_2,
        determiner_noun_agreement_irregular_1, determiner_noun_agreement_irregular_2,
        determiner_noun_agreement_with_adj_2, determiner_noun_agreement_with_adj_irregular_1,
        determiner_noun_agreement_with_adj_irregular_2,
        determiner_noun_agreement_with_adjective_1,
        distractor_agreement_relational_noun, distractor_agreement_relative_clause,
        drop_argument, ellipsis_n_bar_1, ellipsis_n_bar_2,
        existential_there_object_raising, existential_there_quantifiers_1,
        existential_there_quantifiers_2, existential_there_subject_raising,
        expletive_it_object_raising, inchoative, intransitive,
        irregular_past_participle_adjectives, irregular_past_participle_verbs,
        irregular_plural_subject_verb_agreement_1, irregular_plural_subject_verb_agreement_2,
        left_branch_island_echo_question, left_branch_island_simple_question,
        matrix_question_npi_licensor_present, npi_present_1, npi_present_2,
        only_npi_licensor_present, only_npi_scope, passive_1, passive_2,
        principle_A_c_command, principle_A_case_1, principle_A_case_2,
        principle_A_domain_1, principle_A_domain_2, principle_A_domain_3,
        principle_A_reconstruction, regular_plural_subject_verb_agreement_1,
        regular_plural_subject_verb_agreement_2, sentential_negation_npi_licensor_present,
        sentential_negation_npi_scope, sentential_subject_island,
        superlative_quantifiers_1, superlative_quantifiers_2,
        tough_vs_raising_1, tough_vs_raising_2, transitive, wh_island,
        wh_questions_object_gap, wh_questions_subject_gap,
        wh_questions_subject_gap_long_distance, wh_vs_that_no_gap,
        wh_vs_that_no_gap_long_distance, wh_vs_that_with_gap,
        wh_vs_that_with_gap_long_distance
    """
    if task not in BLIMP_TASKS:
        available = ", ".join(BLIMP_TASKS)
        raise ValueError(f"Invalid BLIMP task '{task}'. Available: {available}")

    return MCQEval(
        name=f"blimp_{task}",
        dataset_path="blimp",
        subset_name=task,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
    )


# Individual wrapper functions for all 67 tasks
@task
def blimp_adjunct_island(split: str = "train") -> Task:
    """BLIMP: Adjunct island effects"""
    return blimp(task="adjunct_island", split=split)


@task
def blimp_anaphor_gender_agreement(split: str = "train") -> Task:
    """BLIMP: Anaphor gender agreement"""
    return blimp(task="anaphor_gender_agreement", split=split)


@task
def blimp_anaphor_number_agreement(split: str = "train") -> Task:
    """BLIMP: Anaphor number agreement"""
    return blimp(task="anaphor_number_agreement", split=split)


@task
def blimp_animate_subject_passive(split: str = "train") -> Task:
    """BLIMP: Animate subject in passive constructions"""
    return blimp(task="animate_subject_passive", split=split)


@task
def blimp_animate_subject_trans(split: str = "train") -> Task:
    """BLIMP: Animate subject in transitive constructions"""
    return blimp(task="animate_subject_trans", split=split)


@task
def blimp_causative(split: str = "train") -> Task:
    """BLIMP: Causative constructions"""
    return blimp(task="causative", split=split)


@task
def blimp_complex_NP_island(split: str = "train") -> Task:
    """BLIMP: Complex NP island effects"""
    return blimp(task="complex_NP_island", split=split)


@task
def blimp_coordinate_structure_constraint_complex_left_branch(
    split: str = "train",
) -> Task:
    """BLIMP: Coordinate structure constraint - complex left branch"""
    return blimp(
        task="coordinate_structure_constraint_complex_left_branch", split=split
    )


@task
def blimp_coordinate_structure_constraint_object_extraction(
    split: str = "train",
) -> Task:
    """BLIMP: Coordinate structure constraint - object extraction"""
    return blimp(task="coordinate_structure_constraint_object_extraction", split=split)


@task
def blimp_determiner_noun_agreement_1(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement (1)"""
    return blimp(task="determiner_noun_agreement_1", split=split)


@task
def blimp_determiner_noun_agreement_2(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement (2)"""
    return blimp(task="determiner_noun_agreement_2", split=split)


@task
def blimp_determiner_noun_agreement_irregular_1(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement with irregular nouns (1)"""
    return blimp(task="determiner_noun_agreement_irregular_1", split=split)


@task
def blimp_determiner_noun_agreement_irregular_2(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement with irregular nouns (2)"""
    return blimp(task="determiner_noun_agreement_irregular_2", split=split)


@task
def blimp_determiner_noun_agreement_with_adj_2(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement with adjective (2)"""
    return blimp(task="determiner_noun_agreement_with_adj_2", split=split)


@task
def blimp_determiner_noun_agreement_with_adj_irregular_1(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement with adjective and irregular nouns (1)"""
    return blimp(task="determiner_noun_agreement_with_adj_irregular_1", split=split)


@task
def blimp_determiner_noun_agreement_with_adj_irregular_2(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement with adjective and irregular nouns (2)"""
    return blimp(task="determiner_noun_agreement_with_adj_irregular_2", split=split)


@task
def blimp_determiner_noun_agreement_with_adjective_1(split: str = "train") -> Task:
    """BLIMP: Determiner-noun agreement with adjective (1)"""
    return blimp(task="determiner_noun_agreement_with_adjective_1", split=split)


@task
def blimp_distractor_agreement_relational_noun(split: str = "train") -> Task:
    """BLIMP: Distractor agreement with relational nouns"""
    return blimp(task="distractor_agreement_relational_noun", split=split)


@task
def blimp_distractor_agreement_relative_clause(split: str = "train") -> Task:
    """BLIMP: Distractor agreement in relative clauses"""
    return blimp(task="distractor_agreement_relative_clause", split=split)


@task
def blimp_drop_argument(split: str = "train") -> Task:
    """BLIMP: Dropped argument"""
    return blimp(task="drop_argument", split=split)


@task
def blimp_ellipsis_n_bar_1(split: str = "train") -> Task:
    """BLIMP: N-bar ellipsis (1)"""
    return blimp(task="ellipsis_n_bar_1", split=split)


@task
def blimp_ellipsis_n_bar_2(split: str = "train") -> Task:
    """BLIMP: N-bar ellipsis (2)"""
    return blimp(task="ellipsis_n_bar_2", split=split)


@task
def blimp_existential_there_object_raising(split: str = "train") -> Task:
    """BLIMP: Existential 'there' with object raising"""
    return blimp(task="existential_there_object_raising", split=split)


@task
def blimp_existential_there_quantifiers_1(split: str = "train") -> Task:
    """BLIMP: Existential 'there' with quantifiers (1)"""
    return blimp(task="existential_there_quantifiers_1", split=split)


@task
def blimp_existential_there_quantifiers_2(split: str = "train") -> Task:
    """BLIMP: Existential 'there' with quantifiers (2)"""
    return blimp(task="existential_there_quantifiers_2", split=split)


@task
def blimp_existential_there_subject_raising(split: str = "train") -> Task:
    """BLIMP: Existential 'there' with subject raising"""
    return blimp(task="existential_there_subject_raising", split=split)


@task
def blimp_expletive_it_object_raising(split: str = "train") -> Task:
    """BLIMP: Expletive 'it' with object raising"""
    return blimp(task="expletive_it_object_raising", split=split)


@task
def blimp_inchoative(split: str = "train") -> Task:
    """BLIMP: Inchoative constructions"""
    return blimp(task="inchoative", split=split)


@task
def blimp_intransitive(split: str = "train") -> Task:
    """BLIMP: Intransitive verbs"""
    return blimp(task="intransitive", split=split)


@task
def blimp_irregular_past_participle_adjectives(split: str = "train") -> Task:
    """BLIMP: Irregular past participles as adjectives"""
    return blimp(task="irregular_past_participle_adjectives", split=split)


@task
def blimp_irregular_past_participle_verbs(split: str = "train") -> Task:
    """BLIMP: Irregular past participles in verbs"""
    return blimp(task="irregular_past_participle_verbs", split=split)


@task
def blimp_irregular_plural_subject_verb_agreement_1(split: str = "train") -> Task:
    """BLIMP: Subject-verb agreement with irregular plurals (1)"""
    return blimp(task="irregular_plural_subject_verb_agreement_1", split=split)


@task
def blimp_irregular_plural_subject_verb_agreement_2(split: str = "train") -> Task:
    """BLIMP: Subject-verb agreement with irregular plurals (2)"""
    return blimp(task="irregular_plural_subject_verb_agreement_2", split=split)


@task
def blimp_left_branch_island_echo_question(split: str = "train") -> Task:
    """BLIMP: Left branch island effects in echo questions"""
    return blimp(task="left_branch_island_echo_question", split=split)


@task
def blimp_left_branch_island_simple_question(split: str = "train") -> Task:
    """BLIMP: Left branch island effects in simple questions"""
    return blimp(task="left_branch_island_simple_question", split=split)


@task
def blimp_matrix_question_npi_licensor_present(split: str = "train") -> Task:
    """BLIMP: Matrix question NPI licensor present"""
    return blimp(task="matrix_question_npi_licensor_present", split=split)


@task
def blimp_npi_present_1(split: str = "train") -> Task:
    """BLIMP: Negative polarity items present (1)"""
    return blimp(task="npi_present_1", split=split)


@task
def blimp_npi_present_2(split: str = "train") -> Task:
    """BLIMP: Negative polarity items present (2)"""
    return blimp(task="npi_present_2", split=split)


@task
def blimp_only_npi_licensor_present(split: str = "train") -> Task:
    """BLIMP: 'Only' as NPI licensor"""
    return blimp(task="only_npi_licensor_present", split=split)


@task
def blimp_only_npi_scope(split: str = "train") -> Task:
    """BLIMP: 'Only' NPI scope"""
    return blimp(task="only_npi_scope", split=split)


@task
def blimp_passive_1(split: str = "train") -> Task:
    """BLIMP: Passive constructions (1)"""
    return blimp(task="passive_1", split=split)


@task
def blimp_passive_2(split: str = "train") -> Task:
    """BLIMP: Passive constructions (2)"""
    return blimp(task="passive_2", split=split)


@task
def blimp_principle_A_c_command(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - c-command"""
    return blimp(task="principle_A_c_command", split=split)


@task
def blimp_principle_A_case_1(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - case (1)"""
    return blimp(task="principle_A_case_1", split=split)


@task
def blimp_principle_A_case_2(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - case (2)"""
    return blimp(task="principle_A_case_2", split=split)


@task
def blimp_principle_A_domain_1(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - domain (1)"""
    return blimp(task="principle_A_domain_1", split=split)


@task
def blimp_principle_A_domain_2(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - domain (2)"""
    return blimp(task="principle_A_domain_2", split=split)


@task
def blimp_principle_A_domain_3(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - domain (3)"""
    return blimp(task="principle_A_domain_3", split=split)


@task
def blimp_principle_A_reconstruction(split: str = "train") -> Task:
    """BLIMP: Binding Principle A - reconstruction"""
    return blimp(task="principle_A_reconstruction", split=split)


@task
def blimp_regular_plural_subject_verb_agreement_1(split: str = "train") -> Task:
    """BLIMP: Subject-verb agreement with regular plurals (1)"""
    return blimp(task="regular_plural_subject_verb_agreement_1", split=split)


@task
def blimp_regular_plural_subject_verb_agreement_2(split: str = "train") -> Task:
    """BLIMP: Subject-verb agreement with regular plurals (2)"""
    return blimp(task="regular_plural_subject_verb_agreement_2", split=split)


@task
def blimp_sentential_negation_npi_licensor_present(split: str = "train") -> Task:
    """BLIMP: Sentential negation as NPI licensor"""
    return blimp(task="sentential_negation_npi_licensor_present", split=split)


@task
def blimp_sentential_negation_npi_scope(split: str = "train") -> Task:
    """BLIMP: Sentential negation NPI scope"""
    return blimp(task="sentential_negation_npi_scope", split=split)


@task
def blimp_sentential_subject_island(split: str = "train") -> Task:
    """BLIMP: Sentential subject island effects"""
    return blimp(task="sentential_subject_island", split=split)


@task
def blimp_superlative_quantifiers_1(split: str = "train") -> Task:
    """BLIMP: Superlative quantifiers (1)"""
    return blimp(task="superlative_quantifiers_1", split=split)


@task
def blimp_superlative_quantifiers_2(split: str = "train") -> Task:
    """BLIMP: Superlative quantifiers (2)"""
    return blimp(task="superlative_quantifiers_2", split=split)


@task
def blimp_tough_vs_raising_1(split: str = "train") -> Task:
    """BLIMP: Tough vs raising constructions (1)"""
    return blimp(task="tough_vs_raising_1", split=split)


@task
def blimp_tough_vs_raising_2(split: str = "train") -> Task:
    """BLIMP: Tough vs raising constructions (2)"""
    return blimp(task="tough_vs_raising_2", split=split)


@task
def blimp_transitive(split: str = "train") -> Task:
    """BLIMP: Transitive verbs"""
    return blimp(task="transitive", split=split)


@task
def blimp_wh_island(split: str = "train") -> Task:
    """BLIMP: Wh-island effects"""
    return blimp(task="wh_island", split=split)


@task
def blimp_wh_questions_object_gap(split: str = "train") -> Task:
    """BLIMP: Wh-questions with object gap"""
    return blimp(task="wh_questions_object_gap", split=split)


@task
def blimp_wh_questions_subject_gap(split: str = "train") -> Task:
    """BLIMP: Wh-questions with subject gap"""
    return blimp(task="wh_questions_subject_gap", split=split)


@task
def blimp_wh_questions_subject_gap_long_distance(split: str = "train") -> Task:
    """BLIMP: Wh-questions with long-distance subject gap"""
    return blimp(task="wh_questions_subject_gap_long_distance", split=split)


@task
def blimp_wh_vs_that_no_gap(split: str = "train") -> Task:
    """BLIMP: Wh vs that complementizers without gap"""
    return blimp(task="wh_vs_that_no_gap", split=split)


@task
def blimp_wh_vs_that_no_gap_long_distance(split: str = "train") -> Task:
    """BLIMP: Wh vs that complementizers without gap (long-distance)"""
    return blimp(task="wh_vs_that_no_gap_long_distance", split=split)


@task
def blimp_wh_vs_that_with_gap(split: str = "train") -> Task:
    """BLIMP: Wh vs that complementizers with gap"""
    return blimp(task="wh_vs_that_with_gap", split=split)


@task
def blimp_wh_vs_that_with_gap_long_distance(split: str = "train") -> Task:
    """BLIMP: Wh vs that complementizers with gap (long-distance)"""
    return blimp(task="wh_vs_that_with_gap_long_distance", split=split)
