#!/usr/bin/env python3
"""Generate EVAL_GROUPS entries for family benchmarks."""

import re

# Read config.py
with open('src/openbench/config.py', 'r') as f:
    content = f.read()

# Helper to find all tasks matching a pattern
def find_tasks(pattern):
    return sorted(set(re.findall(rf'"({pattern}[^"]+)":', content)))

# Define all families and their patterns
families = {
    "bigbench": {
        "name": "BIG-Bench",
        "description": "Aggregate of 121 BIG-Bench tasks for comprehensive language model evaluation",
        "pattern": "bigbench_"
    },
    "bbh": {
        "name": "BIG-Bench Hard",
        "description": "Aggregate of 27 challenging BIG-Bench tasks that require multi-step reasoning",
        "pattern": "bbh_"
    },
    "agieval": {
        "name": "AGIEval",
        "description": "Aggregate of AGIEval tasks testing human-level reasoning across various domains",
        "pattern": "agieval_"
    },
    "ethics": {
        "name": "ETHICS",
        "description": "Aggregate of ETHICS tasks testing moral reasoning across different scenarios",
        "pattern": "ethics_"
    },
    "blimp": {
        "name": "BLiMP",
        "description": "Aggregate of BLiMP tasks testing linguistic phenomena understanding",
        "pattern": "blimp_"
    },
    "glue": {
        "name": "GLUE",
        "description": "General Language Understanding Evaluation - 10 NLU tasks",
        "pattern": "glue_"
    },
    "superglue": {
        "name": "SuperGLUE",
        "description": "SuperGLUE benchmark - 6 advanced NLU tasks",
        "tasks": ["copa", "rte_superglue", "wic", "wsc", "cb", "multirc"]
    },
    "global_mmlu": {
        "name": "Global-MMLU",
        "description": "Culturally adapted MMLU across 42 languages",
        "pattern": "global_mmlu_"
    },
    "mmmlu": {
        "name": "MMMLU",
        "description": "Multilingual MMLU across 14 languages",
        "tasks": []  # mmmlu doesn't have subtasks, it's parametric
    },
    "xcopa": {
        "name": "XCOPA",
        "description": "Cross-lingual Choice of Plausible Alternatives across 11 languages",
        "pattern": "xcopa_"
    },
    "xstorycloze": {
        "name": "XStoryCloze",
        "description": "Cross-lingual story completion across 11 languages",
        "pattern": "xstorycloze_"
    },
    "xwinograd": {
        "name": "XWinograd",
        "description": "Multilingual Winograd Schema Challenge across 6 languages",
        "pattern": "xwinograd_"
    },
    "headqa": {
        "name": "HeadQA",
        "description": "Healthcare exam questions in English and Spanish",
        "pattern": "headqa_"
    },
    "mgsm": {
        "name": "MGSM",
        "description": "Multilingual Grade School Math across all languages",
        "tasks": ["mgsm", "mgsm_en", "mgsm_latin", "mgsm_non_latin"]
    },
    "mmmu": {
        "name": "MMMU",
        "description": "Massive Multi-discipline Multimodal Understanding across 27 subjects",
        "pattern": "mmmu_"
    },
    "arabic_exams": {
        "name": "Arabic Exams",
        "description": "Arabic MMLU across multiple subjects",
        "pattern": "arabic_exams_"
    },
    "exercism": {
        "name": "Exercism",
        "description": "Multi-language coding tasks across 5 programming languages",
        "pattern": "exercism_"
    },
    "anli": {
        "name": "ANLI",
        "description": "Adversarial Natural Language Inference across 3 rounds",
        "pattern": "anli_"
    },
    "healthbench": {
        "name": "HealthBench",
        "description": "Medical dialogue evaluation across difficulty levels",
        "tasks": ["healthbench", "healthbench_hard", "healthbench_consensus"]
    },
    "openai_mrcr": {
        "name": "OpenAI MRCR",
        "description": "Memory-Recall with Contextual Retrieval across needle counts",
        "pattern": "openai_mrcr"
    },
    "mmmu_pro": {
        "name": "MMMU-Pro",
        "description": "Enhanced MMMU evaluation",
        "tasks": ["mmmu_pro", "mmmu_pro_vision"]
    },
    "arc": {
        "name": "ARC",
        "description": "AI2 Reasoning Challenge across difficulty levels",
        "tasks": ["arc_easy", "arc_challenge"]
    },
    "arc_agi": {
        "name": "ARC-AGI",
        "description": "Abstraction and Reasoning Corpus across versions",
        "tasks": ["arc_agi_1", "arc_agi_2"]
    },
    "hle": {
        "name": "HLE",
        "description": "Humanity's Last Exam variants",
        "tasks": ["hle", "hle_text"]
    },
    "math_dataset": {
        "name": "MATH Dataset",
        "description": "Mathematical problem solving",
        "tasks": ["math", "math_500"]
    },
    "qa4mre": {
        "name": "QA4MRE",
        "description": "Question Answering for Machine Reading Evaluation across years",
        "pattern": "qa4mre_"
    },
    "otis_mock_aime": {
        "name": "OTIS Mock AIME",
        "description": "Mock AIME competition problems",
        "pattern": "otis_mock_aime"
    },
    "matharena": {
        "name": "MathArena",
        "description": "Math competition problems from AIME, HMMT, and more",
        "tasks": [
            "aime_2023_I", "aime_2023_II", "aime_2024", "aime_2024_I", "aime_2024_II",
            "aime_2025", "aime_2025_II", "brumo_2025",
            "hmmt_feb_2023", "hmmt_feb_2024", "hmmt_feb_2025"
        ]
    },
    "cybench": {
        "name": "CyBench",
        "description": "Cybersecurity agent benchmark with 40 capture-the-flag challenges",
        "tasks": []  # cybench is a single eval, not a family
    },
    "race": {
        "name": "RACE",
        "description": "Reading comprehension from exams",
        "tasks": ["race_high"]  # Only race_high exists currently
    },
    "cti_bench": {
        "name": "CTI-Bench",
        "description": "Cyber Threat Intelligence benchmark",
        "pattern": "cti_bench_"
    },
}

# Generate EVAL_GROUPS entries
print('# Family benchmark groups')
print('# Generated by scripts/generate_family_groups.py\n')

for group_key, config in families.items():
    # Get tasks either from explicit list or by pattern
    if "tasks" in config and config["tasks"]:
        tasks = config["tasks"]
    elif "pattern" in config:
        tasks = find_tasks(config["pattern"])
    else:
        continue

    if not tasks:
        print(f"# Warning: No tasks found for {group_key}")
        continue

    print(f'    "{group_key}": EvalGroup(')
    print(f'        name="{config["name"]}",')
    print(f'        description="{config["description"]}",')
    print(f'        benchmarks=[')
    for task in tasks:
        print(f'            "{task}",')
    print(f'        ],')
    print(f'    ),')
    print()

print(f"\n# Total: {len([f for f in families.values() if 'tasks' in f or 'pattern' in f])} families")
