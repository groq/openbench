"""
Minimal configuration for benchmarks.
Only contains human-written metadata that cannot be extracted from code.
Everything else (epochs, temperature, etc.) comes from the actual task definitions.
"""

from dataclasses import dataclass
from functools import lru_cache
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Optional


@dataclass
class BenchmarkMetadata:
    """Minimal metadata for a benchmark - only what can't be extracted."""

    name: str  # Human-readable display name
    description: str  # Human-written description
    category: str  # Category for grouping
    tags: List[str]  # Tags for searchability

    # Registry info - still needed
    module_path: str
    function_name: str

    # Alpha/experimental flag
    is_alpha: bool = False  # Whether this benchmark is experimental/alpha


# Benchmark metadata - minimal, no duplication
BENCHMARKS = {
    "mbpp": BenchmarkMetadata(
        name="MBPP",
        description="Mostly Basic Python Problems — code generation tasks with unit test verification",
        category="core",
        tags=["code", "generation", "sandbox", "reasoning"],
        module_path="openbench.evals.mbpp",
        function_name="mbpp",
        is_alpha=False,
    ),
    # Graphwalks benchmarks (alpha)
    "clockbench": BenchmarkMetadata(
        name="ClockBench",
        description="Clock benchmark - time-based reasoning tasks",
        category="community",
        tags=["time", "analog", "clock", "reasoning"],
        module_path="openbench.evals.clockbench",
        function_name="clockbench",
        is_alpha=False,
    ),
    "graphwalks": BenchmarkMetadata(
        name="GraphWalks",
        description="Multi-hop reasoning on graphs - both BFS and parent finding tasks",
        category="core",
        tags=["long-context", "graphs", "reasoning", "alpha"],
        module_path="openbench.evals.graphwalks",
        function_name="graphwalks",
        is_alpha=True,
    ),
    "graphwalks_bfs": BenchmarkMetadata(
        name="GraphWalks BFS",
        description="Multi-hop reasoning on graphs - BFS traversal tasks only",
        category="core",
        tags=["long-context", "graphs", "reasoning", "bfs", "alpha"],
        module_path="openbench.evals.graphwalks",
        function_name="graphwalks_bfs",
        is_alpha=True,
    ),
    "graphwalks_parents": BenchmarkMetadata(
        name="GraphWalks Parents",
        description="Multi-hop reasoning on graphs - parent finding tasks only",
        category="core",
        tags=["long-context", "graphs", "reasoning", "parents", "alpha"],
        module_path="openbench.evals.graphwalks",
        function_name="graphwalks_parents",
        is_alpha=True,
    ),
    # Core benchmarks
    "mmlu": BenchmarkMetadata(
        name="MMLU (cais/mmlu)",
        description="Massive Multitask Language Understanding - 57 academic subjects from the cais/mmlu dataset. Only supports English (EN-US).",
        category="core",
        tags=["multiple-choice", "knowledge", "reasoning", "multitask"],
        module_path="openbench.evals.mmlu",
        function_name="mmlu",
    ),
    "mmlu-pro": BenchmarkMetadata(
        name="MMLU Pro (TIGER-Lab)",
        description="Enhanced version of MMLU with more challenging, reasoning-focused questions.",
        category="core",
        tags=["multiple-choice", "knowledge", "reasoning", "multitask"],
        module_path="openbench.evals.mmlu_pro",
        function_name="mmlu_pro",
    ),
    "mmmlu": BenchmarkMetadata(
        name="MMMLU (openai/MMMLU)",
        description="MMLU translated to 15 languages.",
        category="core",
        tags=["multiple-choice", "knowledge", "reasoning", "multitask"],
        module_path="openbench.evals.mmmlu",
        function_name="mmmlu",
    ),
    "openai_mrcr": BenchmarkMetadata(
        name="OpenAI MRCR (Full)",
        description="Memory-Recall with Contextual Retrieval - long-context evaluation that measures recall of 2, 4, and 8 needles across million-token contexts",
        category="core",
        tags=["long-context", "retrieval", "needle", "sequence-matching"],
        module_path="openbench.evals.mrcr",
        function_name="openai_mrcr",
    ),
    "openai_mrcr_2n": BenchmarkMetadata(
        name="OpenAI MRCR (2 Needles)",
        description="Memory-Recall with Contextual Retrieval - long-context evaluation that measures recall of 2 needles across million-token contexts",
        category="core",
        tags=["long-context", "retrieval", "needle", "sequence-matching"],
        module_path="openbench.evals.mrcr",
        function_name="openai_mrcr_2n",
    ),
    "openai_mrcr_4n": BenchmarkMetadata(
        name="OpenAI MRCR (4 Needles)",
        description="Memory-Recall with Contextual Retrieval - long-context evaluation that measures recall of 4 needles across million-token contexts",
        category="core",
        tags=["long-context", "retrieval", "needle", "sequence-matching"],
        module_path="openbench.evals.mrcr",
        function_name="openai_mrcr_4n",
    ),
    "openai_mrcr_8n": BenchmarkMetadata(
        name="OpenAI MRCR (8 Needles)",
        description="Memory-Recall with Contextual Retrieval - long-context evaluation that measures recall of 8 needles across million-token contexts",
        category="core",
        tags=["long-context", "retrieval", "needle", "sequence-matching"],
        module_path="openbench.evals.mrcr",
        function_name="openai_mrcr_8n",
    ),
    "gpqa_diamond": BenchmarkMetadata(
        name="GPQA Diamond",
        description="Graduate-level Google-Proof Q&A in biology, chemistry, and physics",
        category="core",
        tags=["multiple-choice", "science", "graduate-level"],
        module_path="openbench.evals.gpqa_diamond",
        function_name="gpqa_diamond",
    ),
    "humaneval": BenchmarkMetadata(
        name="HumanEval",
        description="Code generation benchmark with 164 programming problems",
        category="core",
        tags=["coding", "generation", "execution"],
        module_path="openbench.evals.humaneval",
        function_name="humaneval",
    ),
    # Exercism benchmarks
    "exercism": BenchmarkMetadata(
        name="Exercism",
        description="Multi-language coding benchmark with real-world programming exercises across Python, Go, JavaScript, Java, and Rust",
        category="core",
        tags=["coding", "multi-language", "execution", "docker"],
        module_path="openbench.evals.exercism.exercism",
        function_name="exercism",
    ),
    "exercism_python": BenchmarkMetadata(
        name="Exercism (Python)",
        description="Python coding tasks from the Exercism benchmark",
        category="core",
        tags=["coding", "python", "execution", "docker"],
        module_path="openbench.evals.exercism.exercism",
        function_name="exercism_python",
    ),
    "exercism_javascript": BenchmarkMetadata(
        name="Exercism (JavaScript)",
        description="JavaScript coding tasks from the Exercism benchmark",
        category="core",
        tags=["coding", "javascript", "execution", "docker"],
        module_path="openbench.evals.exercism.exercism",
        function_name="exercism_javascript",
    ),
    "exercism_go": BenchmarkMetadata(
        name="Exercism (Go)",
        description="Go coding tasks from the Exercism benchmark",
        category="core",
        tags=["coding", "go", "execution", "docker"],
        module_path="openbench.evals.exercism.exercism",
        function_name="exercism_go",
    ),
    "exercism_java": BenchmarkMetadata(
        name="Exercism (Java)",
        description="Java coding tasks from the Exercism benchmark",
        category="core",
        tags=["coding", "java", "execution", "docker"],
        module_path="openbench.evals.exercism.exercism",
        function_name="exercism_java",
    ),
    "exercism_rust": BenchmarkMetadata(
        name="Exercism (Rust)",
        description="Rust coding tasks from the Exercism benchmark",
        category="core",
        tags=["coding", "rust", "execution", "docker"],
        module_path="openbench.evals.exercism.exercism",
        function_name="exercism_rust",
    ),
    "ifeval": BenchmarkMetadata(
        name="Instruction Following",
        description="Tests ability to follow specific formatting and content constraints with both strict and loose evaluation metrics",
        category="core",
        tags=["instruction-following", "constraints", "formatting"],
        module_path="openbench.evals.ifeval",
        function_name="ifeval",
    ),
    "openbookqa": BenchmarkMetadata(
        name="OpenBookQA",
        description="Elementary-level science questions probing understanding of core facts",
        category="core",
        tags=["multiple-choice", "science", "elementary", "open-book"],
        module_path="openbench.evals.openbookqa",
        function_name="openbookqa",
    ),
    "musr": BenchmarkMetadata(
        name="MuSR",
        description="Testing the Limits of Chain-of-thought with Multistep Soft Reasoning - includes murder mysteries, object placements, and team allocation tasks",
        category="core",
        tags=["multiple-choice", "reasoning", "commonsense", "chain-of-thought"],
        module_path="openbench.evals.musr",
        function_name="musr",
    ),
    "musr_murder_mysteries": BenchmarkMetadata(
        name="MuSR Murder Mysteries",
        description="MuSR murder mystery scenarios - who is the most likely murderer?",
        category="core",
        tags=[
            "multiple-choice",
            "reasoning",
            "commonsense",
            "chain-of-thought",
            "murder-mysteries",
        ],
        module_path="openbench.evals.musr",
        function_name="musr_murder_mysteries",
    ),
    "musr_object_placements": BenchmarkMetadata(
        name="MuSR Object Placements",
        description="MuSR object placement reasoning - where would someone look for an object?",
        category="core",
        tags=[
            "multiple-choice",
            "reasoning",
            "commonsense",
            "chain-of-thought",
            "object-placements",
        ],
        module_path="openbench.evals.musr",
        function_name="musr_object_placements",
    ),
    "musr_team_allocation": BenchmarkMetadata(
        name="MuSR Team Allocation",
        description="MuSR team allocation problems - how to allocate people to tasks efficiently?",
        category="core",
        tags=[
            "multiple-choice",
            "reasoning",
            "commonsense",
            "chain-of-thought",
            "team-allocation",
        ],
        module_path="openbench.evals.musr",
        function_name="musr_team_allocation",
    ),
    "supergpqa": BenchmarkMetadata(
        name="SuperGPQA",
        description="Scaling LLM Evaluation across 285 Graduate Disciplines - 26,529 multiple-choice questions across science, engineering, medicine, economics, and philosophy",
        category="core",
        tags=["multiple-choice", "knowledge", "graduate-level", "multidisciplinary"],
        module_path="openbench.evals.supergpqa",
        function_name="supergpqa",
    ),
    "simpleqa": BenchmarkMetadata(
        name="SimpleQA",
        description="Measuring short-form factuality in large language models with simple Q&A pairs",
        category="core",
        tags=["factuality", "question-answering", "graded"],
        module_path="openbench.evals.simpleqa",
        function_name="simpleqa",
    ),
    "tumlu": BenchmarkMetadata(
        name="TUMLU",
        description="TUMLU is a comprehensive, multilingual, and natively developed language understanding benchmark specifically designed for Turkic languages.",
        category="community",
        tags=["factuality", "question-answering", "multiple-choice", "reasoning"],
        module_path="openbench.evals.tumlu",
        function_name="tumlu",
    ),
    "detailbench": BenchmarkMetadata(
        name="DetailBench",
        description="Tests whether LLMs notify users about wrong facts in a text while they are tasked to translate said text",
        category="community",
        tags=["knowledge", "graded", "instruction-following"],
        module_path="openbench.evals.detailbench",
        function_name="detailbench",
    ),
    "browsecomp": BenchmarkMetadata(
        name="BrowseComp",
        description="A Simple Yet Challenging Benchmark for Browsing Agents - evaluates model performance on browsing-related tasks",
        category="core",
        tags=["browsing", "web", "reasoning", "graded"],
        module_path="openbench.evals.browsecomp",
        function_name="browsecomp",
    ),
    "hle": BenchmarkMetadata(
        name="Humanity's Last Exam",
        description="Multi-modal benchmark at the frontier of human knowledge - 2,500 questions across mathematics, humanities, and natural sciences designed by subject-matter experts globally",
        category="core",
        tags=["knowledge", "reasoning", "multi-modal", "graded", "frontier"],
        module_path="openbench.evals.hle",
        function_name="hle",
    ),
    "hle_text": BenchmarkMetadata(
        name="Humanity's Last Exam (Text-Only)",
        description="Text-only variant of HLE with multi-modal questions filtered out - evaluates models without vision capabilities on text-based questions from the frontier of human knowledge",
        category="core",
        tags=["knowledge", "reasoning", "text-only", "graded", "frontier"],
        module_path="openbench.evals.hle",
        function_name="hle_text",
    ),
    "mmstar": BenchmarkMetadata(
        name="MMStar",
        description="MMStar benchmark for measuring multi-modal gain and leakage via coordinated vision and text ablations",
        category="core",
        tags=["vision", "multi-modal", "leakage", "perception", "reasoning"],
        module_path="openbench.evals.mmstar",
        function_name="mmstar",
    ),
    "healthbench": BenchmarkMetadata(
        name="HealthBench",
        description="Medical dialogue evaluation using physician-created rubrics for assessing healthcare conversations",
        category="core",
        tags=["medical", "dialogue", "graded", "rubric-based"],
        module_path="openbench.evals.healthbench",
        function_name="healthbench",
    ),
    "healthbench_hard": BenchmarkMetadata(
        name="HealthBench Hard",
        description="Most challenging medical dialogue cases from HealthBench requiring nuanced medical knowledge",
        category="core",
        tags=["medical", "dialogue", "graded", "rubric-based", "hard"],
        module_path="openbench.evals.healthbench",
        function_name="healthbench_hard",
    ),
    "healthbench_consensus": BenchmarkMetadata(
        name="HealthBench Consensus",
        description="Medical dialogue cases with strong physician consensus on appropriate responses",
        category="core",
        tags=["medical", "dialogue", "graded", "rubric-based", "consensus"],
        module_path="openbench.evals.healthbench",
        function_name="healthbench_consensus",
    ),
    "mgsm": BenchmarkMetadata(
        name="MGSM",
        description="Multilingual Grade School Math benchmark across 11 languages for testing mathematical reasoning",
        category="core",
        tags=["math", "multilingual", "reasoning", "chain-of-thought"],
        module_path="openbench.evals.mgsm",
        function_name="mgsm",
    ),
    "mgsm_en": BenchmarkMetadata(
        name="MGSM English",
        description="Grade school math problems in English for testing mathematical reasoning",
        category="core",
        tags=["math", "english", "reasoning", "chain-of-thought"],
        module_path="openbench.evals.mgsm",
        function_name="mgsm_en",
    ),
    "mgsm_latin": BenchmarkMetadata(
        name="MGSM Latin Script",
        description="Grade school math problems in Latin script languages (German, English, Spanish, French, Swahili)",
        category="core",
        tags=["math", "multilingual", "latin-script", "reasoning", "chain-of-thought"],
        module_path="openbench.evals.mgsm",
        function_name="mgsm_latin",
    ),
    "mgsm_non_latin": BenchmarkMetadata(
        name="MGSM Non-Latin Script",
        description="Grade school math problems in non-Latin script languages (Bengali, Japanese, Russian, Telugu, Thai, Chinese)",
        category="core",
        tags=[
            "math",
            "multilingual",
            "non-latin-script",
            "reasoning",
            "chain-of-thought",
        ],
        module_path="openbench.evals.mgsm",
        function_name="mgsm_non_latin",
    ),
    "drop": BenchmarkMetadata(
        name="DROP",
        description="Reading comprehension benchmark requiring discrete reasoning over paragraphs (arithmetic, counting, sorting)",
        category="core",
        tags=[
            "reading-comprehension",
            "reasoning",
            "arithmetic",
            "counting",
            "sorting",
        ],
        module_path="openbench.evals.drop",
        function_name="drop",
    ),
    "math": BenchmarkMetadata(
        name="MATH",
        description="Measuring Mathematical Problem Solving - 5000 competition math problems across 7 subjects and 5 difficulty levels",
        category="core",
        tags=["math", "problem-solving", "reasoning", "competition", "graded"],
        module_path="openbench.evals.math",
        function_name="math",
    ),
    "math_500": BenchmarkMetadata(
        name="MATH-500",
        description="500-problem subset of MATH dataset for faster evaluation of mathematical problem solving",
        category="core",
        tags=[
            "math",
            "problem-solving",
            "reasoning",
            "competition",
            "graded",
            "subset",
        ],
        module_path="openbench.evals.math",
        function_name="math_500",
    ),
    # Math competitions
    "aime_2023_I": BenchmarkMetadata(
        name="AIME 2023 I",
        description="American Invitational Mathematics Examination 2023 (First)",
        category="math",
        tags=["math", "competition", "aime", "2023"],
        module_path="openbench.evals.matharena.aime_2023_I.aime_2023_I",
        function_name="aime_2023_I",
    ),
    "aime_2023_II": BenchmarkMetadata(
        name="AIME 2023 II",
        description="American Invitational Mathematics Examination 2023 (Second)",
        category="math",
        tags=["math", "competition", "aime", "2023"],
        module_path="openbench.evals.matharena.aime_2023_II.aime_2023_II",
        function_name="aime_2023_II",
    ),
    "aime_2024": BenchmarkMetadata(
        name="AIME 2024",
        description="American Invitational Mathematics Examination 2024 (Combined I & II)",
        category="math",
        tags=["math", "competition", "aime", "2024", "combined"],
        module_path="openbench.evals.matharena.aime_2024.aime_2024",
        function_name="aime_2024",
    ),
    "aime_2024_I": BenchmarkMetadata(
        name="AIME 2024 I",
        description="American Invitational Mathematics Examination 2024 (First)",
        category="math",
        tags=["math", "competition", "aime", "2024"],
        module_path="openbench.evals.matharena.aime_2024_I.aime_2024_I",
        function_name="aime_2024_I",
    ),
    "aime_2024_II": BenchmarkMetadata(
        name="AIME 2024 II",
        description="American Invitational Mathematics Examination 2024 (Second)",
        category="math",
        tags=["math", "competition", "aime", "2024"],
        module_path="openbench.evals.matharena.aime_2024_II.aime_2024_II",
        function_name="aime_2024_II",
    ),
    "aime_2025": BenchmarkMetadata(
        name="AIME 2025",
        description="American Invitational Mathematics Examination 2025",
        category="math",
        tags=["math", "competition", "aime", "2025"],
        module_path="openbench.evals.matharena.aime_2025.aime_2025",
        function_name="aime_2025",
    ),
    "aime_2025_II": BenchmarkMetadata(
        name="AIME 2025 II",
        description="American Invitational Mathematics Examination 2025 (Second)",
        category="math",
        tags=["math", "competition", "aime", "2025"],
        module_path="openbench.evals.matharena.aime_2025_II.aime_2025_II",
        function_name="aime_2025_II",
    ),
    "brumo_2025": BenchmarkMetadata(
        name="BRUMO 2025",
        description="Bruno Mathematical Olympiad 2025",
        category="math",
        tags=["math", "competition", "olympiad", "2025"],
        module_path="openbench.evals.matharena.brumo_2025.brumo_2025",
        function_name="brumo_2025",
    ),
    "hmmt_feb_2023": BenchmarkMetadata(
        name="HMMT Feb 2023",
        description="Harvard-MIT Mathematics Tournament February 2023",
        category="math",
        tags=["math", "competition", "hmmt", "2023"],
        module_path="openbench.evals.matharena.hmmt_feb_2023.hmmt_feb_2023",
        function_name="hmmt_feb_2023",
    ),
    "hmmt_feb_2024": BenchmarkMetadata(
        name="HMMT Feb 2024",
        description="Harvard-MIT Mathematics Tournament February 2024",
        category="math",
        tags=["math", "competition", "hmmt", "2024"],
        module_path="openbench.evals.matharena.hmmt_feb_2024.hmmt_feb_2024",
        function_name="hmmt_feb_2024",
    ),
    "hmmt_feb_2025": BenchmarkMetadata(
        name="HMMT Feb 2025",
        description="Harvard-MIT Mathematics Tournament February 2025",
        category="math",
        tags=["math", "competition", "hmmt", "2025"],
        module_path="openbench.evals.matharena.hmmt_feb_2025.hmmt_feb_2025",
        function_name="hmmt_feb_2025",
    ),
    "arc_easy": BenchmarkMetadata(
        name="ARC-Easy",
        description="AI2 Reasoning Challenge - Easy questions from grade-school science exams",
        category="core",
        tags=["multiple-choice", "science", "commonsense-reasoning"],
        module_path="openbench.evals.arc",
        function_name="arc_easy",
    ),
    "arc_challenge": BenchmarkMetadata(
        name="ARC-Challenge",
        description="AI2 Reasoning Challenge - Challenging questions from grade-school science exams",
        category="core",
        tags=["multiple-choice", "science", "commonsense-reasoning"],
        module_path="openbench.evals.arc",
        function_name="arc_challenge",
    ),
    "boolq": BenchmarkMetadata(
        name="BoolQ",
        description="BoolQ: A Question Answering Dataset for Boolean Reasoning",
        category="core",
        tags=["boolean-reasoning", "question-answering"],
        module_path="openbench.evals.boolq",
        function_name="boolq",
    ),
    "hellaswag": BenchmarkMetadata(
        name="HellaSwag",
        description="Adversarially-filtered sentence completion benchmark for commonsense reasoning",
        category="core",
        tags=["multiple-choice", "commonsense-reasoning", "sentence-completion"],
        module_path="openbench.evals.hellaswag",
        function_name="hellaswag",
    ),
    "piqa": BenchmarkMetadata(
        name="PIQA",
        description="Physical Interaction Question Answering - commonsense about physical situations",
        category="core",
        tags=["multiple-choice", "commonsense-reasoning", "physical-reasoning"],
        module_path="openbench.evals.piqa",
        function_name="piqa",
    ),
    "prost": BenchmarkMetadata(
        name="PROST",
        description="Physical Reasoning about Objects through Space and Time",
        category="core",
        tags=["multiple-choice", "commonsense-reasoning", "physical-reasoning"],
        module_path="openbench.evals.prost",
        function_name="prost",
    ),
    "swag": BenchmarkMetadata(
        name="SWAG",
        description="Situations With Adversarial Generations - grounded commonsense inference",
        category="core",
        tags=["multiple-choice", "commonsense-reasoning", "video-captions"],
        module_path="openbench.evals.swag",
        function_name="swag",
    ),
    "winogrande": BenchmarkMetadata(
        name="WinoGrande",
        description="Large-scale Winograd Schema Challenge for commonsense pronoun resolution",
        category="core",
        tags=["multiple-choice", "commonsense-reasoning", "pronoun-resolution"],
        module_path="openbench.evals.winogrande",
        function_name="winogrande",
    ),
    "wsc273": BenchmarkMetadata(
        name="WSC273",
        description="Original Winograd Schema Challenge with 273 expert-crafted questions",
        category="core",
        tags=["multiple-choice", "commonsense-reasoning", "pronoun-resolution"],
        module_path="openbench.evals.wsc273",
        function_name="wsc273",
    ),
    "scicode": BenchmarkMetadata(
        name="SciCode",
        description="Scientific computing and programming challenges",
        category="core",
        tags=["code-generation", "science", "alpha"],
        module_path="openbench.evals.scicode",
        function_name="scicode",
        is_alpha=True,
    ),
    "cti_bench_ate": BenchmarkMetadata(
        name="CTI-Bench ATE",
        description="Extracting MITRE ATT&CK techniques from malware and threat descriptions",
        category="cybersecurity",
        tags=["extraction", "cybersecurity"],
        module_path="openbench.evals.cti_bench",
        function_name="cti_bench_ate",
    ),
    "cti_bench_mcq": BenchmarkMetadata(
        name="CTI-Bench MCQ",
        description="Multiple-choice questions evaluating understanding of CTI standards, threats, detection strategies, and best practices using authoritative sources like NIST and MITRE",
        category="cybersecurity",
        tags=["multiple-choice", "cybersecurity", "knowledge"],
        module_path="openbench.evals.cti_bench",
        function_name="cti_bench_mcq",
    ),
    "cti_bench_rcm": BenchmarkMetadata(
        name="CTI-Bench RCM",
        description="Mapping CVE descriptions to CWE categories to evaluate vulnerability classification ability",
        category="cybersecurity",
        tags=["classification", "cybersecurity"],
        module_path="openbench.evals.cti_bench",
        function_name="cti_bench_rcm",
    ),
    "cti_bench_vsp": BenchmarkMetadata(
        name="CTI-Bench VSP",
        description="Calculating CVSS scores from vulnerability descriptions to assess severity evaluation skills",
        category="cybersecurity",
        tags=["regression", "cybersecurity"],
        module_path="openbench.evals.cti_bench",
        function_name="cti_bench_vsp",
    ),
    "rootly_gmcq": BenchmarkMetadata(
        name="GMCQ",
        description="GitHub Multiple Choice Questions",
        category="core",
        tags=["code-understanding"],
        module_path="openbench.evals.rootly_gmcq",
        function_name="rootly_gmcq",
    ),
    "rootly_terraform": BenchmarkMetadata(
        name="Terraform",
        description="Terraform Multiple Choice Questions",
        category="core",
        tags=["code-understanding"],
        module_path="openbench.evals.rootly_terraform",
        function_name="rootly_terraform",
    ),
    "jsonschemabench": BenchmarkMetadata(
        name="JSONSchemaBench",
        description="JSON Schema generation benchmark with ~10K real-world schemas from GitHub, Kubernetes, and other sources for evaluating constrained decoding",
        category="core",
        tags=["json", "jsonschema", "generation", "constrained-decoding"],
        module_path="openbench.evals.jsonschemabench",
        function_name="jsonschemabench",
        is_alpha=False,
    ),
    "mmmu": BenchmarkMetadata(
        name="MMMU",
        description="Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark with 11.5K questions across 30 subjects from college exams, quizzes, and textbooks",
        category="core",
        tags=["multimodal", "multiple-choice", "reasoning", "college-level", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu",
        is_alpha=False,
    ),
    "mmmu_art": BenchmarkMetadata(
        name="MMMU Art",
        description="MMMU Art subset focusing on art and visual design questions",
        category="core",
        tags=["multimodal", "multiple-choice", "art", "visual-design", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_art",
        is_alpha=False,
    ),
    "mmmu_biology": BenchmarkMetadata(
        name="MMMU Biology",
        description="MMMU Biology subset focusing on biological sciences",
        category="core",
        tags=["multimodal", "multiple-choice", "biology", "science", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_biology",
        is_alpha=False,
    ),
    "mmmu_chemistry": BenchmarkMetadata(
        name="MMMU Chemistry",
        description="MMMU Chemistry subset focusing on chemical sciences",
        category="core",
        tags=["multimodal", "multiple-choice", "chemistry", "science", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_chemistry",
        is_alpha=False,
    ),
    "mmmu_math": BenchmarkMetadata(
        name="MMMU Math",
        description="MMMU Mathematics subset focusing on mathematical reasoning",
        category="math",
        tags=["multimodal", "multiple-choice", "mathematics", "reasoning", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_math",
        is_alpha=False,
    ),
    "mmmu_physics": BenchmarkMetadata(
        name="MMMU Physics",
        description="MMMU Physics subset focusing on physics and physical sciences",
        category="core",
        tags=["multimodal", "multiple-choice", "physics", "science", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_physics",
        is_alpha=False,
    ),
    "mmmu_accounting": BenchmarkMetadata(
        name="MMMU Accounting",
        description="MMMU Accounting subset focusing on accounting principles and practices",
        category="core",
        tags=["multimodal", "multiple-choice", "accounting", "business", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_accounting",
        is_alpha=False,
    ),
    "mmmu_agriculture": BenchmarkMetadata(
        name="MMMU Agriculture",
        description="MMMU Agriculture subset focusing on agricultural sciences and practices",
        category="core",
        tags=["multimodal", "multiple-choice", "agriculture", "science", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_agriculture",
        is_alpha=False,
    ),
    "mmmu_architecture_and_engineering": BenchmarkMetadata(
        name="MMMU Architecture and Engineering",
        description="MMMU Architecture and Engineering subset focusing on engineering design and architecture",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "architecture",
            "engineering",
            "design",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_architecture_and_engineering",
        is_alpha=False,
    ),
    "mmmu_art_theory": BenchmarkMetadata(
        name="MMMU Art Theory",
        description="MMMU Art Theory subset focusing on art history and theoretical concepts",
        category="core",
        tags=["multimodal", "multiple-choice", "art", "theory", "history", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_art_theory",
        is_alpha=False,
    ),
    "mmmu_basic_medical_science": BenchmarkMetadata(
        name="MMMU Basic Medical Science",
        description="MMMU Basic Medical Science subset focusing on fundamental medical knowledge",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "medicine",
            "science",
            "health",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_basic_medical_science",
        is_alpha=False,
    ),
    "mmmu_clinical_medicine": BenchmarkMetadata(
        name="MMMU Clinical Medicine",
        description="MMMU Clinical Medicine subset focusing on clinical medical practice",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "medicine",
            "clinical",
            "health",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_clinical_medicine",
        is_alpha=False,
    ),
    "mmmu_design": BenchmarkMetadata(
        name="MMMU Design",
        description="MMMU Design subset focusing on design principles and practices",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "design",
            "visual",
            "creative",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_design",
        is_alpha=False,
    ),
    "mmmu_diagnostics_and_laboratory_medicine": BenchmarkMetadata(
        name="MMMU Diagnostics and Laboratory Medicine",
        description="MMMU Diagnostics and Laboratory Medicine subset focusing on medical diagnostics",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "medicine",
            "diagnostics",
            "laboratory",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_diagnostics_and_laboratory_medicine",
        is_alpha=False,
    ),
    "mmmu_electronics": BenchmarkMetadata(
        name="MMMU Electronics",
        description="MMMU Electronics subset focusing on electronic systems and circuits",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "electronics",
            "engineering",
            "technology",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_electronics",
        is_alpha=False,
    ),
    "mmmu_energy_and_power": BenchmarkMetadata(
        name="MMMU Energy and Power",
        description="MMMU Energy and Power subset focusing on energy systems and power generation",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "energy",
            "power",
            "engineering",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_energy_and_power",
        is_alpha=False,
    ),
    "mmmu_finance": BenchmarkMetadata(
        name="MMMU Finance",
        description="MMMU Finance subset focusing on financial concepts and analysis",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "finance",
            "business",
            "economics",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_finance",
        is_alpha=False,
    ),
    "mmmu_geography": BenchmarkMetadata(
        name="MMMU Geography",
        description="MMMU Geography subset focusing on geographical knowledge and analysis",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "geography",
            "earth-science",
            "spatial",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_geography",
        is_alpha=False,
    ),
    "mmmu_history": BenchmarkMetadata(
        name="MMMU History",
        description="MMMU History subset focusing on historical knowledge and analysis",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "history",
            "humanities",
            "culture",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_history",
        is_alpha=False,
    ),
    "mmmu_literature": BenchmarkMetadata(
        name="MMMU Literature",
        description="MMMU Literature subset focusing on literary analysis and knowledge",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "literature",
            "humanities",
            "language",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_literature",
        is_alpha=False,
    ),
    "mmmu_manage": BenchmarkMetadata(
        name="MMMU Management",
        description="MMMU Management subset focusing on management principles and practices",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "management",
            "business",
            "leadership",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_manage",
        is_alpha=False,
    ),
    "mmmu_marketing": BenchmarkMetadata(
        name="MMMU Marketing",
        description="MMMU Marketing subset focusing on marketing strategies and concepts",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "marketing",
            "business",
            "communication",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_marketing",
        is_alpha=False,
    ),
    "mmmu_materials": BenchmarkMetadata(
        name="MMMU Materials",
        description="MMMU Materials subset focusing on materials science and engineering",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "materials",
            "science",
            "engineering",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_materials",
        is_alpha=False,
    ),
    "mmmu_mechanical_engineering": BenchmarkMetadata(
        name="MMMU Mechanical Engineering",
        description="MMMU Mechanical Engineering subset focusing on mechanical systems and design",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "mechanical",
            "engineering",
            "design",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_mechanical_engineering",
        is_alpha=False,
    ),
    "mmmu_music": BenchmarkMetadata(
        name="MMMU Music",
        description="MMMU Music subset focusing on music theory and analysis",
        category="core",
        tags=["multimodal", "multiple-choice", "music", "arts", "theory", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_music",
        is_alpha=False,
    ),
    "mmmu_pharmacy": BenchmarkMetadata(
        name="MMMU Pharmacy",
        description="MMMU Pharmacy subset focusing on pharmaceutical sciences and practice",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "pharmacy",
            "medicine",
            "health",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_pharmacy",
        is_alpha=False,
    ),
    "mmmu_public_health": BenchmarkMetadata(
        name="MMMU Public Health",
        description="MMMU Public Health subset focusing on public health concepts and practices",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "public-health",
            "health",
            "population",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_public_health",
        is_alpha=False,
    ),
    "mmmu_sociology": BenchmarkMetadata(
        name="MMMU Sociology",
        description="MMMU Sociology subset focusing on sociological concepts and analysis",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "sociology",
            "social-science",
            "society",
            "images",
        ],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_sociology",
        is_alpha=False,
    ),
    "mmmu_mcq": BenchmarkMetadata(
        name="MMMU MCQ",
        description="MMMU MCQ subset focusing on multiple choice questions",
        category="core",
        tags=["multimodal", "multiple-choice", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_mcq",
        is_alpha=False,
    ),
    "mmmu_open": BenchmarkMetadata(
        name="MMMU Open",
        description="MMMU Open subset focusing on open-ended questions",
        category="core",
        tags=["multimodal", "open-ended", "images"],
        module_path="openbench.evals.mmmu",
        function_name="mmmu_open",
        is_alpha=False,
    ),
    "mmmu_pro": BenchmarkMetadata(
        name="MMMU-Pro",
        description="Enhanced multimodal MMMU-Pro benchmark with multiple-choice across many options and images",
        category="core",
        tags=[
            "multimodal",
            "multiple-choice",
            "reasoning",
            "images",
            "mmmu-pro",
        ],
        module_path="openbench.evals.mmmu_pro",
        function_name="mmmu_pro",
        is_alpha=False,
    ),
    "mmmu_pro_vision": BenchmarkMetadata(
        name="MMMU-Pro (Vision)",
        description="MMMU-Pro vision subset with images and multiple-choice questions",
        category="core",
        tags=["multimodal", "vision", "multiple-choice", "images", "mmmu-pro"],
        module_path="openbench.evals.mmmu_pro",
        function_name="mmmu_pro_vision",
        is_alpha=False,
    ),
    "arc_agi": BenchmarkMetadata(
        name="ARC-AGI",
        description="Abstraction and Reasoning Corpus for Artificial General Intelligence; specify version with -T version=1 or version=2",
        category="core",
        tags=[
            "reasoning",
            "pattern-recognition",
            "abstract-reasoning",
            "visual",
            "logic",
            "agi",
        ],
        module_path="openbench.evals.arc_agi",
        function_name="arc_agi",
    ),
    "arc_agi_1": BenchmarkMetadata(
        name="ARC-AGI-1",
        description="ARC-AGI dataset version 1",
        category="core",
        tags=[
            "reasoning",
            "pattern-recognition",
            "abstract-reasoning",
            "visual",
            "logic",
            "agi",
        ],
        module_path="openbench.evals.arc_agi",
        function_name="arc_agi_1",
    ),
    "arc_agi_2": BenchmarkMetadata(
        name="ARC-AGI-2",
        description="ARC-AGI dataset version 2",
        category="core",
        tags=[
            "reasoning",
            "pattern-recognition",
            "abstract-reasoning",
            "visual",
            "logic",
            "agi",
        ],
        module_path="openbench.evals.arc_agi",
        function_name="arc_agi_2",
    ),
    "livemcpbench": BenchmarkMetadata(
        name="LiveMCPBench",
        description="Benchmark for evaluating LLM agents on real-world tasks using the Model Context Protocol (MCP) - 95 tasks across different categories",
        category="core",
        tags=["mcp", "agents", "real-world", "tools", "graded"],
        module_path="openbench.evals.livemcpbench",
        function_name="livemcpbench",
        is_alpha=False,
    ),
}


def _normalize_benchmark_key(name: str) -> str:
    """Normalize benchmark keys so '-' and '_' are treated the same."""

    return name.replace("-", "_")


def _build_normalized_lookup(names: Iterable[str]) -> dict[str, str]:
    """Build a lookup mapping normalized benchmark keys to canonical names."""

    lookup: dict[str, str] = {}
    for key in names:
        normalized = _normalize_benchmark_key(key)
        existing = lookup.get(normalized)
        if existing and existing != key:
            raise ValueError(
                "Benchmark names cannot differ only by '-' vs '_' ("
                f"conflict between '{existing}' and '{key}')"
            )
        lookup[normalized] = key
    return lookup


_NORMALIZED_BENCHMARK_NAMES = _build_normalized_lookup(BENCHMARKS.keys())


def get_benchmark_metadata(name: str) -> Optional[BenchmarkMetadata]:
    """Get benchmark metadata by name."""
    metadata = BENCHMARKS.get(name)
    if metadata:
        return metadata

    canonical_name = _NORMALIZED_BENCHMARK_NAMES.get(_normalize_benchmark_key(name))
    if canonical_name:
        return BENCHMARKS[canonical_name]

    return None


def get_all_benchmarks(include_alpha: bool = False) -> dict[str, BenchmarkMetadata]:
    """Get all benchmark metadata.

    Args:
        include_alpha: Whether to include alpha/experimental benchmarks
    """
    if include_alpha:
        return BENCHMARKS
    return {name: meta for name, meta in BENCHMARKS.items() if not meta.is_alpha}


def get_benchmarks_by_category(
    category: str, include_alpha: bool = False
) -> dict[str, BenchmarkMetadata]:
    """Get all benchmarks in a category.

    Args:
        category: Category to filter by
        include_alpha: Whether to include alpha/experimental benchmarks
    """
    results = {
        name: meta for name, meta in BENCHMARKS.items() if meta.category == category
    }
    if not include_alpha:
        results = {name: meta for name, meta in results.items() if not meta.is_alpha}
    return results


def get_categories() -> List[str]:
    """Get all available categories."""
    return sorted(list(set(meta.category for meta in BENCHMARKS.values())))


def search_benchmarks(
    query: str, include_alpha: bool = False
) -> dict[str, BenchmarkMetadata]:
    """Search benchmarks by name, description, or tags.

    Args:
        query: Search query
        include_alpha: Whether to include alpha/experimental benchmarks
    """
    query = query.lower()
    results = {}

    for name, meta in BENCHMARKS.items():
        if not include_alpha and meta.is_alpha:
            continue
        if (
            query in meta.name.lower()
            or query in meta.description.lower()
            or any(query in tag.lower() for tag in meta.tags)
        ):
            results[name] = meta

    return results


# ============================================================================
# Task Loading for CLI
# ============================================================================


def _generate_task_registry(include_alpha: bool = True):
    """Generate task registry from config.

    Args:
        include_alpha: Whether to include alpha/experimental benchmarks
    """
    registry = {}
    for name, metadata in get_all_benchmarks(include_alpha=include_alpha).items():
        registry[name] = f"{metadata.module_path}.{metadata.function_name}"
    return registry


# Full registry including alpha benchmarks for backward compatibility
TASK_REGISTRY = _generate_task_registry(include_alpha=True)

_NORMALIZED_TASK_REGISTRY = _build_normalized_lookup(TASK_REGISTRY.keys())


def _import_module_from_path(path: Path) -> ModuleType:
    """
    Import a .py file or package directory as an anonymous module.
    """
    file_path = path
    if path.is_dir():
        file_path = path / "__init__.py"
        if not file_path.exists():
            raise ValueError(f"{path} is a directory but has no __init__.py")

    mod_name = f"_openbench_dyn_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)

    # For packages, set up proper package structure for relative imports
    if path.is_dir():
        module.__package__ = mod_name
        sys.modules[mod_name] = module

        # Pre-load submodules to support relative imports
        for submodule_file in path.glob("*.py"):
            if submodule_file.name != "__init__.py":
                submodule_name = submodule_file.stem
                submodule_full_name = f"{mod_name}.{submodule_name}"
                submodule_spec = importlib.util.spec_from_file_location(
                    submodule_full_name, str(submodule_file)
                )
                if submodule_spec and submodule_spec.loader:
                    submodule = importlib.util.module_from_spec(submodule_spec)
                    submodule.__package__ = mod_name
                    sys.modules[submodule_full_name] = submodule
                    submodule_spec.loader.exec_module(submodule)
    else:
        sys.modules[mod_name] = module

    spec.loader.exec_module(module)
    return module


@lru_cache()
def load_task(benchmark_name: str, allow_alpha: bool = False) -> Callable:
    """
    Loads a task by benchmark name using the registry or from a local path.

    Args:
        benchmark_name (str): The name of the benchmark or path to a local eval.
        allow_alpha (bool): Whether to allow loading alpha/experimental benchmarks.

    Returns:
        Callable: The imported function object.

    Raises:
        ValueError: If the benchmark is not in the registry and not a valid path.
        ImportError: If the module cannot be imported.
        AttributeError: If the function does not exist in the module.
    """
    # Check if this is an alpha benchmark
    benchmark_meta = get_benchmark_metadata(benchmark_name)
    if benchmark_meta and benchmark_meta.is_alpha and not allow_alpha:
        raise ValueError(
            f"'{benchmark_name}' is an experimental/alpha benchmark. "
            f"Use --alpha flag to run it."
        )

    # Try registry first (registry names take precedence)
    import_path = TASK_REGISTRY.get(benchmark_name)
    if import_path is None:
        canonical_name = _NORMALIZED_TASK_REGISTRY.get(
            _normalize_benchmark_key(benchmark_name)
        )
        if canonical_name:
            import_path = TASK_REGISTRY[canonical_name]
            benchmark_name = canonical_name
    if import_path:
        module_path, func_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    # Fallback to path-based loading
    path = Path(benchmark_name).expanduser()
    if path.exists():
        return _load_task_from_local_path(path)

    # Neither registry nor valid path
    raise ValueError(
        f"Unknown benchmark: '{benchmark_name}'. "
        f"Available benchmarks: {', '.join(TASK_REGISTRY.keys())}"
    )


def _load_task_from_local_path(path: Path) -> Callable:
    """
    Load a task from a local path containing __metadata__.

    Args:
        path: Path to a directory or .py file containing an eval

    Returns:
        Callable: The imported function object

    Raises:
        ValueError: If no valid __metadata__ is found
        AttributeError: If the function does not exist in the module
        ImportError: If the module cannot be imported
    """
    root_module = _import_module_from_path(path)
    metadata = getattr(root_module, "__metadata__", None)

    if not isinstance(metadata, BenchmarkMetadata):
        raise ValueError(f"{path} has no valid __metadata__")

    # Resolve module path relative to root module
    # For local evals, module_path is typically relative like "simpleqa.simpleqa"
    # We need to extract just the last part and combine with the root module name
    if metadata.module_path.startswith(root_module.__name__):
        full_module_name = metadata.module_path
    else:
        # For paths like "simpleqa.simpleqa", we want the last component "simpleqa"
        module_components = metadata.module_path.split(".")
        module_name = module_components[-1]  # Take the last component
        full_module_name = f"{root_module.__name__}.{module_name}"

    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{full_module_name}': {e}")

    try:
        return getattr(module, metadata.function_name)
    except AttributeError:
        raise AttributeError(
            f"Function '{metadata.function_name}' not found in module '{full_module_name}'"
        )


def get_eval_metadata(path_like: str) -> BenchmarkMetadata | None:
    """
    Best-effort extraction of __metadata__ for path-based evals.
    Returns None for registry-based benchmarks or when no metadata is present.
    """
    p = Path(path_like).expanduser()
    if not p.exists():
        return None

    try:
        module = _import_module_from_path(p)
        meta = getattr(module, "__metadata__", None)
        return meta if isinstance(meta, BenchmarkMetadata) else None
    except Exception:
        return None
