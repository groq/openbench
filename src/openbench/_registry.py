"""
Registry for inspect_ai extensions (model providers and tasks).
This module is the entry point for inspect_ai to discover our extensions.
"""

from typing import Type
from inspect_ai.model import ModelAPI
from inspect_ai.model._registry import modelapi


# Model Provider Registration


@modelapi(name="cerebras")
def cerebras() -> Type[ModelAPI]:
    """Register Cerebras provider."""
    from .model._providers.cerebras import CerebrasAPI

    return CerebrasAPI


@modelapi(name="sambanova")
def sambanova() -> Type[ModelAPI]:
    """Register SambaNova provider."""
    from .model._providers.sambanova import SambaNovaAPI

    return SambaNovaAPI


# Task Registration

# Core benchmarks
from .evals.drop import drop  # noqa: F401, E402
from .evals.gpqa_diamond import gpqa_diamond  # noqa: F401, E402
from .evals.healthbench import healthbench, healthbench_hard, healthbench_consensus  # noqa: F401, E402
from .evals.hle import hle, hle_text  # noqa: F401, E402
from .evals.humaneval import humaneval  # noqa: F401, E402
from .evals.math import math, math_500  # noqa: F401, E402
from .evals.mgsm import mgsm, mgsm_en, mgsm_latin, mgsm_non_latin  # noqa: F401, E402
from .evals.mmlu import mmlu  # noqa: F401, E402
from .evals.mrcr import openai_mrcr, openai_mrcr_2n, openai_mrcr_4n, openai_mrcr_8n  # noqa: F401, E402
from .evals.musr import musr  # noqa: F401, E402
from .evals.openbookqa import openbookqa  # noqa: F401, E402
from .evals.simpleqa import simpleqa  # noqa: F401, E402
from .evals.supergpqa import supergpqa  # noqa: F401, E402

# MathArena benchmarks
from .evals.matharena.aime_2023_I.aime_2023_I import aime_2023_I  # noqa: F401, E402
from .evals.matharena.aime_2023_II.aime_2023_II import aime_2023_II  # noqa: F401, E402
from .evals.matharena.aime_2024_I.aime_2024_I import aime_2024_I  # noqa: F401, E402
from .evals.matharena.aime_2024_II.aime_2024_II import aime_2024_II  # noqa: F401, E402
from .evals.matharena.aime_2024.aime_2024 import aime_2024  # noqa: F401, E402
from .evals.matharena.aime_2025.aime_2025 import aime_2025  # noqa: F401, E402
from .evals.matharena.aime_2025_II.aime_2025_II import aime_2025_II  # noqa: F401, E402
from .evals.matharena.brumo_2025.brumo_2025 import brumo_2025  # noqa: F401, E402
from .evals.matharena.hmmt_feb_2023.hmmt_feb_2023 import hmmt_feb_2023  # noqa: F401, E402
from .evals.matharena.hmmt_feb_2024.hmmt_feb_2024 import hmmt_feb_2024  # noqa: F401, E402
from .evals.matharena.hmmt_feb_2025.hmmt_feb_2025 import hmmt_feb_2025  # noqa: F401, E402
