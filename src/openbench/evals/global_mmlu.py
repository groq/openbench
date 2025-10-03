"""
Global-MMLU: Culturally Adapted Multilingual MMLU Benchmark

Global-MMLU addresses cultural and linguistic biases in multilingual evaluation by
providing culturally adapted MMLU questions across 42 languages. Unlike direct
translations (MMMLU), Global-MMLU includes cultural sensitivity annotations and
adapts questions to be relevant across diverse cultural contexts.

Key Features:
- 42 languages including low-resource languages (Amharic, Igbo, Chichewa, etc.)
- Cultural sensitivity labels (Culturally Agnostic vs Culturally Sensitive)
- Professional translations with community verification
- 589,000+ translations across 57 subjects

Dataset: CohereLabs/Global-MMLU
Paper: https://arxiv.org/abs/2412.03304

Sample usage:
```bash
# Run specific language
bench eval global_mmlu_german --model "groq/llama-3.1-70b"

# Or use family benchmark with parameter
bench eval global_mmlu -T language=de --model "groq/llama-3.1-70b"
```

Citation:
@article{singh2024globalmmlu,
    title={Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation},
    author={Singh, Shivalika and others},
    journal={arXiv preprint arXiv:2412.03304},
    year={2024}
}
"""

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from openbench.utils.mcq import MCQEval, MCQSample
from openbench.utils.text import MULTIPLE_CHOICE_PROMPT_TEMPLATE

# All 42 languages available in Global-MMLU
GLOBAL_MMLU_LANGUAGES = {
    "am": "Amharic",
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian (Farsi)",
    "fil": "Filipino (Tagalog)",
    "fr": "French",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "ig": "Igbo",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ky": "Kyrgyz",
    "lt": "Lithuanian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ne": "Nepali",
    "nl": "Dutch",
    "ny": "Chichewa (Nyanja)",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sn": "Shona",
    "so": "Somali",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "te": "Telugu",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yo": "Yoruba",
    "zh": "Chinese",
}


def record_to_mcq_sample(record: dict) -> MCQSample:
    """Convert a Global-MMLU record to an OpenBench MCQSample.

    Global-MMLU includes additional metadata for cultural analysis:
    - subject: Academic subject (e.g., "history", "mathematics")
    - subject_category: Broader category grouping
    - culture: Cultural context of the question
    - cultural_sensitivity_label: "CA" (Culturally Agnostic) or "CS" (Culturally Sensitive)
    """
    return MCQSample(
        input=MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
            prompt=record["question"],
            option_a=record["option_a"],
            option_b=record["option_b"],
            option_c=record["option_c"],
            option_d=record["option_d"],
        ),
        target=record["answer"],
        metadata={
            "subject": record.get("subject", "unknown"),
            "subject_category": record.get("subject_category", "unknown"),
            "culture": record.get("culture", "unknown"),
            "cultural_sensitivity_label": record.get(
                "cultural_sensitivity_label", "unknown"
            ),
        },
    )


@task
def global_mmlu(language: str = "en", split: str = "test") -> Task:
    """
    Global-MMLU family benchmark - culturally adapted MMLU across 42 languages.

    Unlike MMMLU (direct translations), Global-MMLU addresses Western-centric bias
    by including culturally adapted questions with sensitivity annotations.

    Args:
        language: Language code (default: "en"). Available languages:
            am (Amharic), ar (Arabic), bn (Bengali), cs (Czech), de (German),
            el (Greek), en (English), es (Spanish), fa (Persian), fil (Filipino),
            fr (French), ha (Hausa), he (Hebrew), hi (Hindi), id (Indonesian),
            ig (Igbo), it (Italian), ja (Japanese), ko (Korean), ky (Kyrgyz),
            lt (Lithuanian), mg (Malagasy), ms (Malay), ne (Nepali), nl (Dutch),
            ny (Chichewa), pl (Polish), pt (Portuguese), ro (Romanian), ru (Russian),
            si (Sinhala), sn (Shona), so (Somali), sr (Serbian), sv (Swedish),
            sw (Swahili), te (Telugu), tr (Turkish), uk (Ukrainian), vi (Vietnamese),
            yo (Yoruba), zh (Chinese)
        split: Dataset split (default: "test")

    Returns:
        Task configured for Global-MMLU evaluation in the specified language

    Examples:
        # Via family benchmark
        bench eval global_mmlu -T language=de --model "groq/llama-3.1-70b"

        # Via individual task
        bench eval global_mmlu_german --model "groq/llama-3.1-70b"
    """
    if language not in GLOBAL_MMLU_LANGUAGES:
        available = ", ".join(GLOBAL_MMLU_LANGUAGES.keys())
        raise ValueError(
            f"Invalid language '{language}'. Available languages: {available}"
        )

    return MCQEval(
        name=f"global_mmlu_{language}",
        dataset_path="CohereLabs/Global-MMLU",
        subset_name=language,
        record_to_mcq_sample=record_to_mcq_sample,
        split=split,
        auto_id=True,
        config=GenerateConfig(
            temperature=0.5,
        ),
        group_keys=["subject_category", "cultural_sensitivity_label"],
    )


# ============================================================================
# Individual wrapper functions for all 42 languages
# ============================================================================


@task
def global_mmlu_amharic(split: str = "test") -> Task:
    """Global-MMLU: Amharic (am)"""
    return global_mmlu(language="am", split=split)


@task
def global_mmlu_arabic(split: str = "test") -> Task:
    """Global-MMLU: Arabic (ar)"""
    return global_mmlu(language="ar", split=split)


@task
def global_mmlu_bengali(split: str = "test") -> Task:
    """Global-MMLU: Bengali (bn)"""
    return global_mmlu(language="bn", split=split)


@task
def global_mmlu_czech(split: str = "test") -> Task:
    """Global-MMLU: Czech (cs)"""
    return global_mmlu(language="cs", split=split)


@task
def global_mmlu_german(split: str = "test") -> Task:
    """Global-MMLU: German (de)"""
    return global_mmlu(language="de", split=split)


@task
def global_mmlu_greek(split: str = "test") -> Task:
    """Global-MMLU: Greek (el)"""
    return global_mmlu(language="el", split=split)


@task
def global_mmlu_english(split: str = "test") -> Task:
    """Global-MMLU: English (en)"""
    return global_mmlu(language="en", split=split)


@task
def global_mmlu_spanish(split: str = "test") -> Task:
    """Global-MMLU: Spanish (es)"""
    return global_mmlu(language="es", split=split)


@task
def global_mmlu_persian(split: str = "test") -> Task:
    """Global-MMLU: Persian/Farsi (fa)"""
    return global_mmlu(language="fa", split=split)


@task
def global_mmlu_filipino(split: str = "test") -> Task:
    """Global-MMLU: Filipino/Tagalog (fil)"""
    return global_mmlu(language="fil", split=split)


@task
def global_mmlu_french(split: str = "test") -> Task:
    """Global-MMLU: French (fr)"""
    return global_mmlu(language="fr", split=split)


@task
def global_mmlu_hausa(split: str = "test") -> Task:
    """Global-MMLU: Hausa (ha)"""
    return global_mmlu(language="ha", split=split)


@task
def global_mmlu_hebrew(split: str = "test") -> Task:
    """Global-MMLU: Hebrew (he)"""
    return global_mmlu(language="he", split=split)


@task
def global_mmlu_hindi(split: str = "test") -> Task:
    """Global-MMLU: Hindi (hi)"""
    return global_mmlu(language="hi", split=split)


@task
def global_mmlu_indonesian(split: str = "test") -> Task:
    """Global-MMLU: Indonesian (id)"""
    return global_mmlu(language="id", split=split)


@task
def global_mmlu_igbo(split: str = "test") -> Task:
    """Global-MMLU: Igbo (ig)"""
    return global_mmlu(language="ig", split=split)


@task
def global_mmlu_italian(split: str = "test") -> Task:
    """Global-MMLU: Italian (it)"""
    return global_mmlu(language="it", split=split)


@task
def global_mmlu_japanese(split: str = "test") -> Task:
    """Global-MMLU: Japanese (ja)"""
    return global_mmlu(language="ja", split=split)


@task
def global_mmlu_korean(split: str = "test") -> Task:
    """Global-MMLU: Korean (ko)"""
    return global_mmlu(language="ko", split=split)


@task
def global_mmlu_kyrgyz(split: str = "test") -> Task:
    """Global-MMLU: Kyrgyz (ky)"""
    return global_mmlu(language="ky", split=split)


@task
def global_mmlu_lithuanian(split: str = "test") -> Task:
    """Global-MMLU: Lithuanian (lt)"""
    return global_mmlu(language="lt", split=split)


@task
def global_mmlu_malagasy(split: str = "test") -> Task:
    """Global-MMLU: Malagasy (mg)"""
    return global_mmlu(language="mg", split=split)


@task
def global_mmlu_malay(split: str = "test") -> Task:
    """Global-MMLU: Malay (ms)"""
    return global_mmlu(language="ms", split=split)


@task
def global_mmlu_nepali(split: str = "test") -> Task:
    """Global-MMLU: Nepali (ne)"""
    return global_mmlu(language="ne", split=split)


@task
def global_mmlu_dutch(split: str = "test") -> Task:
    """Global-MMLU: Dutch (nl)"""
    return global_mmlu(language="nl", split=split)


@task
def global_mmlu_chichewa(split: str = "test") -> Task:
    """Global-MMLU: Chichewa/Nyanja (ny)"""
    return global_mmlu(language="ny", split=split)


@task
def global_mmlu_polish(split: str = "test") -> Task:
    """Global-MMLU: Polish (pl)"""
    return global_mmlu(language="pl", split=split)


@task
def global_mmlu_portuguese(split: str = "test") -> Task:
    """Global-MMLU: Portuguese (pt)"""
    return global_mmlu(language="pt", split=split)


@task
def global_mmlu_romanian(split: str = "test") -> Task:
    """Global-MMLU: Romanian (ro)"""
    return global_mmlu(language="ro", split=split)


@task
def global_mmlu_russian(split: str = "test") -> Task:
    """Global-MMLU: Russian (ru)"""
    return global_mmlu(language="ru", split=split)


@task
def global_mmlu_sinhala(split: str = "test") -> Task:
    """Global-MMLU: Sinhala (si)"""
    return global_mmlu(language="si", split=split)


@task
def global_mmlu_shona(split: str = "test") -> Task:
    """Global-MMLU: Shona (sn)"""
    return global_mmlu(language="sn", split=split)


@task
def global_mmlu_somali(split: str = "test") -> Task:
    """Global-MMLU: Somali (so)"""
    return global_mmlu(language="so", split=split)


@task
def global_mmlu_serbian(split: str = "test") -> Task:
    """Global-MMLU: Serbian (sr)"""
    return global_mmlu(language="sr", split=split)


@task
def global_mmlu_swedish(split: str = "test") -> Task:
    """Global-MMLU: Swedish (sv)"""
    return global_mmlu(language="sv", split=split)


@task
def global_mmlu_swahili(split: str = "test") -> Task:
    """Global-MMLU: Swahili (sw)"""
    return global_mmlu(language="sw", split=split)


@task
def global_mmlu_telugu(split: str = "test") -> Task:
    """Global-MMLU: Telugu (te)"""
    return global_mmlu(language="te", split=split)


@task
def global_mmlu_turkish(split: str = "test") -> Task:
    """Global-MMLU: Turkish (tr)"""
    return global_mmlu(language="tr", split=split)


@task
def global_mmlu_ukrainian(split: str = "test") -> Task:
    """Global-MMLU: Ukrainian (uk)"""
    return global_mmlu(language="uk", split=split)


@task
def global_mmlu_vietnamese(split: str = "test") -> Task:
    """Global-MMLU: Vietnamese (vi)"""
    return global_mmlu(language="vi", split=split)


@task
def global_mmlu_yoruba(split: str = "test") -> Task:
    """Global-MMLU: Yoruba (yo)"""
    return global_mmlu(language="yo", split=split)


@task
def global_mmlu_chinese(split: str = "test") -> Task:
    """Global-MMLU: Chinese (zh)"""
    return global_mmlu(language="zh", split=split)
