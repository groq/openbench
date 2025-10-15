"""MGSM (Multilingual Grade School Math) dataset loader.

Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems.
Based on: Language Models are Multilingual Chain-of-Thought Reasoners
Freda Shi et al., 2022
https://arxiv.org/abs/2210.03057
"""

import urllib.request
from typing import List, Optional
from inspect_ai.dataset import Dataset, Sample, MemoryDataset

ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
LATIN_LANGUAGES = ["de", "en", "es", "fr", "sw"]
NON_LATIN_LANGUAGES = ["bn", "ja", "ru", "te", "th", "zh"]

LANG_TO_FPATH = {
    "bn": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_bn.tsv",
    "de": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_de.tsv",
    "en": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_en.tsv",
    "es": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_es.tsv",
    "fr": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_fr.tsv",
    "ja": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_ja.tsv",
    "ru": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_ru.tsv",
    "sw": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_sw.tsv",
    "te": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_te.tsv",
    "th": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_th.tsv",
    "zh": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_zh.tsv",
}

LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}""",
    "de": """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

{input}""",
    "es": """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

{input}""",
    "fr": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

{input}""",
    "ja": """の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。そして最後の行には "答え:" の形式で答えを記述し、その後には整数の答え以外何も追加しないでください。

{input}""",
    "ru": """Решите эту математическую задачу. Объясните шаги рассуждения перед тем, как дать окончательный ответ в последней строке сам по себе в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа после "Ответ:".

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"

{input}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত্তর",
    "de": "Antwort",
    "es": "Respuesta",
    "fr": "Réponse",
    "ja": "答え",
    "ru": "Ответ",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "zh": "答案",
}


def load_language_data(language: str) -> List[Sample]:
    """Load MGSM data for a specific language."""
    url = LANG_TO_FPATH[language]
    samples = []

    # Download and parse TSV data
    with urllib.request.urlopen(url) as response:
        content = response.read()

    # Parse TSV
    lines = content.decode("utf-8").strip().split("\n")
    for idx, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) == 2:
            problem, answer = parts
            # Format the instruction with the problem
            instruction = LANG_TO_INSTRUCTIONS[language].format(input=problem)

            samples.append(
                Sample(
                    input=instruction,
                    target=answer.strip(),
                    id=f"{language}_{idx + 1}",
                    metadata={
                        "language": language,
                        "answer_prefix": LANG_TO_ANSWER_PREFIX[language],
                        "latin_script": language in LATIN_LANGUAGES,
                        "original_problem": problem,
                    },
                )
            )

    return samples


def get_dataset(
    languages: Optional[List[str]] = None, limit_per_language: Optional[int] = None
) -> Dataset:
    """Load the MGSM dataset.

    Args:
        languages: List of language codes to include (defaults to all)
        limit_per_language: Maximum samples per language (defaults to all)

    Returns:
        Dataset with MGSM samples
    """
    if languages is None:
        languages = ALL_LANGUAGES
    else:
        # Validate language codes
        for lang in languages:
            if lang not in ALL_LANGUAGES:
                raise ValueError(
                    f"Invalid language code: {lang}. Must be one of {ALL_LANGUAGES}"
                )

    all_samples = []
    for lang in languages:
        lang_samples = load_language_data(lang)
        if limit_per_language is not None:
            lang_samples = lang_samples[:limit_per_language]
        all_samples.extend(lang_samples)

    return MemoryDataset(
        samples=all_samples,
        name=f"mgsm_{'_'.join(languages)}"
        if len(languages) > 1
        else f"mgsm_{languages[0]}",
    )
