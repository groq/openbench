"""CTI-Bench dataset loaders for cybersecurity threat intelligence benchmarks."""

from typing import Any, Dict
from inspect_ai.dataset import Dataset, Sample, hf_dataset


def mcq_record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert MCQ record to Sample format."""
    question = record["Question"]

    # Format options as A) ... B) ... C) ... D) ...
    formatted_options = [
        f"{chr(65 + i)}) {record[f'Option {chr(65 + i)}']}"
        for i in range(4)  # A, B, C, D
    ]

    prompt = f"{question}\n\n" + "\n".join(formatted_options) + "\n\nAnswer:"

    return Sample(
        input=prompt,
        target=record["GT"],
        metadata={
            "question_type": "multiple_choice",
            "domain": "cybersecurity",
            "url": record.get("URL", ""),
        },
    )


def rcm_record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert RCM (CVE→CWE mapping) record to Sample format."""
    description = record["Description"]

    prompt = f"""Given the following vulnerability description, identify the most appropriate CWE (Common Weakness Enumeration) category.

Description: {description}

Respond with only the CWE ID (e.g., CWE-79):"""

    return Sample(
        input=prompt,
        target=record["GT"],
        metadata={
            "task_type": "classification",
            "domain": "vulnerability_mapping",
            "url": record.get("URL", ""),
        },
    )


def vsp_record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert VSP (CVSS severity prediction) record to Sample format."""
    description = record["Description"]

    prompt = f"""Given the following vulnerability description, predict the CVSS (Common Vulnerability Scoring System) base score.

Description: {description}

The CVSS base score ranges from 0.0 to 10.0, where:
- 0.1-3.9: Low severity
- 4.0-6.9: Medium severity  
- 7.0-8.9: High severity
- 9.0-10.0: Critical severity

Respond with only the numeric CVSS score (e.g., 7.5):"""

    return Sample(
        input=prompt,
        target=record["GT"],
        metadata={
            "task_type": "regression",
            "domain": "vulnerability_scoring",
            "url": record.get("URL", ""),
        },
    )


def ate_record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert ATE (ATT&CK Technique Extraction) record to Sample format."""
    prompt = record["Prompt"]

    return Sample(
        input=prompt,
        target=record["GT"],
        metadata={
            "task_type": "technique_extraction",
            "domain": "mitre_attack",
            "url": record.get("URL", ""),
            "platform": record.get("Platform", ""),
            "description": record.get("Description", ""),
        },
    )


def get_cti_bench_mcq_dataset() -> Dataset:
    """Load CTI-Bench MCQ dataset."""
    return hf_dataset(
        path="AI4Sec/cti-bench",
        name="cti-mcq",
        split="test",
        sample_fields=mcq_record_to_sample,
    )


def get_cti_bench_rcm_dataset() -> Dataset:
    """Load CTI-Bench RCM dataset."""
    return hf_dataset(
        path="AI4Sec/cti-bench",
        name="cti-rcm",
        split="test",
        sample_fields=rcm_record_to_sample,
    )


def get_cti_bench_vsp_dataset() -> Dataset:
    """Load CTI-Bench VSP dataset."""
    return hf_dataset(
        path="AI4Sec/cti-bench",
        name="cti-vsp",
        split="test",
        sample_fields=vsp_record_to_sample,
    )


def get_cti_bench_ate_dataset() -> Dataset:
    """Load CTI-Bench ATE (ATT&CK Technique Extraction) dataset."""
    return hf_dataset(
        path="AI4Sec/cti-bench",
        name="cti-ate",
        split="test",
        sample_fields=ate_record_to_sample,
    )
