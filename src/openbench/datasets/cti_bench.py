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
        }
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
        }
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
        }
    )


def taa_record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert TAA (Threat Actor Attribution) record to Sample format."""
    # Use the dataset's prompt directly
    prompt = record["Prompt"]
    
    # The TAA dataset doesn't have explicit ground truth labels.
    # We'll try to extract potential threat actors from the full text
    # and use that as aliases for evaluation.
    text = record["Text"]
    url = record.get("URL", "")
    
    # Extract potential threat actor names from various sources
    potential_actors = []
    aliases = []
    
    # Try to extract from URL (like "stately-taurus", "blind-eagle-apt")
    if url:
        # Handle URLs that end with '/' by taking the second-to-last part
        url_filename = url.rstrip('/').split('/')[-1].replace('.html', '')
        if not url_filename:  # If filename is empty, try the previous part
            url_filename = url.rstrip('/').split('/')[-2] if len(url.rstrip('/').split('/')) > 1 else ''
        url_parts = url_filename.split('-')
        
        # Look for patterns like "stately-taurus", "blind-eagle", etc.
        for i in range(len(url_parts)-1):
            candidate = '-'.join(url_parts[i:i+2])
            if len(candidate) > 5 and not candidate.isdigit() and not any(word in candidate.lower() for word in ['targets', 'attacks', 'analysis', 'malware', 'threat', 'part']):
                potential_actors.append(candidate.replace('-', ' ').title())
        
        # Also check for single-word threat actors in URL
        for part in url_parts:
            if len(part) > 4 and part.lower() not in ['www', 'blog', 'threat', 'research', 'analysis', 'malware', 'part', 'that', 'keeps', 'with', 'from', 'into', 'targets']:
                # Check if it might be a threat actor name (starts with capital or is all caps)
                if part[0].isupper() or part.isupper():
                    potential_actors.append(part.title())
    
    # Extract APT groups and threat group names from text
    import re
    apt_matches = re.findall(r'APT\d+|APT\s+\d+', text)
    threat_groups = re.findall(r'[A-Z][a-zA-Z\s]+(?:Group|Tribe|Team|Panda|Bear|Kitten)', text)
    
    potential_actors.extend(apt_matches)
    potential_actors.extend([name.strip() for name in threat_groups])
    
    # Use the first potential actor as target, or "Unknown" if none found
    target = potential_actors[0] if potential_actors else "Unknown"
    aliases = list(set(potential_actors))  # Remove duplicates
    
    return Sample(
        input=prompt,
        target=target,
        metadata={
            "task_type": "classification",
            "domain": "threat_attribution", 
            "url": url,
            "full_text": text,
            "aliases": aliases,
            "extracted_from": "heuristic" if target != "Unknown" else "none",
        }
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
        }
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


def get_cti_bench_taa_dataset() -> Dataset:
    """Load CTI-Bench TAA dataset.""" 
    return hf_dataset(
        path="AI4Sec/cti-bench",
        name="cti-taa",
        split="test",
        sample_fields=taa_record_to_sample,
    )


def get_cti_bench_ate_dataset() -> Dataset:
    """Load CTI-Bench ATE (ATT&CK Technique Extraction) dataset."""
    return hf_dataset(
        path="AI4Sec/cti-bench",
        name="cti-ate",
        split="test",
        sample_fields=ate_record_to_sample,
    )