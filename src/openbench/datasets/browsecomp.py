"""Dataset loader for BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents.

https://openai.com/index/browsecomp/
"""

import base64
import hashlib
from inspect_ai.dataset import Dataset, csv_dataset, Sample, MemoryDataset


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def record_to_sample(record: dict) -> Sample:
    """Convert a BrowseComp CSV record to an Inspect Sample."""
    # Decrypt the problem and answer using the canary
    problem = decrypt(record.get("problem", ""), record.get("canary", ""))
    answer = decrypt(record.get("answer", ""), record.get("canary", ""))

    # Format the input with the query template
    formatted_input = f"""{problem}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

    return Sample(
        input=formatted_input,
        target=answer,
        metadata={
            "canary": record.get("canary", ""),
            "plain_question": problem,  # Store the plain question for the grader
        },
    )


def get_dataset() -> Dataset:
    """Load the BrowseComp dataset.

    Returns:
        Dataset containing BrowseComp samples
    """
    # Load the full dataset
    dataset = csv_dataset(
        csv_file="https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv",
        sample_fields=record_to_sample,
        auto_id=True,
        name="browsecomp",
    )

    # Convert to list of samples
    samples = list(dataset)

    return MemoryDataset(samples=samples, name="browsecomp")
