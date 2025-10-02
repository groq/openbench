from inspect_ai.dataset import hf_dataset
from inspect_ai.dataset import Sample
from pathlib import Path


def download_h5_file(output_dir: str = "./tmp") -> str:
    """Download the test_data.h5 file from Hugging Face to the tmp directory."""
    from huggingface_hub import hf_hub_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    h5_file_path = output_path / "test_data.h5"

    if not h5_file_path.exists():
        hf_hub_download(
            repo_id="Srimadh/Scicode-test-data-h5",
            filename="test_data.h5",
            repo_type="dataset",
            local_dir=output_path,
        )

    return str(h5_file_path)


def record_to_sample(record):
    return Sample(
        input=record["problem_description_main"],
        target=record.get("general_solution") or "",
        id=record["problem_id"],
        metadata={k: v for k, v in record.items()},
    )


def return_hf_dataset(split: str = "test"):
    return hf_dataset(
        path="SciCode1/SciCode",
        split=split,
        sample_fields=record_to_sample,
        download_mode="force_redownload",
    )
