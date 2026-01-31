"""Unit tests for Terraform generation dataset, scorer extraction, and scorer scoring."""

import shutil
from dataclasses import dataclass

import pytest

from openbench.datasets.terraform_generation import (
    TERRAFORM_PROMPTS,
    load_dataset,
)
from openbench.scorers.terraform_generation import (
    _extract_tf_blocks,
    terraform_generation_scorer,
)
from inspect_ai.scorer import Target


@dataclass
class MockOutput:
    completion: str


@dataclass
class MockMessage:
    role: str
    content: str


@dataclass
class MockTaskState:
    messages: list
    output: MockOutput


def _make_state(completion: str, messages: list | None = None) -> MockTaskState:
    if messages is None:
        messages = [
            MockMessage(role="user", content="Generate Terraform"),
            MockMessage(role="assistant", content=completion),
        ]
    out = MockOutput(completion=completion)
    return MockTaskState(messages=messages, output=out)


class TestExtractTfBlocks:
    """Test _extract_tf_blocks regex extraction."""

    def test_extract_named_blocks(self):
        """Extract blocks with filename in fence (main.tf, variables.tf, outputs.tf)."""
        text = """
**main.tf**
```main.tf
resource "aws_vpc" "x" { cidr_block = "10.0.0.0/16" }
```

```variables.tf
variable "region" { default = "us-east-1" }
```

```outputs.tf
output "vpc_id" { value = aws_vpc.x.id }
```
"""
        files = _extract_tf_blocks(text)
        assert "main.tf" in files
        assert "variables.tf" in files
        assert "outputs.tf" in files
        assert "aws_vpc" in files["main.tf"]
        assert "variable" in files["variables.tf"]
        assert "output" in files["outputs.tf"]

    def test_extract_terraform_lang_block(self):
        """Single ```terraform block becomes main.tf when no other .tf named."""
        text = """
```terraform
terraform { required_providers { aws = { source = "hashicorp/aws" } } }
provider "aws" { region = "us-east-1" }
```
"""
        files = _extract_tf_blocks(text)
        assert "main.tf" in files
        assert "terraform" in files["main.tf"] or "required_providers" in files["main.tf"]
        assert "variables.tf" in files  # placeholder
        assert "outputs.tf" in files  # placeholder

    def test_short_block_ignored(self):
        """Blocks with very little content are ignored."""
        text = "```main.tf\nx\n```"
        files = _extract_tf_blocks(text)
        # Short block (< 20 chars) is skipped; no other blocks so files is empty
        assert not files

    def test_no_blocks_empty(self):
        """No code blocks yields empty dict (no required placeholders if no files)."""
        files = _extract_tf_blocks("No terraform here at all.")
        assert not files

    def test_placeholder_for_missing_required(self):
        """When at least one file is found, missing required files get placeholder."""
        text = """
```main.tf
resource "aws_vpc" "x" { cidr_block = "10.0.0.0/16" }
```
"""
        files = _extract_tf_blocks(text)
        assert "main.tf" in files
        assert "variables.tf" in files
        assert "outputs.tf" in files
        assert files["variables.tf"] == "# placeholder\n"
        assert files["outputs.tf"] == "# placeholder\n"


class TestLoadDataset:
    """Test terraform_generation dataset."""

    def test_load_dataset_returns_memory_dataset(self):
        """load_dataset returns a MemoryDataset."""
        ds = load_dataset()
        assert ds is not None
        assert ds.name == "terraform_generation"

    def test_dataset_has_two_prompts(self):
        """Dataset has two samples (VPC+EC2 and S3 bucket policy)."""
        ds = load_dataset()
        samples = list(ds)
        assert len(samples) == 2
        assert samples[0].metadata.get("task_id") == "task_vpc_3subnets_3ec2"
        assert samples[1].metadata.get("task_id") == "task_s3_bucket_policy"

    def test_prompts_match_config(self):
        """TERRAFORM_PROMPTS matches dataset task_ids."""
        assert len(TERRAFORM_PROMPTS) == 2
        assert TERRAFORM_PROMPTS[0]["task_id"] == "task_vpc_3subnets_3ec2"
        assert TERRAFORM_PROMPTS[1]["task_id"] == "task_s3_bucket_policy"


# Minimal valid Terraform that passes init -backend=false and validate (no apply).
# Each block goes to one file only to avoid duplicate declarations.
VALID_MINIMAL_TF = '''
```main.tf
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 4.0" }
  }
}
provider "aws" {
  region = "us-east-1"
}
```
```variables.tf
variable "name" {
  type    = string
  default = "test"
}
```
```outputs.tf
output "out" {
  value = var.name
}
```
'''

INVALID_TF = '''
```main.tf
resource "invalid_syntax" {
  broken
```
'''


class TestTerraformGenerationScorer:
    """Test terraform_generation_scorer with mock state."""

    @pytest.mark.asyncio
    async def test_no_model_output_returns_zero(self):
        """Empty or missing assistant message returns 0.0."""
        # No assistant message -> scorer gets empty output
        state = _make_state("", messages=[MockMessage(role="user", content="x")])
        target = Target("pass")
        scorer_fn = terraform_generation_scorer()
        result = await scorer_fn(state, target)
        assert result.value == 0.0
        assert "No model output" in (result.explanation or "")

    @pytest.mark.asyncio
    async def test_no_code_blocks_returns_zero(self):
        """Response with no Terraform code blocks returns 0.0."""
        state = _make_state("I cannot generate Terraform for this request.")
        target = Target("pass")
        scorer_fn = terraform_generation_scorer()
        result = await scorer_fn(state, target)
        assert result.value == 0.0
        assert "No Terraform code blocks" in (result.explanation or "")

    @pytest.mark.asyncio
    async def test_invalid_terraform_returns_zero(self):
        """Invalid HCL returns 0.0 (init or validate fails)."""
        if not shutil.which("terraform"):
            pytest.skip("terraform CLI not found")
        state = _make_state(INVALID_TF)
        target = Target("pass")
        scorer_fn = terraform_generation_scorer()
        result = await scorer_fn(state, target)
        assert result.value == 0.0
        assert result.explanation

    @pytest.mark.asyncio
    async def test_valid_minimal_terraform_returns_one(self):
        """Valid minimal Terraform (init + validate) returns 1.0."""
        if not shutil.which("terraform"):
            pytest.skip("terraform CLI not found")
        state = _make_state(VALID_MINIMAL_TF)
        target = Target("pass")
        scorer_fn = terraform_generation_scorer()
        result = await scorer_fn(state, target)
        assert result.value == 1.0, result.explanation
        assert "init and validate passed" in (result.explanation or "")
