"""
Terraform generation prompts: natural-language prompts that ask the model to produce
executable Terraform (HCL) code. Used by the terraform_generation eval.

Run: bench eval terraform_generation --model "groq/llama-3.1-8b-instant"
"""

from inspect_ai.dataset import MemoryDataset, Sample

# Bundled prompts (one sample per task). Can be extended via HuggingFace or file later.
TERRAFORM_PROMPTS = [
    {
        "task_id": "task_vpc_3subnets_3ec2",
        "input": """You are generating Terraform code. Create EXACTLY three files: main.tf, variables.tf, outputs.tf.

Requirements:
- Terraform >= 1.5
- Provider: hashicorp/aws
- Must work with LocalStack at http://localhost:4566
  Configure provider endpoints for ec2, iam, sts to http://localhost:4566 and set:
  skip_credentials_validation=true
  skip_metadata_api_check=true
  skip_requesting_account_id=true
  s3_use_path_style=true
- Variables (ALL must have default values to avoid interactive prompts):
  region (string, default = "us-east-1")
  vpc_cidr (string)
  subnet_cidrs (list(string)) length 3
  instance_type (string)
- Create:
  1 VPC (CIDR = var.vpc_cidr)
  3 Subnets (CIDRs = var.subnet_cidrs) in that VPC
  3 EC2 instances, one in each subnet
- Tag all resources:
  project = "sre-skills-bench"
  task_id = "task_vpc_3subnets_3ec2"
- Outputs:
  vpc_id
  subnet_ids (list)
  instance_ids (list)

Return ONLY code blocks, one per file, labeled with the filename.""",
    },
    {
        "task_id": "task_s3_bucket_policy",
        "input": """You are generating Terraform code. Create EXACTLY three files: main.tf, variables.tf, outputs.tf.

Requirements:
- Terraform >= 1.5
- Provider: hashicorp/aws
- Must work with LocalStack at http://localhost:4566
  Configure provider endpoints for s3, iam, sts to http://localhost:4566 and set:
  skip_credentials_validation=true
  skip_metadata_api_check=true
  skip_requesting_account_id=true
  s3_use_path_style=true
- Variables (ALL must have default values to avoid interactive prompts):
  region (string, default = "us-east-1")
  bucket_name (string)
  allowed_principal_arn (string)
- Create:
  1 S3 bucket with name = var.bucket_name
  1 S3 bucket policy that allows:
    - s3:GetObject for the specified principal (var.allowed_principal_arn)
    - s3:ListBucket for the specified principal
    Use aws_s3_bucket_policy resource to attach the policy
    The policy document should be in JSON format
- Tag all resources:
  project = "sre-skills-bench"
  task_id = "task_s3_bucket_policy"
- Outputs:
  bucket_id
  bucket_arn

Return ONLY code blocks, one per file, labeled with the filename.""",
    },
]


def load_dataset() -> MemoryDataset:
    """Load Terraform generation dataset (prompts as inputs, target = pass after validate)."""
    samples = [
        Sample(
            input=p["input"],
            target="pass",
            metadata={"task_id": p["task_id"]},
        )
        for p in TERRAFORM_PROMPTS
    ]
    return MemoryDataset(samples=samples, name="terraform_generation")
