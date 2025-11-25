#!/usr/bin/env python3
"""
Annotate the ProgressiveMCPBench dataset with server/tool requirements.

This script takes the success annotations from extract_success_logs.py and
adds them to the progressivemcpbench.json dataset file, enabling the
minimal-servers and minimal-tools strategies.

Usage:
    python scripts/annotate_dataset.py success_annotations.json

The script will:
1. Read the existing dataset
2. Add "required_servers" and "required_tools" fields to each task
3. Write the updated dataset back

Example:
    python scripts/extract_success_logs.py logs/progressivemcpbench-2025-01-15 -o annotations.json
    python scripts/annotate_dataset.py annotations.json
"""

import argparse
import json
import sys
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/openbench/datasets/data/progressivemcpbench.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate ProgressiveMCPBench dataset with tool requirements"
    )
    parser.add_argument(
        "annotations",
        type=Path,
        help="Path to success annotations JSON file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to dataset file (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    if not args.annotations.exists():
        print(f"Error: Annotations file not found: {args.annotations}", file=sys.stderr)
        sys.exit(1)

    if not args.dataset.exists():
        print(f"Error: Dataset file not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Load annotations
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} task annotations")

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded dataset with {len(dataset)} tasks")

    # Apply annotations
    updated_count = 0
    already_annotated = 0
    no_annotation = 0

    for task in dataset:
        task_id = task.get("task_id", "")
        if task_id in annotations:
            ann = annotations[task_id]
            old_servers = task.get("required_servers")
            old_tools = task.get("required_tools")

            task["required_servers"] = ann.get("servers_used", [])
            task["required_tools"] = ann.get("tools_used", [])

            if old_servers is None and old_tools is None:
                updated_count += 1
                if args.verbose:
                    print(f"  Added annotation for {task_id}")
            else:
                already_annotated += 1
                if args.verbose:
                    print(f"  Updated annotation for {task_id}")
        else:
            no_annotation += 1
            if args.verbose:
                print(f"  No annotation for {task_id}")

    print("\nSummary:")
    print(f"  New annotations: {updated_count}")
    print(f"  Updated annotations: {already_annotated}")
    print(f"  Tasks without annotations: {no_annotation}")

    if args.dry_run:
        print("\n(Dry run - no changes written)")
    else:
        with open(args.dataset, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"\nDataset updated: {args.dataset}")


if __name__ == "__main__":
    main()
