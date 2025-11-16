"""PRBench dataset loader - using PRBench code directly."""

from typing import Any, Dict, List, Tuple

from datasets import load_dataset  # type: ignore[import-untyped]
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ChatMessageSystem,
    ChatMessageTool,
)


# Copy of Criterion class from PRBench/criteria.py
FIELD_MAP = {"Legal": "law", "Finance": "finance"}


class Criterion:
    def __init__(self, d):
        self.d = d["annotations"]
        self.d["id"] = d["id"]
        self.d["title"] = d["title"]
        self.find_weight()
        self.find_criteria_category()

    def find_weight(self):
        self.d["weight"] = self.d[self.d["weight_class"].replace(" ", "_") + "_weight"]

    def find_criteria_category(self):
        self.d["criteria_category"] = None
        for k, v in self.d.items():
            if "criteria_category" in k:
                self.d["criteria_category"] = v

    def get_weight(self):
        return self.d["weight"]

    def get_title(self):
        return self.d["title"]

    def get_id(self):
        return self.d["id"]

    def get_category(self):
        return self.d["criteria_category"]

    def get_field_for_category(self):
        if "field_for_category" in self.d:
            return self.d["field_for_category"]
        return None

    def __repr__(self):
        ss = "Criterion:"
        ss += f"\nTitle: {self.d['title']}"
        ss += f"\nWeight: {self.d['weight']}"
        ss += f"\nCategory: {self.d['criteria_category']}"
        if "field_for_category" in self.d:
            ss += f"\nField for category: {self.d['field_for_category']}\n"
        return ss


# Copy of process_reference_texts from PRBench/util.py
def process_reference_texts(row):
    for col in row.keys():
        if "reference" in col:
            if isinstance(row[col], list):
                print(
                    f"Processing reference texts for task: {row['task']} in column: {col}, count: {len(row[col])}"
                )
                reference_texts = row[col]
                col_num = col.split("_")[-1]
                reference_text_body = ""
                for i in range(len(reference_texts)):
                    reference_text_body += (
                        f"Reference Text {i}:\n{reference_texts[i]}\n\n"
                    )
                # Safely handle missing/None prompts
                prompt_val = row.get(f"prompt_{col_num}") or ""
                row[f"prompt_{col_num}"] = (reference_text_body or "") + prompt_val
    return row


# Copy of process_row_get_convo from PRBench/util.py
def process_row_get_convo(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    convo = []
    for i in range(10):
        prompt_col = f"prompt_{i}"
        if (
            prompt_col in row
            and isinstance(row[prompt_col], str)
            and len(row[prompt_col].strip()) > 0
        ):
            convo.append(("user", row[prompt_col]))
        response_col = f"response_{i}"
        if (
            response_col in row
            and isinstance(row[response_col], str)
            and len(row[response_col].strip()) > 0
        ):
            convo.append(("assistant", row[response_col]))
    return convo


# Copy of extract_rubric from PRBench/util.py
def extract_rubric(row):
    rubric = row["rubric"]
    return [Criterion(c) for c in rubric]


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a PRBench record to an Inspect Sample using PRBench utilities."""
    # Process reference texts using PRBench's function
    record = process_reference_texts(record)

    # Extract conversation using PRBench's function
    convo = process_row_get_convo(record)

    # Convert conversation to Inspect chat message objects (use widest allowed union)
    conversation_messages: List[
        ChatMessageSystem | ChatMessageUser | ChatMessageAssistant | ChatMessageTool
    ] = []
    for role, content in convo:
        if role == "user":
            conversation_messages.append(ChatMessageUser(content=str(content)))
        else:
            conversation_messages.append(ChatMessageAssistant(content=str(content)))

    # Extract rubrics using PRBench's function (returns list of Criterion objects)
    rubrics = extract_rubric(record)

    # Get field (domain)
    field = record.get("field", "").lower()
    task = record.get("task", "")

    return Sample(
        id=task,
        input=conversation_messages,
        target="",
        metadata={
            "rubrics": rubrics,  # Keep Criterion objects
            "field": field,
            "task": task,
            "conversation": convo,
        },
    )


def get_dataset(split_name: str = "finance") -> Dataset:
    """Load the PRBench dataset from HuggingFace."""
    # Map friendly subset names to actual HF split ids
    split_map = {
        "finance": "finance",
        "legal": "legal",
    }
    hf_split = split_map.get(split_name, split_name)

    # Work around upstream schema mismatch by using streaming (no feature casting)
    # Prefer reading raw parquet directly to avoid dataset_info feature casting
    # Example path seen in errors: hf://datasets/ScaleAI/PRBench/.../data/finance-00000-of-00001.parquet
    parquet_glob = f"hf://datasets/ScaleAI/PRBench/data/{hf_split}-*.parquet"
    try:
        df = load_dataset(
            "parquet",
            data_files=parquet_glob,
            streaming=True,
        )["train"]
    except Exception:
        # Fallback to dataset loader with relaxed verification
        df = load_dataset(
            "ScaleAI/PRBench",
            split=hf_split,
            streaming=True,
            trust_remote_code=True,
            verification_mode="no_checks",
        )

    def _normalize_scalar(v):
        # Some rows store strings as singleton lists; coerce to str
        if isinstance(v, list) and v:
            return v[0]
        return v

    examples = []
    debug_print_max = 3
    seen_debug = 0
    for record in df:
        if seen_debug < debug_print_max:
            try:
                raw_rubric = record.get("rubric", None)
                print("\n[PRBench] Raw record preview:")
                print(f"  Keys: {list(record.keys())}")
                print(f"  rubric type: {type(raw_rubric).__name__}")
                if isinstance(raw_rubric, dict):
                    print(f"  rubric.keys(): {list(raw_rubric.keys())}")
                elif isinstance(raw_rubric, list):
                    print(f"  rubric list len: {len(raw_rubric)}")
                    if raw_rubric:
                        print(f"  rubric[0] type: {type(raw_rubric[0]).__name__}")
                        if isinstance(raw_rubric[0], dict):
                            print(f"  rubric[0].keys(): {list(raw_rubric[0].keys())}")
                            ann = raw_rubric[0].get("annotations", None)
                            print(
                                f"  rubric[0]['annotations'] type: {type(ann).__name__}"
                            )
                            if isinstance(ann, dict):
                                print(
                                    f"  rubric[0]['annotations'].keys(): {list(ann.keys())}"
                                )
                seen_debug += 1
            except Exception as _:
                # Ignore debug print failures
                seen_debug += 1
        try:
            # Normalize rubric field shape across variants
            rubric_val = record.get("rubric")
            if isinstance(rubric_val, dict):
                rubric_list = [rubric_val]
            elif isinstance(rubric_val, list):
                rubric_list = rubric_val
            else:
                rubric_list = []

            normalized_rubric = []
            for item in rubric_list:
                if not isinstance(item, dict):
                    continue
                annotations = item.get("annotations")
                # Some broken rows may have annotations as a list with single struct
                if isinstance(annotations, list) and annotations:
                    annotations = annotations[0]
                if not isinstance(annotations, dict):
                    continue
                normalized_item = {
                    "annotations": {
                        "criteria_category": _normalize_scalar(
                            annotations.get("criteria_category")
                        ),
                        "criteria_description": _normalize_scalar(
                            annotations.get("criteria_description")
                        ),
                        "critically_detrimental_weight": int(
                            _normalize_scalar(
                                annotations.get("critically_detrimental_weight") or 0
                            )
                        ),
                        "critically_important_weight": int(
                            _normalize_scalar(
                                annotations.get("critically_important_weight") or 0
                            )
                        ),
                        "detrimental_weight": int(
                            _normalize_scalar(
                                annotations.get("detrimental_weight") or 0
                            )
                        ),
                        "field_for_category": _normalize_scalar(
                            annotations.get("field_for_category")
                        ),
                        "important_weight": int(
                            _normalize_scalar(annotations.get("important_weight") or 0)
                        ),
                        "slightly_detrimental_weight": int(
                            _normalize_scalar(
                                annotations.get("slightly_detrimental_weight") or 0
                            )
                        ),
                        "slightly_important_weight": int(
                            _normalize_scalar(
                                annotations.get("slightly_important_weight") or 0
                            )
                        ),
                        "weight_class": _normalize_scalar(
                            annotations.get("weight_class")
                        ),
                    },
                    "id": _normalize_scalar(item.get("id") or ""),
                    "title": _normalize_scalar(item.get("title") or ""),
                }
                normalized_rubric.append(normalized_item)

            record["rubric"] = normalized_rubric
            examples.append(record)
        except Exception:
            # Skip any records that fail normalization
            continue

    samples = [record_to_sample(record) for record in examples]

    dataset_name = f"prbench_{split_name}"
    return MemoryDataset(samples=samples, name=dataset_name)
