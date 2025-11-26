"""OCRBench v2 scorer using official evaluation logic.

Reference: https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2
"""

from __future__ import annotations

import re
from typing import Any

from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import TaskState

from openbench.metrics.grouped import grouped

# Optional dependencies for TEDS (Tree Edit Distance Similarity)
try:
    from apted import APTED, Config  # type: ignore[import-untyped, import-not-found]
    from lxml import html  # type: ignore[import-untyped]

    TEDS_AVAILABLE = True
except ImportError:
    TEDS_AVAILABLE = False


# Category definitions
CN_CATEGORIES = {
    "cognition VQA cn",
    "key information extraction cn",
    "formula recognition cn",
    "full-page OCR cn",
    "reasoning VQA cn",
    "text translation cn",
    "table parsing cn",
    "handwritten answer extraction cn",
    "document parsing cn",
}

# VQA categories (exact match, case-insensitive)
VQA_CATEGORIES = {
    "APP agent en",
    "ASCII art classification en",
    "math QA en",
    "reasoning VQA en",
    "science QA en",
    "document classification en",
    "cognition VQA en",
    "diagram QA en",
    "cognition VQA cn",
    "reasoning VQA cn",
    "text recognition en",  # Uses VQA evaluation per official implementation
}

# OCR categories (BLEU + METEOR + F1 + edit distance)
OCR_CATEGORIES = {
    "full-page OCR en",
    "fine-grained text recognition en",
    "full-page OCR cn",
    "text translation cn",
    # Note: "handwritten answer extraction cn" is NOT here - it has special logic
}

# Math/formula categories
MATH_CATEGORIES = {
    "formula recognition en",
    "formula recognition cn",
}

# Table/chart parsing categories (would need TEDS)
TABLE_CATEGORIES = {
    "table parsing en",
    "chart parsing en",
    "table parsing cn",
}

# Document parsing categories
DOC_PARSING_CATEGORIES = {
    "document parsing en",
    "document parsing cn",
}

# Key information extraction categories (F1 score)
KIE_CATEGORIES = {
    "key information extraction en",
    "key information mapping en",
    "key information extraction cn",
}

# ============================================================================
# Capability Mapping: Categories -> 8 Core Capabilities
# ============================================================================

CAPABILITY_MAPPING = {
    # 1. Text Recognition - OCR and handwriting
    "full-page OCR en": "Recognition",
    "full-page OCR cn": "Recognition",
    "fine-grained text recognition en": "Recognition",
    "text recognition en": "Recognition",
    "handwritten answer extraction cn": "Recognition",
    "text translation cn": "Recognition",
    # 2. Text Referring - Locating text positions
    "text grounding en": "Referring",
    # 3. Text Spotting - Detection and recognition
    "text spotting en": "Spotting",
    # 4. Relation Extraction - Key information extraction
    "key information extraction en": "Extraction",
    "key information extraction cn": "Extraction",
    "key information mapping en": "Extraction",
    # 5. Element Parsing - Tables, charts, documents to structured formats
    "table parsing en": "Parsing",
    "table parsing cn": "Parsing",
    "chart parsing en": "Parsing",
    "document parsing en": "Parsing",
    "document parsing cn": "Parsing",
    # 6. Mathematical Calculation - Formulas and math QA
    "formula recognition en": "Calculation",
    "formula recognition cn": "Calculation",
    "math QA en": "Calculation",
    # 7. Visual Text Understanding - VQA, classification, cognition
    "cognition VQA en": "Understanding",
    "cognition VQA cn": "Understanding",
    "document classification en": "Understanding",
    "ASCII art classification en": "Understanding",
    "diagram QA en": "Understanding",
    "VQA with position en": "Understanding",
    "text counting en": "Understanding",
    "APP agent en": "Understanding",
    # 8. Knowledge Reasoning - Complex reasoning VQA
    "reasoning VQA en": "Reasoning",
    "reasoning VQA cn": "Reasoning",
    "science QA en": "Reasoning",
}


def get_capability(category: str) -> str:
    """
    Map a category to its core capability.

    Args:
        category: Category name (e.g., "table parsing en")

    Returns:
        Capability name (e.g., "Parsing")
    """
    return CAPABILITY_MAPPING.get(category, "Unknown")


# ============================================================================
# Helper Functions (from official implementation)
# ============================================================================


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity).

    Used for text matching in VQA tasks.
    """
    distance = levenshtein_distance(prediction, ground_truth)
    max_len = max(len(prediction), len(ground_truth))

    if max_len == 0:
        return 1.0

    normalized_distance = distance / max_len

    if normalized_distance < threshold:
        return 1.0 - normalized_distance
    return 0.0


# ============================================================================
# TEDS Implementation (Tree Edit Distance Similarity for Table Parsing)
# ============================================================================

if TEDS_AVAILABLE:

    class TableTree:
        """Represents a table as a tree structure for TEDS calculation."""

        def __init__(
            self, tag: str, colspan: int = 1, rowspan: int = 1, content: str = ""
        ):
            self.tag = tag
            self.colspan = colspan
            self.rowspan = rowspan
            self.content = content
            self.children: list["TableTree"] = []

        def __repr__(self):
            return f"<{self.tag} colspan={self.colspan} rowspan={self.rowspan}>"

    class TEDSConfig(Config):
        """Custom APTED config for table structure comparison."""

        def __init__(self, structure_only: bool = False):
            self.structure_only = structure_only

        def rename(self, node1: TableTree, node2: TableTree) -> float:
            """Calculate cost of renaming node1 to node2."""
            if node1.tag != node2.tag:
                return 1.0
            if node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
                return 1.0
            if not self.structure_only and node1.content != node2.content:
                # Use normalized Levenshtein distance for content
                distance = levenshtein_distance(node1.content, node2.content)
                max_len = max(len(node1.content), len(node2.content))
                if max_len == 0:
                    return 0.0
                return distance / max_len
            return 0.0

        def delete(self, node: TableTree) -> float:
            """Cost of deleting a node."""
            return 1.0

        def insert(self, node: TableTree) -> float:
            """Cost of inserting a node."""
            return 1.0

        def children(self, node: TableTree) -> list:
            """Return children of a node."""
            return node.children

    def html_to_table_tree(
        html_str: str, structure_only: bool = False
    ) -> TableTree | None:
        """
        Convert HTML table to TableTree structure.

        Args:
            html_str: HTML string containing table
            structure_only: If True, ignore cell content

        Returns:
            TableTree or None if parsing fails
        """
        try:
            # Parse HTML
            doc = html.fromstring(html_str)
            # Find table element
            tables = doc.xpath(".//table")
            if not tables:
                return None
            table = tables[0]

            # Create root node
            root = TableTree(tag="table")

            # Process rows
            for row in table.xpath(".//tr"):
                tr_node = TableTree(tag="tr")
                # Process cells
                for cell in row.xpath(".//td | .//th"):
                    tag = cell.tag
                    colspan = int(cell.get("colspan", 1))
                    rowspan = int(cell.get("rowspan", 1))
                    content = (
                        "" if structure_only else (cell.text_content() or "").strip()
                    )
                    cell_node = TableTree(
                        tag=tag, colspan=colspan, rowspan=rowspan, content=content
                    )
                    tr_node.children.append(cell_node)
                root.children.append(tr_node)

            return root
        except Exception:
            return None

    def calculate_teds(
        pred_html: str, gt_html: str, structure_only: bool = False
    ) -> float:
        """
        Calculate TEDS score between predicted and ground truth HTML tables.

        Args:
            pred_html: Predicted HTML table string
            gt_html: Ground truth HTML table string
            structure_only: If True, only compare structure (ignore content)

        Returns:
            TEDS score between 0.0 and 1.0 (1.0 = perfect match)
        """
        # Convert HTML to tree structures
        pred_tree = html_to_table_tree(pred_html, structure_only=structure_only)
        gt_tree = html_to_table_tree(gt_html, structure_only=structure_only)

        # If either tree is None, return 0
        if pred_tree is None or gt_tree is None:
            return 0.0

        # Count nodes in ground truth tree
        def count_nodes(tree: TableTree) -> int:
            count = 1
            for child in tree.children:
                count += count_nodes(child)
            return count

        n_nodes = count_nodes(gt_tree)
        if n_nodes == 0:
            return 0.0

        # Calculate tree edit distance
        config = TEDSConfig(structure_only=structure_only)
        apted = APTED(pred_tree, gt_tree, config)
        distance = apted.compute_edit_distance()

        # Normalize: TEDS = 1 - (distance / n_nodes)
        teds_score = max(0.0, 1.0 - (float(distance) / n_nodes))

        return teds_score


# ============================================================================
# Evaluation Functions (from official implementation)
# ============================================================================


def vqa_evaluation(
    predict: str, answers: list[str], case_sensitive: bool = False
) -> float:
    """
    VQA evaluation with substring matching or ANLS.

    From official implementation:
    - For short answers (< 5 words): substring matching
    - For longer answers: ANLS score with 0.5 threshold
    - Case-insensitive by default

    Args:
        predict: Model prediction
        answers: List of acceptable answers
        case_sensitive: Whether to preserve case (default: False)

    Returns:
        1.0 if correct, 0.0 if incorrect, or ANLS score for text
    """
    # Normalize prediction
    predict = str(predict).replace("\n", " ").strip()
    if not case_sensitive:
        predict = predict.lower()

    max_score = 0.0
    for answer in answers:
        answer = str(answer).replace("\n", " ").strip()
        if not case_sensitive:
            answer = answer.lower()

        # Short answer: substring matching
        if len(answer.split()) < 5:
            if answer in predict:
                return 1.0
        else:
            # Long answer: ANLS score
            score = anls_score(predict, answer, threshold=0.5)
            max_score = max(max_score, score)

    return max_score


def cn_vqa_evaluation(predict: str, answers: list[str]) -> float:
    """
    Chinese VQA evaluation.

    Similar to VQA but:
    - Removes spaces
    - Uses comma-split length threshold of 4 instead of 5
    """
    # Normalize prediction (remove spaces for Chinese)
    predict = str(predict).replace("\n", " ").replace(" ", "").strip().lower()

    max_score = 0.0
    for answer in answers:
        answer = str(answer).replace("\n", " ").replace(" ", "").strip().lower()

        # Short answer: substring matching (threshold=4 for Chinese)
        if len(answer.split(",")) < 4:
            if answer in predict:
                return 1.0
        else:
            # Long answer: ANLS score
            score = anls_score(predict, answer, threshold=0.5)
            max_score = max(max_score, score)

    return max_score


def counting_evaluation(predict: str, answers: list[str]) -> float:
    """
    Counting evaluation using L1 distance (normalized IoU metric).

    From official implementation (vqa_metric.py):
    - Extracts first number from prediction
    - Calculates: iou = 1 - abs(predict_number - answer) / answer
    - Returns iou if > 0.5, otherwise 0.0
    - Validates prediction is within bounds: 0 < predict_number < 2 * answer
    """
    # Extract numbers from prediction
    numbers = re.findall(r"\d+", predict)

    if not numbers:
        # Try text representations (one, two, three, etc.)
        text_to_num = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        predict_lower = predict.lower().strip()
        for text, num in text_to_num.items():
            if text in predict_lower:
                numbers = [str(num)]
                break

    if not numbers:
        return 0.0

    try:
        predict_number = int(numbers[0])
    except (ValueError, TypeError):
        return 0.0

    # Check against all acceptable answers
    max_score = 0.0
    for answer in answers:
        try:
            answer_number = int(str(answer).strip())
        except (ValueError, TypeError):
            continue

        # Validate prediction is within bounds
        if (
            answer_number == 0
            or predict_number <= 0
            or predict_number >= 2 * answer_number
        ):
            continue

        # Calculate L1 distance normalized IoU: 1 - |pred - gt| / gt
        iou = 1.0 - abs(predict_number - answer_number) / answer_number

        # Return score if IoU > 0.5, otherwise 0
        if iou > 0.5:
            max_score = max(max_score, iou)

    return max_score


def math_expression_evaluation(predict: str, answers: list[str]) -> float:
    """
    Math/formula evaluation - substring matching.

    From official implementation:
    - Strips whitespace and newlines
    - Checks substring matching
    - Normalizes LaTeX formatting
    """
    # Normalize prediction
    predict = str(predict).replace("\n", " ").strip()
    # Remove code block markers if present
    predict = predict.replace("```latex", "").replace("```", "").strip()

    def normalize_latex(s):
        """Normalize LaTeX by removing extra whitespace and standardizing notation."""
        # Remove extra whitespace
        s = " ".join(s.split())

        # Extract content from positioning commands (keep content, remove formatting)
        s = re.sub(
            r"\\overset\{([^}]*)\}\{([^}]*)\}", r"\1\2", s
        )  # \overset{top}{base} -> topbase
        s = re.sub(
            r"\\underset\{([^}]*)\}\{([^}]*)\}", r"\1\2", s
        )  # \underset{bot}{base} -> botbase
        s = re.sub(r"\^\{([^}]*)\}", r"\1", s)  # ^{content} -> content
        s = re.sub(r"_\{([^}]*)\}", r"\1", s)  # _{content} -> content

        # Remove all LaTeX formatting commands (arrows, etc.)
        s = re.sub(r"\\[a-zA-Z]+\[[^\]]*\]\{[^}]*\}", "", s)  # \command[opt]{arg}
        s = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", s)  # \command{arg}
        s = re.sub(r"\\[a-zA-Z]+", "", s)  # \command

        # Normalize some common LaTeX variations
        s = s.replace("\\!", "")  # Remove thin spaces
        s = s.replace("\\,", "")  # Remove thin spaces
        s = s.replace("\\:", "")  # Remove medium spaces
        s = s.replace("\\;", "")  # Remove thick spaces
        s = s.replace("{ }", "").replace("{}", "")  # Remove empty braces

        # Remove all remaining curly braces and underscores/carets
        s = s.replace("{", "").replace("}", "")
        s = s.replace("^", "").replace("_", "")

        # Remove standalone backslashes
        s = s.replace("\\", "")
        return s

    for answer in answers:
        answer = str(answer).replace("\n", " ").strip()

        # Direct substring match
        if answer in predict or predict in answer:
            return 1.0

        # Try normalized versions
        predict_norm = normalize_latex(predict)
        answer_norm = normalize_latex(answer)

        if answer_norm in predict_norm or predict_norm in answer_norm:
            return 1.0

        # Try even more aggressive normalization - remove all whitespace
        predict_compact = "".join(predict_norm.split())
        answer_compact = "".join(answer_norm.split())

        if answer_compact in predict_compact or predict_compact in answer_compact:
            return 1.0

    return 0.0


def cn_math_expression_evaluation(predict: str, answers: list[str]) -> float:
    """
    Chinese math/formula evaluation.

    From official implementation:
    - Extracts content from LaTeX \text{...} tags (keeps the text, removes the command)
    - Then applies math expression evaluation
    """
    # Extract content from LaTeX text tags (replace \text{content} with content)
    predict = re.sub(r"\\text\{([^}]*)\}", r"\1", predict)
    # Also extract from answers
    answers_clean = [re.sub(r"\\text\{([^}]*)\}", r"\1", ans) for ans in answers]
    return math_expression_evaluation(predict, answers_clean)


def ocr_evaluation(predict: str, answers: list[str]) -> float:
    """
    OCR evaluation using averaged metrics.

    From official implementation (cal_per_metrics):
    - Average of: BLEU, METEOR, F-measure, (1 - normalized_edit_distance)

    This implements the full paper specification using NLTK for BLEU and METEOR.
    Evaluates against all target answers and returns the maximum score.
    """
    # Optional NLTK import for BLEU and METEOR
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore[import-not-found,import-untyped]
        from nltk.translate.meteor_score import meteor_score  # type: ignore[import-not-found,import-untyped]
        import nltk  # type: ignore[import-not-found,import-untyped]

        # Ensure required NLTK data is downloaded
        try:
            nltk.data.find("corpora/wordnet.zip")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        NLTK_AVAILABLE = True
    except ImportError:
        NLTK_AVAILABLE = False

    if not answers:
        return 0.0

    pred = str(predict).strip()
    if not pred:
        return 0.0

    # Try each target answer and return the maximum score
    max_score = 0.0

    for gt in answers:
        gt = str(gt).strip()
        if not gt:
            continue

        # Tokenize for metrics
        pred_tokens = pred.split()
        gt_tokens = gt.split()

        # 1. Normalized edit distance (character-level)
        edit_dist = levenshtein_distance(pred, gt)
        max_len = max(len(pred), len(gt))
        normalized_edit_dist = edit_dist / max_len if max_len > 0 else 1.0
        edit_score = 1.0 - normalized_edit_dist

        if NLTK_AVAILABLE and pred_tokens and gt_tokens:
            # 2. BLEU score (with smoothing for short texts)
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothing)

            # 3. METEOR score
            try:
                meteor = meteor_score([gt_tokens], pred_tokens)
            except Exception:
                # Fallback if METEOR fails (e.g., empty tokens after preprocessing)
                meteor = 0.0

            # 4. Token-level F-measure
            pred_set = set(pred_tokens)
            gt_set = set(gt_tokens)
            common = pred_set & gt_set
            precision = len(common) / len(pred_set) if pred_set else 0.0
            recall = len(common) / len(gt_set) if gt_set else 0.0
            f_measure = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Average all 4 metrics as per paper specification
            score = (bleu + meteor + f_measure + edit_score) / 4.0
        else:
            # Fallback if NLTK not available: use F1 + edit distance
            pred_set = set(pred_tokens)
            gt_set = set(gt_tokens)

            if not pred_set or not gt_set:
                f_measure = 0.0
            else:
                common = pred_set & gt_set
                precision = len(common) / len(pred_set) if pred_set else 0.0
                recall = len(common) / len(gt_set) if gt_set else 0.0
                f_measure = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

            score = (f_measure + edit_score) / 2.0

        max_score = max(max_score, score)

    return max_score


def kie_f1_evaluation(predict: str, answers: list[str]) -> float:
    """
    Key Information Extraction F1 score.

    Tries to parse JSON/dict from both prediction and answer, then compares values.
    Falls back to substring matching if parsing fails.
    """
    import json
    import ast

    if not answers or len(answers) == 0:
        return 0.0

    # Get ground truth - should be a dict string
    gt_str = answers[0]

    # Try to parse ground truth as dict
    try:
        # Remove markdown code blocks if present
        gt_clean = gt_str.replace("```json", "").replace("```", "").strip()
        # Try JSON first
        try:
            gt_dict = json.loads(gt_clean)
        except (json.JSONDecodeError, ValueError):
            # Try ast.literal_eval for Python dict format
            gt_dict = ast.literal_eval(gt_clean)
    except (json.JSONDecodeError, ValueError, SyntaxError):
        # If parsing fails, fall back to substring matching
        return 1.0 if gt_str in predict else 0.0

    # Try to parse prediction as dict
    try:
        # Remove markdown code blocks
        pred_clean = predict.replace("```json", "").replace("```", "").strip()
        # Extract JSON if embedded in text
        if "{" in pred_clean and "}" in pred_clean:
            start = pred_clean.find("{")
            end = pred_clean.rfind("}") + 1
            pred_clean = pred_clean[start:end]

        try:
            pred_dict = json.loads(pred_clean)
        except (json.JSONDecodeError, ValueError):
            pred_dict = ast.literal_eval(pred_clean)
    except (json.JSONDecodeError, ValueError, SyntaxError):
        # If we can't parse prediction, try substring matching of values
        matches = sum(1 for v in gt_dict.values() if str(v) in predict)
        return matches / len(gt_dict) if gt_dict else 0.0

    # Compare dictionaries and compute F1
    if not isinstance(pred_dict, dict) or not isinstance(gt_dict, dict):
        return 0.0

    # Flatten nested lists in values for comparison
    def flatten_value(v):
        if isinstance(v, list):
            return str(v[0]) if len(v) > 0 else ""
        return str(v)

    gt_items = {k: flatten_value(v) for k, v in gt_dict.items()}
    pred_items = {k: flatten_value(v) for k, v in pred_dict.items()}

    # Count matches with official normalization
    matches = 0
    for key, gt_val in gt_items.items():
        if key in pred_items:
            # Normalize per official implementation:
            # lowercase, strip, replace newlines with space, remove ALL spaces
            gt_norm = str(gt_val).lower().strip().replace("\n", " ").replace(" ", "")
            pred_norm = (
                str(pred_items[key]).lower().strip().replace("\n", " ").replace(" ", "")
            )
            if gt_norm == pred_norm:
                matches += 1

    # Compute F1
    precision = matches / len(pred_items) if pred_items else 0.0
    recall = matches / len(gt_items) if gt_items else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1


def table_parsing_evaluation(predict: str, answers: list[str]) -> float:
    """
    Table/chart parsing evaluation using TEDS (Tree Edit Distance Similarity).

    If TEDS dependencies are available, uses proper TEDS metric.
    Otherwise, falls back to OCR evaluation.

    Args:
        predict: Predicted HTML table string
        answers: List of ground truth HTML table strings

    Returns:
        TEDS score between 0.0 and 1.0
    """
    if not TEDS_AVAILABLE:
        # Fallback to OCR evaluation if TEDS dependencies not installed
        return ocr_evaluation(predict, answers)

    if not answers:
        return 0.0

    # Try TEDS on each answer, return max score
    max_score = 0.0
    for answer in answers:
        try:
            score = calculate_teds(predict, str(answer), structure_only=False)
            max_score = max(max_score, score)
        except Exception:
            # If TEDS fails, try OCR evaluation as fallback
            continue

    # If TEDS failed for all answers, fall back to OCR
    if max_score == 0.0:
        return ocr_evaluation(predict, answers)

    return max_score


def doc_parsing_evaluation(predict: str, answers: list[str]) -> float:
    """
    Document parsing evaluation.

    Uses TEDS for structured document parsing (similar to tables).
    """
    return table_parsing_evaluation(predict, answers)


def text_grounding_evaluation(predict: str, answers: list[str]) -> float:
    """
    Text grounding/spotting evaluation using IoU.

    Expects bbox coordinates in format: [x1, y1, x2, y2] or similar.
    Extracts numbers from prediction and compares with ground truth.
    """
    if not answers:
        return 0.0

    # Extract all numbers from prediction
    pred_numbers = re.findall(r"\d+", predict)
    if not pred_numbers:
        return 0.0

    # Convert to integers
    try:
        pred_coords = [int(n) for n in pred_numbers[:4]]  # Take first 4 numbers as bbox
    except (ValueError, TypeError):
        return 0.0

    if len(pred_coords) < 4:
        return 0.0

    # Extract ground truth bbox coordinates
    # answers is a list like ['888', '959', '946', '999'] representing [x1, y1, x2, y2]
    gt_numbers = []
    for answer in answers:
        nums = re.findall(r"\d+", str(answer))
        gt_numbers.extend(nums)

    if len(gt_numbers) < 4:
        return 0.0

    try:
        gt_coords = [int(n) for n in gt_numbers[:4]]
    except (ValueError, TypeError):
        return 0.0

    # Calculate IoU (Intersection over Union)
    def calculate_iou(box1, box2):
        # box format: [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    iou = calculate_iou(pred_coords, gt_coords)

    # Return 1.0 if IoU > 0.5, otherwise return the IoU value
    return 1.0 if iou > 0.5 else iou


# ============================================================================
# Main Evaluation Router
# ============================================================================


def evaluate_by_category(
    category: str, prediction: str, answers: list[str], metadata: dict
) -> float:
    """
    Route to appropriate evaluation function based on category.

    Args:
        category: Category name (e.g., "APP agent en", "table parsing cn")
        prediction: Model prediction
        answers: List of acceptable answers
        metadata: Additional metadata (bbox, raw_text, etc.)

    Returns:
        Score between 0.0 and 1.0
    """
    # Special case: handwritten answer extraction cn has TWO paths
    # (must check BEFORE other categories since it's not in any category set)
    if category == "handwritten answer extraction cn":
        # Get question text from metadata to determine path
        question = metadata.get("question", "")

        # Path 1: Essay questions (含"简答") - use OCR metrics
        if "简答" in question:
            return ocr_evaluation(prediction, answers)

        # Path 2: Multiple choice (default) - use pattern matching
        if not answers:
            return 0.0

        target = answers[0]
        if not target:
            return 0.0

        chars = list(target)

        # Multi-character answer: check 10 format patterns
        if len(target) > 1:
            patterns = [
                "".join(chars),  # ABD
                ".".join(chars),  # A.B.D
                ". ".join(chars),  # A. B. D
                ",".join(chars),  # A,B,D
                ", ".join(chars),  # A, B, D
                "、".join(chars),  # A、B、D
                ";".join(chars),  # A;B;D
                "; ".join(chars),  # A; B; D
                " ".join(chars),  # A B D
                "和".join(chars),  # A和B和D
            ]
            # Return 1.0 if any pattern found, 0.0 otherwise
            return 1.0 if any(pattern in prediction for pattern in patterns) else 0.0
        else:
            # Single character: simple substring check
            return 1.0 if target in prediction else 0.0

    # VQA tasks - exact match (case-insensitive)
    if category in VQA_CATEGORIES:
        if category in CN_CATEGORIES:
            return cn_vqa_evaluation(prediction, answers)
        return vqa_evaluation(prediction, answers, case_sensitive=False)

    # OCR tasks - BLEU + METEOR + F1 + edit distance
    elif category in OCR_CATEGORIES:
        return ocr_evaluation(prediction, answers)

    # Math/formula - substring matching
    elif category in MATH_CATEGORIES:
        if category in CN_CATEGORIES:
            return cn_math_expression_evaluation(prediction, answers)
        return math_expression_evaluation(prediction, answers)

    # Table/chart parsing - TEDS (Tree Edit Distance Similarity)
    elif category in TABLE_CATEGORIES:
        return table_parsing_evaluation(prediction, answers)

    # Document parsing - OCR metrics
    elif category in DOC_PARSING_CATEGORIES:
        return doc_parsing_evaluation(prediction, answers)

    # Key information extraction - F1 score
    elif category in KIE_CATEGORIES:
        return kie_f1_evaluation(prediction, answers)

    # Text counting - number extraction
    elif category == "text counting en":
        return counting_evaluation(prediction, answers)

    # Text grounding - IoU (basic implementation for bbox coordinates)
    elif category == "text grounding en":
        return text_grounding_evaluation(prediction, answers)

    # Text spotting - spotting metric (basic implementation)
    elif category == "text spotting en":
        return text_grounding_evaluation(prediction, answers)

    # VQA with position - position-aware VQA
    elif category == "VQA with position en":
        return vqa_evaluation(prediction, answers, case_sensitive=False)

    # Default: exact match, but handle special cases
    else:
        # Handle "<no answer>" case - if target is "<no answer>", check if model says there's no answer
        for answer in answers:
            answer_clean = str(answer).strip().lower()
            pred_clean = prediction.strip().lower()

            if answer_clean == "<no answer>":
                # Model should indicate no answer/not applicable/none/etc
                no_answer_phrases = [
                    "no answer",
                    "not applicable",
                    "n/a",
                    "none",
                    "cannot",
                    "not visible",
                    "not shown",
                ]
                if any(phrase in pred_clean for phrase in no_answer_phrases):
                    return 1.0
                # If model provides content, that's wrong
                return 0.0

            # Regular exact match
            if pred_clean == answer_clean or answer_clean in pred_clean:
                return 1.0

        return 0.0


@scorer(
    metrics=[
        accuracy(),
        grouped(
            metric=[accuracy()],
            group_key="capability",
            all="groups",
            all_label="overall",
        ),
    ]
)
def ocrbenchv2_scorer():
    """
    OCRBench v2 scorer using official evaluation logic.

    Routes to appropriate evaluation function based on category:
    - VQA tasks: Exact match (case-insensitive) or ANLS
    - OCR tasks: Average of F-measure + (1 - edit distance)
    - Math/formula: Substring matching
    - Table/chart: Tree Edit Distance (simplified)
    - Document parsing: OCR metrics
    - Key information: F1 score
    - Text counting: Number extraction

    Returns:
        Scorer function for OCRBench v2 samples
    """

    async def score(state: TaskState, target: Any) -> Score:
        try:
            category = state.metadata.get("type", "")
            dataset_name = state.metadata.get("dataset_name", "")
            prediction = (state.output.completion or "").strip()

            # Map category to core capability
            capability = get_capability(category)

            # Convert Target object to list of strings
            # Target is a sequence-like object from inspect_ai
            # Exclude strings from iterable check to avoid splitting "ABC" into ['A','B','C']
            if isinstance(target, str):
                answers = [target]
            elif hasattr(target, "__iter__"):
                answers = list(target)
            else:
                answers = [str(target)]

            # Evaluate using category-specific function
            score_value = evaluate_by_category(
                category=category,
                prediction=prediction,
                answers=answers,
                metadata=state.metadata,
            )

            return Score(
                value=score_value,
                answer=prediction,
                explanation=f"Capability: {capability} (Category: {category}, Dataset: {dataset_name})",
                metadata={
                    "dataset_name": dataset_name,
                    "category": category,
                    "capability": capability,
                    "prediction": prediction,
                    "answers": answers,
                },
            )

        except Exception as e:
            return Score(
                value=0.0,
                answer=state.output.completion if hasattr(state, "output") else "",
                explanation=f"Error: {str(e)}",
                metadata={
                    "error": str(e),
                    "dataset_name": state.metadata.get("dataset_name", "unknown"),
                },
            )

    return score
