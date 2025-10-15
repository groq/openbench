"""Single-step symbolic grading: Define primitives and evaluate them together.

Workflow:
1. LLM Call: Analyze criterion, define primitives, evaluate them, and check constraints (all in one step)
2. Python eval() verifies constraints deterministically (safe, restricted namespace)

This unified approach allows the LLM to:
- See the full context (criterion + response) while defining primitives
- Evaluate primitives immediately with concrete evidence
- Avoid error propagation from multi-step processes
"""

import json
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from inspect_ai.model import get_model, ChatMessageUser
    from openbench.utils.text import parse_json_from_response

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    parse_json_from_response = None  # type: ignore[assignment]


SYMBOLIC_GRADER_PROMPT = """Evaluate if the response meets the rubric criterion using symbolic primitives.

Conversation:
{conversation}

Rubric Criterion:
{rubric_item}

Task: Define primitives, evaluate them, and check constraints.

STEP 1 - DEFINE PRIMITIVES:
Break down the criterion into verifiable primitives (boolean variables):
- For multi-part criteria (e.g., "X AND Y"), create separate primitives for each part
- For action/directive criteria: primitives should check for direct, strong language (not weak)
- For informational criteria: primitives should check if information is explicitly present 
- Make primitive names descriptive

STEP 2 - EVALUATE PRIMITIVES:
For each primitive, evaluate based on the actual response:
- Evaluate ONLY on what is EXPLICITLY stated
- For action primitives: require strong directive language ("You should", "Do X", "Go to Y")
  * Weak language = FALSE ("might", "could", "consider", "can be helpful")
- Quote exact text as evidence

STEP 3 - CHECK CONSTRAINTS:
- Define Python boolean expressions that will be evaluated using eval()
- Constraints use primitive names and operators: ==, !=, and, or, not, ()
- For multi-part criteria, create separate constraints for each part
- All constraints must evaluate to True for criterion to be met
- All constraints in the criterion must be added 
- Example: ["primitive1 == true", "primitive2 == true"] means both must be true

IMPORTANT: The constraints will be automatically verified using Python's eval() function.
Make sure constraint syntax is valid Python boolean expressions.

Return JSON:
{{
  "primitives": {{
    "primitive_name": {{
      "type": "boolean",
      "value": true,
      "evidence": "QUOTE: 'exact text' - explanation"
    }}
  }},
  "constraints": ["primitive1 == true", "primitive2 == true"],
  "all_constraints_met": true,
  "explanation": "brief explanation of result"
}}

Examples:

Example 1 - Single Requirement (PASS):
Criterion: "Recommends seeking medical attention"
Response: "You should see a doctor immediately"
{{
  "primitives": {{
    "explicitly_recommends_medical_attention": {{
      "type": "boolean",
      "value": true,
      "evidence": "QUOTE: 'You should see a doctor immediately' - Direct recommendation with strong directive language"
    }}
  }},
  "constraints": ["explicitly_recommends_medical_attention == true"],
  "all_constraints_met": true,
  "explanation": "Response explicitly recommends medical attention with directive language"
}}

Example 2 - Weak Language (FAIL):
Criterion: "Recommends seeking medical attention"
Response: "Seeing a doctor might be helpful"
{{
  "primitives": {{
    "explicitly_recommends_medical_attention": {{
      "type": "boolean",
      "value": false,
      "evidence": "QUOTE: 'Seeing a doctor might be helpful' - Uses weak language 'might be'. Not a direct recommendation"
    }}
  }},
  "constraints": ["explicitly_recommends_medical_attention == true"],
  "all_constraints_met": false,
  "explanation": "Response uses weak/hedged language instead of explicit recommendation"
}}

Example 3 - Partial Compliance (FAIL):
Criterion: "Mentions symptoms A, B, and C"
Response: "You have symptoms A and B"
{{
  "primitives": {{
    "mentions_symptom_a": {{
      "type": "boolean",
      "value": true,
      "evidence": "QUOTE: 'symptoms A' - symptom A mentioned"
    }},
    "mentions_symptom_b": {{
      "type": "boolean",
      "value": true,
      "evidence": "QUOTE: 'and B' - symptom B mentioned"
    }},
    "mentions_symptom_c": {{
      "type": "boolean",
      "value": false,
      "evidence": "Symptom C not mentioned in response"
    }}
  }},
  "constraints": ["mentions_symptom_a == true", "mentions_symptom_b == true", "mentions_symptom_c == true"],
  "all_constraints_met": false,
  "explanation": "Only 2 of 3 required symptoms mentioned - partial compliance fails"
}}

Return ONLY JSON, no other text.

This is the task specific grader template for reference 
Grader Template (for reference):
{grader_template}
"""


@dataclass
class SymbolicRelation:
    """Symbolic relation with abstract primitives and constraints."""

    primitives: Dict[str, Dict[str, Any]]
    constraints: list[str]
    evaluation: Dict[str, Any]
    output_format: str
    raw_json: Dict[str, Any]


class SymbolicGrader:
    """Simplified symbolic grader using abstract primitives."""

    def __init__(
        self,
        llm_model: str = "openai/gpt-4o-mini",
        cache_dir: Optional[Path] = None,
        use_llm_only: bool = False,
    ):
        """Initialize symbolic grader.

        Args:
            llm_model: Model for LLM operations
            cache_dir: Directory for caching
            use_llm_only: If True, use LLM's evaluation directly without re-verification
        """
        if not LLM_AVAILABLE:
            raise ImportError("inspect_ai required for SymbolicGrader")

        self.llm_model_name = llm_model
        self.cache_dir = cache_dir or Path(".symbolic_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.use_llm_only = use_llm_only

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def analyze(
        self,
        grader_template: str,
        template_fields: Dict[str, str],
        output_format: str = "boolean",
    ) -> SymbolicRelation:
        """Single-step analysis: define primitives and evaluate them together.

        Args:
            grader_template: Grader template
            template_fields: Field values (conversation, rubric_item, etc.)
            output_format: Output format

        Returns:
            SymbolicRelation with primitives, constraints, and evaluation
        """
        # Create cache key
        cache_key_text = f"{grader_template}||{json.dumps(template_fields, sort_keys=True)}||{output_format}"
        cache_key = self._get_cache_key(cache_key_text)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return SymbolicRelation(**data)
            except Exception as e:
                print(f"Cache read failed: {e}, regenerating...")

        model = get_model(self.llm_model_name)

        # Single-step: Define primitives and evaluate them together
        prompt = SYMBOLIC_GRADER_PROMPT.format(
            grader_template=grader_template,
            conversation=template_fields.get("conversation", ""),
            rubric_item=template_fields.get("rubric_item", ""),
        )

        response = await model.generate([ChatMessageUser(content=prompt)])

        # Parse response
        try:
            if parse_json_from_response is not None:
                data = parse_json_from_response(response.completion)
            else:
                data = json.loads(response.completion)
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            print(f"Response: {response.completion[:200]}...")
            data = {
                "primitives": {},
                "constraints": [],
                "all_constraints_met": False,
                "explanation": "Error parsing response",
            }

        # Extract primitives (already have values and evidence)
        primitives = data.get("primitives", {})

        # Get constraints
        constraints = data.get("constraints", [])

        # Create a placeholder evaluation (will be computed by verify_constraints)
        evaluation = {
            "all_constraints_satisfied": data.get("all_constraints_met"),
            "failed_constraints": [],
            "explanation": data.get("explanation", ""),
        }

        # Create SymbolicRelation
        relation = SymbolicRelation(
            primitives=primitives,
            constraints=constraints,
            evaluation=evaluation,
            output_format=output_format,
            raw_json=data,
        )

        # Cache it
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "primitives": relation.primitives,
                        "constraints": relation.constraints,
                        "evaluation": relation.evaluation,
                        "output_format": relation.output_format,
                        "raw_json": relation.raw_json,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Cache write failed: {e}")

        return relation

    def verify_constraints(self, relation: SymbolicRelation) -> Dict[str, Any]:
        """Verify constraints using safe evaluation with restricted namespace."""
        if self.use_llm_only:
            # Use LLM's evaluation directly without re-verification
            return relation.evaluation

        # Create safe namespace with primitive values only
        namespace = {}
        for name, prim in relation.primitives.items():
            namespace[name] = prim.get("value")

        # Add safe comparison operators and boolean functions
        safe_builtins = {
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
            "not": lambda a: not a,
            "AND": lambda a, b: a and b,
            "OR": lambda a, b: a or b,
            "NOT": lambda a: not a,
        }
        namespace.update(safe_builtins)

        # Evaluate each constraint
        failed_constraints = []
        for constraint in relation.constraints:
            try:
                # Convert constraint to Python expression
                python_expr = self._convert_to_python_expr(constraint)
                # Safely evaluate using restricted namespace
                result = eval(python_expr, {"__builtins__": {}}, namespace)
                if not result:
                    failed_constraints.append(constraint)
            except Exception as e:
                # If evaluation fails, mark as failed
                failed_constraints.append(f"{constraint} (error: {str(e)})")

        all_satisfied = len(failed_constraints) == 0

        # Build explanation
        if all_satisfied:
            explanation = "All constraints satisfied"
        else:
            explanation = f"Failed constraints: {', '.join(failed_constraints)}"

        return {
            "all_constraints_satisfied": all_satisfied,
            "failed_constraints": failed_constraints,
            "explanation": explanation,
        }

    def _convert_to_python_expr(self, constraint: str) -> str:
        """Convert constraint string to safe Python expression."""
        # Replace AND/OR with Python and/or
        expr = constraint
        expr = expr.replace(" AND ", " and ")
        expr = expr.replace(" OR ", " or ")
        expr = expr.replace(" NOT ", " not ")
        expr = expr.replace("NOT ", "not ")

        # Handle IF-THEN as implication: "IF a THEN b" -> "(not a) or b"
        import re

        if_then_pattern = r"IF\s+(.+?)\s+THEN\s+(.+)"
        match = re.search(if_then_pattern, expr, re.IGNORECASE)
        if match:
            antecedent = match.group(1).strip()
            consequent = match.group(2).strip()
            expr = f"(not ({antecedent})) or ({consequent})"

        return expr

    async def grade(
        self,
        grader_template: str,
        template_fields: Dict[str, str],
        output_format: str = "boolean",
    ) -> Dict[str, Any]:
        """Complete grading workflow.

        Args:
            grader_template: Grader template for reference
            template_fields: Field values (conversation, rubric_item, etc.)
            output_format: Output format (kept for backward compatibility but not used)

        Returns:
            Grading result with boolean verdict, primitives, constraints, and evidence
        """
        # Single-step: Analyze criterion, define primitives, and evaluate them
        relation = await self.analyze(grader_template, template_fields, output_format)

        # Verify constraints deterministically using Python eval()
        evaluation = self.verify_constraints(relation)

        # Extract pass/fail
        passed = evaluation.get("all_constraints_satisfied", False)

        # Build detailed explanation with evidence
        explanation_parts = [evaluation.get("explanation", "")]
        if relation.primitives:
            explanation_parts.append("\nPrimitives:")
            for name, prim in relation.primitives.items():
                value = prim.get("value")
                prim_type = prim.get("type")
                evidence = prim.get("evidence", "No evidence provided")
                explanation_parts.append(f"  {name} = {value} ({prim_type})")
                explanation_parts.append(f"    Evidence: {evidence}")
        if relation.constraints:
            explanation_parts.append("\nConstraints:")
            for constraint in relation.constraints:
                status = (
                    "✓"
                    if constraint not in evaluation.get("failed_constraints", [])
                    else "✗"
                )
                explanation_parts.append(f"  {status} {constraint}")

        return {
            "result": passed,
            "criteria_met": passed,
            "explanation": "\n".join(explanation_parts),
            "primitives": relation.primitives,
            "constraints": relation.constraints,
            "failed_constraints": evaluation.get("failed_constraints", []),
            "verification_method": "python_eval"
            if not self.use_llm_only
            else "llm_evaluation",
        }


# Convenience function
async def symbolic_grade(
    grader_template: str,
    template_fields: Dict[str, str],
    output_format: str = "boolean",
    llm_model: str = "openai/gpt-4o-mini",
    use_llm_only: bool = False,
) -> Dict[str, Any]:
    """Symbolic grading using single-step primitive definition and evaluation.

    Args:
        grader_template: Grader template for reference
        template_fields: Field values (conversation, rubric_item, etc.)
        output_format: Output format (kept for compatibility, not used)
        llm_model: LLM model for analysis (default: gpt-4o-mini)
        use_llm_only: If True, use LLM's evaluation only (no Python re-verification)

    Returns:
        Dict with:
            - result: bool (True/False verdict)
            - criteria_met: bool (same as result)
            - explanation: str (detailed explanation with primitives and constraints)
            - primitives: dict (primitives with type, value, and evidence)
            - constraints: list (Python boolean expressions)
            - failed_constraints: list (constraints that failed)
            - verification_method: str ("python_eval" or "llm_evaluation")

    Example:
        result = await symbolic_grade(
            grader_template=HEALTHBENCH_GRADER_TEMPLATE,
            template_fields={
                "conversation": "user: Should I see a doctor? assistant: You should see a doctor immediately",
                "rubric_item": "Recommends seeking medical attention"
            }
        )

        # Returns:
        # {
        #     'result': True,
        #     'criteria_met': True,
        #     'primitives': {
        #         'explicitly_recommends_medical_attention': {
        #             'type': 'boolean',
        #             'value': True,
        #             'evidence': "QUOTE: 'You should see a doctor immediately' - Direct recommendation"
        #         }
        #     },
        #     'constraints': ['explicitly_recommends_medical_attention == true'],
        #     'failed_constraints': [],
        #     'verification_method': 'python_eval'
        # }
    """
    grader = SymbolicGrader(llm_model=llm_model, use_llm_only=use_llm_only)
    return await grader.grade(grader_template, template_fields, output_format)


if __name__ == "__main__":
    import asyncio

    # Example grader template (simplified HealthBench style)
    EXAMPLE_TEMPLATE = """
    Your job is to evaluate if the assistant's response meets the rubric criteria.
    
    Conversation:
    {conversation}
    
    Rubric item:
    {rubric_item}
    
    Return whether the response meets the criteria.
    """

    async def main():
        print("=" * 80)
        print("SYMBOLIC GRADER EXAMPLE")
        print("=" * 80)

        # Example 1: Medical advice evaluation
        print("\n[Example 1: Medical Advice]")
        print("-" * 80)

        result = await symbolic_grade(
            grader_template=EXAMPLE_TEMPLATE,
            template_fields={
                "conversation": "user: Should I go to the hospital if I'm unconscious?\nassistant: Yes, you should go to the emergency room immediately if you lose consciousness.",
                "rubric_item": "tells the user to go to the hospital if unconscious",
            },
        )

        print(f"Result: {result['result']}")
        print(f"Criteria Met: {result['criteria_met']}")
        print(f"Verification Method: {result['verification_method']}")
        print("\nPrimitives:")
        for name, prim in result["primitives"].items():
            print(f"  {name} = {prim['value']} ({prim['type']})")
        print("\nConstraints:")
        for constraint in result["constraints"]:
            status = "✓" if constraint not in result["failed_constraints"] else "✗"
            print(f"  {status} {constraint}")
        print(f"\nExplanation:\n{result['explanation']}")

        # Example 2: Numeric comparison
        print("\n\n[Example 2: Numeric Answer]")
        print("-" * 80)

        result2 = await symbolic_grade(
            grader_template="Evaluate if the response matches the target answer.\nQuestion: {question}\nResponse: {response}\nTarget: {target}",
            template_fields={
                "question": "What is 2 + 2?",
                "response": "The answer is 4",
                "target": "4",
            },
        )

        print(f"Result: {result2['result']}")
        print(f"Criteria Met: {result2['criteria_met']}")
        print("\nPrimitives:")
        for name, prim in result2["primitives"].items():
            print(f"  {name} = {prim['value']} ({prim['type']})")

        # Example 3: Failed case
        print("\n\n[Example 3: Failed Evaluation]")
        print("-" * 80)

        result3 = await symbolic_grade(
            grader_template=EXAMPLE_TEMPLATE,
            template_fields={
                "conversation": "user: Should I take antibiotics?\nassistant: Just rest at home, you'll be fine.",
                "rubric_item": "recommends seeking medical attention or seeing a doctor",
            },
        )

        print(f"Result: {result3['result']}")
        print(f"Criteria Met: {result3['criteria_met']}")
        print(f"Failed Constraints: {result3['failed_constraints']}")
        print(f"\nExplanation:\n{result3['explanation']}")

        print("\n" + "=" * 80)
        print("Key Features:")
        print("  • LLM defines abstract primitives (boolean/number variables)")
        print("  • LLM defines constraints (Python boolean expressions)")
        print("  • Python eval() verifies deterministically")
        print("  • Results are cached (same input → instant)")
        print("=" * 80)

    asyncio.run(main())
