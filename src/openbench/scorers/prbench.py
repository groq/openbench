"""PRBench scorer - using PRBench code directly."""

from typing import Callable, List, Tuple

from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from openbench.utils.text import parse_json_from_response
from openbench.metrics.prbench import prbench_metrics


# Copy of GRADER_TEMPLATE and GRADE_MAP from PRBench/constants.py
GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "The response refers to the need to confirm the business case for the workforce reduction proposed, e.g. is there a genuine reason for fewer employees, is the downturn likely to be long lasting, what reduction in head count is actually required?", and the response just mentions declines in demand, that would still meet the criteria even though it didn't mention one of the examples verbatim.

# Example 1
For example, if the conversation is "user: should I buy bitcoin? assistant: no just buy nvidia stock" and the rubric item is "tells the user to go to buy bitcoin", you should return a json like this:

```json
{
  "explanation": "The assistant does not tell the user to buy bitcoin.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: what are some red fruits? assistant: apples, strawberries" and the criterion is "fails to mention apples", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually mentions apples, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

GRADE_MAP = {
    "true": 1,
    "false": 0,
}

FINANCE_DECISION_TYPE_TEMPLATE = """TASK: Classify the prompt into one FINANCE DECISION TYPE (primary). Add one secondary only if strictly inseparable.

If multi-turn, classify primarily by the LAST user turn.

<<user_turns>>

CATEGORIES (A-Z):

A Governance & Policy

(Description: Set enduring rules or postures such as accounting/tax elections, risk appetite, or disclosure stance.)

(Examples: “Should we elect LIFO or FIFO for tax reporting?” “Do we raise our risk appetite for credit exposure?” “Should dividends be a fixed policy or discretionary?” “Do we disclose climate risks in MD&A this year?”)

B Modeling & Measurement

(Description: Define how value, exposure, or performance is measured, modeled, and interpreted.)

(Examples: “How should we measure portfolio VaR across currencies?” “What's the right discount rate for project valuation?” “Do we model beta using weekly or monthly returns?” “How to estimate expected credit loss under IFRS 9?”)

C Capital & Funding

(Description: Choose balance-sheet structure, financing mix, and capital allocation priorities.)

(Examples: “Should we issue new equity or refinance debt?” “How much leverage can we take without breaching covenants?” “Do we fund expansion from retained earnings or external capital?” “Is it optimal to repurchase shares at current valuation?”)

D Markets & Transactions

(Description: Decide how, when, and at what price to transact in markets or strategic deals.)

(Examples: “When's the best time to execute the bond buyback?” “Should we hedge FX now or wait for better liquidity?” “At what price do we enter the secondary offering?” “Which trading venue minimizes slippage for this order?”)

E Operations, Processes & Controls

(Description: Set repeatable cash, control, and process steps to meet operational and financial obligations.)

(Examples: “How do we automate vendor payment approvals?” “Should we shorten the monthly close cycle?” “What's the best control for petty cash discrepancies?” “How can we speed up receivables collection safely?”)

F Planning & Forecasts

(Description: Set budgets, targets, scenarios, and rolling forecasts to guide performance and risk planning.)

(Examples: “Should we raise our revenue target for next quarter?” “How much buffer to build into cash forecasts?” “Do we base next year's budget on trend or zero-based planning?” “What's the scenario if rates rise by 100 bps?”)

G Compliance & Reporting

(Description: Ensure financial actions, records, and disclosures align with regulatory, accounting, and internal standards.)

(Examples: “Do we meet IFRS 16 lease disclosure requirements?” “Are we compliant with new AML reporting thresholds?” “What filings are due after our debt restructuring?” “Do we need auditor sign-off before publishing results?”)

O Other

(Description: Decision requests that don't fit the above in this lean scheme; use sparingly.)

Z Non-decision / Informational

(Description: General explanation or background without a decision component.)

(Examples: “What's the difference between EBITDA and operating income?” “How do interest rate swaps work?” “What is free cash flow conversion?” “How is goodwill impairment tested?”)

```json
{
  "primary": {"code": "A", "label": "Governance & Policy"},
  "secondary": []
}
```

or if there is a secondary decision type:

```json
{
  "primary": {"code": "F", "label": "Planning & Forecasts"},
  "secondary": [{"code": "A", "label": "Governance & Policy"}]
}
```

Final Instruction:

Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

FINANCE_ECONOMIC_PATHWAY_TEMPLATE = """TASK: Classify the prompt into one FINANCE ECONOMIC PATHWAY (primary). Add one secondary only if strictly inseparable.

If multi-turn, classify primarily by the LAST user turn. Other is a catch-all category for anything that has an economic pathway that is not one of the other categories.

Informational / Educational Only is a catch-all category for anything that doesn't have an economic pathway. For example, if the prompt is "user: what is the capital structure of the company? assistant: the capital structure is 50% debt and 50% equity", then the economic pathway is "Informational / Educational Only".

<<user_turns>>

CATEGORIES (A-Z):

A Value Creation

(Description: Decisions that increase profitability, valuation, or investment performance through higher earnings, NPV, IRR, or ROE.)

(Examples: “Should we invest in automation to boost ROI?” “Does expanding into Asia improve our NPV?” “Will share buybacks lift EPS more than dividends?” “How much value does the new product add to EBITDA?”)

B Operating Efficiency

(Description: Actions that improve cost structure, productivity, or capital utilization to enhance margins and resource use.)

(Examples: “Can we cut logistics costs without hurting service?” “Should we consolidate warehouses to free up capital?” “Will outsourcing payroll improve margin efficiency?” “How do we reduce idle capacity in production?”)

C Risk & Resilience

(Description: Strategies that reduce exposure to market, credit, liquidity, or operational risks, lowering volatility or loss potential.)

(Examples: “Should we hedge commodity exposure at current prices?” “What's the best mix of fixed vs. floating debt now?” “How do we diversify revenue to cushion downturns?” “Can we add liquidity buffers to handle a credit crunch?”)

D Funding Optimization

(Description: Financing, treasury, or strategic choices that improve funding cost, stability, or flexibility through better capital structure or liquidity management.)

(Examples: “Should we issue longer-term bonds at today's rates?” “Do we refinance now or wait for better spreads?” “How can we improve our interest coverage ratio?” “Is a revolving credit facility better than short-term loans?”)

E Compliance and Reporting Integrity

(Description: Efforts ensuring regulatory, accounting, and disclosure accuracy to maintain transparency, trust, and market access.)

(Examples: “Are our revenue disclosures aligned with IFRS 15?” “Do we need to restate last year's tax provision?” “How do we ensure audit trails meet SOX standards?” “What steps prevent misstatement of fair values?”)

O Other

(Description: Economic outcomes not clearly aligned with the main pathways.)

Z Informational / Educational Only

(Description: Purely explanatory or conceptual content with no direct economic consequence.)

(Examples: “What's the difference between NPV and IRR?” “How does leverage amplify returns?” “What is Basel III capital adequacy?” “How do rating agencies assess liquidity risk?”)

```json
{
  "primary": {"code": "E", "label": "Compliance and Reporting Integrity"},
  "secondary": []
}
```

or if there is a secondary decision type:

```json
{
  "primary": {"code": "A", "label": "Value Creation"},
  "secondary": [{"code": "B", "label": "Operating Efficiency"}]
}
```

Final Instruction:

Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


# Copy of create_convo from PRBench/util.py
def create_convo(convo: List[Tuple[str, str]]):
    s = ""
    for speaker, statement in convo:
        s += speaker + ": "
        try:
            s += statement + "\n"
        except Exception as e:
            print(f"Error creating convo: {e}")
            s += "N/A" + "\n"
    return s.strip()


# Copy of get_criteria_met from PRBench/util.py
def get_criteria_met(response):
    if response == "N/A":
        print("Grading timed out")
        return "false"

    if not isinstance(response, str):
        print(f"Response is not a string: {response}")
        return "false"

    grade = response.split('criteria_met":')[-1].split("\n")[0].strip()

    if "true" in grade.lower() and "false" in grade.lower():
        print(f"Unknown grade for response: {response}")
        return "false"
    if "true" in grade.lower():
        return "true"
    elif "false" in grade.lower():
        return "false"
    else:
        print(f"Unknown grade for response: {response}")
        return "false"


# Copy of get_clipped_points from PRBench/util.py
def get_clipped_points(point_total, weights):
    max_weight = sum([w for w in weights if w > 0])
    return point_total / max_weight


@scorer(metrics=[accuracy(), stderr(), prbench_metrics()])
def prbench_scorer(grader_model: str = "openai/gpt-4o-mini") -> Callable:
    """PRBench scorer using PRBench code directly."""
    model = get_model(grader_model)

    async def score(state: TaskState, target: Target) -> Score:
        # Get rubrics from metadata (Criterion objects)
        rubrics = state.metadata.get("rubrics", [])
        if not rubrics:
            return Score(value=0.0, explanation="No rubrics found")

        # Get field and task
        field = state.metadata.get("field", "")
        task = state.metadata.get("task", "")

        # Get conversation and build with response
        conversation = state.metadata.get("conversation", [])
        convo_with_response = conversation + [("assistant", state.output.completion)]
        convo_str = create_convo(convo_with_response)

        # Extract rubric titles and weights using Criterion methods
        rubrics_texts = [r.get_title() for r in rubrics]
        weights = [r.get_weight() for r in rubrics]

        # Grade each rubric
        grades = []
        grading_results = []
        for rubric_title in rubrics_texts:
            # Format grading prompt using PRBench template
            grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str)
            grader_prompt = grader_prompt.replace("<<rubric_item>>", rubric_title)

            # Get grading from model
            result = await model.generate([ChatMessageUser(content=grader_prompt)])
            grading_text = result.completion

            # Extract criteria_met using PRBench's function
            criteria_met_str = get_criteria_met(grading_text)
            grades.append(criteria_met_str)

            # Try to parse JSON for explanation
            try:
                grading_dict = parse_json_from_response(grading_text)
                grading_results.append(grading_dict)
            except Exception:
                grading_results.append(
                    {
                        "criteria_met": criteria_met_str == "true",
                        "explanation": "Could not parse grading response",
                    }
                )

        # Calculate points using PRBench's GRADE_MAP
        points = sum([GRADE_MAP[g] * w for g, w in zip(grades, weights)])

        # Calculate clipped score using PRBench's function
        clipped_score = max(0.0, get_clipped_points(points, weights))

        # Build explanation
        explanations = []
        for rubric, grade_str, grading in zip(rubrics, grades, grading_results):
            met = grade_str == "true"
            exp = grading.get("explanation", "No explanation")
            status = "✓" if met else "✗"
            explanations.append(f"[{status}] {rubric.get_title()}\n  {exp}")

        explanations.sort(key=lambda x: x.startswith("[✗]"), reverse=True)

        return Score(
            value=clipped_score,
            answer=state.output.completion,
            explanation="\n\n".join(explanations),
            metadata={
                "mean_clipped": clipped_score,
                "points": points,
                "weights": weights,
                "grades": grades,
                "field": field,
                "task": task,
            },
        )

    return score
