"""Tests for symbolic grader utility."""

import os
import pytest
from openbench.utils.symbolic_grader import (
    symbolic_grade,
    SymbolicGrader,
)

# Check for API key
@pytest.fixture(scope="session", autouse=True)
def check_api_key():
    """Check that OpenAI API key is set before running tests."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")

# Simple grader template for testing
TEST_TEMPLATE = """
Evaluate if the response meets the criterion.

Determine if the criterion is satisfied based on the conversation.
"""


@pytest.mark.asyncio
class TestSymbolicGrader:
    """Tests for the main symbolic grader."""

    async def test_basic_structure(self):
        """Test that symbolic grader returns correct structure."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: Should I go somewhere?\nAssistant: You should go to the hospital.",
                "rubric_item": "tells user to go to hospital"
            }
        )

        assert "result" in result
        assert "criteria_met" in result
        assert "explanation" in result
        assert "primitives" in result
        assert "constraints" in result
        assert "verification_method" in result

    async def test_basic_pass(self):
        """Test basic evaluation that should pass."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I do?\nAssistant: You should go to the emergency room immediately.",
                "rubric_item": "tells user to seek emergency care"
            }
        )

        assert result["result"] is True
        assert result["criteria_met"] is True

    async def test_basic_fail(self):
        """Test basic evaluation that should fail."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I do?\nAssistant: Just rest at home.",
                "rubric_item": "tells user to go to hospital"
            }
        )

        assert result["result"] is False
        assert result["criteria_met"] is False

    async def test_with_target(self):
        """Test evaluation with target answer."""
        template = "Evaluate if response matches target."
        
        result = await symbolic_grade(
            grader_template=template,
            template_fields={
                "conversation": "User: What is 2+2?\nAssistant: The answer is 4",
                "rubric_item": "response correctly states that 2+2 equals 4"
            }
        )

        # Should have primitives extracted
        assert len(result["primitives"]) > 0

    async def test_primitives_structure(self):
        """Test that primitives have correct structure."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I take?\nAssistant: Take antibiotics for infection.",
                "rubric_item": "recommends antibiotics"
            }
        )

        # Check primitives structure
        for name, prim in result["primitives"].items():
            assert "type" in prim
            assert "value" in prim
            assert prim["type"] in ["boolean", "number", "string"]

    async def test_constraints_list(self):
        """Test that constraints are returned as list."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I do?\nAssistant: Visit the doctor immediately.",
                "rubric_item": "recommends seeing a doctor"
            }
        )

        assert isinstance(result["constraints"], list)
        assert isinstance(result["failed_constraints"], list)

    async def test_grader_instance(self):
        """Test using SymbolicGrader instance directly."""
        grader = SymbolicGrader(llm_model="openai/gpt-4o-mini")

        result = await grader.grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I do?\nAssistant: Call 911 for emergency.",
                "rubric_item": "tells user to call emergency services"
            }
        )

        assert "result" in result
        assert "criteria_met" in result

    async def test_llm_only_mode(self):
        """Test using LLM evaluation only without Python verification."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I do?\nAssistant: Take medicine as prescribed.",
                "rubric_item": "mentions taking medication"
            },
            use_llm_only=True
        )

        assert result["verification_method"] == "llm_evaluation"

    async def test_python_eval_mode(self):
        """Test using Python eval verification."""
        result = await symbolic_grade(
            grader_template=TEST_TEMPLATE,
            template_fields={
                "conversation": "User: What should I do?\nAssistant: Rest and drink water.",
                "rubric_item": "recommends rest and hydration"
            },
            use_llm_only=False
        )

        assert result["verification_method"] == "python_eval"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
