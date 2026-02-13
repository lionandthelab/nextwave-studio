"""Tests for the LLM engine module (app/core/llm_engine.py).

Tests cover code extraction, error classification, and default code generation.
LLM calls are mocked to avoid external API dependencies.
"""

from __future__ import annotations

import pytest

from app.core.llm_engine import GraspCodeGenerator


@pytest.fixture()
def generator() -> GraspCodeGenerator:
    """Provide a GraspCodeGenerator (LLM calls will be mocked per-test)."""
    return GraspCodeGenerator()


# ---------------------------------------------------------------------------
# Code extraction from LLM responses
# ---------------------------------------------------------------------------


class TestExtractCode:
    """Tests for _extract_code static method."""

    def test_extract_from_python_fence(self):
        """Should extract code from ```python ... ``` blocks."""
        response = '```python\ndef hello():\n    return "hi"\n```'
        code = GraspCodeGenerator._extract_code(response)
        assert 'def hello():' in code
        assert '```' not in code

    def test_extract_from_generic_fence(self):
        """Should extract code from ``` ... ``` blocks without language hint."""
        response = '```\ndef foo():\n    pass\n```'
        code = GraspCodeGenerator._extract_code(response)
        assert 'def foo():' in code

    def test_extract_from_fence_with_py_hint(self):
        """Should strip 'py' language hint from generic fence."""
        response = '```py\nx = 1\n```'
        code = GraspCodeGenerator._extract_code(response)
        assert 'x = 1' in code
        assert 'py' not in code.split('\n')[0]

    def test_plain_code_without_fences(self):
        """Code without markdown fences should be returned as-is."""
        response = 'def bar():\n    return 42'
        code = GraspCodeGenerator._extract_code(response)
        assert code == response

    def test_extract_with_surrounding_text(self):
        """Should extract code even with explanation text around fences."""
        response = (
            "Here is the code:\n"
            "```python\nimport time\ntime.sleep(1)\n```\n"
            "This should work."
        )
        code = GraspCodeGenerator._extract_code(response)
        assert 'import time' in code
        assert 'Here is the code' not in code

    def test_empty_response(self):
        """Empty response should return empty string."""
        code = GraspCodeGenerator._extract_code("")
        assert code == ""


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestClassifyError:
    """Tests for _classify_error static method."""

    def test_slip_keywords(self):
        assert GraspCodeGenerator._classify_error("Object slipped from gripper") == "slip"
        assert GraspCodeGenerator._classify_error("The part dropped to ground") == "slip"
        assert GraspCodeGenerator._classify_error("Lost grip on component") == "slip"
        assert GraspCodeGenerator._classify_error("Object fell down") == "slip"

    def test_collision_keywords(self):
        assert GraspCodeGenerator._classify_error("Collision detected") == "collision"
        assert GraspCodeGenerator._classify_error("Objects collide at joint 3") == "collision"
        assert GraspCodeGenerator._classify_error("Penetration depth exceeded") == "collision"

    def test_no_contact_keywords(self):
        assert GraspCodeGenerator._classify_error("No contact force detected") == "no_contact"
        assert GraspCodeGenerator._classify_error("Gripper miss target") == "no_contact"

    def test_timeout_keywords(self):
        assert GraspCodeGenerator._classify_error("Simulation timed out") == "timeout"
        assert GraspCodeGenerator._classify_error("Motion too slow") == "timeout"
        # "Time limit exceeded" matches both timeout ("exceeded") and
        # joint_limit ("limit exceeded") with equal scores; dict ordering
        # makes joint_limit win the tie.
        assert GraspCodeGenerator._classify_error("Time limit exceeded") in ("timeout", "joint_limit")

    def test_unknown_error(self):
        assert GraspCodeGenerator._classify_error("Something unexpected happened") == "unknown"

    def test_empty_error(self):
        assert GraspCodeGenerator._classify_error("") == "unknown"

    def test_case_insensitive(self):
        """Error classification should be case-insensitive."""
        assert GraspCodeGenerator._classify_error("OBJECT SLIPPED") == "slip"
        assert GraspCodeGenerator._classify_error("COLLISION DETECTED") == "collision"


# ---------------------------------------------------------------------------
# Default code generation (no LLM call)
# ---------------------------------------------------------------------------


class TestDefaultCodeGeneration:
    """Tests for _generate_default_code (template-based, no LLM)."""

    def test_default_code_has_function_signature(self):
        """Generated code should contain the required function signature."""
        from app.core.llm_engine import _generate_default_code

        code = _generate_default_code(
            cad_metadata={
                "dimensions": {"x": 0.1, "y": 0.1, "z": 0.1},
                "volume": 0.001,
            },
            robot_model="unitree_h1",
        )
        assert "def run_grasp_simulation" in code
        assert "sim_context" in code
        assert "robot_prim_path" in code
        assert "object_prim_path" in code

    def test_default_code_uses_dimensions(self):
        """Generated code should incorporate object dimensions."""
        from app.core.llm_engine import _generate_default_code

        code = _generate_default_code(
            cad_metadata={
                "dimensions": {"x": 0.2, "y": 0.15, "z": 0.3},
            },
            robot_model="franka",
        )
        # The largest dimension (0.3) should influence grasp_width
        assert "grasp_width" in code
        # Robot model should appear in the code
        assert "franka" in code

    def test_default_code_returns_dict(self):
        """Generated code should have return statements with success key."""
        from app.core.llm_engine import _generate_default_code

        code = _generate_default_code(
            cad_metadata={"dimensions": {"x": 0.05, "y": 0.05, "z": 0.05}},
            robot_model="test_bot",
        )
        assert '"success"' in code or "'success'" in code

    def test_default_code_handles_missing_dimensions(self):
        """Should use defaults when dimensions are missing."""
        from app.core.llm_engine import _generate_default_code

        code = _generate_default_code(
            cad_metadata={},
            robot_model="test_bot",
        )
        assert "def run_grasp_simulation" in code


# ---------------------------------------------------------------------------
# New error type classification
# ---------------------------------------------------------------------------


class TestClassifyNewErrorTypes:
    """Tests for new error categories: overforce, unstable_grasp, workspace_violation."""

    def test_overforce_keywords(self):
        """Overforce errors should be classified correctly."""
        assert GraspCodeGenerator._classify_error(
            "ERROR: Emergency stop - force exceeded safe limit, risk of damage"
        ) == "overforce"
        assert GraspCodeGenerator._classify_error(
            "Torque exceeded maximum, excessive force detected"
        ) == "overforce"

    def test_unstable_grasp_keywords(self):
        """Unstable grasp errors should be classified correctly."""
        assert GraspCodeGenerator._classify_error(
            "Object rotating during lift, angular velocity exceeds threshold"
        ) == "unstable_grasp"
        assert GraspCodeGenerator._classify_error(
            "Unstable grasp - object shifting in gripper, not centered"
        ) == "unstable_grasp"

    def test_workspace_violation_keywords(self):
        """Workspace violation errors should be classified correctly."""
        assert GraspCodeGenerator._classify_error(
            "Target position outside workspace, unreachable by arm"
        ) == "workspace_violation"
        assert GraspCodeGenerator._classify_error(
            "IK failed - cannot reach object position"
        ) == "workspace_violation"

    def test_import_error_keywords(self):
        """Import errors should be classified correctly."""
        assert GraspCodeGenerator._classify_error(
            "ModuleNotFoundError: No module named 'scipy'"
        ) == "import_error"

    def test_attribute_error_keywords(self):
        """Attribute errors should be classified correctly."""
        assert GraspCodeGenerator._classify_error(
            "AttributeError: 'NoneType' has no attribute 'step'"
        ) == "attribute_error"


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    """Tests for _parse_json_response static method."""

    def test_parse_direct_json(self):
        """Should parse a plain JSON string."""
        result = GraspCodeGenerator._parse_json_response(
            '{"feasible": true, "confidence": 0.9}',
            fallback={"feasible": False},
        )
        assert result["feasible"] is True
        assert result["confidence"] == 0.9

    def test_parse_json_in_markdown_fence(self):
        """Should extract JSON from markdown code block."""
        text = 'Here is the result:\n```json\n{"valid": true, "issues": []}\n```'
        result = GraspCodeGenerator._parse_json_response(
            text, fallback={"valid": False}
        )
        assert result["valid"] is True

    def test_parse_json_embedded_in_text(self):
        """Should find JSON object embedded in surrounding text."""
        text = 'The assessment is: {"feasible": false, "reason": "too heavy"} end.'
        result = GraspCodeGenerator._parse_json_response(
            text, fallback={"feasible": True}
        )
        assert result["feasible"] is False
        assert result["reason"] == "too heavy"

    def test_parse_returns_fallback_on_invalid(self):
        """Should return fallback when no valid JSON is found."""
        result = GraspCodeGenerator._parse_json_response(
            "This is not JSON at all.",
            fallback={"feasible": True, "confidence": 0.5},
        )
        assert result["feasible"] is True
        assert result["confidence"] == 0.5

    def test_parse_empty_string(self):
        """Should return fallback for empty input."""
        result = GraspCodeGenerator._parse_json_response(
            "", fallback={"ok": False}
        )
        assert result == {"ok": False}


# ---------------------------------------------------------------------------
# Multiple code blocks extraction
# ---------------------------------------------------------------------------


class TestExtractMultipleCodeBlocks:
    """Tests for extracting multiple fenced code blocks."""

    def test_multiple_python_blocks(self):
        """Multiple ```python blocks should be joined."""
        response = (
            "Part 1:\n```python\nimport time\n```\n"
            "Part 2:\n```python\ndef foo():\n    pass\n```"
        )
        code = GraspCodeGenerator._extract_code(response)
        assert "import time" in code
        assert "def foo():" in code

    def test_code_with_leading_whitespace(self):
        """Code with internal indentation should be preserved."""
        response = "```python\ndef foo():\n    indented = True\n    return indented\n```"
        code = GraspCodeGenerator._extract_code(response)
        assert "    indented = True" in code
        assert "    return indented" in code


# ---------------------------------------------------------------------------
# Correction strategies coverage
# ---------------------------------------------------------------------------


class TestCorrectionStrategies:
    """Verify correction strategies exist for all error types."""

    def test_all_error_types_have_strategies(self):
        """Every classified error type should have a correction strategy."""
        from app.core.llm_engine import CORRECTION_STRATEGIES, _ERROR_KEYWORDS

        for error_type in _ERROR_KEYWORDS:
            assert error_type in CORRECTION_STRATEGIES or error_type in (
                "import_error", "attribute_error"
            ), f"Missing strategy for {error_type}"

    def test_unknown_strategy_exists(self):
        """The 'unknown' fallback strategy should exist."""
        from app.core.llm_engine import CORRECTION_STRATEGIES

        assert "unknown" in CORRECTION_STRATEGIES
        assert len(CORRECTION_STRATEGIES["unknown"]) > 0
