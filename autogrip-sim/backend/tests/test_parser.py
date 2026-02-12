"""Tests for the ManualParser (app/core/parser.py).

Because ``fitz`` (PyMuPDF) may not be installed in the test environment,
we install a lightweight stub into ``sys.modules`` before importing the parser.
This allows testing all regex / chunking logic without the C extension.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Patch fitz before importing ManualParser
# ---------------------------------------------------------------------------
_fitz_stub = types.ModuleType("fitz")
_fitz_stub.open = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("fitz", _fitz_stub)

from app.core.parser import ManualParser  # noqa: E402


@pytest.fixture()
def parser() -> ManualParser:
    """Provide a ManualParser with default settings."""
    return ManualParser(chunk_size=200, chunk_overlap=50)


# ---------------------------------------------------------------------------
# Joint name extraction
# ---------------------------------------------------------------------------


class TestExtractJointNames:
    """Tests for extracting joint names from text."""

    def test_joint_n_pattern(self, parser: ManualParser):
        """Should match 'Joint 1', 'Joint 2', etc."""
        text = "The robot has Joint 1 and Joint 2 for arm motion."
        names = parser._extract_joint_names(text)
        name_lower = [n.lower() for n in names]
        assert any("joint" in n and "1" in n for n in name_lower)
        assert any("joint" in n and "2" in n for n in name_lower)

    def test_j_n_pattern(self, parser: ManualParser):
        """Should match 'J1', 'J2', etc."""
        text = "Connect J1 to the controller. J2 handles rotation."
        names = parser._extract_joint_names(text)
        name_lower = [n.lower() for n in names]
        assert any("j1" in n for n in name_lower)
        assert any("j2" in n for n in name_lower)

    def test_named_joints(self, parser: ManualParser):
        """Should match named joints like 'left_finger', 'right_gripper'."""
        text = "The left_finger and right_gripper are controlled independently."
        names = parser._extract_joint_names(text)
        name_lower = [n.lower() for n in names]
        assert any("left" in n and "finger" in n for n in name_lower)
        assert any("right" in n and "gripper" in n for n in name_lower)

    def test_deduplication(self, parser: ManualParser):
        """Duplicate joint names should be deduplicated."""
        text = "Joint 1 is strong. Joint 1 is also precise."
        names = parser._extract_joint_names(text)
        lower_names = [n.lower() for n in names]
        matches = [n for n in lower_names if "joint" in n and "1" in n]
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# Control function extraction
# ---------------------------------------------------------------------------


class TestExtractControlFunctions:
    """Tests for extracting control function names from text."""

    def test_set_position(self, parser: ManualParser):
        """Should detect set_position function calls."""
        text = "Use set_position(angle) to control the arm."
        funcs = parser._extract_control_functions(text)
        assert "set_position" in funcs

    def test_move_joint(self, parser: ManualParser):
        """Should detect move_joint function calls."""
        text = "Call move_joint(id, target) for each joint."
        funcs = parser._extract_control_functions(text)
        assert "move_joint" in funcs

    def test_method_calls(self, parser: ManualParser):
        """Should detect method calls like robot.set_joint_positions()."""
        text = "robot.set_joint_positions([1.0, 2.0])"
        funcs = parser._extract_control_functions(text)
        assert "set_joint_positions" in funcs

    def test_non_control_functions_excluded(self, parser: ManualParser):
        """Functions that are not robot-control-related should be excluded."""
        text = "print(hello) and len(items) and range(10)"
        funcs = parser._extract_control_functions(text)
        assert "print" not in funcs
        assert "len" not in funcs
        assert "range" not in funcs

    def test_multiple_functions(self, parser: ManualParser):
        """Multiple control functions should be detected."""
        text = (
            "Call set_position(x), then move_joint(j, v), "
            "then open_gripper() and close_gripper()."
        )
        funcs = parser._extract_control_functions(text)
        assert "set_position" in funcs
        assert "move_joint" in funcs
        assert "open_gripper" in funcs
        assert "close_gripper" in funcs


# ---------------------------------------------------------------------------
# Motor spec extraction
# ---------------------------------------------------------------------------


class TestExtractMotorSpecs:
    """Tests for extracting motor specifications from text."""

    def test_torque_extraction(self, parser: ManualParser):
        """Should extract max torque values."""
        text = "Maximum torque: 5.0 Nm for each actuator."
        specs = parser._extract_motor_specs(text)
        assert "torque" in specs
        assert specs["torque"]["max_nm"] == 5.0

    def test_speed_extraction(self, parser: ManualParser):
        """Should extract max speed values."""
        text = "max speed: 180 deg/s across all joints."
        specs = parser._extract_motor_specs(text)
        assert "speed" in specs
        assert specs["speed"]["max_value"] == 180.0

    def test_range_extraction(self, parser: ManualParser):
        """Should extract joint range values."""
        text = "Joint range: -180 to 180 degrees."
        specs = parser._extract_motor_specs(text)
        assert "joint_range" in specs
        assert specs["joint_range"]["min"] == -180.0
        assert specs["joint_range"]["max"] == 180.0

    def test_payload_extraction(self, parser: ManualParser):
        """Should extract payload capacity."""
        text = "Payload: 5.0 kg max lifting capacity."
        specs = parser._extract_motor_specs(text)
        assert "payload" in specs
        assert specs["payload"]["max_kg"] == 5.0

    def test_grip_force_extraction(self, parser: ManualParser):
        """Should extract grip force."""
        text = "Grip force: 100 N for secure holding."
        specs = parser._extract_motor_specs(text)
        assert "grip_force" in specs
        assert specs["grip_force"]["max_n"] == 100.0

    def test_no_specs(self, parser: ManualParser):
        """Text without specs should return an empty dict."""
        text = "This is a general introduction to the robot."
        specs = parser._extract_motor_specs(text)
        assert specs == {}


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------


class TestChunkSplitting:
    """Tests for text chunking with overlap."""

    def test_short_text_single_chunk(self, parser: ManualParser):
        """Short text should produce a single chunk."""
        text = "This is a short text."
        chunks = parser._split_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self, parser: ManualParser):
        """Long text should be split into multiple overlapping chunks."""
        text = "Word " * 100  # ~500 chars
        chunks = parser._split_into_chunks(text)
        assert len(chunks) > 1

    def test_chunk_overlap(self, parser: ManualParser):
        """Adjacent chunks should share overlapping content."""
        text = "A" * 500
        chunks = parser._split_into_chunks(text)
        if len(chunks) >= 2:
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    def test_empty_text(self, parser: ManualParser):
        """Empty text should produce no chunks."""
        chunks = parser._split_into_chunks("")
        assert chunks == []

    def test_chunks_not_empty(self, parser: ManualParser):
        """All chunks should be non-empty strings."""
        text = "Some text. " * 50
        chunks = parser._split_into_chunks(text)
        for chunk in chunks:
            assert len(chunk) > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestParseErrors:
    """Tests for parser error handling."""

    def test_parse_nonexistent_file(self, parser: ManualParser):
        """Parsing a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/manual.pdf")

    def test_parse_non_pdf_file(self, parser: ManualParser):
        """Parsing a non-PDF file should raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a pdf")
            f.flush()
            with pytest.raises(ValueError, match="Expected PDF"):
                parser.parse(f.name)


# ---------------------------------------------------------------------------
# extract_joint_info (public method)
# ---------------------------------------------------------------------------


class TestExtractJointInfo:
    """Tests for the extract_joint_info method that returns structured dicts."""

    def test_returns_dicts_with_required_keys(self, parser: ManualParser):
        """Each result should have name, type, and context keys."""
        text = "Joint 1 is the base joint. Motor 1 drives it."
        info = parser.extract_joint_info(text)
        assert len(info) >= 1
        for entry in info:
            assert "name" in entry
            assert "type" in entry
            assert "context" in entry

    def test_motor_type(self, parser: ManualParser):
        """Motor identifiers should have type 'motor'."""
        text = "Motor 1 is the primary driver."
        info = parser.extract_joint_info(text)
        motor_entries = [e for e in info if e["type"] == "motor"]
        assert len(motor_entries) >= 1

    def test_axis_type(self, parser: ManualParser):
        """Axis/DOF identifiers should have type 'axis'."""
        text = "axis_1 controls the rotation. dof_3 is the wrist."
        info = parser.extract_joint_info(text)
        axis_entries = [e for e in info if e["type"] == "axis"]
        assert len(axis_entries) >= 1
