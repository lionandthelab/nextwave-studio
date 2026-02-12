"""PDF manual parser for extracting robot specifications and control information."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ManualData:
    """Structured data extracted from a robot manual PDF."""

    raw_text: str
    chunks: list[str]
    joint_names: list[str]
    control_functions: list[str]
    motor_specs: dict[str, dict]


class ManualParser:
    """Parses robot manual PDFs and extracts structured information."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, file_path: str) -> ManualData:
        """Parse a PDF manual and return structured data.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ManualData with extracted text, chunks, and robot specifications.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Manual PDF not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {path.suffix}")

        raw_text = self._extract_text(file_path)
        chunks = self._split_into_chunks(raw_text)
        joint_names = self._extract_joint_names(raw_text)
        control_functions = self._extract_control_functions(raw_text)
        motor_specs = self._extract_motor_specs(raw_text)

        logger.info(
            "Parsed manual: %d chars, %d chunks, %d joints, %d functions, %d motor specs",
            len(raw_text),
            len(chunks),
            len(joint_names),
            len(control_functions),
            len(motor_specs),
        )

        return ManualData(
            raw_text=raw_text,
            chunks=chunks,
            joint_names=joint_names,
            control_functions=control_functions,
            motor_specs=motor_specs,
        )

    def _extract_text(self, file_path: str) -> str:
        """Extract all text from a PDF file using PyMuPDF."""
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks for embedding.

        Args:
            text: The full text to split.

        Returns:
            List of text chunks with specified size and overlap.
        """
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for the last sentence-ending punctuation in the chunk
                last_break = max(
                    chunk.rfind(". "),
                    chunk.rfind(".\n"),
                    chunk.rfind("\n\n"),
                )
                if last_break > self.chunk_size // 2:
                    end = start + last_break + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def _extract_joint_names(self, text: str) -> list[str]:
        """Extract joint names from the manual text."""
        joint_info = self.extract_joint_info(text)
        return list({entry["name"] for entry in joint_info})

    def extract_joint_info(self, text: str) -> list[dict]:
        """Extract joint and motor naming patterns from text using regex.

        Args:
            text: The manual text to search.

        Returns:
            List of dicts with keys: name, type, context.
        """
        results = []
        seen = set()

        # Pattern: "Joint N" or "joint_N" or "J1", "J2", etc.
        for m in re.finditer(
            r"(?i)\b(joint[\s_]?\d+|j\d+)\b", text
        ):
            name = m.group(1).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                context = text[max(0, m.start() - 50) : m.end() + 50]
                results.append({"name": name, "type": "joint", "context": context})

        # Pattern: named joints like "left_finger", "right_gripper", "wrist_roll"
        for m in re.finditer(
            r"(?i)\b((?:left|right|upper|lower)[\s_]?"
            r"(?:finger|gripper|thumb|wrist|elbow|shoulder|hip|knee|ankle)"
            r"(?:[\s_]?(?:roll|pitch|yaw|flex|ext))?)\b",
            text,
        ):
            name = m.group(1).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                context = text[max(0, m.start() - 50) : m.end() + 50]
                results.append({"name": name, "type": "named_joint", "context": context})

        # Pattern: motor identifiers like "Motor 1", "M1", "motor_id: 3"
        for m in re.finditer(
            r"(?i)\b(motor[\s_]?\d+|m\d+)\b", text
        ):
            name = m.group(1).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                context = text[max(0, m.start() - 50) : m.end() + 50]
                results.append({"name": name, "type": "motor", "context": context})

        # Pattern: DOF/axis names like "axis_1", "dof_3"
        for m in re.finditer(
            r"(?i)\b((?:axis|dof)[\s_]?\d+)\b", text
        ):
            name = m.group(1).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                context = text[max(0, m.start() - 50) : m.end() + 50]
                results.append({"name": name, "type": "axis", "context": context})

        return results

    def _extract_control_functions(self, text: str) -> list[str]:
        """Extract control function names and API calls from the manual."""
        functions = set()

        # Pattern: function calls like set_position(...), move_joint(...)
        for m in re.finditer(
            r"\b([a-z_][a-z0-9_]*)\s*\(", text
        ):
            func_name = m.group(1)
            # Filter to likely robot control functions
            control_keywords = {
                "set", "get", "move", "rotate", "grip", "grasp", "open", "close",
                "position", "velocity", "torque", "force", "enable", "disable",
                "home", "calibrate", "init", "reset", "stop", "start", "plan",
                "execute", "control", "servo", "motor", "joint", "actuator",
            }
            if any(kw in func_name.lower() for kw in control_keywords):
                functions.add(func_name)

        # Pattern: method calls like robot.set_joint_positions(...)
        for m in re.finditer(
            r"\b\w+\.([a-z_][a-z0-9_]*)\s*\(", text
        ):
            func_name = m.group(1)
            control_keywords = {
                "set", "get", "move", "rotate", "grip", "grasp", "open", "close",
                "position", "velocity", "torque", "force", "enable", "disable",
            }
            if any(kw in func_name.lower() for kw in control_keywords):
                functions.add(func_name)

        return sorted(functions)

    def _extract_motor_specs(self, text: str) -> dict[str, dict]:
        """Extract motor specifications like torque limits, speed, range."""
        specs: dict[str, dict] = {}

        # Extract torque values: "max torque: 5.0 Nm" or "torque limit: 10 N*m"
        for m in re.finditer(
            r"(?i)(?:max(?:imum)?[\s_])?torque[\s:]+(\d+\.?\d*)\s*(?:Nm|N\*m|N\.m)",
            text,
        ):
            specs.setdefault("torque", {})
            specs["torque"]["max_nm"] = float(m.group(1))

        # Extract speed: "max speed: 180 deg/s" or "velocity: 3.14 rad/s"
        for m in re.finditer(
            r"(?i)(?:max(?:imum)?[\s_])?(?:speed|velocity)[\s:]+(\d+\.?\d*)\s*(?:deg/s|rad/s|rpm)",
            text,
        ):
            unit = m.group(0).split()[-1]
            specs.setdefault("speed", {})
            specs["speed"]["max_value"] = float(m.group(1))
            specs["speed"]["unit"] = unit

        # Extract joint range: "range: -180 to 180" or "limits: [-3.14, 3.14]"
        for m in re.finditer(
            r"(?i)(?:range|limits?)[\s:]+\[?\s*(-?\d+\.?\d*)\s*(?:,|to)\s*(-?\d+\.?\d*)\s*\]?",
            text,
        ):
            specs.setdefault("joint_range", {})
            specs["joint_range"]["min"] = float(m.group(1))
            specs["joint_range"]["max"] = float(m.group(2))

        # Extract payload/weight capacity
        for m in re.finditer(
            r"(?i)(?:payload|load[\s_]capacity|max[\s_]weight)[\s:]+(\d+\.?\d*)\s*(?:kg|g|lb)",
            text,
        ):
            unit = m.group(0).split()[-1].lower()
            value = float(m.group(1))
            if unit == "g":
                value /= 1000.0
                unit = "kg"
            elif unit == "lb":
                value *= 0.453592
                unit = "kg"
            specs.setdefault("payload", {})
            specs["payload"]["max_kg"] = value

        # Extract grip force
        for m in re.finditer(
            r"(?i)(?:grip|grasp)[\s_]?force[\s:]+(\d+\.?\d*)\s*(?:N|kN)",
            text,
        ):
            unit = m.group(0).split()[-1]
            value = float(m.group(1))
            if unit == "kN":
                value *= 1000.0
            specs.setdefault("grip_force", {})
            specs["grip_force"]["max_n"] = value

        return specs
