"""Shared pytest fixtures for AutoGrip-Sim Engine tests."""

from __future__ import annotations

import asyncio
import os
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings

# Make sim_server importable from docker/sim_scripts/
_sim_scripts_dir = str(Path(__file__).resolve().parent.parent / "docker" / "sim_scripts")
if _sim_scripts_dir not in sys.path:
    sys.path.insert(0, _sim_scripts_dir)


# ---------------------------------------------------------------------------
# Helpers to generate minimal valid binary files
# ---------------------------------------------------------------------------


def _make_stl_bytes() -> bytes:
    """Generate a minimal valid binary STL file (one triangle)."""
    header = b"\x00" * 80  # 80-byte header
    num_triangles = struct.pack("<I", 1)  # 1 triangle

    # One triangle: normal (3 floats) + 3 vertices (9 floats) + attribute (2 bytes)
    normal = struct.pack("<fff", 0.0, 0.0, 1.0)
    v1 = struct.pack("<fff", 0.0, 0.0, 0.0)
    v2 = struct.pack("<fff", 1.0, 0.0, 0.0)
    v3 = struct.pack("<fff", 0.0, 1.0, 0.0)
    attribute = struct.pack("<H", 0)

    return header + num_triangles + normal + v1 + v2 + v3 + attribute


def _make_pdf_bytes() -> bytes:
    """Generate a minimal valid PDF file.

    Creates a single-page PDF with the text 'Joint 1 set_position torque: 5.0 Nm'.
    This includes keywords useful for parser testing.
    """
    # Minimal valid PDF structure
    content = (
        "Joint 1 specification\n"
        "J2 motor controller\n"
        "set_position(angle)\n"
        "move_joint(id, pos)\n"
        "max torque: 5.0 Nm\n"
        "max speed: 180 deg/s\n"
        "range: -180 to 180\n"
        "left_finger grip\n"
        "right_gripper control\n"
    )

    # Build a valid PDF with the above text in a stream
    stream = f"BT /F1 12 Tf 100 700 Td ({content}) Tj ET"
    stream_bytes = stream.encode("latin-1")

    objects = []
    # Object 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2: Pages
    objects.append(
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    )
    # Object 3: Page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    # Object 4: Stream (page content)
    objects.append(
        b"4 0 obj\n<< /Length "
        + str(len(stream_bytes)).encode()
        + b" >>\nstream\n"
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )
    # Object 5: Font
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    # Build PDF
    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj

    # Cross-reference table
    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += f"0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()

    # Trailer
    pdf += b"trailer\n"
    pdf += f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    pdf += b"startxref\n"
    pdf += f"{xref_offset}\n".encode()
    pdf += b"%%EOF\n"

    return pdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def sample_stl_bytes() -> bytes:
    """Return minimal valid binary STL file bytes."""
    return _make_stl_bytes()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    """Return minimal valid PDF file bytes."""
    return _make_pdf_bytes()


@pytest.fixture()
def tmp_upload_dir(tmp_path: Path):
    """Provide a temporary upload directory and patch settings to use it."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "cad").mkdir()
    (upload_dir / "manuals").mkdir()
    (upload_dir / "results").mkdir()
    return upload_dir


@pytest.fixture()
def mock_settings(tmp_upload_dir: Path):
    """Override app settings to use a temporary upload directory."""
    with patch("app.config.settings", Settings(upload_dir=str(tmp_upload_dir))):
        yield


@pytest.fixture(autouse=True)
def session_manager_reset():
    """Reset the singleton SessionManager state between tests."""
    from app.session_manager import SessionManager

    mgr = SessionManager()
    mgr._sessions = {}
    mgr._file_meta = {}
    # Re-create the lock in case the event loop changed
    mgr._lock = asyncio.Lock()
    yield
    mgr._sessions = {}
    mgr._file_meta = {}
    mgr._lock = asyncio.Lock()


@pytest_asyncio.fixture()
async def client(tmp_upload_dir: Path):
    """Provide an httpx.AsyncClient bound to the FastAPI app.

    Patches settings.upload_dir so that file uploads go to a temp directory.
    """
    with patch("app.config.settings", Settings(upload_dir=str(tmp_upload_dir))):
        # Re-import app after patching so lifespan picks up the temp dir
        from app.main import app

        # Ensure upload dirs exist (lifespan does this, but we patch before startup)
        for sub in ("cad", "manuals", "results"):
            (tmp_upload_dir / sub).mkdir(exist_ok=True)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ---------------------------------------------------------------------------
# Sim-server fixture (for IsaacSimConnector HTTP tests)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def sim_http_client():
    """Provide an httpx.AsyncClient connected to sim_server via ASGI transport.

    The sim_server runs in mock mode (no real Isaac Sim) and provides the
    same REST API that the Docker-based sim_server exposes.
    """
    from sim_server import app as sim_app, sim_manager

    transport = ASGITransport(app=sim_app)
    async with AsyncClient(transport=transport, base_url="http://sim-test") as ac:
        yield ac
    # Reset sim_server state after each test
    await sim_manager.reset()
