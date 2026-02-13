"""Tests for security hotfixes (C1-C4, H3, H6).

Covers: CORS config, filename sanitization, content-type validation,
session ID format validation, and path traversal prevention.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# C1 - CORS configuration
# ---------------------------------------------------------------------------


class TestCORSConfiguration:
    """Verify CORS is not a wildcard and uses settings.cors_origins."""

    async def test_cors_not_wildcard(self, client: AsyncClient):
        """Preflight request should NOT return Access-Control-Allow-Origin: *."""
        resp = await client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # An unknown origin should not be reflected back
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert allow_origin != "*"
        assert "evil.example.com" not in allow_origin


# ---------------------------------------------------------------------------
# C2 - Filename sanitization (path traversal)
# ---------------------------------------------------------------------------


class TestFilenameSanitization:
    """Verify path traversal attempts in filenames are neutralized."""

    async def test_path_traversal_filename_cad(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """A filename with directory traversal should be sanitized."""
        resp = await client.post(
            "/api/v1/upload/cad",
            files={
                "file": (
                    "../../etc/passwd.stl",
                    sample_stl_bytes,
                    "application/octet-stream",
                )
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Filename should be sanitized to just the base name
        assert "/" not in data["filename"]
        assert ".." not in data["filename"]
        assert data["filename"] == "passwd.stl"

    async def test_path_traversal_filename_manual(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ):
        """A manual filename with directory traversal should be sanitized."""
        resp = await client.post(
            "/api/v1/upload/manual",
            files={
                "file": (
                    "../../../tmp/evil.pdf",
                    sample_pdf_bytes,
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "/" not in data["filename"]
        assert data["filename"] == "evil.pdf"


# ---------------------------------------------------------------------------
# C3 - Content-type validation
# ---------------------------------------------------------------------------


class TestContentTypeValidation:
    """Verify MIME type checks reject invalid content types."""

    async def test_cad_wrong_content_type(self, client: AsyncClient):
        """Uploading a CAD file with text/html content type should be rejected."""
        resp = await client.post(
            "/api/v1/upload/cad",
            files={"file": ("model.stl", b"\x00" * 100, "text/html")},
        )
        assert resp.status_code == 400
        assert "content type" in resp.json()["detail"].lower()

    async def test_manual_wrong_content_type(self, client: AsyncClient):
        """Uploading a manual with image/png content type should be rejected."""
        resp = await client.post(
            "/api/v1/upload/manual",
            files={"file": ("manual.pdf", b"%PDF-fake", "image/png")},
        )
        assert resp.status_code == 400
        assert "content type" in resp.json()["detail"].lower()

    async def test_cad_valid_content_type(
        self, client: AsyncClient, sample_stl_bytes: bytes
    ):
        """Uploading a CAD file with application/octet-stream should succeed."""
        resp = await client.post(
            "/api/v1/upload/cad",
            files={
                "file": ("cube.stl", sample_stl_bytes, "application/octet-stream")
            },
        )
        assert resp.status_code == 200

    async def test_manual_valid_content_type(
        self, client: AsyncClient, sample_pdf_bytes: bytes
    ):
        """Uploading a PDF with application/pdf should succeed."""
        resp = await client.post(
            "/api/v1/upload/manual",
            files={
                "file": ("manual.pdf", sample_pdf_bytes, "application/pdf")
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# H6 - Session ID format validation
# ---------------------------------------------------------------------------


class TestSessionIDValidation:
    """Verify session_id path parameters are validated as hex UUIDs."""

    async def test_generate_status_invalid_session_id(self, client: AsyncClient):
        """Non-hex session ID should return 400 on generate status endpoint."""
        resp = await client.get("/api/v1/generate/status/not-a-valid-session-id!!")
        assert resp.status_code == 400
        assert "Invalid session ID" in resp.json()["detail"]

    async def test_generate_stop_invalid_session_id(self, client: AsyncClient):
        """Non-hex session ID should return 400 on generate stop endpoint."""
        resp = await client.post("/api/v1/generate/stop/not-a-valid-uuid!")
        assert resp.status_code == 400
        assert "Invalid session ID" in resp.json()["detail"]

    async def test_generate_code_invalid_session_id(self, client: AsyncClient):
        """Non-hex session ID should return 400 on generate code endpoint."""
        resp = await client.get("/api/v1/generate/code/DROP TABLE sessions")
        assert resp.status_code == 400

    async def test_monitor_stream_invalid_session_id(self, client: AsyncClient):
        """Non-hex session ID should return 400 on monitor stream endpoint."""
        resp = await client.get("/api/v1/monitor/stream/not-a-valid-session-id!!")
        assert resp.status_code == 400

    async def test_monitor_logs_invalid_session_id(self, client: AsyncClient):
        """Non-hex session ID should return 400 on monitor logs endpoint."""
        resp = await client.get("/api/v1/monitor/logs/invalid!")
        assert resp.status_code == 400

    async def test_monitor_gif_invalid_session_id(self, client: AsyncClient):
        """Non-hex session ID should return 400 on monitor GIF endpoint."""
        resp = await client.get(
            "/api/v1/monitor/result/../../etc/passwd/gif"
        )
        # The path pattern may not match the route, which gives 404 or 400
        assert resp.status_code in (400, 404)

    async def test_valid_hex_session_id_passes_validation(
        self, client: AsyncClient
    ):
        """A valid 32-char hex string should pass validation (then 404 for missing session)."""
        valid_id = "a" * 32
        resp = await client.get(f"/api/v1/generate/status/{valid_id}")
        # Should pass validation but session doesn't exist -> 404
        assert resp.status_code == 404
