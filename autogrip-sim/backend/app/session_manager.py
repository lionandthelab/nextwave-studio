"""Thread-safe session state management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.models import LoopStatus, SimulationResult

logger = logging.getLogger(__name__)

# H2: Configurable limits to prevent unbounded memory growth
MAX_SESSIONS = 100
MAX_LOGS_PER_SESSION = 1000

# H5: Explicit whitelist of fields that update_session may modify
UPDATABLE_FIELDS: frozenset[str] = frozenset({
    "status",
    "current_iteration",
    "max_iterations",
    "success_threshold",
    "results",
    "generated_code",
    "task",
})


class _SessionData:
    """Internal representation of a single session."""

    def __init__(
        self,
        session_id: str,
        cad_file_id: str,
        manual_file_id: Optional[str],
        robot_model: str,
    ) -> None:
        self.session_id = session_id
        self.cad_file_id = cad_file_id
        self.manual_file_id = manual_file_id
        self.robot_model = robot_model
        self.status = "created"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.current_iteration = 0
        self.max_iterations = 20
        self.success_threshold = 3
        self.results: list[SimulationResult] = []
        self.logs: list[dict[str, Any]] = []
        self.generated_code: Optional[str] = None
        self.task: Optional[asyncio.Task[None]] = None


# H4: Immutable snapshot returned outside the lock
@dataclass(frozen=True)
class SessionSnapshot:
    """Read-only snapshot of session state, safe to use outside the lock."""

    session_id: str
    cad_file_id: str
    manual_file_id: Optional[str]
    robot_model: str
    status: str
    created_at: str
    current_iteration: int
    max_iterations: int
    success_threshold: int
    generated_code: Optional[str]
    result_count: int
    log_count: int


class SessionManager:
    """Session manager with async-safe access.

    H1: Plain class -- no __new__ singleton. A module-level instance
    (``session_manager``) serves as the single shared instance.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionData] = {}
        self._file_meta: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # File metadata helpers
    # ------------------------------------------------------------------

    async def store_file_meta(self, file_id: str, meta: dict[str, Any]) -> None:
        async with self._lock:
            self._file_meta[file_id] = meta

    async def get_file_meta(self, file_id: str) -> Optional[dict[str, Any]]:
        async with self._lock:
            return self._file_meta.get(file_id)

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(
        self,
        cad_file_id: str,
        manual_file_id: Optional[str],
        robot_model: str,
    ) -> _SessionData:
        session_id = uuid4().hex
        session = _SessionData(
            session_id=session_id,
            cad_file_id=cad_file_id,
            manual_file_id=manual_file_id,
            robot_model=robot_model,
        )
        async with self._lock:
            # H2: Evict oldest sessions when limit is reached
            if len(self._sessions) >= MAX_SESSIONS:
                self._evict_oldest_session()
            self._sessions[session_id] = session
        logger.info("Session created: %s", session_id)
        return session

    async def get_session(self, session_id: str) -> Optional[_SessionData]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_session(
        self, session_id: str, **kwargs: Any
    ) -> Optional[_SessionData]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            # H5: Only allow whitelisted fields to be updated
            for key, value in kwargs.items():
                if key in UPDATABLE_FIELDS:
                    setattr(session, key, value)
                else:
                    logger.warning(
                        "Rejected update to non-updatable field %r on session %s",
                        key, session_id,
                    )
            return session

    # H4: Return an immutable snapshot of session state
    async def get_session_snapshot(self, session_id: str) -> Optional[SessionSnapshot]:
        """Return a frozen, read-only snapshot of the session.

        This is safe to use outside the lock -- it copies all needed scalar
        fields and counts for collections.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionSnapshot(
                session_id=session.session_id,
                cad_file_id=session.cad_file_id,
                manual_file_id=session.manual_file_id,
                robot_model=session.robot_model,
                status=session.status,
                created_at=session.created_at,
                current_iteration=session.current_iteration,
                max_iterations=session.max_iterations,
                success_threshold=session.success_threshold,
                generated_code=session.generated_code,
                result_count=len(session.results),
                log_count=len(session.logs),
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    async def add_log(
        self, session_id: str, level: str, message: str, data: Any = None
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message,
                "data": data,
            }
            session.logs.append(entry)
            # H2: Trim logs when they exceed the per-session limit
            if len(session.logs) > MAX_LOGS_PER_SESSION:
                trimmed = len(session.logs) - MAX_LOGS_PER_SESSION
                session.logs = session.logs[trimmed:]
                logger.debug(
                    "Trimmed %d oldest log entries for session %s",
                    trimmed, session_id,
                )

    async def get_logs(self, session_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return list(session.logs)

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    async def add_result(
        self, session_id: str, result: SimulationResult
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.results.append(result)
            session.current_iteration = result.iteration

    async def get_loop_status(self, session_id: str) -> Optional[LoopStatus]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return LoopStatus(
                session_id=session.session_id,
                current_iteration=session.current_iteration,
                max_iterations=session.max_iterations,
                status=session.status,
                results=list(session.results),
                final_code=session.generated_code,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest_session(self) -> None:
        """Remove the oldest non-running session to make room.

        Must be called while holding ``self._lock``.
        """
        # Prefer evicting completed/failed/stopped sessions first
        candidates = [
            (sid, s) for sid, s in self._sessions.items()
            if s.status in ("success", "failed", "stopped", "created")
        ]
        if not candidates:
            # Fall back to any session (including running) if all are active
            candidates = list(self._sessions.items())

        if not candidates:
            return

        # Sort by created_at and remove the oldest
        oldest_sid, _ = min(candidates, key=lambda x: x[1].created_at)
        del self._sessions[oldest_sid]
        logger.info(
            "Evicted oldest session %s (MAX_SESSIONS=%d reached)",
            oldest_sid, MAX_SESSIONS,
        )


# H1: Module-level instance replaces the singleton __new__ pattern
session_manager = SessionManager()
