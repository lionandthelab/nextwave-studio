"""Thread-safe session state management."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.models import LoopStatus, SimulationResult

logger = logging.getLogger(__name__)


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


class SessionManager:
    """Singleton session manager with async-safe access."""

    _instance: Optional[SessionManager] = None

    def __new__(cls) -> SessionManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
            cls._instance._file_meta = {}
            cls._instance._lock = asyncio.Lock()
        return cls._instance

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
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            return session

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


session_manager = SessionManager()
