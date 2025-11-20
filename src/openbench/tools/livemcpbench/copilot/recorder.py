import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

CallType = Literal["route", "execute"]

class Recorder:
    def __init__(self, db_path: Path, task_id: str, model: str):
        self.db_path = db_path
        self.task_id = task_id
        self.model = model
        self.session_id = None
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()
        self._start_session()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id      TEXT,
                model        TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER NOT NULL REFERENCES sessions(id),
                step_index   INTEGER NOT NULL,
                call_type    TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json  TEXT NOT NULL,
                UNIQUE(session_id, step_index)
            )
        """)
        self._conn.commit()

    def _start_session(self) -> None:
        cur = self._conn.execute(
            "INSERT INTO sessions (task_id, model) VALUES (?, ?)",
            (self.task_id, self.model),
        )
        self.session_id = cur.lastrowid
        self._conn.commit()

    def log_tool_call(
        self,
        step_index: int,
        call_type: CallType,
        request: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if self.session_id is None:
            raise RuntimeError("Session not started")
            
        self._conn.execute(
            "INSERT INTO tool_calls (session_id, step_index, call_type, "
            "request_json, result_json) VALUES (?, ?, ?, ?, ?)",
            (
                self.session_id,
                step_index,
                call_type,
                json.dumps(request),
                json.dumps(result),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
