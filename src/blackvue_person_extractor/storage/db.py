from __future__ import annotations

import sqlite3
from pathlib import Path


def create_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY,
            case_name TEXT NOT NULL,
            archive_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS video_files (
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            original_path TEXT NOT NULL,
            archive_path TEXT,
            filename TEXT NOT NULL,
            start_datetime TEXT,
            end_datetime TEXT,
            recording_type_code TEXT,
            recording_type_label TEXT,
            camera_direction_code TEXT,
            camera_direction_label TEXT,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT,
            width INTEGER,
            height INTEGER,
            fps REAL,
            duration_seconds REAL,
            codec TEXT,
            imported_at TEXT,
            indexed_at TEXT,
            import_status TEXT NOT NULL DEFAULT 'pending',
            processing_status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_video_files_case_id ON video_files(case_id);
        CREATE INDEX IF NOT EXISTS idx_video_files_import_status ON video_files(import_status);
        CREATE INDEX IF NOT EXISTS idx_video_files_processing_status ON video_files(processing_status);
        """
    )
    conn.commit()
