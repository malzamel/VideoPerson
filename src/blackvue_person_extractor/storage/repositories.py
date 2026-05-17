from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from blackvue_person_extractor.core.blackvue_filename import parse_blackvue_filename


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_case(conn: sqlite3.Connection, case_name: str, archive_path: Path, notes: str | None = None) -> int:
    cursor = conn.execute(
        """
        INSERT INTO cases(case_name, archive_path, created_at, notes)
        VALUES (?, ?, ?, ?)
        """,
        (case_name, str(archive_path), utc_now_iso(), notes),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_video_file_pending(
    conn: sqlite3.Connection,
    case_id: int,
    original_path: Path,
    size_bytes: int,
) -> int:
    parsed = parse_blackvue_filename(original_path.name)
    cursor = conn.execute(
        """
        INSERT INTO video_files(
            case_id, original_path, filename, size_bytes,
            start_datetime, recording_type_code, recording_type_label,
            camera_direction_code, camera_direction_label, import_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            case_id,
            str(original_path),
            original_path.name,
            size_bytes,
            parsed.start_datetime.isoformat() if parsed.start_datetime else None,
            parsed.recording_type_code,
            parsed.recording_type_label,
            parsed.camera_direction_code,
            parsed.camera_direction_label,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def mark_video_imported(conn: sqlite3.Connection, video_id: int, archive_path: Path, sha256: str | None) -> None:
    conn.execute(
        """
        UPDATE video_files
        SET archive_path = ?, sha256 = ?, imported_at = ?, import_status = 'imported', error_message = NULL
        WHERE id = ?
        """,
        (str(archive_path), sha256, utc_now_iso(), video_id),
    )
    conn.commit()


def mark_video_skipped(conn: sqlite3.Connection, video_id: int, archive_path: Path) -> None:
    conn.execute(
        """
        UPDATE video_files
        SET archive_path = ?, imported_at = ?, import_status = 'skipped', error_message = NULL
        WHERE id = ?
        """,
        (str(archive_path), utc_now_iso(), video_id),
    )
    conn.commit()


def mark_video_failed(conn: sqlite3.Connection, video_id: int, error_message: str) -> None:
    conn.execute(
        """
        UPDATE video_files
        SET import_status = 'failed', error_message = ?
        WHERE id = ?
        """,
        (error_message[:1000], video_id),
    )
    conn.commit()
