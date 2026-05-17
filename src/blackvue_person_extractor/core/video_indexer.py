from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import sqlite3

from blackvue_person_extractor.core.ffmpeg_utils import ffprobe_json


@dataclass(slots=True)
class VideoMetadata:
    width: int | None
    height: int | None
    fps: float | None
    duration_seconds: float | None
    codec: str | None


def _fraction_to_float(raw: str | None) -> float | None:
    if not raw or "/" not in raw:
        return None
    num, den = raw.split("/", maxsplit=1)
    try:
        den_float = float(den)
        if den_float == 0:
            return None
        return float(num) / den_float
    except ValueError:
        return None


def read_metadata(video_path: Path) -> VideoMetadata:
    data = ffprobe_json(video_path)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = data.get("format", {})
    duration = None
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") is not None else None
    except (ValueError, TypeError):
        duration = None
    return VideoMetadata(
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        fps=_fraction_to_float(video_stream.get("avg_frame_rate")),
        duration_seconds=duration,
        codec=video_stream.get("codec_name"),
    )


def index_imported_video(conn: sqlite3.Connection, video_id: int, video_path: Path) -> None:
    md = read_metadata(video_path)
    row = conn.execute("SELECT start_datetime FROM video_files WHERE id = ?", (video_id,)).fetchone()
    end_datetime = None
    if row and row["start_datetime"] and md.duration_seconds:
        from datetime import datetime

        start_dt = datetime.fromisoformat(row["start_datetime"])
        end_datetime = (start_dt + timedelta(seconds=md.duration_seconds)).isoformat()

    conn.execute(
        """
        UPDATE video_files
        SET width=?, height=?, fps=?, duration_seconds=?, codec=?, end_datetime=?, indexed_at=datetime('now')
        WHERE id=?
        """,
        (md.width, md.height, md.fps, md.duration_seconds, md.codec, end_datetime, video_id),
    )
    conn.commit()
