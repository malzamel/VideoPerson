from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

from .blackvue_filename import parse_blackvue_filename


@dataclass(slots=True)
class ScannedVideoFile:
    path: Path
    size_bytes: int
    is_blackvue: bool
    recording_type_code: str | None
    camera_direction_code: str | None


@dataclass(slots=True)
class ScanSummary:
    source_path: Path
    total_mp4_files: int
    blackvue_files: int
    front_files: int
    rear_files: int
    total_size_bytes: int
    files: list[ScannedVideoFile]


def scan_source_for_videos(
    source_path: str | Path,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_activity: Callable[[str, str], None] | None = None,
    progress_step: int = 100,
) -> ScanSummary:
    source = Path(source_path)
    files: list[ScannedVideoFile] = []
    entries_scanned = 0
    total_size = 0

    def _walk_error(exc: OSError) -> None:
        if on_activity:
            on_activity("error", str(exc))

    if on_activity:
        on_activity("source", str(source))

    for root, _, filenames in os.walk(source, onerror=_walk_error):
        root_path = Path(root)
        if on_activity:
            on_activity("dir", str(root_path))
        for filename in filenames:
            entries_scanned += 1
            full_path = root_path / filename
            if on_progress and entries_scanned % progress_step == 0:
                on_progress(entries_scanned, len(files), total_size)
                if on_activity:
                    on_activity("file", str(full_path))
            if full_path.suffix.lower() != ".mp4":
                continue
            stat = full_path.stat()
            total_size += stat.st_size
            parsed = parse_blackvue_filename(full_path.name)
            files.append(
                ScannedVideoFile(
                    path=full_path,
                    size_bytes=stat.st_size,
                    is_blackvue=parsed.is_valid_blackvue_name,
                    recording_type_code=parsed.recording_type_code,
                    camera_direction_code=parsed.camera_direction_code,
                )
            )
            if on_activity:
                on_activity("match", str(full_path))
            if on_progress:
                on_progress(entries_scanned, len(files), total_size)

    front_count = sum(1 for item in files if item.camera_direction_code == "F")
    rear_count = sum(1 for item in files if item.camera_direction_code == "R")
    blackvue_count = sum(1 for item in files if item.is_blackvue)

    if on_progress:
        on_progress(entries_scanned, len(files), total_size)

    return ScanSummary(
        source_path=source,
        total_mp4_files=len(files),
        blackvue_files=blackvue_count,
        front_files=front_count,
        rear_files=rear_count,
        total_size_bytes=total_size,
        files=sorted(files, key=lambda x: x.path.name),
    )
