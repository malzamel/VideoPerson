from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3
import threading

from blackvue_person_extractor.config import ImportOptions
from blackvue_person_extractor.core.sd_scanner import ScannedVideoFile
from blackvue_person_extractor.storage import repositories
from blackvue_person_extractor.utils.hashing import sha256_file


@dataclass(slots=True)
class ImportStats:
    total_files: int = 0
    copied_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    bytes_copied: int = 0
    persons_found: int = 0


class ImportController:
    def __init__(self) -> None:
        self._paused = threading.Event()
        self._paused.clear()
        self._cancelled = threading.Event()
        self._cancelled.clear()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def wait_if_paused(self) -> None:
        while self._paused.is_set() and not self._cancelled.is_set():
            self._cancelled.wait(0.1)


def _resolve_conflict_path(path: Path) -> Path:
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}.conflict_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _copy_with_progress(
    source: Path,
    target: Path,
    chunk_size: int,
    controller: ImportController,
    on_chunk_copied,
) -> None:
    with source.open("rb") as src, target.open("wb") as dst:
        while True:
            controller.wait_if_paused()
            if controller.cancelled:
                raise RuntimeError("Import cancelled by user.")
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            on_chunk_copied(len(chunk))


def import_files(
    conn: sqlite3.Connection,
    case_id: int,
    files: list[ScannedVideoFile],
    original_dir: Path,
    options: ImportOptions,
    controller: ImportController,
    on_file_start,
    on_file_progress,
    on_file_done,
    on_error,
) -> ImportStats:
    stats = ImportStats(total_files=len(files))
    if not options.process_from_source:
        original_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(files, start=1):
        if controller.cancelled:
            break

        source = item.path
        target = original_dir / source.name
        file_id = repositories.insert_video_file_pending(conn, case_id=case_id, original_path=source, size_bytes=item.size_bytes)
        on_file_start(index, len(files), source.name, item.size_bytes)

        try:
            if options.process_from_source:
                file_hash = sha256_file(source) if options.calculate_sha256 else None
                on_file_progress(index, len(files), source.name, item.size_bytes, item.size_bytes, stats.bytes_copied)
                repositories.mark_video_imported(conn, file_id, source, file_hash)
            else:
                if target.exists():
                    target_size = target.stat().st_size
                    if options.skip_already_copied and (not options.verify_by_size or target_size == item.size_bytes):
                        repositories.mark_video_skipped(conn, file_id, target)
                        stats.skipped_files += 1
                        on_file_done(index, source.name, "skipped")
                        continue
                    target = _resolve_conflict_path(target)

                copied_for_file = 0

                def chunk_cb(bytes_written: int) -> None:
                    nonlocal copied_for_file
                    copied_for_file += bytes_written
                    stats.bytes_copied += bytes_written
                    on_file_progress(index, len(files), source.name, copied_for_file, item.size_bytes, stats.bytes_copied)

                _copy_with_progress(source, target, options.chunk_size, controller, chunk_cb)

                if options.verify_by_size and target.stat().st_size != item.size_bytes:
                    raise IOError("Size verification failed after copy.")

                file_hash = sha256_file(target) if options.calculate_sha256 else None
                repositories.mark_video_imported(conn, file_id, target, file_hash)
            stats.copied_files += 1
            on_file_done(index, source.name, "copied")
        except Exception as exc:  # noqa: BLE001
            if target.exists() and target.stat().st_size == 0:
                target.unlink(missing_ok=True)
            repositories.mark_video_failed(conn, file_id, str(exc))
            stats.failed_files += 1
            on_error(source.name, str(exc))
            on_file_done(index, source.name, "failed")

    return stats
