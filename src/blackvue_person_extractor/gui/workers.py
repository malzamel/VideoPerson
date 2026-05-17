from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from blackvue_person_extractor.config import ImportOptions
from blackvue_person_extractor.core.importer import ImportController, import_files
from blackvue_person_extractor.core.sd_scanner import ScanSummary, scan_source_for_videos
from blackvue_person_extractor.storage.db import create_connection, initialize_database
from blackvue_person_extractor.storage.repositories import create_case


class ScanWorker(QObject):
    started = Signal(str)
    progress = Signal(int, int, "qint64")
    activity = Signal(str, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, source_path: str) -> None:
        super().__init__()
        self.source_path = source_path

    @Slot()
    def run(self) -> None:
        try:
            self.started.emit(self.source_path)
            summary = scan_source_for_videos(
                self.source_path,
                on_progress=lambda entries, mp4_files, total_size: self.progress.emit(entries, mp4_files, total_size),
                on_activity=lambda kind, path: self.activity.emit(kind, path),
            )
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


@dataclass(slots=True)
class ImportRequest:
    source_path: Path
    archive_path: Path
    case_name: str
    options: ImportOptions
    scan_summary: ScanSummary


class ImportWorker(QObject):
    file_started = Signal(int, int, str, "qint64")
    file_progress = Signal(int, int, str, "qint64", "qint64", "qint64")
    file_done = Signal(int, str, str)
    error = Signal(str, str)
    counters = Signal(int, int, int, "qint64", int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, request: ImportRequest) -> None:
        super().__init__()
        self.request = request
        self.controller = ImportController()

    @Slot()
    def run(self) -> None:
        try:
            case_root = self.request.archive_path / self.request.case_name
            original_dir = case_root / "original"
            db_path = case_root / "db" / "blackvue_person_extractor.sqlite"
            (case_root / "logs").mkdir(parents=True, exist_ok=True)
            (case_root / "output").mkdir(parents=True, exist_ok=True)
            if not self.request.options.process_from_source:
                original_dir.mkdir(parents=True, exist_ok=True)

            conn = create_connection(db_path)
            initialize_database(conn)
            case_id = create_case(conn, self.request.case_name, case_root)

            stats = import_files(
                conn=conn,
                case_id=case_id,
                files=self.request.scan_summary.files,
                original_dir=original_dir,
                options=self.request.options,
                controller=self.controller,
                on_file_start=lambda i, t, n, s: self.file_started.emit(i, t, n, s),
                on_file_progress=lambda i, t, n, c, sz, bt: self.file_progress.emit(i, t, n, c, sz, bt),
                on_file_done=lambda i, n, status: self.file_done.emit(i, n, status),
                on_error=lambda n, e: self.error.emit(n, e),
            )
            self.counters.emit(
                stats.copied_files,
                stats.skipped_files,
                stats.failed_files,
                stats.bytes_copied,
                stats.persons_found,
            )
            self.finished.emit(str(case_root))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def pause(self) -> None:
        self.controller.pause()

    def resume(self) -> None:
        self.controller.resume()

    def cancel(self) -> None:
        self.controller.cancel()
