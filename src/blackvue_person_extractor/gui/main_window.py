from __future__ import annotations

from pathlib import Path
import subprocess

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from blackvue_person_extractor.config import ImportOptions
from blackvue_person_extractor.core.sd_scanner import ScanSummary
from blackvue_person_extractor.gui.import_page import ImportPage
from blackvue_person_extractor.gui.import_progress_page import ImportProgressPage
from blackvue_person_extractor.gui.workers import ImportRequest, ImportWorker, ScanWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BlackVue Person Extractor")
        self.resize(1000, 700)

        self._scan_summary: ScanSummary | None = None
        self._scan_thread: QThread | None = None
        self._scan_worker: ScanWorker | None = None
        self._import_thread: QThread | None = None
        self._import_worker: ImportWorker | None = None
        self._last_case_path: Path | None = None
        self._copied = 0
        self._skipped = 0
        self._failed = 0
        self._bytes_copied = 0

        self.pages = QStackedWidget()
        self.import_page = ImportPage()
        self.import_progress_page = ImportProgressPage()
        self.pages.addWidget(self.import_page)
        self.pages.addWidget(self.import_progress_page)
        self.setCentralWidget(self.pages)

        self.import_page.scan_requested.connect(self._start_scan)
        self.import_page.import_requested.connect(self._start_import)
        self.import_progress_page.pause_requested.connect(self._pause_import)
        self.import_progress_page.resume_requested.connect(self._resume_import)
        self.import_progress_page.cancel_requested.connect(self._cancel_import)
        self.import_progress_page.open_archive_requested.connect(self._open_archive)

    def _start_scan(self, source_path: str) -> None:
        if self._scan_thread and self._scan_thread.isRunning():
            self.import_page.append_log("Scan is already running.")
            return
        self.import_page.append_log(f"Scanning {source_path} ...")
        self.import_page.set_scan_running(True)
        self._scan_thread = QThread(self)
        self._scan_worker = ScanWorker(source_path)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.started.connect(self._on_scan_started)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.activity.connect(self._on_scan_activity)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.failed.connect(self._scan_thread.quit)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_worker.failed.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._clear_scan_worker)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.start()

    def _clear_scan_worker(self) -> None:
        self._scan_worker = None
        self._scan_thread = None

    def _on_scan_finished(self, summary: ScanSummary) -> None:
        self._scan_summary = summary
        self.import_page.set_scan_running(False)
        self.import_page.set_scan_summary(summary)
        self.import_page.update_scan_activity("Scan completed.")
        self.import_page.append_log(
            "Scan finished. "
            f"MP4 found: {summary.total_mp4_files}, "
            f"BlackVue: {summary.blackvue_files}, "
            f"Front: {summary.front_files}, Rear: {summary.rear_files}, "
            f"Total size: {summary.total_size_bytes / (1024**3):.2f} GB"
        )

    def _on_scan_failed(self, error: str) -> None:
        self.import_page.set_scan_running(False)
        self.import_page.update_scan_activity("Scan failed.")
        QMessageBox.critical(self, "Scan failed", error)
        self.import_page.append_log(f"Scan failed: {error}")

    def _on_scan_started(self, source_path: str) -> None:
        self.import_page.append_log(f"Started scan: {source_path}")

    def _on_scan_progress(self, entries_scanned: int, mp4_files: int, total_size_bytes: int) -> None:
        self.import_page.update_scan_progress(entries_scanned, mp4_files, total_size_bytes)

    def _on_scan_activity(self, kind: str, path: str) -> None:
        if kind == "source":
            self.import_page.update_scan_activity(f"Source: {path}")
            self.import_page.append_log(f"Source root: {path}")
            return
        if kind == "dir":
            self.import_page.update_scan_activity(f"Scanning folder: {path}")
            return
        if kind == "file":
            self.import_page.update_scan_activity(f"Checking files in: {path}")
            return
        if kind == "match":
            self.import_page.update_scan_activity(f"Found video: {path}")
            self.import_page.append_log(f"Found video: {path}")
            return
        if kind == "error":
            self.import_page.update_scan_activity(f"Scan warning: {path}")
            self.import_page.append_log(f"Scan warning: {path}")

    def _start_import(
        self,
        source: str,
        archive: str,
        case_name: str,
        verify_size: bool,
        calculate_sha: bool,
        skip_already_copied: bool,
        process_from_source: bool,
    ) -> None:
        if self._scan_summary is None:
            QMessageBox.warning(self, "Missing scan", "Please scan source before import.")
            return
        if self._scan_summary.source_path != Path(source):
            QMessageBox.warning(self, "Source changed", "Please scan again after changing source path.")
            return
        if process_from_source:
            source_path = Path(source).resolve()
            archive_path = Path(archive).resolve()
            if archive_path == source_path or source_path in archive_path.parents:
                QMessageBox.warning(
                    self,
                    "Invalid archive location",
                    "Choose a local archive folder outside the SD/source path so the card remains untouched.",
                )
                return

        self._copied = 0
        self._skipped = 0
        self._failed = 0
        self._bytes_copied = 0
        self.import_progress_page.set_stage("Preparing")
        self.import_progress_page.set_mode(
            "Direct from source (SD card read-only)" if process_from_source else "Copy to local archive"
        )
        self.import_progress_page.set_detection_stats(0, "not started")
        self.import_progress_page.set_counters(0, 0, 0, 0)
        self.pages.setCurrentWidget(self.import_progress_page)

        options = ImportOptions(
            verify_by_size=verify_size and not process_from_source,
            calculate_sha256=calculate_sha,
            skip_already_copied=skip_already_copied and not process_from_source,
            process_from_source=process_from_source,
        )
        request = ImportRequest(
            source_path=Path(source),
            archive_path=Path(archive),
            case_name=case_name,
            options=options,
            scan_summary=self._scan_summary,
        )
        self._import_worker = ImportWorker(request)
        self._import_thread = QThread(self)
        self._import_worker.moveToThread(self._import_thread)
        self._import_thread.started.connect(self._import_worker.run)

        self._import_worker.file_started.connect(self._on_import_file_started)
        self._import_worker.file_progress.connect(self._on_import_file_progress)
        self._import_worker.file_done.connect(self._on_import_file_done)
        self._import_worker.error.connect(self._on_import_error)
        self._import_worker.counters.connect(self._on_import_counters)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_worker.failed.connect(self._on_import_failed)
        self._import_worker.finished.connect(self._import_thread.quit)
        self._import_worker.failed.connect(self._import_thread.quit)
        self._import_worker.finished.connect(self._import_worker.deleteLater)
        self._import_worker.failed.connect(self._import_worker.deleteLater)
        self._import_thread.finished.connect(self._import_thread.deleteLater)
        self._import_thread.start()

    def _on_import_file_started(self, _index: int, _total: int, filename: str, _size: int) -> None:
        self.import_progress_page.set_stage("Processing files")
        self.import_progress_page.set_current_file(filename)
        self.import_progress_page.append_log(f"Processing: {filename}")

    def _on_import_file_progress(
        self,
        file_index: int,
        total_files: int,
        _filename: str,
        copied_for_file: int,
        file_size: int,
        bytes_total: int,
    ) -> None:
        self._bytes_copied = bytes_total
        self.import_progress_page.update_progress(file_index, total_files, copied_for_file, file_size)
        self.import_progress_page.set_counters(self._copied, self._skipped, self._failed, self._bytes_copied)

    def _on_import_file_done(self, _index: int, filename: str, status: str) -> None:
        if status == "copied":
            self._copied += 1
        elif status == "skipped":
            self._skipped += 1
        else:
            self._failed += 1
        self.import_progress_page.set_counters(self._copied, self._skipped, self._failed, self._bytes_copied)
        self.import_progress_page.append_log(f"{status.upper()}: {filename}")

    def _on_import_error(self, filename: str, error: str) -> None:
        self.import_progress_page.append_log(f"ERROR [{filename}]: {error}")

    def _on_import_counters(
        self,
        copied: int,
        skipped: int,
        failed: int,
        bytes_copied: int,
        persons_found: int,
    ) -> None:
        self._copied = copied
        self._skipped = skipped
        self._failed = failed
        self._bytes_copied = bytes_copied
        self.import_progress_page.set_counters(copied, skipped, failed, bytes_copied)
        self.import_progress_page.set_detection_stats(persons_found, "pending (person detection not implemented yet)")

    def _on_import_finished(self, case_path: str) -> None:
        self._last_case_path = Path(case_path)
        self.import_progress_page.set_stage("Completed")
        self.import_progress_page.set_detection_stats(0, "pending (person detection not implemented yet)")
        self.import_progress_page.append_log(f"Import finished. Case path: {case_path}")
        self.import_progress_page.append_log("All DB/log/output artifacts were stored locally on this PC.")
        QMessageBox.information(self, "Import finished", f"Import completed.\n{case_path}")

    def _on_import_failed(self, error: str) -> None:
        self.import_progress_page.set_stage("Failed")
        self.import_progress_page.append_log(f"Import failed: {error}")
        QMessageBox.critical(self, "Import failed", error)

    def _pause_import(self) -> None:
        if self._import_worker:
            self._import_worker.pause()
            self.import_progress_page.append_log("Paused.")

    def _resume_import(self) -> None:
        if self._import_worker:
            self._import_worker.resume()
            self.import_progress_page.append_log("Resumed.")

    def _cancel_import(self) -> None:
        if self._import_worker:
            self._import_worker.cancel()
            self.import_progress_page.append_log("Cancel requested.")

    def _open_archive(self) -> None:
        if not self._last_case_path:
            QMessageBox.information(self, "No archive yet", "Run import first.")
            return
        subprocess.Popen(["explorer", str(self._last_case_path)])
