from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ImportProgressPage(QWidget):
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    open_archive_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.stage_label = QLabel("Stage: idle")
        self.mode_label = QLabel("Mode: -")
        root.addWidget(self.stage_label)
        root.addWidget(self.mode_label)
        self.current_file_label = QLabel("Current file: -")
        root.addWidget(self.current_file_label)

        self.overall_progress = QProgressBar()
        self.current_file_progress = QProgressBar()
        root.addWidget(QLabel("Overall progress"))
        root.addWidget(self.overall_progress)
        root.addWidget(QLabel("Current file progress"))
        root.addWidget(self.current_file_progress)

        self.counters_label = QLabel("Copied: 0 | Skipped: 0 | Failed: 0 | Bytes copied: 0")
        root.addWidget(self.counters_label)
        self.stats_label = QLabel("Persons found: 0 | Detection: not started")
        root.addWidget(self.stats_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        root.addWidget(self.log_view)

        buttons = QHBoxLayout()
        pause_btn = QPushButton("Pause")
        resume_btn = QPushButton("Resume")
        cancel_btn = QPushButton("Cancel")
        open_archive_btn = QPushButton("Open Archive Folder")
        pause_btn.clicked.connect(self.pause_requested.emit)
        resume_btn.clicked.connect(self.resume_requested.emit)
        cancel_btn.clicked.connect(self.cancel_requested.emit)
        open_archive_btn.clicked.connect(self.open_archive_requested.emit)
        buttons.addWidget(pause_btn)
        buttons.addWidget(resume_btn)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(open_archive_btn)
        root.addLayout(buttons)

    def set_current_file(self, filename: str) -> None:
        self.current_file_label.setText(f"Current file: {filename}")

    def update_progress(self, file_index: int, total_files: int, copied_for_file: int, file_size: int) -> None:
        self.overall_progress.setMaximum(max(total_files, 1))
        self.overall_progress.setValue(file_index)
        self.current_file_progress.setMaximum(max(file_size, 1))
        self.current_file_progress.setValue(copied_for_file)

    def set_counters(self, copied: int, skipped: int, failed: int, bytes_copied: int) -> None:
        self.counters_label.setText(
            f"Copied: {copied} | Skipped: {skipped} | Failed: {failed} | Bytes copied: {bytes_copied}"
        )

    def set_stage(self, stage_text: str) -> None:
        self.stage_label.setText(f"Stage: {stage_text}")

    def set_mode(self, mode_text: str) -> None:
        self.mode_label.setText(f"Mode: {mode_text}")

    def set_detection_stats(self, persons_found: int, note: str) -> None:
        self.stats_label.setText(f"Persons found: {persons_found} | Detection: {note}")

    def append_log(self, text: str) -> None:
        self.log_view.append(text)
