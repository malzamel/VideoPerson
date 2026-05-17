from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
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

        self.counters_label = QLabel("Processed: 0 | Skipped: 0 | Failed: 0 | Bytes copied: 0")
        root.addWidget(self.counters_label)
        self.stats_label = QLabel("Candidate windows: 0 | Clear faces accepted: 0 | Rejected candidates: 0 | Detection: not started")
        root.addWidget(self.stats_label)
        previews_row = QHBoxLayout()
        previews_row.setSpacing(10)

        rejected_box = QFrame()
        rejected_box.setFrameShape(QFrame.Shape.StyledPanel)
        rejected_layout = QVBoxLayout(rejected_box)
        self.rejected_preview_title = QLabel("Rejected preview (left): -")
        self.rejected_preview_image = QLabel("No rejected preview yet.")
        self.rejected_preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rejected_preview_image.setMinimumHeight(220)
        self.rejected_preview_image.setStyleSheet("border: 1px solid #444;")
        rejected_layout.addWidget(self.rejected_preview_title)
        rejected_layout.addWidget(self.rejected_preview_image)

        accepted_box = QFrame()
        accepted_box.setFrameShape(QFrame.Shape.StyledPanel)
        accepted_layout = QVBoxLayout(accepted_box)
        self.accepted_preview_title = QLabel("Accepted preview (right): -")
        self.accepted_preview_image = QLabel("No accepted face yet.")
        self.accepted_preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accepted_preview_image.setMinimumHeight(220)
        self.accepted_preview_image.setStyleSheet("border: 1px solid #444;")
        accepted_layout.addWidget(self.accepted_preview_title)
        accepted_layout.addWidget(self.accepted_preview_image)

        previews_row.addWidget(rejected_box)
        previews_row.addWidget(accepted_box)
        root.addLayout(previews_row)

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
            f"Processed: {copied} | Skipped: {skipped} | Failed: {failed} | Bytes copied: {bytes_copied}"
        )

    def set_stage(self, stage_text: str) -> None:
        self.stage_label.setText(f"Stage: {stage_text}")

    def set_mode(self, mode_text: str) -> None:
        self.mode_label.setText(f"Mode: {mode_text}")

    def set_detection_stats(self, candidate_windows: int, accepted_faces: int, rejected_candidates: int, note: str) -> None:
        self.stats_label.setText(
            "Candidate windows: "
            f"{candidate_windows} | Clear faces accepted: {accepted_faces} | "
            f"Rejected candidates: {rejected_candidates} | Detection: {note}"
        )

    def _load_preview_pixmap(self, image_path: str) -> QPixmap | None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return None
        scaled = pixmap.scaled(
            420,
            240,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return scaled

    def set_accepted_preview(self, image_path: str, title: str) -> None:
        scaled = self._load_preview_pixmap(image_path)
        if scaled is None:
            self.accepted_preview_title.setText("Accepted preview (right): failed to load image")
            return
        self.accepted_preview_title.setText(f"Accepted preview (right): {title}")
        self.accepted_preview_image.setPixmap(scaled)

    def set_rejected_preview(self, image_path: str, title: str) -> None:
        scaled = self._load_preview_pixmap(image_path)
        if scaled is None:
            self.rejected_preview_title.setText("Rejected preview (left): failed to load image")
            return
        self.rejected_preview_title.setText(f"Rejected preview (left): {title}")
        self.rejected_preview_image.setPixmap(scaled)

    def set_rejected_note(self, note: str) -> None:
        self.rejected_preview_title.setText(f"Rejected preview (left): {note}")

    def append_log(self, text: str) -> None:
        self.log_view.append(text)
