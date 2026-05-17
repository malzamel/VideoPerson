from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from blackvue_person_extractor.core.sd_scanner import ScanSummary


class ImportPage(QWidget):
    scan_requested = Signal(str)
    import_requested = Signal(str, str, str, bool, bool, bool, bool)

    def __init__(self) -> None:
        super().__init__()
        self._last_summary: ScanSummary | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.source_path_input = QLineEdit()
        source_btn = QPushButton("Select SD Card / Source Folder")
        source_btn.clicked.connect(self._select_source)
        source_layout = QHBoxLayout()
        source_layout.addWidget(self.source_path_input)
        source_layout.addWidget(source_btn)
        source_widget = QWidget()
        source_widget.setLayout(source_layout)
        form.addRow("Source:", source_widget)

        self.archive_path_input = QLineEdit()
        archive_btn = QPushButton("Select Archive Folder")
        archive_btn.clicked.connect(self._select_archive)
        archive_layout = QHBoxLayout()
        archive_layout.addWidget(self.archive_path_input)
        archive_layout.addWidget(archive_btn)
        archive_widget = QWidget()
        archive_widget.setLayout(archive_layout)
        form.addRow("Archive:", archive_widget)

        self.case_name_input = QLineEdit()
        self.case_name_input.setPlaceholderText("2026-05-17")
        self.case_name_input.setText(date.today().isoformat())
        form.addRow("Case Name:", self.case_name_input)
        root.addLayout(form)

        options_box = QGroupBox("Import Options")
        options_layout = QVBoxLayout(options_box)
        self.verify_size_checkbox = QCheckBox("Verify copied files by size")
        self.verify_size_checkbox.setChecked(True)
        self.sha_checkbox = QCheckBox("Calculate SHA256 hash after copy")
        self.skip_checkbox = QCheckBox("Skip files already copied")
        self.skip_checkbox.setChecked(True)
        self.process_from_source_checkbox = QCheckBox("Process directly from source (no copy)")
        self.direct_mode_note = QLabel(
            "Direct mode keeps SD card files untouched and stores DB/logs/results in the selected local archive folder."
        )
        self.direct_mode_note.setWordWrap(True)
        self.process_from_source_checkbox.setChecked(True)
        self.process_from_source_checkbox.toggled.connect(self._on_process_from_source_toggled)
        options_layout.addWidget(self.verify_size_checkbox)
        options_layout.addWidget(self.sha_checkbox)
        options_layout.addWidget(self.skip_checkbox)
        options_layout.addWidget(self.process_from_source_checkbox)
        options_layout.addWidget(self.direct_mode_note)
        self._on_process_from_source_toggled(self.process_from_source_checkbox.isChecked())
        root.addWidget(options_box)

        button_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Scan Source")
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        self.import_btn = QPushButton("Import to Computer")
        self.import_btn.clicked.connect(self._on_import_clicked)
        self.import_btn.setEnabled(False)
        button_layout.addWidget(self.scan_btn)
        button_layout.addWidget(self.import_btn)
        root.addLayout(button_layout)

        self.summary_label = QLabel("No scan run yet.")
        self.scan_activity_label = QLabel("Activity: -")
        self.scan_activity_label.setWordWrap(True)
        self.scan_busy_bar = QProgressBar()
        self.scan_busy_bar.setRange(0, 1)
        self.scan_busy_bar.setValue(0)
        self.scan_busy_bar.hide()
        root.addWidget(self.summary_label)
        root.addWidget(self.scan_activity_label)
        root.addWidget(self.scan_busy_bar)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setPlaceholderText("Scan and import status will appear here.")
        root.addWidget(self.error_log)

    def _select_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.source_path_input.setText(folder)

    def _select_archive(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Archive Folder")
        if folder:
            self.archive_path_input.setText(folder)

    def _on_scan_clicked(self) -> None:
        source = self.source_path_input.text().strip()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Invalid source", "Please select a valid source folder.")
            return
        self.scan_requested.emit(source)

    def _on_import_clicked(self) -> None:
        source = self.source_path_input.text().strip()
        archive = self.archive_path_input.text().strip()
        case_name = self.case_name_input.text().strip()
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Invalid source", "Please select a valid source folder.")
            return
        if not archive:
            QMessageBox.warning(self, "Missing archive", "Please choose an archive folder.")
            return
        if not case_name:
            QMessageBox.warning(self, "Missing case name", "Please enter a case name.")
            return

        self.import_requested.emit(
            source,
            archive,
            case_name,
            self.verify_size_checkbox.isChecked(),
            self.sha_checkbox.isChecked(),
            self.skip_checkbox.isChecked(),
            self.process_from_source_checkbox.isChecked(),
        )

    def _on_process_from_source_toggled(self, checked: bool) -> None:
        # Size verification and skip logic apply only when files are copied into archive.
        self.verify_size_checkbox.setEnabled(not checked)
        self.skip_checkbox.setEnabled(not checked)
        self.direct_mode_note.setVisible(checked)

    def set_scan_running(self, running: bool) -> None:
        self.scan_btn.setEnabled(not running)
        self.import_btn.setEnabled(not running and self._last_summary is not None and self._last_summary.total_mp4_files > 0)
        if running:
            self.summary_label.setText("Scanning source... 0 files checked | MP4 found: 0 | Size: 0.00 GB")
            self.scan_activity_label.setText("Activity: starting scan...")
            self.scan_busy_bar.setRange(0, 0)
            self.scan_busy_bar.show()
        else:
            self.scan_busy_bar.setRange(0, 1)
            self.scan_busy_bar.setValue(1)
            self.scan_busy_bar.hide()

    def update_scan_progress(self, entries_scanned: int, mp4_files: int, total_size_bytes: int) -> None:
        self.summary_label.setText(
            f"Scanning source... {entries_scanned} files checked | MP4 found: {mp4_files} | "
            f"Size: {total_size_bytes / (1024**3):.2f} GB"
        )

    def update_scan_activity(self, activity_text: str) -> None:
        self.scan_activity_label.setText(f"Activity: {activity_text}")

    def set_scan_summary(self, summary: ScanSummary) -> None:
        self._last_summary = summary
        self.import_btn.setEnabled(summary.total_mp4_files > 0)
        self.summary_label.setText(
            (
                f"MP4: {summary.total_mp4_files} | "
                f"BlackVue: {summary.blackvue_files} | "
                f"Front: {summary.front_files} | Rear: {summary.rear_files} | "
                f"Total size: {summary.total_size_bytes / (1024**3):.2f} GB"
            )
        )

    def append_log(self, text: str) -> None:
        self.error_log.append(text)
