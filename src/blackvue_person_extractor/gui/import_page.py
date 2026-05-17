from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from blackvue_person_extractor.core.sd_scanner import ScanSummary


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self._content = content
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; text-align: left; padding: 4px 2px; }"
        )
        self._toggle.toggled.connect(self._on_toggled)

        self._content_frame = QFrame()
        self._content_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._content_frame.setStyleSheet("QFrame { border: 1px solid #3a3a3a; border-radius: 4px; }")
        content_layout = QVBoxLayout(self._content_frame)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(6)
        content_layout.addWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content_frame)
        self._on_toggled(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._content_frame.setVisible(checked)
        self.adjustSize()

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)


class ImportPage(QWidget):
    scan_requested = Signal(str)
    import_requested = Signal(str, str, str, bool, bool, bool, bool, object)
    clear_cache_requested = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._last_summary: ScanSummary | None = None
        self._reprocess_scope = "none"
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
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

        options_box = QWidget()
        options_layout = QVBoxLayout(options_box)
        options_layout.setContentsMargins(0, 0, 0, 0)
        self.verify_size_checkbox = QCheckBox("Verify copied files by size")
        self.verify_size_checkbox.setChecked(True)
        self.sha_checkbox = QCheckBox("Calculate SHA256 hash after copy")
        self.skip_checkbox = QCheckBox("Skip already processed files")
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
        self.options_section = CollapsibleSection("Import Options", options_box, expanded=False)
        root.addWidget(self.options_section)

        perf_box = QWidget()
        perf_form = QFormLayout(perf_box)
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItem("Ultra Fast", "ultra_fast")
        self.processing_mode_combo.addItem("Balanced (Default)", "balanced")
        self.processing_mode_combo.addItem("Accurate", "accurate")
        self.processing_mode_combo.setCurrentIndex(1)

        self.detection_strategy_combo = QComboBox()
        self.detection_strategy_combo.addItem("Face only - fastest", "face_only")
        self.detection_strategy_combo.addItem("Face first, then person - recommended", "face_then_person")
        self.detection_strategy_combo.addItem("Person + face - slowest", "person_and_face")
        self.detection_strategy_combo.setCurrentIndex(1)

        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Front only", "front")
        self.camera_combo.addItem("Rear only", "rear")
        self.camera_combo.addItem("Both", "both")
        self.camera_combo.setCurrentIndex(0)

        self.normal_fps_spin = QDoubleSpinBox()
        self.normal_fps_spin.setRange(0.1, 10.0)
        self.normal_fps_spin.setSingleStep(0.5)
        self.normal_fps_spin.setValue(1.0)

        self.event_fps_spin = QDoubleSpinBox()
        self.event_fps_spin.setRange(0.1, 10.0)
        self.event_fps_spin.setSingleStep(0.5)
        self.event_fps_spin.setValue(3.0)

        self.detailed_fps_spin = QDoubleSpinBox()
        self.detailed_fps_spin.setRange(1.0, 20.0)
        self.detailed_fps_spin.setSingleStep(1.0)
        self.detailed_fps_spin.setValue(5.0)

        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(480, 3840)
        self.max_width_spin.setSingleStep(160)
        self.max_width_spin.setValue(960)

        rec_types_widget = QWidget()
        rec_types_layout = QHBoxLayout(rec_types_widget)
        rec_types_layout.setContentsMargins(0, 0, 0, 0)
        self.rec_e_checkbox = QCheckBox("E")
        self.rec_i_checkbox = QCheckBox("I")
        self.rec_m_checkbox = QCheckBox("M")
        self.rec_n_checkbox = QCheckBox("N")
        self.rec_p_checkbox = QCheckBox("P")
        for cb in (self.rec_e_checkbox, self.rec_i_checkbox, self.rec_m_checkbox, self.rec_n_checkbox, self.rec_p_checkbox):
            cb.setChecked(True)
            rec_types_layout.addWidget(cb)

        self.use_gpu_checkbox = QCheckBox("Use GPU if available")
        self.use_gpu_checkbox.setChecked(True)
        self.important_first_checkbox = QCheckBox("Process important recordings first (E/I/M)")
        self.important_first_checkbox.setChecked(True)
        self.ai_status_label = QLabel("AI acceleration: detecting...")

        cache_buttons_widget = QWidget()
        cache_buttons_layout = QHBoxLayout(cache_buttons_widget)
        cache_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.reprocess_selected_btn = QPushButton("Reprocess selected files")
        self.reprocess_selected_btn.clicked.connect(self._on_reprocess_selected_clicked)
        self.reprocess_all_btn = QPushButton("Reprocess all")
        self.reprocess_all_btn.clicked.connect(self._on_reprocess_all_clicked)
        self.clear_cache_btn = QPushButton("Clear cache")
        self.clear_cache_btn.clicked.connect(self._on_clear_cache_clicked)
        cache_buttons_layout.addWidget(self.reprocess_selected_btn)
        cache_buttons_layout.addWidget(self.reprocess_all_btn)
        cache_buttons_layout.addWidget(self.clear_cache_btn)

        perf_form.addRow("Processing mode:", self.processing_mode_combo)
        perf_form.addRow("Detection strategy:", self.detection_strategy_combo)
        perf_form.addRow("Camera:", self.camera_combo)
        perf_form.addRow("Recording types:", rec_types_widget)
        perf_form.addRow("Normal sample FPS:", self.normal_fps_spin)
        perf_form.addRow("Event sample FPS:", self.event_fps_spin)
        perf_form.addRow("Detailed rescan FPS:", self.detailed_fps_spin)
        perf_form.addRow("Max detection width:", self.max_width_spin)
        perf_form.addRow("", self.use_gpu_checkbox)
        perf_form.addRow("", self.important_first_checkbox)
        perf_form.addRow("AI acceleration:", self.ai_status_label)
        perf_form.addRow("Cache actions:", cache_buttons_widget)
        self.perf_section = CollapsibleSection("Processing Performance", perf_box, expanded=False)
        root.addWidget(self.perf_section)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        self.scan_btn = QPushButton("Scan Source")
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        self.import_btn = QPushButton("Start Processing")
        self.import_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self._on_import_clicked)
        self.import_btn.setEnabled(False)
        self.scan_btn.setObjectName("secondaryButton")
        button_layout.addWidget(self.scan_btn)
        button_layout.addWidget(self.import_btn)
        root.addLayout(button_layout)
        self._on_process_from_source_toggled(self.process_from_source_checkbox.isChecked())

        self.summary_label = QLabel("No scan run yet.")
        self.scan_activity_label = QLabel("Activity: -")
        self.scan_activity_label.setWordWrap(True)
        results_box = QWidget()
        results_layout = QFormLayout(results_box)
        self.checked_count_label = QLabel("0")
        self.mp4_count_label = QLabel("0")
        self.size_label = QLabel("0.00 GB")
        self.blackvue_count_label = QLabel("0")
        self.front_count_label = QLabel("0")
        self.rear_count_label = QLabel("0")
        results_layout.addRow("Files checked:", self.checked_count_label)
        results_layout.addRow("MP4 found:", self.mp4_count_label)
        results_layout.addRow("Total size:", self.size_label)
        results_layout.addRow("BlackVue:", self.blackvue_count_label)
        results_layout.addRow("Front camera:", self.front_count_label)
        results_layout.addRow("Rear camera:", self.rear_count_label)
        self.scan_busy_bar = QProgressBar()
        self.scan_busy_bar.setRange(0, 1)
        self.scan_busy_bar.setValue(0)
        self.scan_busy_bar.hide()
        root.addWidget(self.summary_label)
        root.addWidget(self.scan_activity_label)
        self.results_section = CollapsibleSection("Live Scan Results", results_box, expanded=False)
        root.addWidget(self.results_section)
        root.addWidget(self.scan_busy_bar)
        events_box = QWidget()
        events_layout = QVBoxLayout(events_box)
        events_layout.setContentsMargins(0, 0, 0, 0)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setMinimumHeight(180)
        self.error_log.setPlaceholderText("Scan and import status will appear here.")
        events_layout.addWidget(self.error_log)
        self.events_section = CollapsibleSection("Live Events", events_box, expanded=False)
        root.addWidget(self.events_section)
        root.addStretch(1)
        self._apply_visual_polish()

    def _apply_visual_polish(self) -> None:
        control_height = 34
        compact_control_height = 30

        for widget in (
            self.source_path_input,
            self.archive_path_input,
            self.case_name_input,
            self.processing_mode_combo,
            self.detection_strategy_combo,
            self.camera_combo,
            self.max_width_spin,
            self.normal_fps_spin,
            self.event_fps_spin,
            self.detailed_fps_spin,
        ):
            widget.setMinimumHeight(compact_control_height)

        for widget in (
            self.scan_btn,
            self.import_btn,
            self.reprocess_selected_btn,
            self.reprocess_all_btn,
            self.clear_cache_btn,
        ):
            widget.setMinimumHeight(control_height)

        # Keep cache action buttons visually balanced.
        self.reprocess_selected_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.reprocess_all_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.clear_cache_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.setStyleSheet(
            """
            QToolButton {
                font-size: 13px;
            }
            QPushButton#primaryButton {
                background-color: #2d6cdf;
                color: white;
                border: 1px solid #275ec2;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton#primaryButton:disabled {
                background-color: #4a4a4a;
                color: #b5b5b5;
                border: 1px solid #555555;
            }
            QPushButton#secondaryButton {
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 4px 8px;
            }
            """
        )

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
        recording_types = tuple(
            code
            for code, enabled in (
                ("E", self.rec_e_checkbox.isChecked()),
                ("I", self.rec_i_checkbox.isChecked()),
                ("M", self.rec_m_checkbox.isChecked()),
                ("N", self.rec_n_checkbox.isChecked()),
                ("P", self.rec_p_checkbox.isChecked()),
            )
            if enabled
        )
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "Invalid source", "Please select a valid source folder.")
            return
        if not archive:
            QMessageBox.warning(self, "Missing archive", "Please choose an archive folder.")
            return
        if not case_name:
            QMessageBox.warning(self, "Missing case name", "Please enter a case name.")
            return
        if not recording_types:
            QMessageBox.warning(self, "Missing recording types", "Select at least one recording type.")
            return

        advanced_settings = {
            "processing_mode": self.processing_mode_combo.currentData(),
            "detection_strategy": self.detection_strategy_combo.currentData(),
            "camera_filter": self.camera_combo.currentData(),
            "recording_type_filter": recording_types,
            "normal_sample_fps": float(self.normal_fps_spin.value()),
            "event_sample_fps": float(self.event_fps_spin.value()),
            "detailed_rescan_fps": float(self.detailed_fps_spin.value()),
            "max_detection_width": int(self.max_width_spin.value()),
            "use_gpu": self.use_gpu_checkbox.isChecked(),
            "prioritize_important_first": self.important_first_checkbox.isChecked(),
            "reprocess_scope": self._reprocess_scope,
        }
        if self._reprocess_scope != "none":
            self._reprocess_scope = "none"

        self.import_requested.emit(
            source,
            archive,
            case_name,
            self.verify_size_checkbox.isChecked(),
            self.sha_checkbox.isChecked(),
            self.skip_checkbox.isChecked(),
            self.process_from_source_checkbox.isChecked(),
            advanced_settings,
        )

    def _on_reprocess_selected_clicked(self) -> None:
        self._reprocess_scope = "selected"
        self.append_log("Reprocess selected enabled for the next run.")

    def _on_reprocess_all_clicked(self) -> None:
        self._reprocess_scope = "all"
        self.append_log("Reprocess all enabled for the next run.")

    def _on_clear_cache_clicked(self) -> None:
        archive = self.archive_path_input.text().strip()
        case_name = self.case_name_input.text().strip()
        if not archive or not case_name:
            QMessageBox.warning(self, "Missing inputs", "Select archive folder and case name before clearing cache.")
            return
        self.clear_cache_requested.emit(archive, case_name)

    def set_ai_status(self, status_text: str) -> None:
        self.ai_status_label.setText(status_text)

    def _on_process_from_source_toggled(self, checked: bool) -> None:
        # Size verification applies only when files are copied into archive.
        self.verify_size_checkbox.setEnabled(not checked)
        self.skip_checkbox.setEnabled(True)
        self.direct_mode_note.setVisible(checked)
        if hasattr(self, "import_btn"):
            if checked:
                self.import_btn.setText("Process from SD Card (No Copy)")
            else:
                self.import_btn.setText("Import to Computer (Copy Videos)")

    def set_scan_running(self, running: bool) -> None:
        self.scan_btn.setEnabled(not running)
        self.import_btn.setEnabled(not running and self._last_summary is not None and self._last_summary.total_mp4_files > 0)
        if running:
            self.summary_label.setText("Scanning source... 0 files checked | MP4 found: 0 | Size: 0.00 GB")
            self.scan_activity_label.setText("Activity: starting scan...")
            self.checked_count_label.setText("0")
            self.mp4_count_label.setText("0")
            self.size_label.setText("0.00 GB")
            self.blackvue_count_label.setText("0")
            self.front_count_label.setText("0")
            self.rear_count_label.setText("0")
            self.scan_busy_bar.setRange(0, 0)
            self.scan_busy_bar.show()
        else:
            self.scan_busy_bar.setRange(0, 1)
            self.scan_busy_bar.setValue(1)
            self.scan_busy_bar.hide()

    def update_scan_progress(self, entries_scanned: int, mp4_files: int, total_size_bytes: int) -> None:
        size_text = f"{total_size_bytes / (1024**3):.2f} GB"
        self.summary_label.setText(
            f"Scanning source... {entries_scanned} files checked | MP4 found: {mp4_files} | "
            f"Size: {size_text}"
        )
        self.checked_count_label.setText(str(entries_scanned))
        self.mp4_count_label.setText(str(mp4_files))
        self.size_label.setText(size_text)

    def update_scan_activity(self, activity_text: str) -> None:
        self.scan_activity_label.setText(f"Activity: {activity_text}")

    def set_scan_summary(self, summary: ScanSummary) -> None:
        self._last_summary = summary
        self.import_btn.setEnabled(summary.total_mp4_files > 0)
        total_size_text = f"{summary.total_size_bytes / (1024**3):.2f} GB"
        self.summary_label.setText(
            (
                f"MP4: {summary.total_mp4_files} | "
                f"BlackVue: {summary.blackvue_files} | "
                f"Front: {summary.front_files} | Rear: {summary.rear_files} | "
                f"Total size: {total_size_text}"
            )
        )
        self.mp4_count_label.setText(str(summary.total_mp4_files))
        self.blackvue_count_label.setText(str(summary.blackvue_files))
        self.front_count_label.setText(str(summary.front_files))
        self.rear_count_label.setText(str(summary.rear_files))
        self.size_label.setText(total_size_text)

    def append_log(self, text: str) -> None:
        self.error_log.append(text)
