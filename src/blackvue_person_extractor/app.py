from __future__ import annotations

import faulthandler
from pathlib import Path
import sys

import cv2
from PySide6.QtWidgets import QApplication

from blackvue_person_extractor.gui.main_window import MainWindow
from blackvue_person_extractor.logging_config import get_logger, setup_file_logger

_CRASH_LOG_FILE = None


def _setup_runtime_logging() -> None:
    global _CRASH_LOG_FILE
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_log = log_dir / "runtime.log"
    crash_log = log_dir / "crash.log"
    setup_file_logger(runtime_log)
    _CRASH_LOG_FILE = crash_log.open("a", encoding="utf-8")
    faulthandler.enable(file=_CRASH_LOG_FILE, all_threads=True)


def main() -> int:
    _setup_runtime_logging()
    logger = get_logger()
    logger.info("Application startup.")
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version)
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        logger.info("OpenCV threading guard enabled (threads=1, OpenCL disabled).")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to configure OpenCV threading guard: %r", exc)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
