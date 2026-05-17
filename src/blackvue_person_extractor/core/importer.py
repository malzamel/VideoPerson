from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

from blackvue_person_extractor.config import ImportOptions
from blackvue_person_extractor.core.person_detection import DetectionConfig, PersonDetector
from blackvue_person_extractor.core.sd_scanner import ScannedVideoFile
from blackvue_person_extractor.logging_config import get_logger
from blackvue_person_extractor.storage import repositories
from blackvue_person_extractor.utils.hashing import sha256_file

logger = get_logger()


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
    output_dir: Path,
    options: ImportOptions,
    controller: ImportController,
    on_file_start,
    on_file_progress,
    on_file_done,
    on_error,
    on_counters=None,
    on_detection=None,
) -> ImportStats:
    logger.info("import_files start: files=%s process_from_source=%s", len(files), options.process_from_source)
    filtered_files = [
        item
        for item in files
        if (options.camera_filter == "both" or item.camera_direction_code == options.camera_filter[:1].upper())
        and (not options.recording_type_filter or (item.recording_type_code in options.recording_type_filter))
    ]
    if options.prioritize_important_first:
        priority = {"E": 0, "I": 1, "M": 2, "P": 3, "N": 4}
        filtered_files.sort(key=lambda f: (priority.get(f.recording_type_code or "N", 9), f.path.name))
    else:
        filtered_files.sort(key=lambda f: f.path.name)
    stats = ImportStats(total_files=len(filtered_files))
    detector = PersonDetector()
    snapshot_dir = output_dir / "person_shots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    settings_hash = hashlib.sha256(
        json.dumps(
            {
                "processing_mode": options.processing_mode,
                "detection_strategy": options.detection_strategy,
                "camera_filter": options.camera_filter,
                "recording_type_filter": list(options.recording_type_filter),
                "normal_sample_fps": options.normal_sample_fps,
                "event_sample_fps": options.event_sample_fps,
                "detailed_rescan_fps": options.detailed_rescan_fps,
                "max_detection_width": options.max_detection_width,
                "use_gpu": options.use_gpu,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if not options.process_from_source:
        original_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(filtered_files, start=1):
        if controller.cancelled:
            break

        source = item.path
        logger.info("import_files file %s/%s: %s", index, len(filtered_files), source)
        target = original_dir / source.name
        file_id = repositories.insert_video_file_pending(conn, case_id=case_id, original_path=source, size_bytes=item.size_bytes)
        on_file_start(index, len(filtered_files), source.name, item.size_bytes)
        try:
            source_stat = source.stat()
        except Exception as exc:  # noqa: BLE001
            repositories.mark_video_failed(conn, file_id, f"Source stat failed: {exc}")
            stats.failed_files += 1
            on_error(source.name, f"Source file unavailable: {exc}")
            on_file_done(index, source.name, "failed")
            if on_counters:
                on_counters(
                    stats.copied_files,
                    stats.skipped_files,
                    stats.failed_files,
                    stats.bytes_copied,
                    stats.persons_found,
                )
            continue
        cache_row = repositories.get_processing_cache(
            conn,
            file_path=source,
            file_size_bytes=item.size_bytes,
            modified_time_ns=source_stat.st_mtime_ns,
            settings_hash=settings_hash,
        )
        if options.skip_already_copied and not options.reprocess_all and cache_row:
            cached_persons = int(cache_row["persons_found"] or 0)
            stats.persons_found += cached_persons
            repositories.mark_video_skipped(conn, file_id, source if options.process_from_source else target)
            stats.skipped_files += 1
            on_file_done(index, source.name, "skipped")
            if on_detection:
                on_detection(
                    source.name,
                    cached_persons,
                    stats.persons_found,
                    str(cache_row["snapshot_path"] or ""),
                    0,
                    0,
                )
            if on_counters:
                on_counters(
                    stats.copied_files,
                    stats.skipped_files,
                    stats.failed_files,
                    stats.bytes_copied,
                    stats.persons_found,
                )
            continue

        try:
            imported_video_path = source
            if options.process_from_source:
                file_hash = sha256_file(source) if options.calculate_sha256 else None
                on_file_progress(
                    index,
                    len(filtered_files),
                    source.name,
                    item.size_bytes,
                    item.size_bytes,
                    stats.bytes_copied,
                )
                repositories.mark_video_imported(conn, file_id, source, file_hash)
            else:
                if target.exists():
                    target_size = target.stat().st_size
                    if options.skip_already_copied and (not options.verify_by_size or target_size == item.size_bytes):
                        repositories.mark_video_skipped(conn, file_id, target)
                        stats.skipped_files += 1
                        on_file_done(index, source.name, "skipped")
                        if on_counters:
                            on_counters(
                                stats.copied_files,
                                stats.skipped_files,
                                stats.failed_files,
                                stats.bytes_copied,
                                stats.persons_found,
                            )
                        continue
                    target = _resolve_conflict_path(target)

                copied_for_file = 0

                def chunk_cb(bytes_written: int) -> None:
                    nonlocal copied_for_file
                    copied_for_file += bytes_written
                    stats.bytes_copied += bytes_written
                    on_file_progress(
                        index,
                        len(filtered_files),
                        source.name,
                        copied_for_file,
                        item.size_bytes,
                        stats.bytes_copied,
                    )

                _copy_with_progress(source, target, options.chunk_size, controller, chunk_cb)

                if options.verify_by_size and target.stat().st_size != item.size_bytes:
                    raise IOError("Size verification failed after copy.")

                file_hash = sha256_file(target) if options.calculate_sha256 else None
                repositories.mark_video_imported(conn, file_id, target, file_hash)
                imported_video_path = target

            try:
                detection = detector.detect_people_and_snapshot(
                    imported_video_path,
                    output_dir=snapshot_dir / source.stem,
                    recording_type_code=item.recording_type_code,
                    camera_direction_code=item.camera_direction_code,
                    config=DetectionConfig(
                        mode=options.processing_mode,
                        strategy=options.detection_strategy,
                        use_gpu=options.use_gpu,
                        normal_sample_fps=options.normal_sample_fps,
                        event_sample_fps=options.event_sample_fps,
                        detailed_rescan_fps=options.detailed_rescan_fps,
                        max_detection_width=options.max_detection_width,
                        debug_mode=options.debug_mode,
                    ),
                )
                persons_in_video = detection.max_people
                logger.info(
                    "detection complete for %s: persons=%s sampled=%s windows=%s",
                    source.name,
                    persons_in_video,
                    detection.sampled_frames,
                    detection.candidate_windows,
                )
            except Exception as detection_exc:  # noqa: BLE001
                persons_in_video = 0
                detection = None
                on_error(source.name, f"Person detection warning: {detection_exc!r}")
                logger.exception("Detection failed for %s: %r", source, detection_exc)
            stats.persons_found += persons_in_video
            if on_detection:
                on_detection(
                    source.name,
                    persons_in_video,
                    stats.persons_found,
                    str(detection.snapshot_path) if detection and detection.snapshot_path else "",
                    detection.sampled_frames if detection else 0,
                    detection.candidate_windows if detection else 0,
                )
            repositories.upsert_processing_cache(
                conn,
                file_path=source,
                file_size_bytes=item.size_bytes,
                modified_time_ns=source_stat.st_mtime_ns,
                settings_hash=settings_hash,
                status="completed",
                persons_found=persons_in_video,
                snapshot_path=detection.snapshot_path if detection else None,
                metadata_path=detection.metadata_path if detection else None,
            )

            stats.copied_files += 1
            on_file_done(index, source.name, "copied")
            if on_counters:
                on_counters(
                    stats.copied_files,
                    stats.skipped_files,
                    stats.failed_files,
                    stats.bytes_copied,
                    stats.persons_found,
                )
        except Exception as exc:  # noqa: BLE001
            if target.exists() and target.stat().st_size == 0:
                target.unlink(missing_ok=True)
            repositories.mark_video_failed(conn, file_id, str(exc))
            stats.failed_files += 1
            on_error(source.name, str(exc))
            repositories.upsert_processing_cache(
                conn,
                file_path=source,
                file_size_bytes=item.size_bytes,
                modified_time_ns=source_stat.st_mtime_ns,
                settings_hash=settings_hash,
                status="failed",
                persons_found=0,
                snapshot_path=None,
                metadata_path=None,
            )
            on_file_done(index, source.name, "failed")
            if on_counters:
                on_counters(
                    stats.copied_files,
                    stats.skipped_files,
                    stats.failed_files,
                    stats.bytes_copied,
                    stats.persons_found,
                )

    return stats
