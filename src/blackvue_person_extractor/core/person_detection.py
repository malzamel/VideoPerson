from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from blackvue_person_extractor.core.ffmpeg_utils import choose_hwaccel, extract_frame_ffmpeg, ffprobe_json, sample_frames_ffmpeg
from blackvue_person_extractor.logging_config import get_logger

logger = get_logger()


DetectionMode = Literal["ultra_fast", "balanced", "accurate"]
DetectionStrategy = Literal["face_only", "face_then_person", "person_and_face"]


@dataclass(slots=True)
class PersonDetectionResult:
    max_people: int
    snapshot_path: Path | None
    face_path: Path | None
    person_path: Path | None
    full_frame_path: Path | None
    contact_sheet_path: Path | None
    metadata_path: Path | None
    sampled_frames: int
    candidate_windows: int


@dataclass(slots=True)
class DetectionConfig:
    mode: DetectionMode = "balanced"
    strategy: DetectionStrategy = "face_then_person"
    use_gpu: bool = True
    normal_sample_fps: float | None = None
    event_sample_fps: float | None = None
    detailed_rescan_fps: float | None = None
    max_detection_width: int | None = None

    def fast_scan_fps(self, recording_type: str | None) -> float:
        if recording_type in {"E", "I", "M"} and self.event_sample_fps is not None:
            return self.event_sample_fps
        if recording_type not in {"E", "I", "M"} and self.normal_sample_fps is not None:
            return self.normal_sample_fps
        if self.mode == "ultra_fast":
            return 0.5
        if self.mode == "accurate":
            return 5.0 if recording_type in {"E", "I", "M"} else 3.0
        return 3.0 if recording_type in {"E", "I", "M"} else 1.0

    def fast_scan_width(self) -> int:
        if self.max_detection_width is not None:
            return self.max_detection_width
        if self.mode == "ultra_fast":
            return 720
        if self.mode == "accurate":
            return 1280
        return 960

    def detailed_fps(self) -> float:
        if self.detailed_rescan_fps is not None:
            return self.detailed_rescan_fps
        if self.mode == "accurate":
            return 8.0
        return 5.0

    def detailed_width(self) -> int:
        if self.max_detection_width is not None:
            return min(max(self.max_detection_width, 960), 1920)
        if self.mode == "accurate":
            return 1920
        return 1280


@dataclass(slots=True)
class _Candidate:
    timestamp_sec: float
    face_bbox: tuple[int, int, int, int]
    face_conf: float
    source_shape: tuple[int, int]
    detection_shape: tuple[int, int]


@dataclass(slots=True)
class _ScoredCandidate:
    timestamp_sec: float
    face_bbox: tuple[int, int, int, int]
    person_bbox: tuple[int, int, int, int] | None
    face_conf: float
    score: float
    sharpness: float
    brightness: float


class PersonDetector:
    """Face-first cascaded detector with sampled FFmpeg pipeline."""

    def __init__(self) -> None:
        logger.info("Initializing PersonDetector")
        self._face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @staticmethod
    def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    @staticmethod
    def _pad_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_x: float, pad_y: float) -> tuple[int, int, int, int]:
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * pad_x)
        py = int(bh * pad_y)
        return PersonDetector._clamp_bbox(x1 - px, y1 - py, x2 + px, y2 + py, w, h)

    def _detect_faces(
        self,
        frame: np.ndarray,
        min_neighbors: int = 4,
        min_relative_area: float = 0.0015,
    ) -> list[tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(20, 20))
        results: list[tuple[int, int, int, int, float]] = []
        frame_area = float(frame.shape[0] * frame.shape[1])
        for (x, y, w, h) in faces:
            area = float(w * h)
            rel_area = area / frame_area if frame_area > 0 else 0.0
            aspect = float(w) / float(h) if h > 0 else 0.0
            # Reject tiny/odd-shape detections to reduce false positives.
            if rel_area < min_relative_area:
                continue
            if aspect < 0.55 or aspect > 1.8:
                continue
            # Haar does not expose confidence; use bbox area as proxy.
            conf = area
            results.append((x, y, x + w, y + h, conf))
        return results

    def _detect_person_bboxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        rects, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return []
        boxes: list[tuple[int, int, int, int]] = []
        for x, y, w, h in rects:
            boxes.append((x, y, x + w, y + h))
        return boxes

    def _detect_person_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        rects, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return None
        best_idx = int(np.argmax(weights)) if len(weights) > 0 else 0
        x, y, w, h = rects[best_idx]
        return x, y, x + w, y + h

    def _merge_windows(self, timestamps: list[float], duration_sec: float) -> list[tuple[float, float]]:
        if not timestamps:
            return []
        ordered = sorted(timestamps)
        merged: list[tuple[float, float]] = []
        start = ordered[0]
        end = ordered[0]
        for ts in ordered[1:]:
            if ts - end <= 5.0:
                end = ts
            else:
                merged.append((max(0.0, start - 3.0), min(duration_sec, end + 3.0)))
                start, end = ts, ts
        merged.append((max(0.0, start - 3.0), min(duration_sec, end + 3.0)))
        return merged

    @staticmethod
    def _quality_score(face_crop: np.ndarray, face_conf: float) -> tuple[float, float, float]:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        # Favor sharp, properly exposed, and larger-confidence candidates.
        exposure = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        score = (sharpness * 0.45) + (face_conf * 0.35) + (exposure * 100.0 * 0.20)
        return score, sharpness, brightness

    def detect_people_and_snapshot(
        self,
        video_path: Path,
        output_dir: Path,
        recording_type_code: str | None = None,
        camera_direction_code: str | None = None,
        config: DetectionConfig | None = None,
    ) -> PersonDetectionResult:
        """Face-first sampled scan with detailed rescan and original-quality exports."""
        logger.info("Detection start: video=%s mode=%s strategy=%s", video_path, (config.mode if config else "balanced"), (config.strategy if config else "face_then_person"))
        cfg = config or DetectionConfig()
        hwaccel = choose_hwaccel(cfg.use_gpu)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration_sec = 3600.0
        try:
            probe = ffprobe_json(video_path)
            fmt = probe.get("format", {})
            if fmt.get("duration") is not None:
                duration_sec = float(fmt.get("duration"))
        except Exception:
            # Fall back to OpenCV metadata when ffprobe is unavailable.
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    duration_sec = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            cap.release()

        fast_fps = cfg.fast_scan_fps(recording_type_code)
        fast_width = cfg.fast_scan_width()
        detailed_fps = cfg.detailed_fps()
        detailed_width = cfg.detailed_width()
        fast_max_frames = 120 if cfg.mode == "ultra_fast" else (220 if cfg.mode == "balanced" else 360)
        detail_max_frames_per_window = 120 if cfg.mode != "accurate" else 220
        logger.info(
            "Detection config: hwaccel=%s fast_fps=%s fast_width=%s detailed_fps=%s detailed_width=%s",
            hwaccel or "cpu",
            fast_fps,
            fast_width,
            detailed_fps,
            detailed_width,
        )

        sampled_frames = 0
        candidate_ts: list[float] = []
        person_seeded_windows = False
        for ts, frame in sample_frames_ffmpeg(
            video_path=video_path,
            sample_fps=fast_fps,
            max_width=fast_width,
            hwaccel=hwaccel,
            max_frames=fast_max_frames,
        ):
            sampled_frames += 1
            faces = self._detect_faces(
                frame,
                min_neighbors=5,
                min_relative_area=0.0025,
            )
            if faces:
                candidate_ts.append(ts)

        # Person-seeded windows only for person+face strategy (avoids false positives in default flow).
        if not candidate_ts and cfg.strategy == "person_and_face":
            for ts, frame in sample_frames_ffmpeg(
                video_path=video_path,
                sample_fps=max(0.5, fast_fps),
                max_width=max(720, min(fast_width, 960)),
                hwaccel=hwaccel,
                max_frames=fast_max_frames,
            ):
                sampled_frames += 1
                person_boxes = self._detect_person_bboxes(frame)
                if person_boxes:
                    candidate_ts.append(ts)
                    person_seeded_windows = True

        windows = self._merge_windows(candidate_ts, duration_sec=duration_sec)
        if len(windows) > 6:
            windows = windows[:6]
        logger.info(
            "Fast pass done: sampled_frames=%s candidate_timestamps=%s windows=%s",
            sampled_frames,
            len(candidate_ts),
            len(windows),
        )
        scored: list[_ScoredCandidate] = []
        max_people = 0

        for start, end in windows:
            for ts, frame in sample_frames_ffmpeg(
                video_path=video_path,
                sample_fps=detailed_fps,
                max_width=detailed_width,
                start_time=start,
                duration=max(0.1, end - start),
                hwaccel=hwaccel,
                max_frames=detail_max_frames_per_window,
            ):
                faces = self._detect_faces(
                    frame,
                    min_neighbors=4,
                    min_relative_area=0.0015,
                )
                if not faces and cfg.strategy == "person_and_face" and person_seeded_windows:
                    person_box = self._detect_person_bbox(frame)
                    if person_box is not None:
                        x1, y1, x2, y2 = person_box
                        max_people = max(max_people, 1)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            score, sharpness, brightness = self._quality_score(crop, 1.0)
                            scored.append(
                                _ScoredCandidate(
                                    timestamp_sec=ts,
                                    face_bbox=person_box,
                                    person_bbox=person_box,
                                    face_conf=1.0,
                                    score=score,
                                    sharpness=sharpness,
                                    brightness=brightness,
                                )
                            )
                    continue

                for (x1, y1, x2, y2, conf) in faces:
                    max_people = max(max_people, 1)
                    fx1, fy1, fx2, fy2 = self._pad_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0], 0.30, 0.35)
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    if face_crop.size == 0:
                        continue
                    score, sharpness, brightness = self._quality_score(face_crop, conf)
                    person_bbox = None
                    if cfg.strategy in {"face_then_person", "person_and_face"}:
                        person_bbox = self._detect_person_bbox(frame)
                    scored.append(
                        _ScoredCandidate(
                            timestamp_sec=ts,
                            face_bbox=(fx1, fy1, fx2, fy2),
                            person_bbox=person_bbox,
                            face_conf=conf,
                            score=score,
                            sharpness=sharpness,
                            brightness=brightness,
                        )
                    )

        if not scored:
            logger.info("Detection finished without candidates: video=%s", video_path)
            return PersonDetectionResult(
                max_people=max_people,
                snapshot_path=None,
                face_path=None,
                person_path=None,
                full_frame_path=None,
                contact_sheet_path=None,
                metadata_path=None,
                sampled_frames=sampled_frames,
                candidate_windows=len(windows),
            )

        top = sorted(scored, key=lambda c: c.score, reverse=True)[:5]
        best = top[0]

        original_frame = extract_frame_ffmpeg(video_path, best.timestamp_sec, hwaccel=hwaccel)
        scale_x = original_frame.shape[1] / detailed_width
        scale_y = original_frame.shape[0] / max(1, int(round(detailed_width * (9 / 16))))

        bx1 = int(best.face_bbox[0] * scale_x)
        by1 = int(best.face_bbox[1] * scale_y)
        bx2 = int(best.face_bbox[2] * scale_x)
        by2 = int(best.face_bbox[3] * scale_y)
        bx1, by1, bx2, by2 = self._clamp_bbox(bx1, by1, bx2, by2, original_frame.shape[1], original_frame.shape[0])
        face_crop = original_frame[by1:by2, bx1:bx2]

        person_crop = None
        person_bbox_original = None
        if best.person_bbox is not None:
            px1 = int(best.person_bbox[0] * scale_x)
            py1 = int(best.person_bbox[1] * scale_y)
            px2 = int(best.person_bbox[2] * scale_x)
            py2 = int(best.person_bbox[3] * scale_y)
            px1, py1, px2, py2 = self._clamp_bbox(px1, py1, px2, py2, original_frame.shape[1], original_frame.shape[0])
            person_bbox_original = [px1, py1, px2, py2]
            person_crop = original_frame[py1:py2, px1:px2]

        face_path = output_dir / "best_face.png"
        person_path = output_dir / "best_person_crop.png"
        full_frame_path = output_dir / "best_full_frame.jpg"
        contact_sheet_path = output_dir / "contact_sheet.jpg"
        metadata_path = output_dir / "metadata.json"

        cv2.imwrite(str(face_path), face_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if person_crop is not None and person_crop.size > 0:
            cv2.imwrite(str(person_path), person_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(str(full_frame_path), original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        thumbs: list[np.ndarray] = []
        for cand in top:
            frm = extract_frame_ffmpeg(video_path, cand.timestamp_sec, hwaccel=hwaccel)
            tx1 = int(cand.face_bbox[0] * scale_x)
            ty1 = int(cand.face_bbox[1] * scale_y)
            tx2 = int(cand.face_bbox[2] * scale_x)
            ty2 = int(cand.face_bbox[3] * scale_y)
            tx1, ty1, tx2, ty2 = self._clamp_bbox(tx1, ty1, tx2, ty2, frm.shape[1], frm.shape[0])
            thumb = frm[ty1:ty2, tx1:tx2]
            if thumb.size > 0:
                thumbs.append(cv2.resize(thumb, (320, 240)))
        if thumbs:
            while len(thumbs) < 5:
                thumbs.append(np.zeros_like(thumbs[0]))
            contact = np.hstack(thumbs[:5])
            cv2.imwrite(str(contact_sheet_path), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        metadata = {
            "source_video_path": str(video_path),
            "source_filename": video_path.name,
            "camera_direction": camera_direction_code,
            "recording_type": recording_type_code,
            "source_timestamp_ms": int(best.timestamp_sec * 1000),
            "original_frame_width": int(original_frame.shape[1]),
            "original_frame_height": int(original_frame.shape[0]),
            "detection_frame_width": int(detailed_width),
            "detection_frame_height": int(round(detailed_width * (9 / 16))),
            "scale_factor_x": scale_x,
            "scale_factor_y": scale_y,
            "face_bbox_original": [bx1, by1, bx2, by2],
            "person_bbox_original": person_bbox_original,
            "quality_score": best.score,
            "sharpness_score": best.sharpness,
            "brightness_score": best.brightness,
            "face_confidence": best.face_conf,
            "processing_mode": cfg.mode,
            "detection_strategy": cfg.strategy,
            "sampled_frames": sampled_frames,
            "candidate_windows": len(windows),
            "hwaccel": hwaccel or "cpu",
            "output_image_paths": {
                "best_face": str(face_path),
                "best_person_crop": str(person_path) if person_crop is not None and person_crop.size > 0 else None,
                "best_full_frame": str(full_frame_path),
                "contact_sheet": str(contact_sheet_path) if thumbs else None,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(
            "Detection finished: video=%s persons=%s sampled_frames=%s windows=%s output=%s",
            video_path,
            max_people,
            sampled_frames,
            len(windows),
            face_path,
        )

        return PersonDetectionResult(
            max_people=max_people,
            snapshot_path=face_path,
            face_path=face_path,
            person_path=person_path if person_crop is not None and person_crop.size > 0 else None,
            full_frame_path=full_frame_path,
            contact_sheet_path=contact_sheet_path if thumbs else None,
            metadata_path=metadata_path,
            sampled_frames=sampled_frames,
            candidate_windows=len(windows),
        )
