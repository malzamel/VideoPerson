from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # noqa: BLE001
    mp = None

try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None

from blackvue_person_extractor.core.ffmpeg_utils import choose_hwaccel, extract_frame_ffmpeg, ffprobe_json, sample_frames_ffmpeg
from blackvue_person_extractor.logging_config import get_logger

logger = get_logger()

DetectionMode = Literal["ultra_fast", "balanced", "accurate"]
DetectionStrategy = Literal["face_only", "person_first_then_face", "person_only"]


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
    strategy: DetectionStrategy = "person_first_then_face"
    use_gpu: bool = True
    normal_sample_fps: float | None = None
    event_sample_fps: float | None = None
    detailed_rescan_fps: float | None = None
    max_detection_width: int | None = None
    yolo_conf: float = 0.35
    min_face_conf: float = 0.75
    min_face_size_px: int = 60
    min_sharpness: float = 25.0
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    debug_mode: bool = False

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
        return 8.0 if self.mode == "accurate" else 5.0

    def detailed_width(self) -> int:
        if self.max_detection_width is not None:
            return min(max(self.max_detection_width, 960), 1920)
        return 1920 if self.mode == "accurate" else 1280


@dataclass(slots=True)
class _ScoredCandidate:
    timestamp_sec: float
    person_bbox: tuple[int, int, int, int] | None
    face_bbox: tuple[int, int, int, int] | None
    person_conf: float
    face_conf: float
    detection_shape: tuple[int, int]  # (height, width)
    score: float
    sharpness: float
    brightness: float


class PersonDetector:
    def __init__(self) -> None:
        logger.info("Initializing PersonDetector")
        if YOLO is None:
            raise RuntimeError("ultralytics is required. Install with: pip install ultralytics")
        if mp is None:
            raise RuntimeError("mediapipe is required. Install with: pip install mediapipe")
        self._person_model = YOLO("yolo11n.pt")
        self._person_class_id = None
        for cid, name in self._person_model.names.items():
            if str(name).lower() == "person":
                self._person_class_id = int(cid)
                break
        if self._person_class_id is None:
            raise RuntimeError("YOLO model does not contain a person class.")
        self._face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6,
        )

    @staticmethod
    def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    @staticmethod
    def _pad_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_x: float, pad_y: float) -> tuple[int, int, int, int]:
        bw = x2 - x1
        bh = y2 - y1
        return PersonDetector._clamp_bbox(
            x1 - int(bw * pad_x),
            y1 - int(bh * pad_y),
            x2 + int(bw * pad_x),
            y2 + int(bh * pad_y),
            w,
            h,
        )

    def _detect_persons(self, frame: np.ndarray, conf_threshold: float, imgsz: int) -> list[tuple[int, int, int, int, float]]:
        results = self._person_model.predict(
            frame,
            conf=conf_threshold,
            classes=[self._person_class_id],
            imgsz=imgsz,
            verbose=False,
        )
        out: list[tuple[int, int, int, int, float]] = []
        if not results:
            return out
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            if cls_id != self._person_class_id:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, conf))
        return out

    def _detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_detector.process(rgb)
        if not result.detections:
            return []
        h, w = frame.shape[:2]
        out: list[tuple[int, int, int, int, float]] = []
        for det in result.detections:
            rel = det.location_data.relative_bounding_box
            x1 = int(rel.xmin * w)
            y1 = int(rel.ymin * h)
            x2 = int((rel.xmin + rel.width) * w)
            y2 = int((rel.ymin + rel.height) * h)
            x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, w, h)
            conf = float(det.score[0]) if det.score else 0.0
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, conf))
        return out

    def _detect_faces_in_person(self, frame: np.ndarray, person_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int, float]]:
        px1, py1, px2, py2 = person_bbox
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return []
        local = self._detect_faces(crop)
        return [(px1 + x1, py1 + y1, px1 + x2, py1 + y2, conf) for (x1, y1, x2, y2, conf) in local]

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
                start = ts
                end = ts
        merged.append((max(0.0, start - 3.0), min(duration_sec, end + 3.0)))
        return merged

    @staticmethod
    def _quality_score(face_crop: np.ndarray, face_conf: float, person_conf: float, face_width_px: int) -> tuple[float, float, float]:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        sharpness_norm = min(sharpness / 300.0, 1.0)
        brightness_norm = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        face_size_norm = min(float(face_width_px) / 180.0, 1.0)
        score = (
            0.30 * min(max(face_conf, 0.0), 1.0)
            + 0.25 * face_size_norm
            + 0.25 * sharpness_norm
            + 0.10 * brightness_norm
            + 0.10 * min(max(person_conf, 0.0), 1.0)
        )
        return score, sharpness, brightness

    @staticmethod
    def _save_overlay(
        frame: np.ndarray,
        output_path: Path,
        reason: str,
        person_bbox: tuple[int, int, int, int] | None,
        face_bbox: tuple[int, int, int, int] | None,
    ) -> None:
        overlay = frame.copy()
        if person_bbox is not None:
            cv2.rectangle(overlay, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), (0, 255, 0), 2)
        if face_bbox is not None:
            cv2.rectangle(overlay, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 180, 0), 2)
        cv2.putText(overlay, reason, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def detect_people_and_snapshot(
        self,
        video_path: Path,
        output_dir: Path,
        recording_type_code: str | None = None,
        camera_direction_code: str | None = None,
        config: DetectionConfig | None = None,
    ) -> PersonDetectionResult:
        cfg = config or DetectionConfig()
        logger.info("Detection start: video=%s mode=%s strategy=%s", video_path, cfg.mode, cfg.strategy)
        hwaccel = choose_hwaccel(cfg.use_gpu)
        output_dir.mkdir(parents=True, exist_ok=True)

        duration_sec = 3600.0
        try:
            probe = ffprobe_json(video_path)
            if probe.get("format", {}).get("duration") is not None:
                duration_sec = float(probe["format"]["duration"])
        except Exception:
            pass

        fast_fps = cfg.fast_scan_fps(recording_type_code)
        fast_width = cfg.fast_scan_width()
        detailed_fps = cfg.detailed_fps()
        detailed_width = cfg.detailed_width()
        fast_max_frames = 90 if cfg.mode == "ultra_fast" else (180 if cfg.mode == "balanced" else 320)
        detail_max_frames = 90 if cfg.mode != "accurate" else 180
        yolo_imgsz = 640 if fast_width <= 960 else 960
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
        for ts, frame in sample_frames_ffmpeg(
            video_path=video_path,
            sample_fps=fast_fps,
            max_width=fast_width,
            hwaccel=hwaccel,
            max_frames=fast_max_frames,
        ):
            sampled_frames += 1
            if cfg.strategy == "face_only":
                if self._detect_faces(frame):
                    candidate_ts.append(ts)
                continue
            persons = self._detect_persons(frame, conf_threshold=cfg.yolo_conf, imgsz=yolo_imgsz)
            if not persons:
                continue
            if cfg.strategy == "person_only":
                candidate_ts.append(ts)
            else:
                if any(self._detect_faces_in_person(frame, (x1, y1, x2, y2)) for x1, y1, x2, y2, _ in persons):
                    candidate_ts.append(ts)

        windows = self._merge_windows(candidate_ts, duration_sec)[:6]
        logger.info(
            "Fast pass done: sampled_frames=%s candidate_timestamps=%s windows=%s",
            sampled_frames,
            len(candidate_ts),
            len(windows),
        )

        scored: list[_ScoredCandidate] = []
        max_people = 0
        rejected = 0
        accepted = 0
        for start, end in windows:
            for ts, frame in sample_frames_ffmpeg(
                video_path=video_path,
                sample_fps=detailed_fps,
                max_width=detailed_width,
                start_time=start,
                duration=max(0.1, end - start),
                hwaccel=hwaccel,
                max_frames=detail_max_frames,
            ):
                det_shape = (frame.shape[0], frame.shape[1])
                if cfg.strategy == "face_only":
                    for fx1, fy1, fx2, fy2, fconf in self._detect_faces(frame):
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fconf < cfg.min_face_conf or fw < cfg.min_face_size_px or fh < cfg.min_face_size_px:
                            continue
                        face_box = self._pad_bbox(fx1, fy1, fx2, fy2, frame.shape[1], frame.shape[0], 0.30, 0.35)
                        face_crop = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                        if face_crop.size == 0:
                            continue
                        score, sharpness, brightness = self._quality_score(face_crop, fconf, 0.0, fw)
                        if sharpness < cfg.min_sharpness or brightness < cfg.min_brightness or brightness > cfg.max_brightness:
                            continue
                        scored.append(
                            _ScoredCandidate(ts, None, face_box, 0.0, fconf, det_shape, score, sharpness, brightness)
                        )
                    continue

                persons = self._detect_persons(frame, conf_threshold=cfg.yolo_conf, imgsz=max(640, min(detailed_width, 960)))
                for px1, py1, px2, py2, pconf in persons:
                    max_people = max(max_people, 1)
                    person_box = (px1, py1, px2, py2)
                    faces = self._detect_faces_in_person(frame, person_box)
                    if not faces:
                        if cfg.debug_mode and rejected < 80:
                            self._save_overlay(
                                frame,
                                output_dir / "debug" / "rejected" / f"{video_path.stem}_{int(ts*1000)}_no_face_inside_person.jpg",
                                "no_face_inside_person",
                                person_box,
                                None,
                            )
                            rejected += 1
                        if cfg.strategy == "person_only":
                            person_crop = frame[py1:py2, px1:px2]
                            if person_crop.size == 0:
                                continue
                            score, sharpness, brightness = self._quality_score(person_crop, 0.0, pconf, px2 - px1)
                            scored.append(
                                _ScoredCandidate(ts, person_box, None, pconf, 0.0, det_shape, score, sharpness, brightness)
                            )
                        continue

                    for fx1, fy1, fx2, fy2, fconf in faces:
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fconf < cfg.min_face_conf or fw < cfg.min_face_size_px or fh < cfg.min_face_size_px:
                            if cfg.debug_mode and rejected < 80:
                                self._save_overlay(
                                    frame,
                                    output_dir / "debug" / "rejected" / f"{video_path.stem}_{int(ts*1000)}_face_reject.jpg",
                                    "face_confidence_low" if fconf < cfg.min_face_conf else "face_too_small",
                                    person_box,
                                    (fx1, fy1, fx2, fy2),
                                )
                                rejected += 1
                            continue
                        face_box = self._pad_bbox(fx1, fy1, fx2, fy2, frame.shape[1], frame.shape[0], 0.30, 0.35)
                        face_crop = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                        if face_crop.size == 0:
                            continue
                        score, sharpness, brightness = self._quality_score(face_crop, fconf, pconf, fw)
                        if sharpness < cfg.min_sharpness or brightness < cfg.min_brightness or brightness > cfg.max_brightness:
                            if cfg.debug_mode and rejected < 80:
                                self._save_overlay(
                                    frame,
                                    output_dir / "debug" / "rejected" / f"{video_path.stem}_{int(ts*1000)}_quality_reject.jpg",
                                    "blur_too_high" if sharpness < cfg.min_sharpness else "bad_brightness",
                                    person_box,
                                    face_box,
                                )
                                rejected += 1
                            continue
                        scored.append(
                            _ScoredCandidate(ts, person_box, face_box, pconf, fconf, det_shape, score, sharpness, brightness)
                        )
                        if cfg.debug_mode and accepted < 80:
                            self._save_overlay(
                                frame,
                                output_dir / "debug" / "accepted" / f"{video_path.stem}_{int(ts*1000)}_accepted.jpg",
                                "accepted",
                                person_box,
                                face_box,
                            )
                            accepted += 1

        if not scored:
            logger.info("Detection finished without candidates: video=%s", video_path)
            return PersonDetectionResult(0, None, None, None, None, None, None, sampled_frames, len(windows))

        top = sorted(scored, key=lambda c: c.score, reverse=True)[:5]
        best = top[0]
        original = extract_frame_ffmpeg(video_path, best.timestamp_sec, hwaccel=hwaccel)
        det_h, det_w = best.detection_shape
        scale_x = original.shape[1] / max(det_w, 1)
        scale_y = original.shape[0] / max(det_h, 1)

        person_bbox_original = None
        person_crop = None
        if best.person_bbox is not None:
            px1 = int(best.person_bbox[0] * scale_x)
            py1 = int(best.person_bbox[1] * scale_y)
            px2 = int(best.person_bbox[2] * scale_x)
            py2 = int(best.person_bbox[3] * scale_y)
            px1, py1, px2, py2 = self._clamp_bbox(px1, py1, px2, py2, original.shape[1], original.shape[0])
            person_bbox_original = [px1, py1, px2, py2]
            person_crop = original[py1:py2, px1:px2]

        face_bbox_original = None
        face_crop = None
        if best.face_bbox is not None:
            fx1 = int(best.face_bbox[0] * scale_x)
            fy1 = int(best.face_bbox[1] * scale_y)
            fx2 = int(best.face_bbox[2] * scale_x)
            fy2 = int(best.face_bbox[3] * scale_y)
            fx1, fy1, fx2, fy2 = self._clamp_bbox(fx1, fy1, fx2, fy2, original.shape[1], original.shape[0])
            face_bbox_original = [fx1, fy1, fx2, fy2]
            face_crop = original[fy1:fy2, fx1:fx2]

        confirmed_person = person_bbox_original is not None
        confirmed_face = face_bbox_original is not None and face_crop is not None and face_crop.size > 0
        if cfg.strategy != "face_only" and not (confirmed_person and confirmed_face):
            return PersonDetectionResult(0, None, None, None, None, None, None, sampled_frames, len(windows))

        face_path = output_dir / "best_face.png"
        person_path = output_dir / "best_person_crop.png"
        full_frame_path = output_dir / "best_full_frame_overlay.jpg"
        contact_sheet_path = output_dir / "contact_sheet.jpg"
        metadata_path = output_dir / "metadata.json"

        overlay = original.copy()
        if person_bbox_original is not None:
            px1, py1, px2, py2 = person_bbox_original
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 255, 0), 3)
        if face_bbox_original is not None:
            fx1, fy1, fx2, fy2 = face_bbox_original
            cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (255, 180, 0), 2)
        cv2.putText(
            overlay,
            f"person={best.person_conf:.2f} face={best.face_conf:.2f} score={best.score:.2f}",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(full_frame_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if confirmed_face:
            cv2.imwrite(str(face_path), face_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        if confirmed_person and person_crop is not None and person_crop.size > 0:
            cv2.imwrite(str(person_path), person_crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        thumbs: list[np.ndarray] = []
        for cand in top:
            if cand.face_bbox is None:
                continue
            frm = extract_frame_ffmpeg(video_path, cand.timestamp_sec, hwaccel=hwaccel)
            ch, cw = cand.detection_shape
            sx = frm.shape[1] / max(cw, 1)
            sy = frm.shape[0] / max(ch, 1)
            tx1 = int(cand.face_bbox[0] * sx)
            ty1 = int(cand.face_bbox[1] * sy)
            tx2 = int(cand.face_bbox[2] * sx)
            ty2 = int(cand.face_bbox[3] * sy)
            tx1, ty1, tx2, ty2 = self._clamp_bbox(tx1, ty1, tx2, ty2, frm.shape[1], frm.shape[0])
            thumb = frm[ty1:ty2, tx1:tx2]
            if thumb.size > 0:
                thumbs.append(cv2.resize(thumb, (320, 240)))
        if thumbs:
            while len(thumbs) < 5:
                thumbs.append(np.zeros_like(thumbs[0]))
            cv2.imwrite(str(contact_sheet_path), np.hstack(thumbs[:5]), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        metadata = {
            "source_video_path": str(video_path),
            "source_filename": video_path.name,
            "camera_direction": camera_direction_code,
            "recording_type": recording_type_code,
            "source_timestamp_ms": int(best.timestamp_sec * 1000),
            "original_frame_width": int(original.shape[1]),
            "original_frame_height": int(original.shape[0]),
            "detection_frame_width": int(det_w),
            "detection_frame_height": int(det_h),
            "scale_factor_x": scale_x,
            "scale_factor_y": scale_y,
            "face_bbox_original": face_bbox_original,
            "person_bbox_original": person_bbox_original,
            "quality_score": best.score,
            "sharpness_score": best.sharpness,
            "brightness_score": best.brightness,
            "face_confidence": best.face_conf,
            "person_confidence": best.person_conf,
            "confirmed_person": confirmed_person,
            "confirmed_face": confirmed_face,
            "processing_mode": cfg.mode,
            "detection_strategy": cfg.strategy,
            "sampled_frames": sampled_frames,
            "candidate_windows": len(windows),
            "hwaccel": hwaccel or "cpu",
            "output_image_paths": {
                "best_face": str(face_path) if confirmed_face else None,
                "best_person_crop": str(person_path) if confirmed_person and person_crop is not None and person_crop.size > 0 else None,
                "best_full_frame": str(full_frame_path),
                "contact_sheet": str(contact_sheet_path) if thumbs else None,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return PersonDetectionResult(
            max_people=max_people if confirmed_person else 0,
            snapshot_path=person_path if confirmed_person and person_crop is not None and person_crop.size > 0 else full_frame_path,
            face_path=face_path if confirmed_face else None,
            person_path=person_path if confirmed_person and person_crop is not None and person_crop.size > 0 else None,
            full_frame_path=full_frame_path,
            contact_sheet_path=contact_sheet_path if thumbs else None,
            metadata_path=metadata_path,
            sampled_frames=sampled_frames,
            candidate_windows=len(windows),
        )
