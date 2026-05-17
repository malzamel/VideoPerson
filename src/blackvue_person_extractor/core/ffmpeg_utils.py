from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO, Generator

import ctypes
import cv2
import numpy as np

from blackvue_person_extractor.logging_config import get_logger

logger = get_logger()
_HWACCEL_DISABLE: set[str] = set()
_HWACCEL_LOCK = threading.Lock()


def _ffmpeg_candidates() -> list[str]:
    candidates: list[str] = []
    env_ffmpeg = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg:
        candidates.append(env_ffmpeg)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(str(Path(local_app_data) / "Microsoft" / "WindowsApps" / "ffmpeg.exe"))
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.exists():
            for package_dir in winget_packages.glob("Gyan.FFmpeg_*"):
                for exe in package_dir.glob("**/bin/ffmpeg.exe"):
                    candidates.append(str(exe))
    candidates.extend(
        [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
    )
    return candidates


def resolve_ffmpeg_exe() -> str | None:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    for candidate in _ffmpeg_candidates():
        if Path(candidate).exists():
            return candidate
    return None


def resolve_ffprobe_exe() -> str | None:
    on_path = shutil.which("ffprobe")
    if on_path:
        return on_path
    ffmpeg_exe = resolve_ffmpeg_exe()
    if ffmpeg_exe:
        sibling = Path(ffmpeg_exe).with_name("ffprobe.exe")
        if sibling.exists():
            return str(sibling)
    return None


def ffmpeg_available() -> bool:
    return resolve_ffmpeg_exe() is not None


def ffprobe_json(video_path: Path) -> dict:
    ffprobe_exe = resolve_ffprobe_exe()
    if ffprobe_exe is None:
        raise RuntimeError("ffprobe not found. Install FFmpeg or set FFMPEG_PATH to ffmpeg.exe.")
    cmd = [
        ffprobe_exe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout or "{}")


def ffmpeg_hwaccels() -> list[str]:
    ffmpeg_exe = resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        return []
    cmd = [ffmpeg_exe, "-hide_banner", "-hwaccels"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines()]
    return [line for line in lines if line and not line.lower().startswith("hardware acceleration methods")]


def choose_hwaccel(prefer_gpu: bool = True) -> str | None:
    if resolve_ffmpeg_exe() is None:
        return None
    if not prefer_gpu:
        return None
    try:
        available = set(ffmpeg_hwaccels())
    except Exception:
        return None
    for candidate in ("cuda", "d3d11va", "dxva2"):
        if candidate in _HWACCEL_DISABLE:
            continue
        if candidate == "cuda":
            try:
                ctypes.WinDLL("nvcuda.dll")
            except Exception:
                with _HWACCEL_LOCK:
                    _HWACCEL_DISABLE.add("cuda")
                logger.info("Disabling CUDA hwaccel: nvcuda.dll not available.")
                continue
        if candidate in available:
            return candidate
    return None


def _iter_mjpeg_frames(stream: bytes) -> Generator[np.ndarray, None, None]:
    marker_soi = b"\xff\xd8"
    marker_eoi = b"\xff\xd9"
    i = 0
    while True:
        start = stream.find(marker_soi, i)
        if start < 0:
            return
        end = stream.find(marker_eoi, start + 2)
        if end < 0:
            return
        jpg = stream[start : end + 2]
        arr = np.frombuffer(jpg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            yield frame
        i = end + 2


def _iter_mjpeg_frames_stream(stream: BinaryIO) -> Generator[np.ndarray, None, None]:
    marker_soi = b"\xff\xd8"
    marker_eoi = b"\xff\xd9"
    buffer = b""
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        buffer += chunk
        while True:
            start = buffer.find(marker_soi)
            if start < 0:
                # Keep small tail only; SOI is 2 bytes.
                buffer = buffer[-1:] if len(buffer) > 1 else buffer
                break
            end = buffer.find(marker_eoi, start + 2)
            if end < 0:
                # Keep from SOI onward for next chunk.
                buffer = buffer[start:]
                break
            jpg = buffer[start : end + 2]
            buffer = buffer[end + 2 :]
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def sample_frames_ffmpeg(
    video_path: Path,
    sample_fps: float,
    max_width: int,
    start_time: float | None = None,
    duration: float | None = None,
    hwaccel: str | None = None,
    max_frames: int | None = None,
) -> Generator[tuple[float, np.ndarray], None, None]:
    ffmpeg_exe = resolve_ffmpeg_exe()
    logger.info(
        "sample_frames_ffmpeg start: file=%s fps=%s width=%s start=%s duration=%s hwaccel=%s max_frames=%s",
        video_path,
        sample_fps,
        max_width,
        start_time,
        duration,
        hwaccel or "cpu",
        max_frames,
    )
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg not found. Install FFmpeg or set FFMPEG_PATH to ffmpeg.exe.")

    vf = f"fps={sample_fps},scale='min({max_width},iw)':-2:flags=lanczos"
    def _build_cmd(use_hwaccel: str | None) -> list[str]:
        cmd_local = [ffmpeg_exe, "-hide_banner", "-loglevel", "error"]
        if use_hwaccel:
            cmd_local += ["-hwaccel", use_hwaccel]
        if start_time is not None:
            cmd_local += ["-ss", f"{start_time:.3f}"]
        cmd_local += ["-i", str(video_path)]
        if duration is not None:
            cmd_local += ["-t", f"{duration:.3f}"]
        cmd_local += [
            "-vf",
            vf,
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            "-",
        ]
        return cmd_local

    def _run_sampler_once(use_hwaccel: str | None) -> tuple[int, int | None, str]:
        cmd_local = _build_cmd(use_hwaccel)
        produced_local = 0
        proc_local: subprocess.Popen | None = None
        stderr_text = ""
        try:
            proc_local = subprocess.Popen(cmd_local, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc_local.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe unavailable")
            for idx, frame in enumerate(_iter_mjpeg_frames_stream(proc_local.stdout)):
                ts = idx / sample_fps if sample_fps > 0 else float(idx)
                if start_time is not None:
                    ts += start_time
                yield ts, frame
                produced_local += 1
                if max_frames is not None and produced_local >= max_frames:
                    proc_local.terminate()
                    break
            if proc_local.stderr is not None:
                stderr_text = proc_local.stderr.read().decode("utf-8", errors="replace").strip()
            proc_local.wait(timeout=5)
            return produced_local, proc_local.returncode, stderr_text
        except Exception:
            logger.exception("sample_frames_ffmpeg failed (hwaccel=%s).", use_hwaccel or "cpu")
            if proc_local is not None and proc_local.poll() is None:
                proc_local.kill()
            raise

    first_attempt = _run_sampler_once(hwaccel)
    produced, returncode, stderr_text = yield from first_attempt
    logger.info("sample_frames_ffmpeg end: produced=%s returncode=%s", produced, returncode)
    if produced > 0:
        return

    # Retry on CPU decode when HW acceleration yields no frames.
    if hwaccel:
        logger.warning(
            "ffmpeg sampler produced 0 frames with hwaccel=%s (returncode=%s). Retrying on CPU. stderr=%s",
            hwaccel,
            returncode,
            stderr_text[:500],
        )
        with _HWACCEL_LOCK:
            _HWACCEL_DISABLE.add(hwaccel)
        logger.info("Temporarily disabled hwaccel '%s' for this run.", hwaccel)
        second_attempt = _run_sampler_once(None)
        produced_cpu, returncode_cpu, stderr_cpu = yield from second_attempt
        logger.info(
            "sample_frames_ffmpeg CPU retry end: produced=%s returncode=%s",
            produced_cpu,
            returncode_cpu,
        )
        if produced_cpu > 0:
            return
        raise RuntimeError(
            f"ffmpeg sampler produced 0 frames on both hwaccel={hwaccel} (rc={returncode}) and CPU "
            f"(rc={returncode_cpu}). stderr(cpu)={stderr_cpu[:500]}"
        )

    raise RuntimeError(f"ffmpeg sampler produced 0 frames (returncode={returncode}). stderr={stderr_text[:500]}")


def extract_frame_ffmpeg(video_path: Path, timestamp_sec: float, hwaccel: str | None = None) -> np.ndarray:
    ffmpeg_exe = resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg not found. Install FFmpeg or set FFMPEG_PATH to ffmpeg.exe.")

    def _extract_once(use_hwaccel: str | None) -> np.ndarray:
        cmd = [ffmpeg_exe, "-hide_banner", "-loglevel", "error"]
        if use_hwaccel:
            cmd += ["-hwaccel", use_hwaccel]
        cmd += [
            "-ss",
            f"{max(timestamp_sec, 0.0):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        arr = np.frombuffer(result.stdout, dtype=np.uint8)
        frame_local = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_local is None:
            raise RuntimeError(f"Failed to decode frame at {timestamp_sec:.3f}s from {video_path}")
        return frame_local

    try:
        return _extract_once(hwaccel)
    except Exception:
        if hwaccel:
            logger.warning("extract_frame_ffmpeg failed with hwaccel=%s; retrying on CPU.", hwaccel)
            return _extract_once(None)
        logger.exception("extract_frame_ffmpeg failed.")
        raise


def _sample_frames_opencv(
    video_path: Path,
    sample_fps: float,
    max_width: int,
    start_time: float | None = None,
    duration: float | None = None,
    max_frames: int | None = None,
) -> Generator[tuple[float, np.ndarray], None, None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        step = max(int(round(fps / max(sample_fps, 0.1))), 1)

        start_idx = int(max(start_time or 0.0, 0.0) * fps)
        end_idx = None
        if duration is not None:
            end_idx = start_idx + int(max(duration, 0.1) * fps)

        frame_idx = start_idx
        produced = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        while True:
            if end_idx is not None and frame_idx > end_idx:
                break
            if max_frames is not None and produced >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                new_h = int(h * (max_width / float(w)))
                frame = cv2.resize(frame, (max_width, max(new_h, 1)), interpolation=cv2.INTER_AREA)
            yield frame_idx / fps, frame
            produced += 1
            frame_idx += step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    finally:
        cap.release()


def _extract_frame_opencv(video_path: Path, timestamp_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        target_idx = int(max(timestamp_sec, 0.0) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to decode frame at {timestamp_sec:.3f}s from {video_path}")
        return frame
    finally:
        cap.release()
