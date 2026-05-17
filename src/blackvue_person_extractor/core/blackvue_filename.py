from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

BLACKVUE_FILENAME_RE = re.compile(r"^(\d{8})_(\d{6})_([A-Z])([FR])([A-Z0-9]*)?\.mp4$", re.IGNORECASE)

RECORDING_TYPE_LABELS = {
    "N": "Normal",
    "P": "Parking Motion / Parking Time-lapse",
    "M": "Manual",
    "E": "Driving Impact",
    "I": "Parking Impact",
    "O": "Overspeed",
    "A": "Acceleration",
    "T": "Hard Cornering",
    "B": "Hard Braking",
    "R": "Geofence Enter",
    "X": "Geofence Exit",
    "G": "Geofence Pass",
    "D": "Drowsiness",
    "L": "Distraction",
    "Y": "Seatbelt",
    "F": "Undetected",
}

CAMERA_DIRECTION_LABELS = {"F": "Front", "R": "Rear"}


@dataclass(slots=True)
class BlackVueFilenameInfo:
    filename: str
    recording_date: str | None
    recording_time: str | None
    start_datetime: datetime | None
    recording_type_code: str | None
    recording_type_label: str | None
    camera_direction_code: str | None
    camera_direction_label: str | None
    other_code: str | None
    extension: str
    is_valid_blackvue_name: bool


def parse_blackvue_filename(filename_or_path: str | Path) -> BlackVueFilenameInfo:
    filename = Path(filename_or_path).name
    match = BLACKVUE_FILENAME_RE.match(filename)
    extension = Path(filename).suffix.lower()
    if not match:
        return BlackVueFilenameInfo(
            filename=filename,
            recording_date=None,
            recording_time=None,
            start_datetime=None,
            recording_type_code=None,
            recording_type_label=None,
            camera_direction_code=None,
            camera_direction_label=None,
            other_code=None,
            extension=extension,
            is_valid_blackvue_name=False,
        )

    date_code, time_code, rec_code, cam_code, other = match.groups()
    rec_code = rec_code.upper()
    cam_code = cam_code.upper()

    try:
        start_dt = datetime.strptime(f"{date_code}{time_code}", "%Y%m%d%H%M%S")
    except ValueError:
        start_dt = None

    return BlackVueFilenameInfo(
        filename=filename,
        recording_date=date_code,
        recording_time=time_code,
        start_datetime=start_dt,
        recording_type_code=rec_code,
        recording_type_label=RECORDING_TYPE_LABELS.get(rec_code, "Other"),
        camera_direction_code=cam_code,
        camera_direction_label=CAMERA_DIRECTION_LABELS.get(cam_code, "Unknown"),
        other_code=(other or ""),
        extension=extension,
        is_valid_blackvue_name=True,
    )
