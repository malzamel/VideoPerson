# blackvue-person-extractor-gui

Local Windows GUI app to import and process BlackVue videos without cloud upload.

## Current status

This first implementation includes:

- GUI startup with PySide6.
- Import page for source/archive/case selection.
- Recursive MP4 scan with BlackVue filename parsing.
- Safe chunked copy to local archive (64 MB chunks).
- Optional direct processing from source path (no copy to archive/original).
- SQLite setup and import logging.
- Import progress UI with pause/resume/cancel.

Processing (person/face detection and reporting) is scaffolded for the next steps.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run GUI

```bash
python -m blackvue_person_extractor.app
```

## FFmpeg requirement

Install `ffmpeg` and `ffprobe` and make sure they are available on `PATH`.

## Privacy notice

- All processing is local on your machine.
- No cloud upload and no external API calls.
- No face recognition by person name.
- User is responsible for lawful usage and secure storage of outputs.
