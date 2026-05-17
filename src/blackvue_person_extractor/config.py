from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_COPY_CHUNK_SIZE = 64 * 1024 * 1024


@dataclass(slots=True)
class ImportOptions:
    verify_by_size: bool = True
    calculate_sha256: bool = False
    skip_already_copied: bool = True
    process_from_source: bool = True
    chunk_size: int = DEFAULT_COPY_CHUNK_SIZE


@dataclass(slots=True)
class CasePaths:
    root: Path
    original_dir: Path
    db_dir: Path
    logs_dir: Path
    output_dir: Path


def build_case_paths(archive_root: Path, case_name: str) -> CasePaths:
    case_root = archive_root / case_name
    return CasePaths(
        root=case_root,
        original_dir=case_root / "original",
        db_dir=case_root / "db",
        logs_dir=case_root / "logs",
        output_dir=case_root / "output",
    )
