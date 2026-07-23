from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "1.0"
FORMAT_NAME = "VGArm Trajectory Dataset v1"


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    scene: Path
    robot: str
    tasks_file: Path
    episodes: int
    seed: int
    position_jitter: float
    modalities: tuple[str, ...] = ("state",)
    no_viewer: bool = True
    overwrite: bool = False
    resume: bool = False
    fail_fast: bool = False
    quiet: bool = False
    verbose: bool = False
