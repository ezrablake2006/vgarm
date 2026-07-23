from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "1.1"
FORMAT_NAME = "VGArm Trajectory Dataset v1.1"


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
    cameras: tuple[str, ...] = ()
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: float = 20.0
    no_viewer: bool = True
    overwrite: bool = False
    resume: bool = False
    fail_fast: bool = False
    quiet: bool = False
    verbose: bool = False
