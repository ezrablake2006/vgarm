from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "1.2"
FORMAT_NAME = "VGArm Trajectory Dataset v1.2"


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
    visual_width: int = 640
    visual_height: int = 480
    visual_fps: float = 20.0
    visual_chunk_frames: int = 64
    no_viewer: bool = True
    overwrite: bool = False
    resume: bool = False
    fail_fast: bool = False
    quiet: bool = False
    verbose: bool = False

    # Source-compatible names for callers written against v0.4.0.
    @property
    def rgb_width(self) -> int:
        return self.visual_width

    @property
    def rgb_height(self) -> int:
        return self.visual_height

    @property
    def rgb_fps(self) -> float:
        return self.visual_fps
