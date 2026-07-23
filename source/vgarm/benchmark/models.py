from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    id: str
    instruction: str


@dataclass(frozen=True)
class BenchmarkConfig:
    scene: Path
    robots: tuple[str, ...]
    episodes: int
    seed: int
    output: Path
    tasks: tuple[TaskSpec, ...]
    position_jitter: float = 0.0
    no_viewer: bool = True
    viewer_speed: float = 1.0
    quiet: bool = False
    verbose: bool = False
    fail_fast: bool = False
    overwrite: bool = False


@dataclass
class VerificationResult:
    predicate: str
    passed: bool
    measured_value: float | None
    required_margin: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    episode_id: int
    episode_seed: int
    robot: str
    task_id: str
    instruction: str
    scene: str
    program_version: str
    started_at: str
    duration_seconds: float
    object_positions: dict[str, list[float]]
    initial_object_positions: dict[str, list[float]] = field(default_factory=dict)
    planned_target_position: list[float] | None = None
    original_target_position: list[float] | None = None
    transport_waypoints: list[list[float]] = field(default_factory=list)
    final_object_positions: dict[str, list[float]] = field(default_factory=dict)
    held_object: str | None = None
    obstacle_objects: list[str] = field(default_factory=list)
    path_diagnostics: dict[str, Any] = field(default_factory=lambda: {
        "direct_path_clear": None,
        "target_region_clear": None,
        "minimum_clearance": None,
        "required_clearance": None,
    })
    collision_diagnostics: dict[str, Any] | None = None
    grasp_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    object_geometry: dict[str, dict[str, Any]] = field(default_factory=dict)
    parse_success: bool = False
    pick_attempted: bool = False
    pick_success: bool | None = None
    place_attempted: bool = False
    place_success: bool | None = None
    task_success: bool = False
    failure_stage: str | None = None
    failure_reason: str | None = None
    failure_category: str | None = None
    verification: VerificationResult | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
