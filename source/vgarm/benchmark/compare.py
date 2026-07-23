from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DETERMINISTIC_FIELDS = (
    "episode_id", "episode_seed", "robot", "task_id", "instruction",
    "initial_object_positions", "planned_target_position",
    "original_target_position", "transport_waypoints", "final_object_positions",
    "task_success", "failure_stage", "failure_category", "verification",
)


def deterministic_payload(directory: Path) -> list[dict[str, Any]]:
    episodes = []
    with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            episodes.append({key: source.get(key) for key in DETERMINISTIC_FIELDS})
    return episodes


def fingerprint(directory: Path) -> str:
    payload = deterministic_payload(directory)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_results(first: Path, second: Path) -> tuple[bool, str, str]:
    first_hash = fingerprint(first)
    second_hash = fingerprint(second)
    return first_hash == second_hash, first_hash, second_hash
