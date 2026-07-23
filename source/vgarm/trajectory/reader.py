from __future__ import annotations

import json
from pathlib import Path

import duckdb


def read_episodes(root: Path) -> list[dict]:
    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_episode_rows(root: Path, metadata: dict) -> list[dict]:
    path = root / metadata["trajectory_file"]
    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY frame_index", [str(path)]
        )
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def find_episode(root: Path, episode_id: int) -> dict:
    for episode in read_episodes(root):
        if episode["episode_id"] == episode_id:
            return episode
    raise ValueError(f"episode id {episode_id} was not found")
