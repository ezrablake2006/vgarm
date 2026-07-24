from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np


class VisualDataError(ValueError):
    pass


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
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        for row in rows:
            visual = row.get("visual_observation")
            if (
                isinstance(visual, dict)
                and "frame_index" not in visual
                and "rgb_frame_index" in visual
            ):
                row["visual_observation"] = {
                    "frame_index": visual["rgb_frame_index"],
                    "timestamp": visual.get("rgb_timestamp"),
                }
        return rows
    finally:
        connection.close()


def find_episode(root: Path, episode_id: int) -> dict:
    for episode in read_episodes(root):
        if episode["episode_id"] == episode_id:
            return episode
    raise ValueError(f"episode id {episode_id} was not found")


class Episode:
    def __init__(self, root: Path, metadata: dict):
        self.root, self.metadata = root, metadata

    def _read(self, camera: str, modality: str, frame_index):
        cameras = self.metadata.get("array_files", {})
        if camera not in cameras:
            raise VisualDataError(f"camera '{camera}' has no array data")
        if modality not in cameras[camera]:
            raise VisualDataError(
                f"modality '{modality}' is unavailable for camera '{camera}'"
            )
        single = isinstance(frame_index, (int, np.integer))
        indices = [int(frame_index)] if single else list(range(*frame_index.indices(
            self._frame_count(camera, modality))))
        found = {}
        for shard in cameras[camera][modality]:
            needed = [i for i in indices if shard["first_frame_index"] <= i <= shard["last_frame_index"]]
            if not needed:
                continue
            path = self.root / shard["path"]
            with path.open("rb") as stream, np.load(
                stream, allow_pickle=False) as data:
                lookup = {int(value): pos for pos, value in enumerate(data["frame_index"])}
                for index in needed:
                    if index not in lookup:
                        raise VisualDataError(f"frame {index} is missing")
                    pos = lookup[index]
                    item = {
                        "frame_index": int(data["frame_index"][pos]),
                        "physics_row": int(data["physics_row"][pos]),
                        "timestamp": float(data["timestamp"][pos]),
                    }
                    if modality == "depth":
                        item["depth_m"] = np.asarray(data["depth_m"][pos]).copy()
                    else:
                        item["object_id"] = np.asarray(data["object_id"][pos]).copy()
                        item["object_type"] = np.asarray(data["object_type"][pos]).copy()
                    found[index] = item
        missing = [i for i in indices if i not in found]
        if missing:
            raise VisualDataError(f"frame index out of range or missing: {missing[0]}")
        result = [found[i] for i in indices]
        return result[0] if single else result

    def _frame_count(self, camera, modality):
        return sum(x["frame_count"] for x in self.metadata["array_files"][camera][modality])

    def read_depth(self, *, camera: str, frame_index):
        return self._read(camera, "depth", frame_index)

    def read_segmentation(self, *, camera: str, frame_index):
        return self._read(camera, "segmentation", frame_index)

    def segmentation_manifest(self) -> dict:
        path = self.metadata.get("segmentation_index_file")
        if not path:
            raise VisualDataError("segmentation label manifest is unavailable")
        return json.loads((self.root / path).read_text(encoding="utf-8"))


def open_episode(dataset: str | Path, episode_id: int = 0) -> Episode:
    root = Path(dataset)
    return Episode(root, find_episode(root, episode_id))
