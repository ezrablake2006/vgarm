"""Join a sampled physics row to the exact corresponding MP4 frame."""

import argparse
import json
from pathlib import Path

import duckdb
import imageio.v2 as imageio


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--episode-id", type=int, default=0)
parser.add_argument("--camera", default="camera_front")
args = parser.parse_args()

episodes = [
    json.loads(line)
    for line in (args.dataset / "meta" / "episodes.jsonl").read_text().splitlines()
    if line.strip()
]
episode = next(item for item in episodes if item["episode_id"] == args.episode_id)
trajectory = args.dataset / episode["trajectory_file"]
row = duckdb.connect().execute(
    """
    SELECT sim_step, timestamp, action.ctrl, visual_observation
    FROM read_parquet(?)
    WHERE visual_observation IS NOT NULL
    ORDER BY visual_observation.rgb_frame_index
    LIMIT 1
    """,
    [str(trajectory)],
).fetchone()
frame_index = row[3]["rgb_frame_index"]
video = args.dataset / episode["video_files"][args.camera]["path"]
reader = imageio.get_reader(str(video), format="ffmpeg")
try:
    frame = reader.get_data(frame_index)
finally:
    reader.close()
print(
    f"frame={frame_index} shape={frame.shape} sim_step={row[0]} "
    f"timestamp={row[1]:.6f} action.ctrl={row[2]}"
)
