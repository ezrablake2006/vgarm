"""Load the first native VGArm trajectory without materializing other episodes."""

import argparse
import json
from pathlib import Path

import duckdb


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--episode-id", type=int, default=0)
args = parser.parse_args()
metadata = [
    json.loads(line)
    for line in (args.dataset / "meta" / "episodes.jsonl").read_text().splitlines()
    if line.strip()
]
episode = next(item for item in metadata if item["episode_id"] == args.episode_id)
path = args.dataset / episode["trajectory_file"]
row = duckdb.connect().execute(
    "SELECT * FROM read_parquet(?) LIMIT 1", [str(path)]
).fetchone()
print(f"episode={episode['episode_id']} steps={episode['num_steps']}")
print(f"first row columns={len(row)} trajectory={path}")

