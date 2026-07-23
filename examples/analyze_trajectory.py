"""Print task/phase counts from a native VGArm trajectory dataset."""

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
args = parser.parse_args()
episodes = [
    json.loads(line)
    for line in (args.dataset / "meta" / "episodes.jsonl").read_text().splitlines()
    if line.strip()
]
tasks = Counter(item["task_id"] for item in episodes)
phases = Counter()
connection = duckdb.connect()
for episode in episodes:
    path = args.dataset / episode["trajectory_file"]
    for phase, count in connection.execute(
        "SELECT control.phase, count(*) FROM read_parquet(?) GROUP BY 1",
        [str(path)],
    ).fetchall():
        phases[phase or "unlabelled"] += count
print("tasks", dict(tasks))
print("phases", dict(phases))

