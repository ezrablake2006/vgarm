"""Read raw v1.2 visual arrays without knowing the chunk layout."""

import argparse
import numpy as np

from vgarm.trajectory.reader import open_episode


parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--episode-id", type=int, default=0)
parser.add_argument("--camera", default="camera_front")
args = parser.parse_args()

episode = open_episode(args.dataset, args.episode_id)
shards = episode.metadata["array_files"][args.camera]
modality = "depth" if "depth" in shards else "segmentation"
count = sum(item["frame_count"] for item in shards[modality])
manifest = (
    episode.segmentation_manifest()
    if "segmentation" in shards else {"labels": []}
)
names = {
    (item["object_id"], item["object_type"]): item.get("name")
    for item in manifest["labels"]
}

for index in sorted({0, count // 2, count - 1}):
    if "depth" in shards:
        item = episode.read_depth(camera=args.camera, frame_index=index)
        depth = item["depth_m"]
        print(index, "depth", depth.shape, depth.dtype,
              float(depth.min()), float(depth.max()), "meter")
    if "segmentation" in shards:
        item = episode.read_segmentation(camera=args.camera, frame_index=index)
        pairs = sorted(set(zip(
            item["object_id"].reshape(-1).tolist(),
            item["object_type"].reshape(-1).tolist())))
        print(index, "segmentation", item["object_id"].shape,
              item["object_id"].dtype,
              [(pair, names.get(pair)) for pair in pairs])
    print("mapping", item["physics_row"], item["timestamp"])
