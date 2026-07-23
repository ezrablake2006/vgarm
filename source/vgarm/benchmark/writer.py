from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import TextIO

from .models import BenchmarkConfig, EpisodeResult


EPISODE_FIELDS = [
    "episode_id", "episode_seed", "robot", "task_id", "instruction", "scene",
    "program_version", "started_at", "duration_seconds", "object_positions",
    "initial_object_positions", "planned_target_position",
    "original_target_position", "transport_waypoints", "final_object_positions",
    "held_object", "obstacle_objects", "path_diagnostics",
    "collision_diagnostics", "grasp_diagnostics",
    "object_geometry",
    "parse_success", "pick_attempted", "pick_success", "place_attempted",
    "place_success", "task_success", "failure_stage", "failure_category",
    "failure_reason", "verification", "traceback",
]


class ResultWriter:
    def __init__(self, config: BenchmarkConfig):
        self.output = config.output
        if self.output.exists() and any(self.output.iterdir()) and not config.overwrite:
            raise FileExistsError(f"output directory is not empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        config_payload = asdict(config)
        config_payload["scene"] = str(config.scene)
        config_payload["output"] = str(config.output)
        (self.output / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self._jsonl: TextIO = (self.output / "episodes.jsonl").open("w", encoding="utf-8")
        self._csv_file: TextIO = (self.output / "episodes.csv").open("w", encoding="utf-8", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=EPISODE_FIELDS)
        self._csv.writeheader()
        self._log: TextIO = (self.output / "benchmark.log").open("w", encoding="utf-8")

    def write_episode(self, result: EpisodeResult) -> None:
        payload = result.to_dict()
        self._jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        csv_payload = payload.copy()
        for field in (
            "object_positions", "initial_object_positions",
            "transport_waypoints", "final_object_positions", "obstacle_objects",
            "path_diagnostics", "collision_diagnostics", "grasp_diagnostics",
            "object_geometry", "verification",
        ):
            csv_payload[field] = json.dumps(csv_payload[field], ensure_ascii=False)
        self._csv.writerow(csv_payload)
        self._csv_file.flush()

    def log(self, message: str) -> None:
        self._log.write(message + "\n")
        self._log.flush()

    def write_summary(self, summary: dict) -> None:
        (self.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        columns = [
            "robot", "episodes", "parse_success_rate", "pick_success_rate",
            "place_success_rate", "overall_success_rate", "collision_rate",
            "timeout_rate", "average_duration_seconds",
        ]
        with (self.output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for robot, metrics in summary["by_robot"].items():
                writer.writerow({"robot": robot, **{key: metrics[key] for key in columns[1:]}})
        def pct(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.1%}"
        lines = [
            "| Robot | Episodes | Parse success | Pick success | Place success | Overall success | Collision rate | Timeout rate | Average duration |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for robot, metrics in summary["by_robot"].items():
            duration = metrics["average_duration_seconds"]
            duration_text = "N/A" if duration is None else f"{duration:.3f}s"
            lines.append(
                f"| {robot} | {metrics['episodes']} | {pct(metrics['parse_success_rate'])} | "
                f"{pct(metrics['pick_success_rate'])} | {pct(metrics['place_success_rate'])} | "
                f"{pct(metrics['overall_success_rate'])} | {pct(metrics['collision_rate'])} | "
                f"{pct(metrics['timeout_rate'])} | {duration_text} |"
            )
        (self.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        task_summary = summary.get("by_task", {})
        (self.output / "summary_by_task.json").write_text(
            json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        task_columns = [
            "task_id", "episodes", "parse_success_rate", "pick_success_rate",
            "place_success_rate", "overall_success_rate", "collision_rate",
            "timeout_rate", "average_duration_seconds", "median_duration_seconds",
        ]
        with (self.output / "summary_by_task.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=task_columns)
            writer.writeheader()
            for task_id, metrics in task_summary.items():
                writer.writerow({"task_id": task_id, **{
                    key: metrics[key] for key in task_columns[1:]
                }})
        task_lines = [
            "| Task | Episodes | Parse success | Pick success | Place success | Overall success | Collision rate | Timeout rate | Average duration | Median duration |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for task_id, metrics in task_summary.items():
            avg = metrics["average_duration_seconds"]
            med = metrics["median_duration_seconds"]
            task_lines.append(
                f"| {task_id} | {metrics['episodes']} | {pct(metrics['parse_success_rate'])} | "
                f"{pct(metrics['pick_success_rate'])} | {pct(metrics['place_success_rate'])} | "
                f"{pct(metrics['overall_success_rate'])} | {pct(metrics['collision_rate'])} | "
                f"{pct(metrics['timeout_rate'])} | "
                f"{'N/A' if avg is None else f'{avg:.3f}s'} | "
                f"{'N/A' if med is None else f'{med:.3f}s'} |"
            )
        (self.output / "summary_by_task.md").write_text(
            "\n".join(task_lines) + "\n", encoding="utf-8"
        )
        all_tasks = sorted(task_summary)
        with (self.output / "summary_robot_task_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["robot", *all_tasks])
            for robot, task_metrics in summary.get("by_robot_task", {}).items():
                writer.writerow([
                    robot,
                    *[
                        task_metrics.get(task, {}).get("overall_success_rate")
                        for task in all_tasks
                    ],
                ])

    def close(self) -> None:
        self._jsonl.close()
        self._csv_file.close()
        self._log.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
