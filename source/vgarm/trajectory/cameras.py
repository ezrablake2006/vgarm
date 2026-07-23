from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


class DatasetConfigurationError(ValueError):
    """A predictable user-supplied dataset configuration error."""


class RgbDependencyError(DatasetConfigurationError):
    pass


class RationalFrameScheduler:
    """Select the first physics row at or after each ideal RGB timestamp.

    Frame zero is sampled at physics row zero. Integer fractions avoid both a
    divisibility assumption and long-running floating-point phase drift.
    """

    def __init__(self, physics_timestep: float, fps: float):
        if physics_timestep <= 0 or fps <= 0:
            raise ValueError("physics timestep and RGB FPS must be positive")
        self.timestep = Fraction(str(physics_timestep))
        self.fps = Fraction(str(fps))
        self.next_frame = 0

    def sample_index(self, sim_step: int) -> int | None:
        if sim_step < 0:
            raise ValueError("sim_step must be non-negative")
        elapsed = sim_step * self.timestep
        target = Fraction(self.next_frame, 1) / self.fps
        if elapsed < target:
            return None
        index = self.next_frame
        self.next_frame += 1
        return index


def require_rgb_dependencies():
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RgbDependencyError(
            'RGB recording requires: pip install "vgarm[rgb]"'
        ) from error
    return imageio


def validate_rgb_configuration(
    modalities: tuple[str, ...],
    cameras: tuple[str, ...],
    width: int,
    height: int,
    fps: float,
) -> None:
    supported = {"state", "rgb"}
    unsupported = set(modalities) - supported
    if unsupported:
        raise DatasetConfigurationError(
            "unsupported modalities: " + ", ".join(sorted(unsupported))
        )
    if "state" not in modalities:
        raise DatasetConfigurationError("the state modality is required")
    if len(set(modalities)) != len(modalities):
        raise DatasetConfigurationError("--modalities contains duplicate values")
    if "rgb" not in modalities:
        return
    if not cameras:
        raise DatasetConfigurationError("--modalities rgb requires --cameras")
    if len(set(cameras)) != len(cameras):
        raise DatasetConfigurationError("--cameras contains duplicate values")
    valid_dimensions = (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
        and width > 0
        and height > 0
        and width % 2 == 0
        and height % 2 == 0
    )
    if not valid_dimensions:
        raise DatasetConfigurationError(
            "RGB width and height must be positive even integers for H.264 "
            f"yuv420p; received {width}x{height}"
        )
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise DatasetConfigurationError("--rgb-fps must be greater than zero")


def named_cameras(model) -> dict[str, int]:
    return {
        name: camera_id
        for camera_id in range(model.ncam)
        if (name := mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
        ))
    }


def validate_camera_names(model, requested: tuple[str, ...]) -> dict[str, int]:
    available = named_cameras(model)
    missing = [name for name in requested if name not in available]
    if missing:
        choices = ", ".join(sorted(available)) or "(none)"
        label = (
            f"unknown RGB camera '{missing[0]}'"
            if len(missing) == 1
            else "unknown RGB cameras " + ", ".join(f"'{name}'" for name in missing)
        )
        raise DatasetConfigurationError(
            f"{label}; available cameras: {choices}"
        )
    return {name: available[name] for name in requested}


def probe_video(path: Path) -> dict:
    imageio = require_rgb_dependencies()
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        frame_count = int(reader.count_frames())
        if frame_count <= 0:
            raise ValueError(f"video has no decodable frames: {path}")
        indices = sorted({0, frame_count // 2, frame_count - 1})
        decoded = [np.asarray(reader.get_data(index)) for index in indices]
        height, width = decoded[0].shape[:2]
        if any(
            frame.shape != (height, width, 3) or frame.dtype != np.uint8
            for frame in decoded
        ):
            raise ValueError(f"video contains invalid RGB frames: {path}")
        metadata = reader.get_meta_data()
        return {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": float(metadata.get("fps", 0.0)),
            "decoded_indices": indices,
        }
    finally:
        reader.close()


class EpisodeRgbRecorder:
    def __init__(
        self,
        model,
        data,
        directory: Path,
        *,
        cameras: tuple[str, ...],
        width: int,
        height: int,
        fps: float,
        renderer_factory: Callable | None = None,
        writer_factory: Callable | None = None,
    ):
        validate_rgb_configuration(("state", "rgb"), cameras, width, height, fps)
        self.imageio = require_rgb_dependencies()
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_ids = validate_camera_names(model, cameras)
        self.scheduler = RationalFrameScheduler(float(model.opt.timestep), fps)
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = {
            name: directory / f"{name}.tmp.mp4" for name in cameras
        }
        make_renderer = renderer_factory or (
            lambda current_model, h, w: mujoco.Renderer(
                current_model, height=h, width=w
            )
        )
        self.renderer = None
        self.writers = {}
        make_writer = writer_factory or (
            lambda path: self.imageio.get_writer(
                str(path),
                format="ffmpeg",
                mode="I",
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
        )
        try:
            self.renderer = make_renderer(model, height, width)
            self.writers = {
                name: make_writer(path) for name, path in self.paths.items()
            }
        except Exception:
            for writer in self.writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            if self.renderer is not None:
                self.renderer.close()
            raise
        self.frame_count = 0
        self.closed = False

    def sample(self, sim_step: int, timestamp: float) -> dict | None:
        frame_index = self.scheduler.sample_index(sim_step)
        if frame_index is None:
            return None
        if frame_index != self.frame_count:
            raise RuntimeError("RGB frame scheduler produced a discontinuity")
        for name, writer in self.writers.items():
            self.renderer.update_scene(self.data, camera=name)
            frame = np.asarray(self.renderer.render())
            if frame.shape != (self.height, self.width, 3):
                raise RuntimeError(
                    f"camera {name} rendered {frame.shape}; expected "
                    f"{(self.height, self.width, 3)}"
                )
            if frame.dtype != np.uint8:
                raise RuntimeError(
                    f"camera {name} rendered {frame.dtype}; expected uint8"
                )
            writer.append_data(frame)
        self.frame_count += 1
        return {
            "rgb_frame_index": frame_index,
            "rgb_timestamp": float(timestamp),
        }

    def close(self) -> dict[str, dict]:
        if self.closed:
            raise RuntimeError("RGB recorder is already closed")
        errors = []
        for name, writer in self.writers.items():
            try:
                writer.close()
            except Exception as error:
                errors.append(f"{name}: {error}")
        self.renderer.close()
        self.closed = True
        if errors:
            raise RuntimeError("video encoder close failed: " + "; ".join(errors))
        result = {}
        for name, path in self.paths.items():
            probe = probe_video(path)
            if probe["frame_count"] != self.frame_count:
                raise RuntimeError(
                    f"camera {name}: encoded {probe['frame_count']} frames; "
                    f"expected {self.frame_count}"
                )
            camera_id = self.camera_ids[name]
            result[name] = {
                **probe,
                "camera": name,
                "camera_id": camera_id,
                "requested_fps": self.fps,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "container": "mp4",
                "fovy": float(self.model.cam_fovy[camera_id]),
                "model_position": np.asarray(
                    self.model.cam_pos[camera_id]
                ).tolist(),
                "model_quaternion": np.asarray(
                    self.model.cam_quat[camera_id]
                ).tolist(),
                "world_position_at_start": np.asarray(
                    self.data.cam_xpos[camera_id]
                ).tolist(),
                "world_rotation_at_start": np.asarray(
                    self.data.cam_xmat[camera_id]
                ).reshape(3, 3).tolist(),
                "temporary_path": str(path),
            }
        return result

    def abort(self) -> None:
        if self.closed:
            return
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:
                pass
        try:
            self.renderer.close()
        except Exception:
            pass
        self.closed = True
