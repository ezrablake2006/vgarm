from __future__ import annotations

from fractions import Fraction
import math
import os
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from .util import sha256_file


class DatasetConfigurationError(ValueError):
    """A predictable user-supplied dataset configuration error."""


class RgbDependencyError(DatasetConfigurationError):
    pass


class RationalFrameScheduler:
    """First physics row at/after n/fps, without floating point drift."""

    def __init__(self, physics_timestep: float, fps: float):
        if physics_timestep <= 0 or fps <= 0:
            raise ValueError("physics timestep and visual FPS must be positive")
        self.timestep = Fraction(str(physics_timestep))
        self.fps = Fraction(str(fps))
        self.next_frame = 0

    def sample_index(self, sim_step: int) -> int | None:
        if sim_step < 0:
            raise ValueError("sim_step must be non-negative")
        if sim_step * self.timestep < Fraction(self.next_frame, 1) / self.fps:
            return None
        result = self.next_frame
        self.next_frame += 1
        return result


def require_rgb_dependencies():
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RgbDependencyError(
            'RGB recording requires: pip install "vgarm[rgb]"'
        ) from error
    return imageio


def _close_ffmpeg_resource(resource, generator_attribute: str) -> None:
    """Close imageio and the subprocess pipes retained by imageio-ffmpeg.

    imageio-ffmpeg 0.6.x leaves already-exited subprocess pipe objects open
    because its generator finalizer only closes them while the process is
    running. Python 3.14 reports those descriptors as ResourceWarning. Capture
    the subprocess before closing the generator, then deterministically close
    every pipe regardless of process state.
    """
    generator = getattr(resource, generator_attribute, None)
    process = None
    frame = getattr(generator, "gi_frame", None)
    if frame is not None:
        process = frame.f_locals.get("process") or frame.f_locals.get("p")
    try:
        resource.close()
    finally:
        if process is not None:
            for pipe_name in ("stdin", "stdout", "stderr"):
                pipe = getattr(process, pipe_name, None)
                if pipe is not None and not pipe.closed:
                    pipe.close()
            if process.poll() is None:
                process.wait(timeout=5)


def _close_ffmpeg_generator(generator) -> None:
    frame = getattr(generator, "gi_frame", None)
    process = frame.f_locals.get("process") if frame is not None else None
    try:
        generator.close()
    finally:
        if process is not None:
            for pipe_name in ("stdin", "stdout", "stderr"):
                pipe = getattr(process, pipe_name, None)
                if pipe is not None and not pipe.closed:
                    pipe.close()
            if process.poll() is None:
                process.wait(timeout=5)


DEFAULT_VISUAL_CHUNK_FRAMES = 64


def validate_visual_configuration(
    modalities: tuple[str, ...], cameras: tuple[str, ...],
    width: int, height: int, fps: float,
    chunk_frames: int = DEFAULT_VISUAL_CHUNK_FRAMES,
) -> None:
    supported = ("state", "rgb", "depth", "segmentation")
    unknown = [item for item in modalities if item not in supported]
    if unknown:
        raise DatasetConfigurationError(
            "unsupported modalities: " + ", ".join(sorted(set(unknown)))
        )
    if len(set(modalities)) != len(modalities):
        raise DatasetConfigurationError("--modalities contains duplicate values")
    if "state" not in modalities:
        raise DatasetConfigurationError("the state modality is required")
    visual = any(item != "state" for item in modalities)
    if visual and not cameras:
        raise DatasetConfigurationError("visual modalities require --cameras")
    if len(set(cameras)) != len(cameras):
        raise DatasetConfigurationError("--cameras contains duplicate values")
    if not visual:
        return
    valid_width = isinstance(width, int) and not isinstance(width, bool) and width > 0
    valid_height = isinstance(height, int) and not isinstance(height, bool) and height > 0
    if "rgb" in modalities and (
        not valid_width or not valid_height or width % 2 or height % 2
    ):
        raise DatasetConfigurationError(
            "RGB width and height must be positive even integers for H.264 "
            f"yuv420p; received {width}x{height}"
        )
    if not valid_width:
        raise DatasetConfigurationError("--visual-width must be a positive integer")
    if not valid_height:
        raise DatasetConfigurationError("--visual-height must be a positive integer")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise DatasetConfigurationError("--visual-fps must be greater than zero")
    if not isinstance(chunk_frames, int) or chunk_frames <= 0:
        raise DatasetConfigurationError("--visual-chunk-frames must be positive")


# v0.4.0 public compatibility.
validate_rgb_configuration = validate_visual_configuration


def named_cameras(model) -> dict[str, int]:
    return {
        name: camera_id for camera_id in range(model.ncam)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id))
    }


def validate_camera_names(model, requested: tuple[str, ...]) -> dict[str, int]:
    available = named_cameras(model)
    missing = [name for name in requested if name not in available]
    if missing:
        choices = ", ".join(sorted(available)) or "(none)"
        label = (
            f"unknown visual camera '{missing[0]}'" if len(missing) == 1
            else "unknown visual cameras " + ", ".join(f"'{x}'" for x in missing)
        )
        legacy = (
            f"; unknown RGB camera '{missing[0]}'"
            if len(missing) == 1 else ""
        )
        raise DatasetConfigurationError(
            f"{label}{legacy}; available cameras: {choices}"
        )
    return {name: available[name] for name in requested}


def probe_video(path: Path) -> dict:
    require_rgb_dependencies()
    import imageio_ffmpeg

    count, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    if count <= 0:
        raise ValueError(f"video has no decodable frames: {path}")
    indices = sorted({0, count // 2, count - 1})
    generator = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(generator)
        width, height = metadata["size"]
        sampled = {}
        wanted = set(indices)
        for index, frame_bytes in enumerate(generator):
            if index in wanted:
                sampled[index] = np.frombuffer(
                    frame_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
            if index >= indices[-1]:
                break
        frames = [sampled[index] for index in indices]
        if any(x.shape != (height, width, 3) or x.dtype != np.uint8 for x in frames):
            raise ValueError(f"video contains invalid RGB frames: {path}")
        return {"frame_count": count, "width": width, "height": height,
                "fps": float(metadata.get("fps", 0.0)),
                "decoded_indices": indices}
    finally:
        _close_ffmpeg_generator(generator)


class _ChunkWriter:
    def __init__(self, directory: Path, modality: str, chunk_frames: int):
        self.directory = directory / modality
        self.directory.mkdir(parents=True, exist_ok=True)
        self.modality, self.limit = modality, chunk_frames
        self.items: list[dict] = []
        self.shards: list[dict] = []

    def append(self, metadata: dict, **arrays) -> None:
        self.items.append({**metadata, **arrays})
        if len(self.items) == self.limit:
            self._flush()

    def _flush(self) -> None:
        if not self.items:
            return
        index = len(self.shards)
        final = self.directory / f"chunk_{index:06d}.npz"
        temporary = self.directory / f"chunk_{index:06d}.tmp.npz"
        keys = self.items[0].keys()
        values = {key: np.asarray([item[key] for item in self.items]) for key in keys}
        try:
            np.savez_compressed(temporary, **values)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            with temporary.open("rb") as stream, np.load(
                stream, allow_pickle=False) as loaded:
                for key, value in values.items():
                    if (
                        loaded[key].shape != value.shape
                        or loaded[key].dtype != value.dtype
                    ):
                        raise RuntimeError(
                            f"invalid {self.modality} chunk {index}")
            os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.shards.append({
            "path": str(final), "frame_count": len(self.items),
            "first_frame_index": int(values["frame_index"][0]),
            "last_frame_index": int(values["frame_index"][-1]),
            "sha256": sha256_file(final), "bytes": final.stat().st_size,
        })
        self.items.clear()

    def close(self) -> list[dict]:
        self._flush()
        return self.shards


class EpisodeVisualRecorder:
    """One scheduler and one renderer lifecycle for every camera/modality."""

    def __init__(self, model, data, directory: Path, *, modalities, cameras,
                 width, height, fps,
                 chunk_frames=DEFAULT_VISUAL_CHUNK_FRAMES,
                 renderer_factory=None,
                 writer_factory=None, video_directory=None):
        validate_visual_configuration(
            modalities, cameras, width, height, fps, chunk_frames
        )
        self.model, self.data = model, data
        self.modalities, self.cameras = modalities, cameras
        self.width, self.height, self.fps = width, height, fps
        self.camera_ids = validate_camera_names(model, cameras)
        self.scheduler = RationalFrameScheduler(float(model.opt.timestep), fps)
        self.frame_count, self.closed = 0, False
        make_renderer = renderer_factory or (
            lambda m, h, w: mujoco.Renderer(m, height=h, width=w)
        )
        self.renderer = make_renderer(model, height, width)
        self.rgb_paths, self.rgb_writers = {}, {}
        self.chunks = {}
        try:
            if "rgb" in modalities:
                imageio = require_rgb_dependencies()
                video_dir = Path(video_directory) if video_directory else directory / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                make_writer = writer_factory or (
                    lambda path: imageio.get_writer(
                        str(path), format="ffmpeg", mode="I", fps=fps,
                        codec="libx264", pixelformat="yuv420p",
                        macro_block_size=1, ffmpeg_log_level="error")
                )
                for camera in cameras:
                    path = video_dir / f"{camera}.tmp.mp4"
                    self.rgb_paths[camera] = path
                    self.rgb_writers[camera] = make_writer(path)
            for camera in cameras:
                base = directory / "arrays" / camera
                for modality in ("depth", "segmentation"):
                    if modality in modalities:
                        self.chunks[camera, modality] = _ChunkWriter(
                            base, modality, chunk_frames
                        )
        except Exception:
            self.abort()
            raise

    def sample(self, sim_step: int, timestamp: float) -> dict | None:
        frame_index = self.scheduler.sample_index(sim_step)
        if frame_index is None:
            return None
        if frame_index != self.frame_count:
            raise RuntimeError("visual frame scheduler produced a discontinuity")
        meta = {"frame_index": np.int64(frame_index),
                "timestamp": np.float64(timestamp),
                "physics_row": np.int64(sim_step)}
        for camera in self.cameras:
            self.renderer.update_scene(self.data, camera=camera)
            if "rgb" in self.modalities:
                self.renderer.disable_depth_rendering()
                self.renderer.disable_segmentation_rendering()
                rgb = np.asarray(self.renderer.render())
                if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                    raise RuntimeError(f"camera {camera} returned invalid RGB")
                self.rgb_writers[camera].append_data(rgb)
            if "depth" in self.modalities:
                self.renderer.disable_segmentation_rendering()
                self.renderer.enable_depth_rendering()
                depth = np.asarray(self.renderer.render(), dtype=np.float32)
                if depth.shape != (self.height, self.width):
                    raise RuntimeError(f"camera {camera} returned invalid depth")
                self.chunks[camera, "depth"].append(meta, depth_m=depth)
            if "segmentation" in self.modalities:
                self.renderer.disable_depth_rendering()
                self.renderer.enable_segmentation_rendering()
                seg = np.asarray(self.renderer.render(), dtype=np.int32)
                if seg.shape != (self.height, self.width, 2):
                    raise RuntimeError(f"camera {camera} returned invalid segmentation")
                self.chunks[camera, "segmentation"].append(
                    meta, object_id=seg[..., 0], object_type=seg[..., 1])
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.frame_count += 1
        return {"frame_index": frame_index, "timestamp": float(timestamp)}

    def close(self) -> dict:
        errors = []
        for writer in self.rgb_writers.values():
            try:
                _close_ffmpeg_resource(writer, "_write_gen")
            except Exception as error:
                errors.append(str(error))
        arrays = {}
        for key, writer in self.chunks.items():
            arrays.setdefault(key[0], {})[key[1]] = writer.close()
        self.renderer.close()
        self.closed = True
        if errors:
            raise RuntimeError("video encoder close failed: " + "; ".join(errors))
        videos = {}
        for camera, path in self.rgb_paths.items():
            info = probe_video(path)
            if info["frame_count"] != self.frame_count:
                raise RuntimeError(f"camera {camera}: RGB frame count mismatch")
            cid = self.camera_ids[camera]
            videos[camera] = {
                **info, "camera": camera, "camera_id": cid,
                "requested_fps": self.fps, "codec": "h264",
                "pixel_format": "yuv420p", "container": "mp4",
                "fovy": float(self.model.cam_fovy[cid]),
                "temporary_path": str(path)}
        return {"videos": videos, "arrays": arrays}

    def abort(self) -> None:
        if getattr(self, "closed", False):
            return
        for writer in getattr(self, "rgb_writers", {}).values():
            try:
                _close_ffmpeg_resource(writer, "_write_gen")
            except Exception:
                pass
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        self.closed = True


class EpisodeRgbRecorder(EpisodeVisualRecorder):
    """Compatibility facade for the v0.4.0 RGB-only API."""

    def __init__(self, model, data, directory, *, cameras, width, height, fps,
                 renderer_factory=None, writer_factory=None):
        super().__init__(
            model, data, Path(directory).parent,
            modalities=("state", "rgb"), cameras=cameras, width=width,
            height=height, fps=fps, renderer_factory=renderer_factory,
            writer_factory=writer_factory, video_directory=directory)

    def sample(self, sim_step, timestamp):
        item = super().sample(sim_step, timestamp)
        return None if item is None else {
            "rgb_frame_index": item["frame_index"],
            "rgb_timestamp": item["timestamp"]}

    def close(self):
        return super().close()["videos"]
