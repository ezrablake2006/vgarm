from __future__ import annotations

import time
from typing import Callable


class ViewerClosed(RuntimeError):
    """Raised when the user closes a passive MuJoCo viewer."""


class SimulationSession:
    """Own one passive viewer and synchronize it from the physics step loop."""

    def __init__(
        self,
        model,
        data,
        *,
        enabled: bool = False,
        speed: float = 1.0,
        viewer_factory: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if speed <= 0:
            raise ValueError("viewer speed must be greater than zero")
        self.model = model
        self.data = data
        self.enabled = enabled
        self.speed = speed
        self._viewer_factory = viewer_factory
        self._clock = clock
        self._sleeper = sleeper
        self._context = None
        self.viewer = None
        self._last_sync: float | None = None

    def __enter__(self) -> "SimulationSession":
        if not self.enabled:
            return self
        if self._viewer_factory is None:
            from mujoco import viewer

            self._viewer_factory = viewer.launch_passive
        handle = self._viewer_factory(self.model, self.data)
        if hasattr(handle, "__enter__"):
            self._context = handle
            self.viewer = handle.__enter__()
        else:
            self.viewer = handle
        self._last_sync = self._clock()
        return self

    def sync(self) -> None:
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            raise ViewerClosed("MuJoCo viewer was closed by the user")
        self.viewer.sync()
        target_period = float(self.model.opt.timestep) / self.speed
        now = self._clock()
        if self._last_sync is not None:
            remaining = target_period - (now - self._last_sync)
            if remaining > 0:
                self._sleeper(remaining)
                now = self._clock()
        self._last_sync = now

    def wait_until_closed(self, step: Callable[[], None]) -> None:
        while self.viewer is not None and self.viewer.is_running():
            step()

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        elif self.viewer is not None and hasattr(self.viewer, "close"):
            self.viewer.close()
        self._context = None
        self.viewer = None

    def __exit__(self, *_args) -> None:
        self.close()
