"""Config → the objects one camera worker runs. Wiring, and nothing else.

Split from `worker` so the loop can be tested without a network and the wiring can be
tested without a loop. Everything here is constructed but not started: a `FrameSource`
opens its socket when it is iterated, so building a pipeline is cheap and side-effect
free — which is what lets these paths be exercised in CI with no camera on the LAN.

Errors raised here are deliberately generic about *values*: an RTSP URL carries the
camera's credentials, so a message may name the camera, the env var, or the source kind,
and never the URL itself (ADR-0005, CLAUDE.md).

Implements P2.7 (engine-architecture.md §9).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from muster.analytics.geometry import GeometryAnalytics
from muster.analytics.site_geometry import SiteGeometry
from muster.config.schema import CameraConfig, MusterConfig, RtspSource
from muster.detector.detector import Detector
from muster.detector.model_manager import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_CACHE,
    MODEL_CACHE_ENV_VAR,
    ModelManager,
)
from muster.detector.onnx_detector import OnnxDetector
from muster.errors import ConfigError
from muster.ingest.rtsp import RtspFrameSource
from muster.ingest.source import FrameSource
from muster.sampler.sampler import FrameSampler
from muster.tracker.bytetrack import ByteTrackTracker
from muster.tracker.tracker import Tracker
from muster.types import CameraId


@dataclass(frozen=True, slots=True)
class CameraPipeline:
    """Everything one worker needs, already wired to one camera's config."""

    camera_id: CameraId
    source: FrameSource
    sampler: FrameSampler
    detector: Detector
    tracker: Tracker
    analytics: GeometryAnalytics


def build_pipeline(config: MusterConfig, camera_id: CameraId) -> CameraPipeline:
    """Construct one camera's pipeline, or refuse with a reason that names no secret."""
    camera = _camera(config, camera_id)
    cache_dir = Path(os.environ.get(MODEL_CACHE_ENV_VAR, DEFAULT_MODEL_CACHE)).expanduser()
    model = camera.detector.model or DEFAULT_MODEL
    return CameraPipeline(
        camera_id=camera_id,
        source=_source(camera),
        # Start at the ceiling and let backpressure find the rate the box can hold.
        sampler=FrameSampler(target_fps=config.budget.fps_max),
        detector=OnnxDetector(
            ModelManager(cache_dir).ensure(model).path,
            confidence=config.thresholds.detection_confidence,
        ),
        tracker=ByteTrackTracker(
            track_thresh=config.thresholds.track_thresh,
            max_cost=config.thresholds.max_cost,
            max_cost_low=config.thresholds.max_cost_low,
            track_memory_s=config.thresholds.track_memory_s,
        ),
        analytics=GeometryAnalytics(
            SiteGeometry.compile(config), dwell_min_s=config.thresholds.dwell_min_s
        ),
    )


def _camera(config: MusterConfig, camera_id: CameraId) -> CameraConfig:
    for camera in config.cameras:
        if camera.camera_id == camera_id:
            return camera
    msg = f"no camera {camera_id!r} in config"
    raise ConfigError(msg)


def _source(camera: CameraConfig) -> FrameSource:
    if not isinstance(camera.source, RtspSource):
        msg = (
            f"camera {camera.camera_id!r}: source kind {camera.source.kind.value!r} "
            "is not supported by the engine yet"
        )
        raise ConfigError(msg)
    return RtspFrameSource(camera.camera_id, _url(camera, camera.source))


def _url(camera: CameraConfig, source: RtspSource) -> str:
    if source.url is not None:
        return source.url
    if source.url_env is None:  # pragma: no cover - the schema requires exactly one
        msg = f"camera {camera.camera_id!r}: neither url nor url_env is set"
        raise ConfigError(msg)
    url = os.environ.get(source.url_env)
    if not url:
        # Names the variable, never its value: that value is a credential.
        msg = f"camera {camera.camera_id!r}: ${source.url_env} is unset or empty"
        raise ConfigError(msg)
    return url
