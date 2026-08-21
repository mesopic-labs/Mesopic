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

from mesopic.analytics.geometry import GeometryAnalytics
from mesopic.analytics.site_geometry import SiteGeometry
from mesopic.config.schema import CameraConfig, FrigateSource, MesopicConfig, RtspSource
from mesopic.detector.detector import Detector
from mesopic.detector.model_manager import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_CACHE,
    MODEL_CACHE_ENV_VAR,
    ModelManager,
)
from mesopic.detector.onnx_detector import OnnxDetector
from mesopic.errors import ConfigError
from mesopic.ingest.frigate import FrigateObjects, FrigateTrackSource
from mesopic.ingest.rtsp import RtspFrameSource
from mesopic.ingest.source import FrameSource, TrackSource
from mesopic.sampler.sampler import FrameSampler
from mesopic.tracker.bytetrack import ByteTrackTracker
from mesopic.tracker.tracker import Tracker
from mesopic.types import CameraId


@dataclass(frozen=True, slots=True)
class CameraPipeline:
    """Everything one worker needs, already wired to one camera's config."""

    camera_id: CameraId
    source: FrameSource
    sampler: FrameSampler
    detector: Detector
    tracker: Tracker
    analytics: GeometryAnalytics


@dataclass(frozen=True, slots=True)
class TrackPipeline:
    """A camera whose upstream already detected and tracked (P4.3).

    No detector and no tracker, because there is nothing to run them on — the absence is
    the point, and a pipeline carrying unused ones would invite somebody to use them.
    """

    camera_id: CameraId
    source: TrackSource
    sampler: FrameSampler
    analytics: GeometryAnalytics


def build_pipeline(config: MesopicConfig, camera_id: CameraId) -> CameraPipeline | TrackPipeline:
    """Construct one camera's pipeline, or refuse with a reason that names no secret."""
    camera = _camera(config, camera_id)
    if isinstance(camera.source, FrigateSource):
        return _frigate_pipeline(config, camera, camera.source)
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


def _frigate_pipeline(
    config: MesopicConfig, camera: CameraConfig, source: FrigateSource
) -> TrackPipeline:
    """Wire a Frigate camera. The broker is site-wide; the topic and name are the camera's."""
    if not config.frigate.broker:  # pragma: no cover - the schema validator rejects this
        msg = f"camera {camera.camera_id!r}: no frigate.broker is configured"
        raise ConfigError(msg)
    width, height = camera.reference_resolution
    return TrackPipeline(
        camera_id=camera.camera_id,
        source=FrigateTrackSource(
            camera.camera_id,
            broker=config.frigate.broker,
            port=config.frigate.port,
            topic=source.mqtt_topic,
            credentials=_credentials(config),
            objects=FrigateObjects(
                camera.camera_id,
                # Frigate's own name for the camera, which the shared events topic makes
                # load-bearing: it is the only thing saying whose a message is.
                frigate_camera=source.camera or camera.camera_id,
                width=width,
                height=height,
            ),
        ),
        sampler=FrameSampler(target_fps=config.budget.fps_max),
        analytics=GeometryAnalytics(
            SiteGeometry.compile(config), dwell_min_s=config.thresholds.dwell_min_s
        ),
    )


def _credentials(config: MesopicConfig) -> tuple[str, str] | None:
    """Broker credentials, by env-var reference. Names a variable, never its value."""
    if config.frigate.username_env is None or config.frigate.password_env is None:
        return None
    username = os.environ.get(config.frigate.username_env)
    password = os.environ.get(config.frigate.password_env)
    if not username or not password:
        msg = (
            f"frigate broker credentials: ${config.frigate.username_env} or "
            f"${config.frigate.password_env} is unset or empty"
        )
        raise ConfigError(msg)
    return (username, password)


def _camera(config: MesopicConfig, camera_id: CameraId) -> CameraConfig:
    for camera in config.cameras:
        if camera.camera_id == camera_id:
            return camera
    msg = f"no camera {camera_id!r} in config"
    raise ConfigError(msg)


def _source(camera: CameraConfig) -> FrameSource:
    if not isinstance(camera.source, RtspSource):
        # ONVIF only: Frigate returned above, and its absence here is what keeps this
        # branch honest about what is actually unbuilt.
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
