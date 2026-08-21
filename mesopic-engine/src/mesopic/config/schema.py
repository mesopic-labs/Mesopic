"""The `mesopic.yaml` schema as pydantic models.

`mesopic.yaml` is the authoritative site description; the store's `cameras`/`zones`/
`lines` tables are its compiled form. Validation is strict and happens once, at load: a
bad config fails loud at startup rather than silently mis-counting.

Three conventions the models must enforce, not merely document:

* **Secrets are `*_env` references only.** An inline secret is a validation error.
* **All geometry is normalized** `[0, 1]`, so it survives a resolution change.
* **Referential integrity**: every `camera_id` on a line or zone must exist.

The first of those is carried by `extra="forbid"` rather than by a rule per secret: a
`secret` key next to `secret_env` is not a field this schema knows, so it is rejected
because it is *unknown*, and so is the next inline secret nobody has thought of yet.
Enumerating the forbidden spellings would only ever cover the ones already imagined.

Implements P2.1. Fields are the shape from engine-architecture.md §13.1.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mesopic.types import (
    CameraId,
    Direction,
    LineId,
    MetricName,
    SiteId,
    SourceKind,
    ZoneId,
    ZoneRole,
)

NormCoord = Annotated[float, Field(ge=0.0, le=1.0)]
"""One normalized axis value. The range check lives here so every point inherits it."""

NormalizedPoint = tuple[NormCoord, NormCoord]
"""``(x, y)`` in ``[0, 1]``, origin top-left — the only geometry this file accepts."""

PixelExtent = Annotated[int, Field(gt=0)]

_HOURS_IN_A_YEAR = 8760
"""Ceiling on the raw-event retention window. `events` is the short-retention log."""

_MAX_TCP_PORT = 65535


class ConfigSection(BaseModel):
    """Base for every section: closed to unknown keys, and immutable once loaded.

    ``frozen`` because a loaded config is a fact about the site, not a scratchpad — the
    hot-reload path (§13.1) validates a *new* tree and swaps it, so nothing needs to
    mutate this one in place.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Site and budget --------------------------------------------------------


class SiteConfig(ConfigSection):
    """Who this engine is. ``site_id`` is also the cloud-sync identity."""

    site_id: SiteId
    timezone: str = "UTC"
    """The zone every time the dashboard renders is shown in (P3.9).

    Display only, and that is the whole of it: buckets, the sync wire format, `/healthz`,
    `/api/metrics` and the CSV export are UTC regardless, and conversion happens at one
    boundary (`mesopic.api.board.clock_label`, plus `data-timezone` for the chart axes).

    Set to the shop's own zone rather than left at `UTC`: an operator asked whether 09:30
    was the lunch rush should not be doing the arithmetic. Both surfaces move together —
    a page with one clock localised and one not is worse than either alone."""

    @field_validator("timezone")
    @classmethod
    def _must_be_a_real_zone(cls, value: str) -> str:
        """A typo here would silently mislabel every hour the dashboard ever renders."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            msg = f"unknown IANA timezone {value!r}"
            raise ValueError(msg) from None
        return value


class BudgetConfig(ConfigSection):
    """The effective-fps envelope the supervisor schedules within (ADR-0003)."""

    cpu_budget: Annotated[float, Field(gt=0.0, le=1.0)] = 0.75
    fps_min: Annotated[float, Field(gt=0.0)] = 1.0
    fps_max: Annotated[float, Field(gt=0.0)] = 5.0

    @model_validator(mode="after")
    def _envelope_must_be_orderable(self) -> Self:
        if self.fps_min > self.fps_max:
            msg = f"fps_min ({self.fps_min}) is above fps_max ({self.fps_max})"
            raise ValueError(msg)
        return self


# --- Cameras ----------------------------------------------------------------


class RtspSource(ConfigSection):
    """A camera we decode ourselves. Prefer its sub-stream — see the worked example.

    The URL is the one credential the config file can legitimately hold, because it is
    also the address. `url_env` is the form a real deployment should use: an RTSP URL
    carries the camera's username and password, and a config with a live one in it
    cannot safely be committed, shared in an issue, or rendered on `/config`.
    """

    kind: Literal[SourceKind.RTSP]
    url: str | None = None
    url_env: str | None = None
    transport: Literal["tcp", "udp"] = "tcp"
    """TCP by default: UDP loss on a cheap LAN reads as detector noise (§4)."""

    @model_validator(mode="after")
    def _exactly_one_url_form(self) -> Self:
        """Both set is a silent "which one won?"; neither is a camera with no address."""
        if (self.url is None) == (self.url_env is None):
            msg = "an rtsp source needs exactly one of url or url_env"
            raise ValueError(msg)
        return self


class OnvifSource(ConfigSection):
    """A camera whose RTSP URI is resolved from its ONVIF profile at startup.

    Provisional shape: engine-architecture.md §13.1 names the `onvif` kind but does not
    spell out its keys, and `ingest.onvif_discovery` is still setup-time only. Whoever
    implements the ingest path owns extending this — the credential-by-env rule is the
    part that is not negotiable.
    """

    kind: Literal[SourceKind.ONVIF]
    host: str
    port: Annotated[int, Field(gt=0, le=65535)] = 80
    username_env: str
    password_env: str


class FrigateSource(ConfigSection):
    """Frigate has already detected; we consume its objects over MQTT (ADR-0006)."""

    kind: Literal[SourceKind.FRIGATE]
    mqtt_topic: str = "frigate/events"
    """Where Frigate publishes objects. Its default carries **every** camera on the box,
    which is why `camera` below is what decides whose messages these are."""
    camera: str | None = None
    """Frigate's own name for this camera, when it differs from `camera_id`.

    The events topic is shared, so the payload's `camera` field is the only thing that
    says which camera a message is about — the topic cannot. Defaults to `camera_id`,
    which is right whenever the two systems were named consistently.
    """


CameraSource = Annotated[RtspSource | OnvifSource | FrigateSource, Field(discriminator="kind")]


class DetectorOverrides(ConfigSection):
    """Per-camera overrides of the detector defaults (§6)."""

    model: str | None = None
    input_size: Annotated[int, Field(gt=0)] = 640
    roi_from_geometry: bool = True


class CameraConfig(ConfigSection):
    camera_id: CameraId
    name: str
    source: CameraSource
    reference_resolution: tuple[PixelExtent, PixelExtent]
    """The frame size this camera's geometry was authored against, for the record only —
    zones and lines are normalized, so a resolution change does not invalidate them."""
    stall_timeout_s: Annotated[float, Field(gt=0.0)] = 5.0
    detector: DetectorOverrides = Field(default_factory=DetectorOverrides)
    enabled: bool = True


# --- Geometry ---------------------------------------------------------------


class LineConfig(ConfigSection):
    """A directional counting line, in normalized coordinates."""

    line_id: LineId
    camera_id: CameraId
    a: NormalizedPoint
    b: NormalizedPoint
    positive_dir: Direction = Direction.IN
    metrics: list[MetricName] = Field(default_factory=list)

    @model_validator(mode="after")
    def _endpoints_must_differ(self) -> Self:
        """A zero-length segment has no side to cross, so it can never count."""
        if self.a == self.b:
            msg = f"line {self.line_id!r} has identical endpoints"
            raise ValueError(msg)
        return self


class ZoneConfig(ConfigSection):
    """A polygonal region, in normalized coordinates."""

    zone_id: ZoneId
    camera_id: CameraId
    role: ZoneRole = ZoneRole.AREA
    polygon: Annotated[list[NormalizedPoint], Field(min_length=3)]
    """At least a triangle — fewer vertices enclose no area to be inside of."""
    metrics: list[MetricName] = Field(default_factory=list)


# --- Tuning -----------------------------------------------------------------


class ThresholdsConfig(ConfigSection):
    """Detector and tracker tuning (algorithms.md §3)."""

    detection_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    track_thresh: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    max_cost: Annotated[float, Field(gt=0.0)] = 0.4
    """Stage-1 association gate: accept a match iff its COST is below this.

    Named for the tracker's own units rather than for `match_thresh`, the name
    engine-architecture.md §13.1 used to carry. That name is an IoU-cost number, and
    ADR-0014 chose centre distance as the cost function — whose 0.4 is its own *measured*
    operating point on a different scale entirely, explicitly "not a scaled-down guess"
    from an IoU number. A config key that cannot be translated into what the tracker
    reads is a key that either does nothing or breaks association silently; this one is
    the number the tracker actually uses.
    """
    max_cost_low: Annotated[float, Field(gt=0.0)] = 0.25
    """Stage-2 (low-confidence) gate. TIGHTER than stage 1, which is the opposite of what
    the BYTE paper's names suggest — see algorithms.md §3.3."""
    track_memory_s: Annotated[float, Field(gt=0.0)] = 2.0
    dwell_min_s: Annotated[float, Field(ge=0.0)] = 3.0
    queue_dwell_weight: bool = True


class StorageConfig(ConfigSection):
    """How long the local SQLite file keeps what it is allowed to keep.

    Separate from `thresholds` because that section tunes what the detector and tracker
    see (algorithms.md §3), and this one bounds what survives on disk.

    Only `events` has a retention window here. `metrics_minute` is disk-bounded rather
    than policy-bounded on the edge — a self-hoster's own history on their own disk —
    and the cap that enforces that ceiling is still an open question in
    engine-architecture.md §11.
    """

    event_retention_hours: Annotated[int, Field(gt=0, le=_HOURS_IN_A_YEAR)] = 72
    """How long the raw event log is kept for local debugging and re-aggregation.

    Bounded at both ends deliberately: zero would delete every event as it was written,
    which reads like "off" but silently defeats re-aggregation, and an unbounded window
    is a full disk on a box nobody is watching.
    """


class ApiConfig(ConfigSection):
    """Where the local dashboard and its JSON listen (engine-architecture.md §13).

    §13 asks for "minimal and local" auth and this section carries both halves of it: the
    bind address is the access control for *reads*, and `password_env` is the credential
    that guards every write (ADR-0019).
    """

    enabled: bool = True
    host: Annotated[str, Field(min_length=1)] = "127.0.0.1"
    """Loopback by default, because the read surface carries no credential and exposes
    camera topology and occupancy history. Reaching it from another machine is the
    operator's deliberate act. In a container this is set to `0.0.0.0` and the port
    published on the host's loopback instead, which keeps the same boundary one layer
    out (P3.6)."""
    port: Annotated[int, Field(gt=0, le=_MAX_TCP_PORT)] = 8080
    password_env: Annotated[str, Field(min_length=1)] | None = None
    """Name of the environment variable holding the operator's password (ADR-0019).

    A name, never the value — the rule every other secret in this file follows. Absent is
    a legitimate config and does not fail startup: it fails *writes*, so an engine with no
    credential is one nobody can reconfigure over HTTP."""


# --- Exporters and sync -----------------------------------------------------


class MqttExporterConfig(ConfigSection):
    enabled: bool = False
    broker: str | None = None
    port: Annotated[int, Field(gt=0, le=65535)] = 1883
    base_topic: str = "mesopic"
    discovery: bool = True
    """Announce sensors to Home Assistant (P4.4).

    On by default, because the exporter is already opt-in behind `enabled` and a second
    switch to find is how a day-one integration becomes a support question. Turn it off
    for a shared broker where someone else owns the `homeassistant/` prefix."""
    discovery_prefix: str = "homeassistant"
    """Where Home Assistant listens. Its own default; changed only if HA's was."""

    @model_validator(mode="after")
    def _enabled_needs_a_broker(self) -> Self:
        if self.enabled and not self.broker:
            msg = "mqtt exporter is enabled but no broker is set"
            raise ValueError(msg)
        return self


class PrometheusExporterConfig(ConfigSection):
    enabled: bool = False


class WebhookExporterConfig(ConfigSection):
    """Deliveries are HMAC-signed, so an enabled webhook without a secret is a hole."""

    enabled: bool = False
    url: str | None = None
    secret_env: str | None = None

    @model_validator(mode="after")
    def _enabled_needs_a_url_and_a_secret_reference(self) -> Self:
        if not self.enabled:
            return self
        if not self.url:
            msg = "webhook exporter is enabled but no url is set"
            raise ValueError(msg)
        if not self.secret_env:
            msg = "webhook exporter is enabled but no secret_env is set"
            raise ValueError(msg)
        return self


class CsvExporterConfig(ConfigSection):
    enabled: bool = False
    dir: str | None = None

    @model_validator(mode="after")
    def _enabled_needs_a_directory(self) -> Self:
        if self.enabled and not self.dir:
            msg = "csv exporter is enabled but no dir is set"
            raise ValueError(msg)
        return self


class FrigateConfig(ConfigSection):
    """How to reach the Frigate broker (ADR-0006, ADR-0022).

    Site-wide rather than per camera, because a site runs one Frigate; the *topic* stays
    on the camera because that is the part that differs.

    Deliberately **not** `exporters.mqtt`: a site can consume Frigate without publishing
    anything, and reading ingest configuration out of an exporter would mean turning on a
    publisher to make a camera work.
    """

    broker: str | None = None
    port: Annotated[int, Field(gt=0, le=65535)] = 1883
    username_env: str | None = None
    password_env: str | None = None
    """By reference, never inline — a broker password is a secret like any other."""


class ExportersConfig(ConfigSection):
    mqtt: MqttExporterConfig = Field(default_factory=MqttExporterConfig)
    prometheus: PrometheusExporterConfig = Field(default_factory=PrometheusExporterConfig)
    webhook: WebhookExporterConfig = Field(default_factory=WebhookExporterConfig)
    csv: CsvExporterConfig = Field(default_factory=CsvExporterConfig)


class CloudSyncConfig(ConfigSection):
    """Off by default — the local engine is whole on its own (ADR-0001)."""

    enabled: bool = False
    endpoint: str | None = None
    site_token_env: str | None = None

    @model_validator(mode="after")
    def _enabled_needs_a_tls_endpoint_and_a_token_reference(self) -> Self:
        if not self.enabled:
            return self
        if not self.endpoint or not self.endpoint.startswith("https://"):
            msg = "cloud_sync endpoint must be an https:// URL"
            raise ValueError(msg)
        if not self.site_token_env:
            msg = "cloud_sync is enabled but no site_token_env is set"
            raise ValueError(msg)
        return self


# --- Root -------------------------------------------------------------------


class MesopicConfig(ConfigSection):
    """Root of the validated configuration tree."""

    site: SiteConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    cameras: Annotated[list[CameraConfig], Field(min_length=1)]
    lines: list[LineConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    exporters: ExportersConfig = Field(default_factory=ExportersConfig)
    frigate: FrigateConfig = Field(default_factory=FrigateConfig)
    cloud_sync: CloudSyncConfig = Field(default_factory=CloudSyncConfig)

    @model_validator(mode="after")
    def _identities_must_be_unique(self) -> Self:
        """Every id becomes a PRIMARY KEY in the store, so a duplicate loses a row."""
        _reject_duplicates("camera_id", [camera.camera_id for camera in self.cameras])
        _reject_duplicates("line_id", [line.line_id for line in self.lines])
        _reject_duplicates("zone_id", [zone.zone_id for zone in self.zones])
        return self

    @model_validator(mode="after")
    def _a_frigate_camera_needs_a_broker(self) -> Self:
        """Refuse at load rather than let the worker die in a restart loop.

        A `kind: frigate` camera with nowhere to connect is not a camera that degrades —
        it is one that can never produce a single event, and the honest place to say so is
        where the operator is still looking at the error.
        """
        frigate = [camera for camera in self.cameras if isinstance(camera.source, FrigateSource)]
        if frigate and not self.frigate.broker:
            named = ", ".join(sorted(camera.camera_id for camera in frigate))
            msg = f"camera(s) {named} use Frigate but no frigate.broker is set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _geometry_must_name_a_real_camera(self) -> Self:
        """A dangling reference is a zone that silently never counts anything."""
        known = {camera.camera_id for camera in self.cameras}
        for line in self.lines:
            if line.camera_id not in known:
                msg = f"line {line.line_id!r} references unknown camera {line.camera_id!r}"
                raise ValueError(msg)
        for zone in self.zones:
            if zone.camera_id not in known:
                msg = f"zone {zone.zone_id!r} references unknown camera {zone.camera_id!r}"
                raise ValueError(msg)
        return self


def _reject_duplicates(field: str, values: list[str]) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        msg = f"duplicate {field}: {', '.join(repr(value) for value in duplicates)}"
        raise ValueError(msg)
