"""The dashboard board: the polled fragment, its tiles, and the assets it loads.

Three things these tests hold in place, each of which is a way the board could be wrong
while looking right:

* **A scope with no data renders `—`, never `0`.** Zero is a measurement — it says nobody
  walked past. No data says the camera never reported. A dashboard that prints `0` for a
  dead camera is telling the operator the shop is empty.
* **The tiles are derived from config, not from the rows.** Same rule `camera_health`
  follows: a zone absent from the board reads as a zone that does not exist, so what the
  operator configured decides who appears and the store only fills in values.
* **One poll, not two.** The fragment carries its own `hx-trigger` and the series as a
  JSON island, so the tiles and the charts can never disagree about which window they are
  showing.

Red-first for P3.2.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from muster.api.app import create_app
from muster.api.board import (
    EXPOSURE_CELLS,
    BoardWindow,
    charts_of,
    exposure_of,
    freshness_for,
    human_duration,
    scope_slots,
    tiles_for,
)
from muster.api.health import CameraHealth
from muster.config.schema import MusterConfig
from muster.store.store import Store
from muster.supervisor.handle import WorkerReport
from muster.types import (
    CameraId,
    CameraState,
    FrameTs,
    MetricName,
    MetricRow,
    MinuteBucket,
    ScopeId,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"
STATIC_DIR = REPO_ROOT / "muster-engine" / "src" / "muster" / "api" / "static"
VENDOR_DIR = STATIC_DIR / "vendor"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")
SHOP_FLOOR = ScopeId("shop-floor")
DOOR_LINE = ScopeId("door-count")


@pytest.fixture
def config() -> MusterConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MusterConfig) -> Iterator[Store]:
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


def _all_streaming() -> dict[CameraId, WorkerReport]:
    return {
        FRONT_DOOR: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
        TILL: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
    }


@pytest.fixture
async def client(config: MusterConfig, store: Store) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(config=config, store=store, camera_reports=_all_streaming)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        yield client


def _row(
    metric: MetricName,
    value: float,
    *,
    minutes_ago: int,
    camera_id: CameraId = FRONT_DOOR,
    scope_id: ScopeId | None = SHOP_FLOOR,
) -> MetricRow:
    bucket = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=minutes_ago)
    return MetricRow(
        camera_id=camera_id,
        bucket=MinuteBucket(bucket),
        metric=metric,
        scope_id=scope_id,
        value=value,
    )


# --- Tiles ------------------------------------------------------------------


def test_a_configured_scope_with_no_data_shows_no_value(config: MusterConfig) -> None:
    """The difference between "nobody came" and "nothing reported" is the whole point.

    A tile that renders `0` for a camera that has never reported tells the operator the
    shop is empty, which is a measurement it has not got.
    """
    tiles = tiles_for(config, rows=[])

    occupancy = [tile for tile in tiles if tile.metric is MetricName.OCCUPANCY]
    assert occupancy, "a configured occupancy zone must appear even with an empty store"
    assert all(tile.value is None for tile in occupancy)


def test_a_counting_tile_sums_the_window(config: MusterConfig) -> None:
    rows = [
        _row(MetricName.FOOTFALL, 2.0, minutes_ago=3, scope_id=DOOR_LINE),
        _row(MetricName.FOOTFALL, 3.0, minutes_ago=2, scope_id=DOOR_LINE),
        _row(MetricName.FOOTFALL, 1.0, minutes_ago=1, scope_id=DOOR_LINE),
    ]

    tiles = tiles_for(config, rows=rows)

    footfall = next(tile for tile in tiles if tile.metric is MetricName.FOOTFALL)
    assert footfall.value == 6.0


def test_a_stateful_tile_shows_the_newest_bucket(config: MusterConfig) -> None:
    """Occupancy is a level, not a total. Summing it would report a shop-day as a crowd."""
    rows = [
        _row(MetricName.OCCUPANCY, 2.0, minutes_ago=3),
        _row(MetricName.OCCUPANCY, 7.0, minutes_ago=2),
        _row(MetricName.OCCUPANCY, 5.0, minutes_ago=1),
    ]

    tiles = tiles_for(config, rows=rows)

    occupancy = next(tile for tile in tiles if tile.metric is MetricName.OCCUPANCY)
    assert occupancy.value == 5.0


def test_the_heatmap_gets_no_tile(config: MusterConfig) -> None:
    """`shop-floor` declares `heatmap`, which is a packed blob and not a number. P4.1's."""
    tiles = tiles_for(config, rows=[])

    assert all(tile.metric is not MetricName.HEATMAP for tile in tiles)


def test_a_tile_exists_for_every_configured_scope_and_metric(config: MusterConfig) -> None:
    tiles = tiles_for(config, rows=[])

    assert (TILL, MetricName.QUEUE_LEN, ScopeId("queue-till")) in {
        (tile.camera_id, tile.metric, tile.scope_id) for tile in tiles
    }


# --- Charts -----------------------------------------------------------------


def test_a_configured_metric_with_no_data_still_gets_a_chart(config: MusterConfig) -> None:
    """An empty chart is a visible absence. No chart is indistinguishable from a metric
    the operator never asked for."""
    charts = charts_of(config, rows=[])

    assert {chart["metric"] for chart in charts} >= {"footfall", "occupancy", "queue_len"}


def test_a_chart_puts_every_scope_on_one_time_axis(config: MusterConfig) -> None:
    """uPlot takes `[xs, ys…]` with one shared x. Per-series axes would plot two scopes
    against each other's timestamps."""
    rows = [
        _row(MetricName.DWELL_SECONDS, 10.0, minutes_ago=2),
        _row(MetricName.DWELL_SECONDS, 20.0, minutes_ago=1),
        _row(
            MetricName.DWELL_SECONDS,
            30.0,
            minutes_ago=1,
            camera_id=TILL,
            scope_id=ScopeId("queue-till"),
        ),
    ]

    chart = next(c for c in charts_of(config, rows=rows) if c["metric"] == "dwell_seconds")

    assert len(chart["t"]) == 2
    assert all(len(values) == len(chart["t"]) for values in chart["v"])


def test_a_scope_missing_a_bucket_gets_a_hole_not_a_shift(config: MusterConfig) -> None:
    """The failure this prevents: a camera that dropped a minute has every later point
    slide one bucket left, and the chart reads as an event that happened a minute early."""
    rows = [
        _row(MetricName.DWELL_SECONDS, 10.0, minutes_ago=2),
        _row(MetricName.DWELL_SECONDS, 20.0, minutes_ago=1),
        _row(
            MetricName.DWELL_SECONDS,
            30.0,
            minutes_ago=1,
            camera_id=TILL,
            scope_id=ScopeId("queue-till"),
        ),
    ]

    chart = next(c for c in charts_of(config, rows=rows) if c["metric"] == "dwell_seconds")
    till = chart["v"][chart["labels"].index("queue-till")]

    assert till == [None, 30.0]


# --- Colour slots -----------------------------------------------------------


def test_a_scope_keeps_one_colour_across_every_chart(config: MusterConfig) -> None:
    """Colour follows the scope, never its rank within a chart.

    Assigning by position means `shop-floor` is the first series in the occupancy chart
    and the second in the dwell chart, so it changes colour between two plates the
    operator reads side by side.
    """
    slots = scope_slots(config)

    charts = charts_of(config, rows=[])
    for chart in charts:
        for label, slot in zip(chart["labels"], chart["slots"], strict=True):
            assert slot == slots[label]


def test_the_slots_come_from_the_config_not_the_rows(config: MusterConfig) -> None:
    """Same rule the tiles follow. A scope that is silent today must not be handed a
    different colour tomorrow when it starts reporting."""
    slots = scope_slots(config)

    assert slots == scope_slots(config)
    assert set(slots) == {"door-count", "shop-floor", "queue-till"}
    assert sorted(slots.values()) == list(range(len(slots)))


# --- The exposure strip -----------------------------------------------------


def _at(minutes_ago: int) -> datetime:
    return datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=minutes_ago)


def test_a_window_with_no_rows_is_wholly_unexposed(config: MusterConfig) -> None:
    """The strip's job is to make a blind box obvious before any number is read."""
    exposure = exposure_of([], end=_at(0), window=BoardWindow.HOUR)

    assert exposure.covered == 0
    assert not any(exposure.cells)


def test_a_bucket_marks_only_the_cell_it_falls_in(config: MusterConfig) -> None:
    """One hour across 60 cells is a cell a minute, so a single bucket lights one cell."""
    end = _at(0)
    rows = [_row(MetricName.OCCUPANCY, 3.0, minutes_ago=30)]

    exposure = exposure_of(rows, end=end, window=BoardWindow.HOUR)

    assert exposure.covered == 1
    assert exposure.cells[30] is True


def test_the_newest_bucket_lands_inside_the_strip(config: MusterConfig) -> None:
    """A bucket at exactly `end` belongs to the last cell, not one past the end."""
    end = _at(0)
    rows = [_row(MetricName.OCCUPANCY, 3.0, minutes_ago=0)]

    exposure = exposure_of(rows, end=end, window=BoardWindow.HOUR)

    assert exposure.cells[-1] is True
    assert len(exposure.cells) == 60


def test_a_bucket_older_than_the_window_is_not_drawn(config: MusterConfig) -> None:
    """The strip describes the window on screen. A row from before it is not evidence
    that the window has data."""
    rows = [_row(MetricName.OCCUPANCY, 3.0, minutes_ago=180)]

    exposure = exposure_of(rows, end=_at(0), window=BoardWindow.HOUR)

    assert exposure.covered == 0


def test_coverage_counts_cells_and_not_rows(config: MusterConfig) -> None:
    """Six cameras reporting the same minute is one exposed minute, not six."""
    rows = [
        _row(MetricName.OCCUPANCY, 3.0, minutes_ago=10),
        _row(MetricName.QUEUE_LEN, 1.0, minutes_ago=10, camera_id=TILL, scope_id=ScopeId("q")),
        _row(MetricName.OCCUPANCY, 4.0, minutes_ago=11),
    ]

    exposure = exposure_of(rows, end=_at(0), window=BoardWindow.HOUR)

    assert exposure.covered == 2


# --- Uptime -----------------------------------------------------------------


def test_uptime_reads_as_a_duration_not_a_count_of_seconds() -> None:
    """`51720.4s` is a number the operator has to divide. The board is read at a glance."""
    assert human_duration(45.0) == "45s"
    assert human_duration(90.0) == "1m 30s"
    assert human_duration(3600.0) == "1h 0m"
    assert human_duration(51720.4) == "14h 22m"


def test_uptime_never_renders_a_negative_duration() -> None:
    """The clock behind it is monotonic, but a formatter that can print `-1s` is one
    clock change away from saying the engine started in the future."""
    assert human_duration(-5.0) == "0s"


# --- The fragment -----------------------------------------------------------


async def test_a_window_the_page_does_not_offer_is_refused(client: httpx.AsyncClient) -> None:
    """The window is a closed enum, so there is no unbounded range to bound."""
    response = await client.get("/fragments/board", params={"window": "7d"})

    assert response.status_code == httpx.codes.BAD_REQUEST


async def test_the_default_window_is_six_hours(client: httpx.AsyncClient) -> None:
    response = await client.get("/fragments/board")

    assert response.status_code == httpx.codes.OK
    assert BoardWindow.SIX_HOURS.value in response.text


async def test_the_fragment_is_a_fragment(client: httpx.AsyncClient) -> None:
    """HTMX swaps this into a live page; a whole document would nest `<html>` inside it."""
    text = (await client.get("/fragments/board")).text

    assert "<html" not in text
    assert "<body" not in text


async def test_the_fragment_polls_itself(client: httpx.AsyncClient) -> None:
    """The trigger rides on the swapped element, so it survives its own replacement."""
    text = (await client.get("/fragments/board")).text

    assert 'hx-get="/fragments/board' in text
    assert "every 30s" in text
    assert 'hx-swap="outerHTML"' in text


async def test_the_fragment_carries_the_series_for_the_charts(client: httpx.AsyncClient) -> None:
    """One request feeds both halves, so the tiles and the charts cannot disagree."""
    text = (await client.get("/fragments/board")).text

    assert 'type="application/json"' in text


async def test_the_fragment_draws_the_exposure_strip(client: httpx.AsyncClient) -> None:
    """The strip is the one panel that distinguishes a quiet shop from a blind box, and
    it is inside the polled fragment so it ages with the data it describes."""
    text = (await client.get("/fragments/board")).text

    assert text.count('class="cell') == EXPOSURE_CELLS
    assert "intervals with data" in text


async def test_the_fragment_separates_levels_from_counts(client: httpx.AsyncClient) -> None:
    """Reading a level as a count is the mistake the two columns exist to prevent: one is
    what the floor is like now, the other is what the window accumulated."""
    text = (await client.get("/fragments/board", params={"window": "1h"})).text

    assert ">Now<" in text
    assert ">This 1h<" in text


async def test_the_fragment_reports_engine_health(client: httpx.AsyncClient) -> None:
    """Health lives inside the polled fragment; rendered once at page load it only ages."""
    text = (await client.get("/fragments/board")).text

    assert "engine ok" in text


async def test_the_fragment_loads_nothing_from_the_internet(client: httpx.AsyncClient) -> None:
    text = (await client.get("/fragments/board")).text

    assert "http://" not in text
    assert "https://" not in text


async def test_a_dead_camera_still_shows_its_tiles(config: MusterConfig, store: Store) -> None:
    """A camera in backoff is a gap in the series, not a scope that stopped existing."""
    app = create_app(config=config, store=store, camera_reports=dict)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        text = (await client.get("/fragments/board")).text

    assert "shop-floor" in text
    assert "queue-till" in text


# --- The page ---------------------------------------------------------------


async def test_the_page_embeds_the_board_on_first_load(client: httpx.AsyncClient) -> None:
    """Rendered inline, not fetched: a dashboard that is blank until the first poll lands
    looks broken for as long as the poll interval."""
    text = (await client.get("/")).text

    assert "shop-floor" in text
    assert 'hx-get="/fragments/board' in text


async def test_the_page_serves_its_scripts_from_this_box(client: httpx.AsyncClient) -> None:
    text = (await client.get("/")).text

    sources = re.findall(r'<script[^>]*src="([^"]+)"', text)
    assert sources, "the board needs htmx and uPlot"
    assert all(source.startswith("/static/") for source in sources)


# --- Vendored assets --------------------------------------------------------


def _recorded_digests() -> dict[str, str]:
    """`(filename, sha256)` pairs parsed out of the vendor manifest's table.

    One row: the filename in the first cell, the digest in the last.
    """
    manifest = (VENDOR_DIR / "VENDOR.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*`([\w.\-]+)`\s*\|.*\|\s*`([0-9a-f]{64})`\s*\|$", manifest, re.M)
    return dict(rows)


def test_every_vendored_asset_matches_its_recorded_digest() -> None:
    """A vendored library is code nobody reviews again. The digest is what makes a silent
    swap — an edit, a bad re-download, a supply-chain substitution — fail CI instead of
    shipping."""
    digests = _recorded_digests()

    assert digests, "VENDOR.md must record a digest per asset"
    for filename, expected in digests.items():
        actual = hashlib.sha256((VENDOR_DIR / filename).read_bytes()).hexdigest()
        assert actual == expected, f"{filename} does not match VENDOR.md"


def test_every_vendored_asset_is_recorded() -> None:
    """The digest check is only as good as its coverage: an unrecorded file is unchecked."""
    on_disk = {path.name for path in VENDOR_DIR.iterdir() if path.suffix in {".js", ".css"}}

    assert on_disk == set(_recorded_digests())


def test_the_vendored_licences_are_present() -> None:
    """MIT and 0BSD both require the notice to travel with the code."""
    licences = {path.name for path in VENDOR_DIR.iterdir() if path.name.startswith("LICENSE")}

    assert len(licences) >= 2


def test_the_stylesheet_honours_reduced_motion() -> None:
    css = (STATIC_DIR / "hud.css").read_text(encoding="utf-8")

    assert "prefers-reduced-motion" in css


# --- Per-camera freshness (P3.7) --------------------------------------------
#
# Age rather than a wall-clock time, deliberately: what the reader wants is "is this
# camera current", and an age answers it without picking a timezone. Which clock the
# dashboard prints times in is still open (P3.9), and a freshness indicator should not
# quietly decide it.


def test_freshness_reports_how_long_ago_a_camera_last_delivered() -> None:
    now = datetime(2026, 8, 18, 9, 43, tzinfo=UTC)
    cameras = [
        CameraHealth(
            camera_id=FRONT_DOOR,
            state=CameraState.STREAMING,
            last_frame_ts=FrameTs(now - timedelta(seconds=8)),
            consecutive_failures=0,
            effective_fps=2.5,
        )
    ]

    (first,) = freshness_for(cameras, now=now)

    assert first.age == "8s"
    assert first.effective_fps == pytest.approx(2.5)


def test_a_camera_that_has_never_reported_has_no_age() -> None:
    """Same rule the tiles follow: absent is not zero.

    A `0s` age on a camera that has never sent a frame would read as the freshest thing
    on the page, which is the exact inversion of the truth.
    """
    cameras = [
        CameraHealth(
            camera_id=FRONT_DOOR,
            state=CameraState.CONNECT,
            last_frame_ts=None,
            consecutive_failures=0,
            effective_fps=None,
        )
    ]

    (first,) = freshness_for(cameras, now=datetime(2026, 8, 18, 9, 43, tzinfo=UTC))

    assert first.age is None


def test_a_frame_timestamped_in_the_future_does_not_report_a_negative_age() -> None:
    """A camera whose clock runs fast is a skewed clock, not a frame from the future."""
    now = datetime(2026, 8, 18, 9, 43, tzinfo=UTC)
    cameras = [
        CameraHealth(
            camera_id=FRONT_DOOR,
            state=CameraState.STREAMING,
            last_frame_ts=FrameTs(now + timedelta(seconds=30)),
            consecutive_failures=0,
            effective_fps=2.5,
        )
    ]

    (first,) = freshness_for(cameras, now=now)

    assert first.age == "0s"


async def test_the_fragment_shows_each_cameras_freshness(client: httpx.AsyncClient) -> None:
    text = (await client.get("/fragments/board")).text

    assert FRONT_DOOR in text
    assert TILL in text
    assert "streaming" in text


async def test_the_fragment_names_a_stalled_camera(config: MusterConfig, store: Store) -> None:
    """The whole point of P3.7 reaching the dashboard: a wedged camera is visible.

    Its tiles keep rendering — a stalled camera is a gap, not a scope that stopped
    existing — so without this row the page looks exactly like a quiet shop.

    The clock is pinned rather than read: an age asserted against `datetime.now()` is a
    test that reads `3m 59s` whenever the machine is a millisecond slow.
    """
    now = datetime(2026, 8, 18, 9, 43, tzinfo=UTC)

    def _one_stalled() -> dict[CameraId, WorkerReport]:
        return {
            FRONT_DOOR: WorkerReport(
                state=CameraState.STALLED,
                consecutive_failures=0,
                last_frame_ts=FrameTs(now - timedelta(minutes=4)),
                effective_fps=0.0,
            ),
            TILL: WorkerReport(state=CameraState.STREAMING, consecutive_failures=0),
        }

    app = create_app(config=config, store=store, camera_reports=_one_stalled, clock=lambda: now)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        text = (await client.get("/fragments/board")).text

    assert "stalled" in text
    assert "4m 0s ago" in text
