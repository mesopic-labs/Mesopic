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
    Cohort,
    Tile,
    charts_of,
    clock_label,
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
    staff_value: float | None = None,
) -> MetricRow:
    bucket = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=minutes_ago)
    return MetricRow(
        camera_id=camera_id,
        bucket=MinuteBucket(bucket),
        metric=metric,
        scope_id=scope_id,
        value=value,
        staff_value=staff_value,
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


# --- The live durations -----------------------------------------------------


def test_freshness_carries_the_age_as_a_number_as_well_as_text() -> None:
    """The board's ticker ages the number between swaps; the text is the first frame."""
    seen = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
    (fresh,) = freshness_for(
        [
            CameraHealth(
                camera_id=CameraId("front-door"),
                state=CameraState.STREAMING,
                last_frame_ts=FrameTs(seen),
                consecutive_failures=0,
                effective_fps=2.5,
            )
        ],
        now=seen + timedelta(seconds=90),
    )
    assert fresh.age == "1m 30s"
    assert fresh.age_s == pytest.approx(90.0)


def test_a_camera_that_never_reported_carries_no_number_to_tick() -> None:
    """The em dash must not be tickable into `0s`. A ticker handed a number for a camera
    that has never sent a frame renders the freshest thing on the page out of nothing."""
    (fresh,) = freshness_for(
        [
            CameraHealth(
                camera_id=CameraId("front-door"),
                state=CameraState.CONNECT,
                last_frame_ts=None,
                consecutive_failures=0,
                effective_fps=None,
            )
        ],
        now=datetime(2026, 8, 18, 10, 0, tzinfo=UTC),
    )
    assert fresh.age is None
    assert fresh.age_s is None


def test_a_clock_skewed_into_the_future_ages_to_zero_not_below() -> None:
    """Same rule the rendered text already follows: a frame from the future is a skewed
    camera clock, and `-30s ago` would be a bug report about the dashboard."""
    now = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)
    (fresh,) = freshness_for(
        [
            CameraHealth(
                camera_id=CameraId("front-door"),
                state=CameraState.STREAMING,
                last_frame_ts=FrameTs(now + timedelta(seconds=30)),
                consecutive_failures=0,
                effective_fps=2.5,
            )
        ],
        now=now,
    )
    assert fresh.age_s == 0.0


@pytest.mark.parametrize(
    ("seconds", "rendered"),
    [
        (0.0, "0s"),
        (0.4, "0s"),
        (59.9, "59s"),
        (60.0, "1m 0s"),
        (3599.0, "59m 59s"),
        (3600.0, "1h 0m"),
        (51720.4, "14h 22m"),
    ],
)
def test_human_duration_boundaries_the_ticker_mirrors(seconds: float, rendered: str) -> None:
    """`humanDuration` in `board.js` is a hand copy of this function, because a live
    ticker cannot call the server every second. These cases pin the contract it copies:
    if one changes here, the browser starts disagreeing with the frame it was handed.
    """
    assert human_duration(seconds) == rendered


async def test_the_fragment_carries_the_numbers_the_ticker_needs(
    config: MusterConfig, store: Store
) -> None:
    """Without these attributes the ticker has nothing to age and the durations freeze
    again — silently, because the page still renders correct-looking numbers once."""
    seen = datetime.now(UTC) - timedelta(seconds=5)

    def reporting() -> dict[CameraId, WorkerReport]:
        return {
            FRONT_DOOR: WorkerReport(
                state=CameraState.STREAMING,
                consecutive_failures=0,
                last_frame_ts=FrameTs(seen),
                effective_fps=2.5,
            )
        }

    app = create_app(config=config, store=store, camera_reports=reporting)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        body = (await client.get("/fragments/board")).text

    assert "data-uptime-s=" in body
    assert "data-age-s=" in body


async def test_a_camera_with_no_frame_yet_carries_no_age_attribute(
    client: httpx.AsyncClient,
) -> None:
    """`_all_streaming` reports no `last_frame_ts`, so both cameras render an em dash.
    The attribute must be absent, not zero — the ticker keys off its presence."""
    body = (await client.get("/fragments/board")).text
    assert "data-age-s=" not in body
    assert "data-uptime-s=" in body


# --- Cohort (P4.6) ----------------------------------------------------------


def _tile(tiles: tuple[Tile, ...], metric: MetricName, scope_id: ScopeId) -> Tile:
    return next(tile for tile in tiles if tile.metric is metric and tile.scope_id == scope_id)


def _config_without(role: str) -> MusterConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    parsed["zones"] = [zone for zone in parsed["zones"] if zone.get("role") != role]
    return MusterConfig.model_validate(parsed)


def _config_also_collecting(metric: str, *, on_zone: str) -> MusterConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    for zone in parsed["zones"]:
        if zone["zone_id"] == on_zone:
            zone.setdefault("metrics", []).append(metric)
    return MusterConfig.model_validate(parsed)


def test_the_default_cohort_leaves_every_reading_alone(config: MusterConfig) -> None:
    """`value` is the total and `all` is the default, so a board nobody has filtered shows
    exactly what it showed before this control existed."""
    rows = [_row(MetricName.OCCUPANCY, 10.0, staff_value=3.0, minutes_ago=0)]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.ALL)

    assert _tile(tiles, MetricName.OCCUPANCY, SHOP_FLOOR).value == 10.0


def test_customers_are_the_total_less_the_staff_portion(config: MusterConfig) -> None:
    """`value` is the total, staff included (ADR-0021), so the customer figure is a
    subtraction — and it happens here at the presentation boundary, never in the store."""
    rows = [_row(MetricName.OCCUPANCY, 10.0, staff_value=3.0, minutes_ago=0)]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)

    assert _tile(tiles, MetricName.OCCUPANCY, SHOP_FLOOR).value == 7.0


def test_the_staff_cohort_shows_the_stored_staff_portion(config: MusterConfig) -> None:
    rows = [_row(MetricName.OCCUPANCY, 10.0, staff_value=3.0, minutes_ago=0)]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.STAFF)

    assert _tile(tiles, MetricName.OCCUPANCY, SHOP_FLOOR).value == 3.0


def test_a_counting_metric_sums_its_cohort_across_the_window(config: MusterConfig) -> None:
    """The window's total is still a total once filtered — the subtraction is per bucket
    and the sum is over the results, not the other way round."""
    rows = [
        _row(MetricName.FOOTFALL, 5.0, staff_value=1.0, minutes_ago=2, scope_id=DOOR_LINE),
        _row(MetricName.FOOTFALL, 4.0, staff_value=2.0, minutes_ago=1, scope_id=DOOR_LINE),
    ]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)

    assert _tile(tiles, MetricName.FOOTFALL, DOOR_LINE).value == 6.0


def test_one_unmeasured_bucket_makes_the_whole_total_absent(config: MusterConfig) -> None:
    """A `None` that drops silently out of a sum is worse than an em dash: it renders a
    confident number short by however much was never measured."""
    rows = [
        _row(MetricName.FOOTFALL, 5.0, staff_value=1.0, minutes_ago=2, scope_id=DOOR_LINE),
        _row(MetricName.FOOTFALL, 4.0, staff_value=None, minutes_ago=1, scope_id=DOOR_LINE),
    ]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)

    assert _tile(tiles, MetricName.FOOTFALL, DOOR_LINE).value is None


def test_a_scope_whose_staff_portion_was_never_measured_shows_no_value(
    config: MusterConfig,
) -> None:
    """`None` is not zero here either. A row with no staff figure cannot answer a cohort
    question, and `value - None` is not a subtraction."""
    rows = [_row(MetricName.OCCUPANCY, 10.0, staff_value=None, minutes_ago=0)]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)

    assert _tile(tiles, MetricName.OCCUPANCY, SHOP_FLOOR).value is None


def test_a_mean_over_two_populations_cannot_be_split(config: MusterConfig) -> None:
    """`dwell_seconds` stores a mean over all dwells beside a mean over staff dwells, and
    `n_staff` is stored nowhere — so the difference of the two averages is not the
    customers' mean dwell. An em dash is the honest answer; a number would not be."""
    rows = [_row(MetricName.DWELL_SECONDS, 40.0, staff_value=10.0, minutes_ago=0)]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)

    assert _tile(tiles, MetricName.DWELL_SECONDS, SHOP_FLOOR).value is None


def test_a_peak_cannot_be_split_either() -> None:
    """`occupancy_raw` stores two independent maxima, and `max(a) - max(b)` understates
    `max(a - b)`: five customers at one moment and three staff at another would render as
    two. Wrong in a known direction is still wrong."""
    config = _config_also_collecting("occupancy_raw", on_zone="shop-floor")
    rows = [_row(MetricName.OCCUPANCY_RAW, 5.0, staff_value=3.0, minutes_ago=0)]

    tiles = tiles_for(config, rows=rows, cohort=Cohort.CUSTOMERS)

    assert _tile(tiles, MetricName.OCCUPANCY_RAW, SHOP_FLOOR).value is None


def test_conversion_is_unaffected_by_the_cohort() -> None:
    """Conversion's denominator is still total footfall — ADR-0021 leaves that open — so it
    has no staff portion to subtract. Blanking it under a cohort would read as a metric
    that stopped working, which is the one thing it must not do."""
    config = _config_also_collecting("conversion", on_zone="shop-floor")
    rows = [_row(MetricName.CONVERSION, 0.25, staff_value=None, minutes_ago=0)]

    for cohort in Cohort:
        tiles = tiles_for(config, rows=rows, cohort=cohort)

        assert _tile(tiles, MetricName.CONVERSION, SHOP_FLOOR).value == 0.25


def test_the_charts_follow_the_cohort_too(config: MusterConfig) -> None:
    """The tiles and the plates read the same rows in the same request, so a cohort that
    moved one and not the other would put two populations on one screen."""
    rows = [_row(MetricName.OCCUPANCY, 10.0, staff_value=3.0, minutes_ago=0)]

    charts = charts_of(config, rows=rows, cohort=Cohort.CUSTOMERS)

    occupancy = next(chart for chart in charts if chart["metric"] == MetricName.OCCUPANCY.value)
    assert occupancy["v"] == [[7.0]]


async def test_a_cohort_the_page_does_not_offer_is_refused(client: httpx.AsyncClient) -> None:
    """A closed set, for the reason the window is one: the split is a fixed vocabulary,
    not a filter expression the caller gets to compose."""
    response = await client.get("/fragments/board", params={"cohort": "managers"})

    assert response.status_code == httpx.codes.BAD_REQUEST


async def test_the_default_cohort_is_everyone(client: httpx.AsyncClient) -> None:
    response = await client.get("/fragments/board")

    assert response.status_code == httpx.codes.OK
    assert 'data-cohort="all"' in response.text


async def test_the_fragment_names_the_cohort_it_is_showing(client: httpx.AsyncClient) -> None:
    """One element owns the cohort, for the reason one element owns the range: a second
    copy is the one that silently disagrees."""
    for cohort in Cohort:
        text = (await client.get("/fragments/board", params={"cohort": cohort.value})).text

        assert f'data-cohort="{cohort.value}"' in text


async def test_every_control_carries_both_the_range_and_the_cohort(
    client: httpx.AsyncClient,
) -> None:
    """The poll URL and both navs render a URL each way. One that drops a parameter
    silently resets the other control — click a range and lose your cohort."""
    text = (await client.get("/fragments/board", params={"cohort": "staff"})).text

    urls = [url for url in re.findall(r'(?:href|hx-get)="([^"]+)"', text) if "?" in url]
    assert urls, "the fragment renders its own poll URL and both navs"
    for url in urls:
        assert "window=" in url, url
        assert "cohort=" in url, url


async def test_the_cohort_controls_are_real_links(client: httpx.AsyncClient) -> None:
    """With scripting off the chips still have to work, so each carries an `href` beside
    its `hx-get` — the precedent the range chips set."""
    text = (await client.get("/fragments/board", params={"cohort": "staff"})).text

    assert 'href="/?window=6h&amp;cohort=customers"' in text


async def test_the_page_accepts_a_cohort_without_scripting(client: httpx.AsyncClient) -> None:
    """The link target has to be a real route, not just an htmx endpoint."""
    response = await client.get("/", params={"window": "6h", "cohort": "staff"})

    assert response.status_code == httpx.codes.OK
    assert 'data-cohort="staff"' in response.text


async def test_a_filtered_board_says_the_heatmaps_are_not_filtered(
    client: httpx.AsyncClient,
) -> None:
    """`heatmap_minute` carries no staff dimension at all, so the overlays cannot follow
    the toggle. Saying so is the difference between a known limit and two populations
    rendered side by side with nothing to tell them apart."""
    text = (await client.get("/fragments/board", params={"cohort": "customers"})).text

    assert "heatmaps show everyone" in text


async def test_an_unfiltered_board_needs_no_heatmap_caveat(client: httpx.AsyncClient) -> None:
    """The caveat is about a mismatch, and at `all` there is none to warn about."""
    text = (await client.get("/fragments/board")).text

    assert "heatmaps show everyone" not in text


async def test_the_caveat_lives_inside_the_swapped_fragment(client: httpx.AsyncClient) -> None:
    """Rendered once in the page shell it would go stale on the first chip click, which is
    the bug PR #51 was written about. The fragment owns everything that describes it."""
    fragment = (await client.get("/fragments/board", params={"cohort": "staff"})).text
    page = (await client.get("/", params={"cohort": "staff"})).text

    assert fragment.count("heatmaps show everyone") == 1
    assert page.count("heatmaps show everyone") == 1


async def test_a_site_with_no_staff_zone_is_offered_no_cohort(store: Store) -> None:
    """Every `staff_value` is `0.0` or `None` without a `role: staff` zone, so the control
    could only ever answer zero — and a control that cannot answer is worse than none."""
    app = create_app(config=_config_without("staff"), store=store, camera_reports=_all_streaming)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        text = (await client.get("/fragments/board")).text

    assert 'aria-label="Cohort"' not in text
    assert "shop-floor" in text, "the rest of the board must still render"


async def test_a_site_with_a_staff_zone_is_offered_the_cohort(client: httpx.AsyncClient) -> None:
    text = (await client.get("/fragments/board")).text

    assert 'aria-label="Cohort"' in text


COHORT_NOW = datetime(2026, 8, 18, 10, 0, 30, tzinfo=UTC)
"""Mid-minute on purpose: the tiles bucket by the minute, so a clock sitting exactly on a
boundary hides a whole class of off-by-a-minute error."""


async def test_an_unsplittable_reading_renders_an_em_dash(
    config: MusterConfig, store: Store
) -> None:
    """The assertion the rest of this file makes about missing rows, made about a missing
    *split*: the number is there, the cohort's share of it is not, and the difference has
    to survive all the way to the page."""
    store.upsert_metrics(
        [
            MetricRow(
                camera_id=FRONT_DOOR,
                bucket=MinuteBucket(COHORT_NOW.replace(second=0, microsecond=0)),
                metric=MetricName.DWELL_SECONDS,
                scope_id=SHOP_FLOOR,
                value=40.0,
                staff_value=10.0,
            )
        ]
    )
    app = create_app(
        config=config, store=store, camera_reports=_all_streaming, clock=lambda: COHORT_NOW
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        everyone = (await client.get("/fragments/board")).text
        customers = (await client.get("/fragments/board", params={"cohort": "customers"})).text

    assert "40" in everyone, "the total is measured and must still render"
    assert customers.count("&mdash;") > everyone.count("&mdash;")
    assert "is-absent" in customers


# --- The site's clock (P3.9) --------------------------------------------------


def test_a_rendered_time_is_the_sites_local_time() -> None:
    """P3.9's decision: the operator reads their own clock, not the storage layer's.

    09:30 UTC in July is 10:30 in London, and a shop owner asked whether that was the
    lunch rush should not have to do the arithmetic.
    """
    moment = datetime(2026, 7, 15, 9, 30, tzinfo=UTC)

    assert clock_label(moment, "Europe/London") == "10:30 BST"


def test_the_label_names_the_offset_it_is_in() -> None:
    """The same zone, six months apart, is an hour apart and says so.

    Without the abbreviation the two readings are indistinguishable on the page, which is
    the whole failure mode of rendering a local time with no label.
    """
    winter = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)

    assert clock_label(winter, "Europe/London") == "09:30 GMT"


def test_a_site_that_is_utc_still_says_so() -> None:
    """The default is not a special case: `UTC` is a zone like any other, and a label
    that vanished for it would leave the reader guessing on exactly the sites that never
    configured one."""
    moment = datetime(2026, 7, 15, 9, 30, tzinfo=UTC)

    assert clock_label(moment, "UTC") == "09:30 UTC"


def test_the_label_converts_rather_than_relabels() -> None:
    """A zone west of UTC crosses a date boundary, which relabelling would not."""
    moment = datetime(2026, 7, 15, 2, 30, tzinfo=UTC)

    assert clock_label(moment, "America/Los_Angeles") == "19:30 PDT"
