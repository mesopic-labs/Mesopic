"""The heatmap read path — decay, normalization, and the store round trip.

These are render choices, and the reason they are tested is that each one fails as a
*plausible picture* rather than as an error. A view with no decay looks like a heatmap;
it is just a heatmap of all time. A linearly normalized view looks like a heatmap; it is
just a picture of the single hottest cell with everything else rendered as floor.

Red-first for P4.1.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from muster.analytics.metrics.heatmap import pack_counts
from muster.api.app import create_app
from muster.api.heatmap import DEFAULT_DECAY, HeatmapView, normalize, roll_up
from muster.config.schema import MusterConfig
from muster.store.store import Store
from muster.types import CameraId, HeatmapRow, MinuteBucket, ZoneId

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "muster.yaml"

CAMERA = CameraId("front-door")
FLOOR = ZoneId("floor")
BACK = ZoneId("back-room")
END = datetime(2026, 8, 18, 10, 0, tzinfo=UTC)


def _row(
    cells: list[float],
    *,
    minutes_ago: int = 0,
    zone_id: ZoneId = FLOOR,
    width: int = 2,
    height: int = 2,
) -> HeatmapRow:
    return HeatmapRow(
        camera_id=CAMERA,
        bucket=MinuteBucket(END - timedelta(minutes=minutes_ago)),
        zone_id=zone_id,
        grid_w=width,
        grid_h=height,
        counts=pack_counts(cells, width=width, height=height),
    )


def _log_ratio(value: float, peak: float) -> float:
    return math.log1p(value) / math.log1p(peak)


def _only(views: list[HeatmapView]) -> HeatmapView:
    assert len(views) == 1
    return views[0]


# --- Normalization ----------------------------------------------------------


def test_an_empty_grid_renders_uniformly_cold() -> None:
    """Not an error, and not a division by zero: nobody walked there (algorithms.md §10)."""
    assert normalize([0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0)


def test_the_hottest_cell_reaches_full_intensity() -> None:
    assert normalize([0.0, 10.0])[1] == pytest.approx(1.0)


def test_normalization_is_logarithmic_not_linear() -> None:
    """A till cell is routinely 100x any other. Linearly normalized, the modest cell
    renders at 1% — indistinguishable from floor nobody walked on."""
    modest = normalize([1000.0, 10.0])[1]
    assert modest > 0.3
    assert modest == pytest.approx(_log_ratio(10.0, 1000.0))


def test_one_outlier_cell_does_not_flatten_the_rest() -> None:
    """The p99 clip is what stops a single hot cell from setting the whole scale."""
    grid = [5.0] * 100 + [100000.0]
    normalized = normalize(grid)
    assert normalized[-1] == pytest.approx(1.0)  # clipped, still the hottest
    assert normalized[0] == pytest.approx(1.0)  # and the body of the grid is visible


# --- Roll-up ----------------------------------------------------------------


def test_minutes_sum_into_one_grid() -> None:
    minutes = [_row([10.0, 0, 0, 0]), _row([10.0, 0, 0, 0], minutes_ago=1)]
    view = _only(roll_up(minutes, end=END, decay=1.0))
    assert view.peak_ds == pytest.approx(20.0)
    assert view.minutes == 2


def test_older_minutes_count_for_less() -> None:
    """Without decay a window is an all-time integral and yesterday drowns this morning."""
    recent = _only(roll_up([_row([10.0, 0, 0, 0])], end=END))
    old = _only(roll_up([_row([10.0, 0, 0, 0], minutes_ago=10)], end=END))
    assert old.peak_ds == pytest.approx(recent.peak_ds * DEFAULT_DECAY**10)


def test_each_zone_gets_its_own_view() -> None:
    views = roll_up([_row([10.0, 0, 0, 0]), _row([5.0, 0, 0, 0], zone_id=BACK)], end=END)
    assert sorted(view.zone_id for view in views) == [BACK, FLOOR]


def test_a_window_with_no_rows_produces_no_views() -> None:
    assert roll_up([], end=END) == []


def test_minutes_at_a_stale_grid_size_are_dropped_not_misplaced() -> None:
    """Summing a 2x2 into a 4x4 would put every historical cell somewhere it never was.
    A visibly shorter history is the honest outcome of changing the grid constants."""
    view = _only(
        roll_up(
            [_row([1.0] * 16, width=4, height=4), _row([9.0] * 4, minutes_ago=1)],
            end=END,
            decay=1.0,
        )
    )
    assert (view.grid_w, view.grid_h) == (4, 4)
    assert view.minutes == 1


# --- The route --------------------------------------------------------------


@pytest.fixture
def config() -> MusterConfig:
    """The worked example: `shop-floor` on `front-door` asks for a heatmap."""
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MusterConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MusterConfig) -> Iterator[Store]:
    with Store(tmp_path / "muster.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


NOW = datetime(2026, 8, 18, 10, 0, 30, tzinfo=UTC)
"""Deliberately mid-minute: the route decays by *fractional* minute age, so a clock on an
exact boundary would hide a whole class of off-by-a-minute error."""


@pytest.fixture
async def client(config: MusterConfig, store: Store) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(config=config, store=store, clock=lambda: NOW)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        yield client


def _stored(store: Store, *, zone_id: str, hot: float, minutes_ago: int = 0) -> None:
    cells = [0.0] * 16
    cells[5] = hot
    store.upsert_heatmaps(
        [
            HeatmapRow(
                camera_id=CameraId("front-door"),
                bucket=MinuteBucket(
                    NOW.replace(second=0, microsecond=0) - timedelta(minutes=minutes_ago)
                ),
                zone_id=ZoneId(zone_id),
                grid_w=4,
                grid_h=4,
                counts=pack_counts(cells, width=4, height=4),
            )
        ]
    )


async def test_the_route_returns_a_normalized_grid(client: httpx.AsyncClient, store: Store) -> None:
    _stored(store, zone_id="shop-floor", hot=120.0)

    response = await client.get("/api/heatmap")

    assert response.status_code == 200
    (zone,) = response.json()["zones"]
    assert zone["zone_id"] == "shop-floor"
    assert (zone["grid_w"], zone["grid_h"]) == (4, 4)
    assert zone["cells"][5] == pytest.approx(1.0)
    # Decayed by half a minute of age, not reported raw: 120 * 0.95**0.5.
    assert zone["peak_ds"] == round(120 * DEFAULT_DECAY**0.5)


async def test_the_route_can_be_narrowed_to_one_zone(
    client: httpx.AsyncClient, store: Store
) -> None:
    _stored(store, zone_id="shop-floor", hot=120.0)

    response = await client.get("/api/heatmap", params={"zone_id": "till-queue"})

    assert response.json()["zones"] == []


async def test_a_zone_with_no_stored_minutes_is_absent_rather_than_zero(
    client: httpx.AsyncClient,
) -> None:
    """The figure is still rendered — the dashboard builds its shells from config — so an
    absent zone reads as cold rather than as a zone that stopped being watched."""
    response = await client.get("/api/heatmap")
    assert response.json() == {"zones": [], "truncated": False}


async def test_an_unknown_query_key_is_refused(client: httpx.AsyncClient) -> None:
    """`extra="forbid"` at the edge: a misspelled filter must not silently widen the read.

    400 rather than 422 because the app answers every validation failure generically —
    the detail goes to the log, never to the caller.
    """
    response = await client.get("/api/heatmap", params={"zone": "shop-floor"})
    assert response.status_code == 400


async def test_the_dashboard_renders_a_figure_per_heatmap_zone(
    client: httpx.AsyncClient,
) -> None:
    """The acceptance criterion's overlay. Built from config, so it is there before any
    traffic is — and outside `#board`, so the 30-second poll cannot destroy the canvas."""
    body = (await client.get("/")).text
    assert 'data-zone="shop-floor"' in body
    assert 'class="floor-grid"' in body
