"""`/api/metrics`: a window of the store's series, and nothing computed on the way out.

The read side of engine-architecture.md §13. Three things these tests hold in place:

* the window is bounded — an unbounded range against a year of minute buckets is an OOM
  on the box the engine is supposed to be invisible on;
* a truncated answer says so, because a chart that silently plots the first 5,000 rows of
  a longer range is a chart that lies about a quiet afternoon;
* a rejected request gets a generic message. The P2.1 lesson: a validation error that
  echoes the offending input is one config key away from printing an RTSP URL, and
  `input_value=` is exactly how pydantic likes to do that.

Red-first for P3.1.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from mesopic.api.app import MAX_LIMIT, MAX_WINDOW, create_app
from mesopic.config.schema import MesopicConfig
from mesopic.store.store import Store
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket, ScopeId

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

FRONT_DOOR = CameraId("front-door")
TILL = CameraId("till")
DOOR_LINE = ScopeId("door-count")

# `rtsp://user:pass@` is the one credential form allowed in this repository — the
# documented placeholder, exempted by name in `.gitleaks.toml`. Any other spelling is a
# leak as far as the scanner is concerned, and it is right to insist. The host is in RFC
# 5737's documentation range, so this URL cannot address a real camera either. Same
# constants, same reason, as `test_cli_spike.py`.
CREDENTIAL = "user:pass"
CAMERA_HOST = "192.0.2.10"
BUCKET = MinuteBucket(datetime(2026, 8, 16, 9, 30, tzinfo=UTC))
WINDOW = {"from": "2026-08-16T09:00:00Z", "to": "2026-08-16T10:00:00Z"}


@pytest.fixture
def config() -> MesopicConfig:
    parsed: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    return MesopicConfig.model_validate(parsed)


@pytest.fixture
def store(tmp_path: Path, config: MesopicConfig) -> Iterator[Store]:
    with Store(tmp_path / "mesopic.db") as store:
        store.migrate()
        store.apply_config(config)
        yield store


@pytest.fixture
async def client(config: MesopicConfig, store: Store) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(config=config, store=store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as client:
        yield client


def _row(
    value: float,
    *,
    minute: int = 0,
    camera: CameraId = FRONT_DOOR,
    metric: MetricName = MetricName.FOOTFALL,
    scope: ScopeId | None = DOOR_LINE,
) -> MetricRow:
    return MetricRow(
        camera_id=camera,
        bucket=MinuteBucket(BUCKET + timedelta(minutes=minute)),
        metric=metric,
        scope_id=scope,
        value=value,
        sample_count=1,
    )


# --- The happy path ---------------------------------------------------------


async def test_a_window_comes_back_as_one_series_per_scope(
    client: httpx.AsyncClient, store: Store
) -> None:
    store.upsert_metrics([_row(1.0, minute=0), _row(2.0, minute=1)])

    body = (await client.get("/api/metrics", params=WINDOW)).json()

    assert len(body["series"]) == 1
    (series,) = body["series"]
    assert series["camera_id"] == FRONT_DOOR
    assert series["metric"] == MetricName.FOOTFALL.value
    assert series["scope_id"] == "door-count"


async def test_points_are_columnar_and_time_ordered(
    client: httpx.AsyncClient, store: Store
) -> None:
    """uPlot consumes parallel arrays; row-tuples here would be rewritten in P3.2."""
    store.upsert_metrics([_row(2.0, minute=1), _row(1.0, minute=0)])

    (series,) = (await client.get("/api/metrics", params=WINDOW)).json()["series"]

    assert series["v"] == [1.0, 2.0]
    assert series["t"] == [
        int(BUCKET.timestamp()),
        int((BUCKET + timedelta(minutes=1)).timestamp()),
    ]


async def test_a_camera_wide_metric_reports_a_null_scope(
    client: httpx.AsyncClient, store: Store
) -> None:
    """`''` is the upsert key's spelling of "no scope" (P2.5) and must not reach a caller."""
    store.upsert_metrics([_row(3.0, metric=MetricName.OCCUPANCY, scope=None)])

    (series,) = (await client.get("/api/metrics", params=WINDOW)).json()["series"]

    assert series["scope_id"] is None


async def test_series_are_split_by_camera(client: httpx.AsyncClient, store: Store) -> None:
    """One flat list of rows would make two cameras' footfall plot as one sawtooth."""
    store.upsert_metrics([_row(1.0), _row(9.0, camera=TILL, scope=ScopeId("queue-till"))])

    body = (await client.get("/api/metrics", params=WINDOW)).json()

    assert {series["camera_id"] for series in body["series"]} == {FRONT_DOOR, TILL}


async def test_a_camera_can_be_asked_for_by_id(client: httpx.AsyncClient, store: Store) -> None:
    store.upsert_metrics([_row(1.0), _row(9.0, camera=TILL, scope=ScopeId("queue-till"))])

    body = (await client.get("/api/metrics", params={**WINDOW, "camera_id": TILL})).json()

    assert {series["camera_id"] for series in body["series"]} == {TILL}


async def test_metrics_can_be_named_more_than_once(client: httpx.AsyncClient, store: Store) -> None:
    """`?metric=footfall&metric=occupancy` — the dashboard asks for several at a time."""
    store.upsert_metrics(
        [
            _row(1.0),
            _row(3.0, metric=MetricName.OCCUPANCY, scope=None),
            _row(5.0, metric=MetricName.DWELL_SECONDS, scope=None),
        ]
    )

    body = (
        await client.get(
            "/api/metrics",
            params=[*WINDOW.items(), ("metric", "footfall"), ("metric", "occupancy")],
        )
    ).json()

    assert {series["metric"] for series in body["series"]} == {"footfall", "occupancy"}


async def test_an_empty_window_is_an_empty_series_list_not_an_error(
    client: httpx.AsyncClient,
) -> None:
    """A new box has no history; a 404 here would read as a broken dashboard."""
    response = await client.get("/api/metrics", params=WINDOW)

    assert response.status_code == httpx.codes.OK
    assert response.json()["series"] == []


# --- Bounds -----------------------------------------------------------------


async def test_a_truncated_answer_says_so(client: httpx.AsyncClient, store: Store) -> None:
    """Silently plotting the first N rows of a longer range misreports a quiet afternoon."""
    store.upsert_metrics([_row(float(minute), minute=minute) for minute in range(5)])

    body = (await client.get("/api/metrics", params={**WINDOW, "limit": 2})).json()

    assert body["truncated"] is True


async def test_a_complete_answer_says_so(client: httpx.AsyncClient, store: Store) -> None:
    store.upsert_metrics([_row(1.0)])

    body = (await client.get("/api/metrics", params=WINDOW)).json()

    assert body["truncated"] is False


async def test_a_limit_beyond_the_cap_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/metrics", params={**WINDOW, "limit": MAX_LIMIT + 1})

    assert response.status_code == httpx.codes.BAD_REQUEST


async def test_a_window_longer_than_the_cap_is_rejected(client: httpx.AsyncClient) -> None:
    """Minute buckets over an unbounded range is how a dashboard poll OOMs an N100."""
    end = datetime(2026, 8, 16, 9, 0, tzinfo=UTC) + MAX_WINDOW + timedelta(minutes=1)
    response = await client.get(
        "/api/metrics", params={"from": WINDOW["from"], "to": end.isoformat()}
    )

    assert response.status_code == httpx.codes.BAD_REQUEST


async def test_an_inverted_window_is_rejected(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/metrics", params={"from": WINDOW["to"], "to": WINDOW["from"]})

    assert response.status_code == httpx.codes.BAD_REQUEST


async def test_an_unknown_metric_name_is_rejected(client: httpx.AsyncClient) -> None:
    """The metric vocabulary is a locked contract shared with the cloud (ADR-0010)."""
    response = await client.get("/api/metrics", params={**WINDOW, "metric": "shoplifting"})

    assert response.status_code == httpx.codes.BAD_REQUEST


async def test_a_naive_timestamp_is_rejected(client: httpx.AsyncClient) -> None:
    """Everything stored is UTC; accepting a bare local time would silently shift a day."""
    response = await client.get(
        "/api/metrics", params={"from": "2026-08-16T09:00:00", "to": "2026-08-16T10:00:00"}
    )

    assert response.status_code == httpx.codes.BAD_REQUEST


# --- What an error is allowed to say ----------------------------------------


async def test_a_rejected_request_gets_a_generic_message(client: httpx.AsyncClient) -> None:
    """Detail goes to the log, not to the caller — and pydantic's default echoes input."""
    response = await client.get("/api/metrics", params={**WINDOW, "camera_id": "x" * 300})

    assert response.status_code == httpx.codes.BAD_REQUEST
    assert response.json() == {"detail": "invalid request"}


async def test_no_rejected_input_is_echoed_back(client: httpx.AsyncClient) -> None:
    """The shape that leaks a secret: a value quoted back inside a validation error.

    An RTSP URL carries the camera's credentials, and `/config` (P3.4) will validate one
    through this same handler stack. The rule has to hold before that lands, not after.

    Padded past `MAX_ID_LENGTH` so the value is genuinely rejected — an accepted one would
    make this pass without the error path ever running.
    """
    credential_url = f"rtsp://{CREDENTIAL}@{CAMERA_HOST}:554/Streaming/Channels/101"

    response = await client.get(
        "/api/metrics", params={**WINDOW, "camera_id": credential_url + "x" * 300}
    )

    assert response.status_code == httpx.codes.BAD_REQUEST
    assert "rtsp://" not in response.text
    assert CREDENTIAL not in response.text
