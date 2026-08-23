"""`mesopic truth score` — the command P2.9 reads the M1 accuracy number off.

`score()` has been tested since MK.2; what was missing was the seam between it and the
run it is supposed to measure. The engine writes `metrics_minute` rows into SQLite and
the scorer takes a sequence of `MetricRow`, so this command is the bridge: open the
store, read the run, and hand it to the scorer with the labels a human wrote.

The window it reads is deliberately wider than the clip. `score()` counts the minutes
the engine spoke about that the footage does not span, and a query clipped to the truth
file's own span could never return one — the guard would be structurally dead rather
than passing.

Written red-first.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from mesopic import cli
from mesopic.config.schema import MesopicConfig
from mesopic.store.store import Store
from mesopic.types import CameraId, MetricName, MetricRow, MinuteBucket

RUNNER = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mesopic.yaml"

CAMERA = CameraId("front-door")
STREAM_START = datetime(2026, 8, 23, 10, 0, tzinfo=UTC)
CLIP_SECONDS = 180.0


def _manifest_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "clip_id": "doorway-daylight-01",
        "sha256": "a" * 64,
        "duration_s": CLIP_SECONDS,
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "provenance": {
            "kind": "own_rig",
            "licence": "Owned",
            "licence_verified_utc": "2026-08-16",
            "url": None,
        },
        "consent": {"model_release": "obtained", "note": None},
        "scene": {
            "reference": "good_doorway",
            "mount_height_m": 2.8,
            "mount_angle_deg": 42.0,
            "lighting": "even_daylight",
        },
    }
    payload.update(overrides)
    return payload


def _truth_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "clip_id": "doorway-daylight-01",
        "labelled_by": "markos",
        "labelled_at_utc": "2026-08-23T09:00:00Z",
        "duration_s": CLIP_SECONDS,
        "crossings": [
            {"t_s": 10.0, "line_id": "entrance", "direction": "in"},
            {"t_s": 70.0, "line_id": "entrance", "direction": "in"},
            {"t_s": 130.0, "line_id": "entrance", "direction": "in"},
        ],
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _seed(data_dir: Path, rows: list[MetricRow]) -> None:
    """A migrated store with a real camera in it — `metrics_minute` cascades from it."""
    parsed = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    config = MesopicConfig.model_validate(parsed)
    with Store(data_dir / "mesopic.db") as store:
        store.migrate()
        store.apply_config(config)
        store.upsert_metrics(rows)


def _row(minute: int, value: float, metric: MetricName = MetricName.FOOTFALL) -> MetricRow:
    return MetricRow(
        camera_id=CAMERA,
        bucket=MinuteBucket(STREAM_START + timedelta(minutes=minute)),
        metric=metric,
        scope_id=None,
        value=value,
    )


def _invoke(tmp_path: Path, *extra: str) -> Any:
    truth = _write(tmp_path / "doorway-daylight-01.truth.json", _truth_payload())
    clip = _write(tmp_path / "doorway-daylight-01.clip.json", _manifest_payload())
    return RUNNER.invoke(
        cli.app,
        [
            "truth",
            "score",
            "--truth",
            str(truth),
            "--clip",
            str(clip),
            "--data-dir",
            str(tmp_path),
            "--stream-start",
            "2026-08-23T10:00:00Z",
            *extra,
        ],
    )


def test_scores_a_run_the_engine_left_in_the_store(tmp_path: Path) -> None:
    """The whole point of the command: labels plus a real run, one number out."""
    _seed(tmp_path, [_row(0, 1.0), _row(1, 1.0), _row(2, 1.0)])

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "total_error_pct: 0.0" in result.stdout


def test_gating_refuses_a_clip_that_may_not_back_a_published_number(tmp_path: Path) -> None:
    """The guard that makes the flag worth having: a `hard` clip cannot produce an M1
    number, and the refusal names the reason rather than printing a plausible figure."""
    _seed(tmp_path, [_row(0, 1.0), _row(1, 1.0), _row(2, 1.0)])
    truth = _write(tmp_path / "doorway-daylight-01.truth.json", _truth_payload())
    payload = _manifest_payload()
    payload["scene"]["reference"] = "hard"
    clip = _write(tmp_path / "doorway-daylight-01.clip.json", payload)

    result = RUNNER.invoke(
        cli.app,
        [
            "truth",
            "score",
            "--truth",
            str(truth),
            "--clip",
            str(clip),
            "--data-dir",
            str(tmp_path),
            "--stream-start",
            "2026-08-23T10:00:00Z",
            "--gating",
        ],
    )

    assert result.exit_code == 1
    assert "hard" in result.stdout
    assert "total_error_pct" not in result.stdout


def test_counts_minutes_the_engine_invented_after_the_footage_ended(tmp_path: Path) -> None:
    """The reason the store is read wider than the clip.

    A run that keeps emitting after the footage stops is over-counting, and it is the
    over-count a gate cannot see that is dangerous, because it reads as a pass. Reading
    only the clip's own minutes would make `stray_minutes` structurally zero — a guard
    that is dead rather than passing.
    """
    _seed(tmp_path, [_row(0, 1.0), _row(1, 1.0), _row(2, 1.0), _row(9, 40.0)])

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.stdout
    assert "stray_minutes: 1" in result.stdout
    assert "total_error_pct: 0.0" not in result.stdout


def test_a_stream_start_off_a_minute_boundary_is_refused_not_guessed(tmp_path: Path) -> None:
    """Media minute 0 spans the clip's first 60 s and a `MinuteBucket` is a wall-clock
    minute. Off a boundary the two do not correspond and every crossing lands in the
    wrong bucket, so this exits rather than picking one."""
    _seed(tmp_path, [_row(0, 1.0)])
    truth = _write(tmp_path / "doorway-daylight-01.truth.json", _truth_payload())
    clip = _write(tmp_path / "doorway-daylight-01.clip.json", _manifest_payload())

    result = RUNNER.invoke(
        cli.app,
        [
            "truth",
            "score",
            "--truth",
            str(truth),
            "--clip",
            str(clip),
            "--data-dir",
            str(tmp_path),
            "--stream-start",
            "2026-08-23T10:00:30Z",
        ],
    )

    assert result.exit_code == 1
    assert "minute boundary" in result.stdout


def test_a_stream_start_without_an_offset_is_refused(tmp_path: Path) -> None:
    """A naive instant means the number silently depends on the box's timezone."""
    _seed(tmp_path, [_row(0, 1.0)])
    truth = _write(tmp_path / "doorway-daylight-01.truth.json", _truth_payload())
    clip = _write(tmp_path / "doorway-daylight-01.clip.json", _manifest_payload())

    result = RUNNER.invoke(
        cli.app,
        [
            "truth",
            "score",
            "--truth",
            str(truth),
            "--clip",
            str(clip),
            "--data-dir",
            str(tmp_path),
            "--stream-start",
            "2026-08-23T10:00:00",
        ],
    )

    assert result.exit_code == 1
    assert "UTC offset" in result.stdout


def test_a_truncated_read_is_refused_rather_than_scored(tmp_path: Path, monkeypatch: Any) -> None:
    """A short read is indistinguishable from silence downstream, and silence is a claim
    that nothing happened — so a run that hits the ceiling cannot be scored at all."""
    monkeypatch.setattr(cli, "SCORE_ROW_LIMIT", 2)
    _seed(tmp_path, [_row(0, 1.0), _row(1, 1.0), _row(2, 1.0)])

    result = _invoke(tmp_path)

    assert result.exit_code == 1
    assert "truncated" in result.stdout


def test_scores_traffic_when_asked_for_it(tmp_path: Path) -> None:
    """`--metric` selects both sides of the comparison. Footfall counts inward crossings
    only; `line_cross` counts an exit as traffic too, and the two are read against
    different bands — so a run's rows are filtered to the metric being measured."""
    _seed(
        tmp_path,
        [
            _row(0, 1.0, MetricName.LINE_CROSS),
            _row(1, 1.0, MetricName.LINE_CROSS),
            _row(2, 1.0, MetricName.LINE_CROSS),
            _row(0, 99.0, MetricName.FOOTFALL),
        ],
    )

    result = _invoke(tmp_path, "--metric", "line_cross")

    assert result.exit_code == 0, result.stdout
    assert "metric: line_cross" in result.stdout
    assert "total_error_pct: 0.0" in result.stdout


def test_a_data_dir_with_a_tilde_finds_the_store(tmp_path: Path, monkeypatch: Any) -> None:
    """`mesopic run` canonicalises the same flag, and a path that works for one command
    and not the other is the kind of difference nobody reports as a bug."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _seed(tmp_path, [_row(0, 1.0), _row(1, 1.0), _row(2, 1.0)])

    result = _invoke(tmp_path, "--data-dir", "~")

    assert result.exit_code == 0, result.stdout
    assert "total_error_pct: 0.0" in result.stdout
