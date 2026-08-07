-- Muster edge store — authoritative DDL (engine-architecture.md §11, ADR-0009).
--
-- NOTHING IN THIS SCHEMA CAN HOLD A PIXEL. There is no image column, no crop column,
-- no bbox-pixel column, and there never will be. Foot-points live only long enough to
-- derive events; the frame is discarded in the camera worker (ADR-0005). A schema audit
-- test asserts this, so adding such a column fails CI rather than review.
--
-- Migrations are forward-only and live in ./migrations/. This file is the current shape.

PRAGMA journal_mode = WAL;      -- many readers, one writer
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;    -- WAL + NORMAL is durable enough for a metrics store

-- One row per configured camera on this site. Mirror of config; config wins on conflict.
CREATE TABLE IF NOT EXISTS cameras (
    camera_id     TEXT PRIMARY KEY,          -- CameraId, stable, from config
    name          TEXT NOT NULL,
    source_kind   TEXT NOT NULL
                    CHECK (source_kind IN ('rtsp','onvif','frigate')),
    enabled       INTEGER NOT NULL DEFAULT 1,
    ref_width     INTEGER NOT NULL,          -- reference frame size geometry was authored against
    ref_height    INTEGER NOT NULL,
    updated_at    TEXT NOT NULL              -- ISO-8601 UTC
) STRICT;

-- Polygonal regions (incl. staff-zones and queue-zones) in NORMALIZED coords.
CREATE TABLE IF NOT EXISTS zones (
    zone_id       TEXT PRIMARY KEY,          -- ZoneId
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'area'
                    CHECK (role IN ('area','queue','staff')),
    polygon       TEXT NOT NULL,             -- JSON: [[x,y],...] normalized [0,1], closed
    updated_at    TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);

-- Directed counting lines in NORMALIZED coords. positive_dir labels the +1 sense.
CREATE TABLE IF NOT EXISTS lines (
    line_id       TEXT PRIMARY KEY,          -- LineId
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    ax REAL NOT NULL, ay REAL NOT NULL,      -- segment endpoint A (normalized)
    bx REAL NOT NULL, by REAL NOT NULL,      -- segment endpoint B (normalized)
    positive_dir  TEXT NOT NULL DEFAULT 'in'
                    CHECK (positive_dir IN ('in','out')),
    updated_at    TEXT NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS idx_lines_camera ON lines(camera_id);

-- Short-retention raw event log: local debugging + re-aggregation. NOT synced.
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY,
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    ts            TEXT NOT NULL,             -- capture time, ISO-8601 UTC
    kind          TEXT NOT NULL,             -- 'line_cross' | 'zone_enter' | 'zone_exit' | ...
    track_id      INTEGER NOT NULL,          -- TrackId, per-camera per-run only
    line_id       TEXT REFERENCES lines(line_id) ON DELETE SET NULL,
    zone_id       TEXT REFERENCES zones(zone_id) ON DELETE SET NULL,
    direction     INTEGER,                   -- +1 / -1 for line crossings
    is_staff      INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_cam_kind_ts ON events(camera_id, kind, ts);

-- The durable, sync-source time series. One row per (camera, metric, scope, minute).
CREATE TABLE IF NOT EXISTS metrics_minute (
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    bucket        TEXT NOT NULL,             -- MinuteBucket, ISO-8601 UTC, floor-to-minute
    metric        TEXT NOT NULL,             -- see muster.types.MetricName
    scope_id      TEXT,                      -- zone_id / line_id, nullable for camera-wide
    value         REAL NOT NULL,
    staff_value   REAL,                      -- staff sub-count, nullable
    sample_count  INTEGER NOT NULL DEFAULT 0,-- contributing frames: confidence + gap detection
    synced_at     TEXT,                      -- NULL until shipped to cloud; the sync cursor
    PRIMARY KEY (camera_id, metric, scope_id, bucket)
) STRICT, WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_metrics_unsynced ON metrics_minute(synced_at) WHERE synced_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_metrics_bucket ON metrics_minute(bucket);

-- Heatmaps are a grid blob, kept separate so metrics_minute stays scalar and queryable.
CREATE TABLE IF NOT EXISTS heatmap_minute (
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    zone_id       TEXT NOT NULL REFERENCES zones(zone_id) ON DELETE CASCADE,
    bucket        TEXT NOT NULL,
    grid_w        INTEGER NOT NULL,
    grid_h        INTEGER NOT NULL,
    -- Packed uint16, grid_w*grid_h. UNIT: DECISECONDS of foot-point presence per cell.
    -- NOT a frame count: accumulation is dt-weighted, because the adaptive sampler
    -- samples LESS when the scene is busy — a per-frame counter would render the load
    -- controller, not the floor. Saturates at 65535 ds (~1.8h), unreachable in a minute.
    counts        BLOB NOT NULL,
    synced_at     TEXT,
    PRIMARY KEY (camera_id, zone_id, bucket)
) STRICT, WITHOUT ROWID;

-- Schema version for the sync contract (ADR-0009: edge and cloud schemas co-evolve).
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;
