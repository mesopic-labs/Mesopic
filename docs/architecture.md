<!--include:README.md#How it fits together-->

---

## Repository layout

**This repository is the engine, and only the engine.** It is MIT, top to bottom, with
no proprietary code in it. The hosted cloud is a separate service in a separate,
closed repository — it is not required to run anything here, and nothing here depends
on it.

```
mesopic/
├── mesopic-engine/          # the MIT engine — the whole open-source product
│   ├── src/mesopic/
│   │   ├── types.py            domain vocabulary: ids, UTC time, normalized geometry
│   │   ├── config/             mesopic.yaml schema + loader (validated, fail-loud)
│   │   ├── ingest/             RTSP/ONVIF via PyAV; Frigate via MQTT; one interface
│   │   ├── sampler/            adaptive fps — the CPU-budget lever
│   │   ├── detector/           ONNX Runtime; model fetch/export/quantize/cache
│   │   ├── tracker/            ByteTrack; owns foot-point + pixel→normalized
│   │   ├── analytics/          tracks + geometry → raw events; metric plugins
│   │   ├── aggregator/         events → minute buckets, idempotent
│   │   ├── store/              SQLite (WAL, STRICT) — single writer, sync buffer
│   │   ├── exporters/          CSV · webhook · MQTT · Prometheus
│   │   ├── api/                local HUD dashboard, /healthz, /metrics
│   │   ├── sync/               metrics-only push to the cloud — optional, off by default
│   │   ├── supervisor/         process-per-camera, backpressure, restart
│   │   └── cli.py              `mesopic run | discover | calibrate | export | doctor`
│   └── tests/
├── docker/                 # engine image + the local dev stack (mediamtx, mosquitto)
├── examples/mesopic.yaml    # a worked config: one camera, one line, one zone
└── pyproject.toml          # the shared lint / type / test / boundary config
```

The one piece of cloud-facing code here is `mesopic/sync/` — the client that pushes your
own metrics to the hosted dashboard if you choose to use it. It is MIT like everything
else, it is off by default, and you can read exactly what it sends. It reads the metrics
tables and nothing else: an enforced import boundary means it has no path to a frame at
all, so "video never leaves the building" is a structural property of the code rather
than a promise in a README.
