<div align="center">

# Muster

### A billion cameras already watching. Almost none of them counting.

**Open-source video-intelligence for the cameras you already own.**

<!-- Badges (CI, licence, stars, discussions) go in when the repository goes public;
     they render broken until then and would assert things that are not true yet. -->

> **Status:** pre-launch. APIs, config, and metric definitions will change before `v1.0`.

</div>

---

## What is Muster?

Muster turns the **RTSP/ONVIF security cameras you already have** into business sensors — footfall,
queues, dwell, occupancy — **without new hardware and without a sales call**. The heavy computer
vision runs *on your box* (a cheap CPU-only mini-PC is enough); **footage never leaves the building.**
Only anonymous foot-point coordinates and derived metrics are ever computed, and frames are discarded
the instant they're processed. It's the Frigate/Plausible open-source playbook aimed at the top-down,
sales-led market that Verkada ($5.8B, Dec 2025) and Spot AI grew into — except Muster is MIT-licensed,
self-serve, and yours to run.

- **Engine** — MIT-licensed, Python-first, CPU-only capable. It does the vision.
- **Cloud** — a thin, optional, paid layer for multi-site dashboards, history, and alerts. It only ever
  sees metrics JSON, never pixels.

If you never want the cloud, you never need it. The engine and a local dashboard are free forever.

---

## What it measures

**The core six** — the metrics a shop, café, gym, clinic, or venue actually acts on:

- **Footfall** — people entering/exiting over time.
- **Live occupancy** — how many people are in a space right now.
- **Queue length** — number of people waiting in a defined region.
- **Dwell time** — how long people linger in a zone.
- **Line-crossings** — directional counts across a virtual tripwire.
- **Conversion** — entries vs. a downstream action (e.g. footfall → till zone).

**Two cheap adjacencies** that fall out of the same pipeline nearly for free:

- **Zone heatmaps** — spatial density of foot-points over a period.
- **Staff-vs-customer filtering** — separate the people you employ from the people you serve.

> **Not in v1 (roadmap):** loss-prevention/theft detection, face recognition, pose estimation,
> ANPR/parking. See [Roadmap](#roadmap).

---

## Quickstart

Point Muster at any RTSP camera and watch it count. No account, no cloud, no config file to start.

```bash
make image                      # build the engine container

docker run -d --name muster \
  -e MUSTER_RTSP_URL="rtsp://user:pass@192.168.1.64:554/stream1" \
  -p 8080:8080 \
  -v muster-data:/data \
  muster-engine:dev
```

Then open **http://localhost:8080** for the local dashboard.

> A published image, so the first step becomes a single `docker run`, ships with the
> first release.

- `MUSTER_RTSP_URL` — the ONVIF/RTSP stream to analyse. That's the only thing you *must* provide.
- `-p 8080:8080` — the local HUD dashboard + metrics API.
- `-v muster-data:/data` — persists the SQLite metric store and your `muster.yaml` config.

By default the engine samples ~2–5 effective FPS and runs an INT8-quantized YOLO model on CPU — sized
so an **Intel N100-class mini-PC (4 cores, no GPU)** handles a couple of cameras. A GPU, Coral TPU, or
NPU is an **optional** speed-up, never a requirement.

> For more than the single-camera demo, mount a config file (`-v ./muster.yaml:/data/muster.yaml`)
> and define your cameras, lines, and zones — see [Example config](#example-config).

---

## How it fits together

Video stays local. Only metrics can (optionally) sync to the hosted dashboard.

```
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │   Camera A   │        │   Camera B   │        │   Camera C   │
   │  RTSP/ONVIF  │        │  RTSP/ONVIF  │        │  RTSP/ONVIF  │
   └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
          │  H.264/H.265         │                      │
          └──────────────────────┼──────────────────────┘
                                  ▼
        ┌───────────────────────────────────────────────────┐
        │            MUSTER ENGINE  (your box, MIT)           │
        │                                                     │
        │  PyAV/FFmpeg ingest → sample ~2–5 fps               │
        │  YOLO (nano/small, INT8) via ONNX RT / OpenVINO     │
        │  ByteTrack multi-object tracking                    │
        │  lines · zones · dwell · occupancy · conversion     │
        │                                                     │
        │  ⛔ frames discarded immediately — never stored     │
        │  ✅ only foot-points + metrics → SQLite (/data)      │
        └───────────────┬──────────────────────┬──────────────┘
                        │                      │
             ┌──────────▼─────────┐   metrics JSON only
             │   Local dashboard   │   (authenticated,
             │   HUD @ :8080       │    per-site channel)
             │   CSV · MQTT ·      │            │
             │   webhook · Prom.   │            ▼
             └────────────────────┘   ┌───────────────────────┐
                                       │   HOSTED CLOUD (paid)  │
                                       │   FastAPI + Timescale  │
                                       │   multi-site · history │
                                       │   email/webhook alerts │
                                       │   ⛔ never sees pixels  │
                                       └───────────────────────┘
```

**The one thing to internalise:** inference never runs in Muster's cloud in v1. The cloud is a
metrics viewer. Cloud-side inference is not built, not on by default, and would need its own
decision record before it ever were — it would break both the cost model and the privacy posture.

---

## Supported cameras

**Any camera that speaks RTSP or ONVIF** — which is essentially every IP security camera made in the
last decade (Hikvision, Dahua, Reolink, Amcrest, Axis, Ubiquiti, generic ONVIF, and re-badges of all
of the above). Muster ingests the existing stream; it does not need its own sensor, unlike
FootfallCam or Milesight, which require dedicated hardware per door.

- Prefer a **sub-stream** (e.g. 640×480–1280×720) for counting — it's cheaper to decode and plenty for
  foot-point detection. Muster does not need your 4K main stream.
- Both **H.264** and **H.265/HEVC** are supported via FFmpeg.

---

## Integrations

Day-one, first-class:

- **[Frigate](https://frigate.video/)** — run alongside your existing Frigate NVR; consume the same
  cameras. Muster adds the business analytics Frigate deliberately doesn't do.
- **[Home Assistant](https://www.home-assistant.io/) (MQTT)** — Muster publishes live occupancy,
  counts, and queue state to MQTT for automations and Lovelace dashboards.

Exports, everywhere:

| Channel        | Use it for                                        |
| -------------- | ------------------------------------------------- |
| **CSV**        | ad-hoc analysis, spreadsheets, BI import          |
| **Webhook**    | push events/rollups to your own backend           |
| **MQTT**       | Home Assistant, Node-RED, and the broader IoT bus |
| **Prometheus** | `/metrics` scrape endpoint for Grafana/alerting   |

---

## Example config

`muster.yaml` — one camera, one counting line, one dwell zone. Coordinates are normalized `[0,1]`
image space, so they survive resolution changes.

```yaml
# muster.yaml — mount at /data/muster.yaml
site:
  site_id: "front-of-house"
  timezone: "Europe/London"        # display only; everything is stored in UTC

cameras:
  - camera_id: entrance
    name: "Front door"
    source:
      kind: rtsp                   # rtsp | onvif | frigate
      url: "rtsp://user:pass@192.168.1.64:554/stream1"
      transport: tcp
    reference_resolution: [1280, 720]

# Directional tripwire: A→B counts as "in", B→A as "out".
lines:
  - line_id: door
    camera_id: entrance
    a: [0.20, 0.85]
    b: [0.80, 0.85]
    positive_dir: in
    metrics: [line_cross, footfall]

# Region people linger in; emits dwell, live occupancy, and queue length.
zones:
  - zone_id: waiting_area
    camera_id: entrance
    role: queue                    # area | queue | staff
    polygon: [[0.10, 0.40], [0.60, 0.40], [0.60, 0.95], [0.10, 0.95]]
    metrics: [dwell_seconds, occupancy, queue_len]

exporters:
  mqtt: { enabled: true, broker: "192.168.1.10", base_topic: "muster" }
  prometheus: { enabled: true }    # scrape at :8080/metrics

# Optional: push metrics-only to the hosted dashboard. Omit to stay fully local.
cloud_sync:
  enabled: false
  # site_token_env: "MUSTER_SITE_TOKEN"   # by reference; never the token itself
```

A fuller worked example, with two cameras and every metric wired up, is in
[`examples/muster.yaml`](./examples/muster.yaml).

---

## Muster Cloud (optional, paid)

The engine and local dashboard are **free forever.** Cloud is the thin, self-serve layer for people who
want multi-site rollups, 90-day history, and alerts without running their own Postgres — built on
FastAPI + TimescaleDB, magic-link auth, and Stripe self-serve. **It only ever receives metrics JSON
over an authenticated per-site channel; it never receives, requests, or stores video.**

| Tier                       | Price          | What you get                                                          |
| -------------------------- | -------------- | --------------------------------------------------------------------- |
| **Self-hosted**            | **Free**       | Full engine + local dashboard + all exports. No cloud.                |
| **Cloud — single site**    | **£19/mo**     | Hosted dashboard, 90-day metric history, email + webhook alerts.      |
| **Cloud — multi-site**     | **£49/mo**     | Everything above, across all your locations, in one view.             |
| **Vertical models** (later)| **£50/yr**     | Fine-tuned models for specific verticals (roadmap).                   |
| **Managed appliance** (later) | hardware + sub, TBD | A pre-flashed mini-PC that self-registers — "you only ever see the dashboard." For the non-technical buyer. |

Three onboarding paths, **all keeping video local:** (1) self-hosted free; (2) hosted dashboard +
local engine (the primary paid path — only metrics sync); (3) the managed appliance (roadmap) for
buyers who never want to touch Docker.

> **Pricing is not final.** Muster has no paying customers yet; these tiers are what we intend to
> charge, and we would rather say so than pretend otherwise. The self-hosted tier being free forever
> is the part that is not going to change.

---

## Privacy

Muster is built **UK/EU-first and GDPR-by-design.**

- **Footage never leaves the building.** All inference is local. Muster's cloud does not run inference
  in v1 and cannot receive video.
- **Frames are discarded immediately** after they're processed. Muster stores only foot-point
  coordinates and the metrics derived from them.
- **No face recognition, no biometric storage in v1.** Muster counts and tracks anonymous points, not
  identities.
- **Metrics-only retention.** The hosted cloud keeps derived metrics for **90 days**; never video.

If you self-host and disable the cloud sync, nothing leaves your network at all.

---

## Roadmap

| Horizon     | What                                                                                     |
| ----------- | ---------------------------------------------------------------------------------------- |
| **MVP (6–8 wks)** | Engine (core six + two adjacencies), local HUD dashboard, Docker one-command run, Frigate + HA/MQTT integration, CSV/webhook/MQTT/Prometheus exports. |
| **Month 2–3** | Cloud v0: hosted dashboard, metrics-only sync, magic-link auth, Stripe, email/webhook alerts, TimescaleDB rollups. |
| **Later**   | Managed appliance (pre-flashed, self-registering); vertical fine-tuned models (£50/yr).  |
| **Deferred** | Loss-prevention/theft, face recognition, pose estimation, ANPR/parking. Explicitly **not** v1. |
| **ADR-flagged future** | Optional, separately-priced cloud-side inference — never on by default, never required. |

Dates are targets, not commitments — this is early software built in the open.

---

## Repository layout

**This repository is the engine, and only the engine.** It is MIT, top to bottom, with
no proprietary code in it. The hosted cloud is a separate service in a separate,
closed repository — it is not required to run anything here, and nothing here depends
on it.

```
muster/
├── muster-engine/          # the MIT engine — the whole open-source product
│   ├── src/muster/
│   │   ├── types.py            domain vocabulary: ids, UTC time, normalized geometry
│   │   ├── config/             muster.yaml schema + loader (validated, fail-loud)
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
│   │   └── cli.py              `muster run | discover | calibrate | export | doctor`
│   └── tests/
├── docker/                 # engine image + the local dev stack (mediamtx, mosquitto)
├── examples/muster.yaml    # a worked config: one camera, one line, one zone
└── pyproject.toml          # the shared lint / type / test / boundary config
```

The one piece of cloud-facing code here is `muster/sync/` — the client that pushes your
own metrics to the hosted dashboard if you choose to use it. It is MIT like everything
else, it is off by default, and you can read exactly what it sends. It reads the metrics
tables and nothing else: an enforced import boundary means it has no path to a frame at
all, so "video never leaves the building" is a structural property of the code rather
than a promise in a README.

## Documentation

Architecture, algorithms, and the decision records are published alongside the docs site
as it lands (see the roadmap). The short version of the design lives in this README; the
things most people want next are:

| Question | Where |
|---|---|
| How do I run it? | [Quickstart](#quickstart) above |
| How do I configure cameras, lines, and zones? | [Example config](#example-config), and `examples/muster.yaml` |
| How do I contribute? | [CONTRIBUTING.md](./CONTRIBUTING.md) |
| I found a security issue | [SECURITY.md](./SECURITY.md) |
| What does it store about people? | [Privacy](#privacy) above |

---

## Contributing

Muster is MIT-licensed and contributions are welcome — especially camera compatibility reports,
integration adapters, and metric-accuracy validation on real footage.

- Read **[CONTRIBUTING.md](./CONTRIBUTING.md)** for dev setup, coding conventions, and the PR process.
- All contributors are held to our **[Code of Conduct](./CODE_OF_CONDUCT.md)**.
- Found a bug or have a camera that misbehaves? Open an
  [issue](../../issues). Questions and ideas go in [Discussions](../../discussions).

**Conventions in brief:** Python-first, small cohesive functions (≤40 lines), single responsibility,
DRY, self-documenting code (comments explain *why*), imports at the top of the file, type aliases for
domain values, and static-analysis-clean code. Significant decisions are recorded as ADRs.

---

## License

Muster's engine is released under the **[MIT License](./LICENSE)** — use it, fork it, ship it. The
hosted cloud service is a separate, optional, paid offering; running your own engine and dashboard
never requires it.

<div align="center">
<sub>Muster · started 13 July 2026 · open core, video-intelligence for cameras you already own.</sub>
</div>
