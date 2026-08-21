<div align="center">

# Mesopic

### A billion cameras already watching. Almost none of them counting.

**Open-source video-intelligence for the cameras you already own.**

<!-- Badges (CI, licence, stars, discussions) go in when the repository goes public;
     they render broken until then and would assert things that are not true yet. -->

<!-- The 60-second demo GIF goes here (P5.5), as ![Mesopic counting a storefront](docs/assets/demo.gif).
     Absent until it is recorded, because a broken image is worse than no image on the one
     screen a launch visitor actually looks at. Record it with `make gif` +
     scripts/record-demo-gif.sh, which enforces the < 5 MB / <= 60 s budget the M3 gate sets. -->

> **Status:** pre-launch. APIs, config, and metric definitions will change before `v1.0`.

</div>

---

## What is Mesopic?

Mesopic turns the **RTSP/ONVIF security cameras you already have** into business sensors — footfall,
queues, dwell, occupancy — **without new hardware and without a sales call**. The heavy computer
vision runs *on your box* (a cheap CPU-only mini-PC is enough); **footage never leaves the building.**
Only anonymous foot-point coordinates and derived metrics are ever computed, and frames are discarded
the instant they're processed. It's the Frigate/Plausible open-source playbook aimed at the top-down,
sales-led market that Verkada ($5.8B, Dec 2025) and Spot AI grew into — except Mesopic is MIT-licensed,
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

Point Mesopic at any RTSP camera and watch it count. No account, no cloud, no config file to start.

```bash
docker run -d --name mesopic \
  -e MESOPIC_RTSP_URL="rtsp://user:pass@192.168.1.64:554/stream1" \
  -p 8080:8080 \
  -v mesopic-data:/data \
  ghcr.io/mesopic-labs/mesopic-engine:latest
```

Then open **http://localhost:8080** for the local dashboard.

The image is multi-arch (`linux/amd64`, `linux/arm64`), so the same command works on a
mini-PC and on an ARM board. It ships with **no model weights baked in** — the first
start downloads the detector, quantizes it to INT8, and caches it in `/data`, so give
that first run a minute longer than later ones. Every release is signed; if you want to
check that before running it:

```bash
cosign verify ghcr.io/mesopic-labs/mesopic-engine:latest \
  --certificate-identity-regexp '^https://github\.com/.+/Mesopic/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Prefer to build it yourself? `make image` does exactly what CI does.

### Try it without a camera

No RTSP stream to hand? One command brings up the engine, the dashboard, a synthetic
camera and an MQTT broker together:

```bash
make demo                       # engine + dashboard + sample stream + broker
```

It prints the dashboard URL and a generated password, then runs in the foreground;
`Ctrl-C` stops it and `docker compose --profile demo down -v` removes it.

**The counts will read zero.** The bundled stream is a test pattern, not footage of a
room, so there is nobody in it to count — what the demo shows is the pipeline running end
to end: the stream connects, the dashboard renders live, MQTT publishes retained topics
and `/metrics` scrapes. Point it at something real to see real numbers:

```bash
MESOPIC_RTSP_URL="rtsp://user:pass@192.168.1.64:554/stream1" make demo
```

Reading the dashboard never asks for a password. The password guards *changes* — drawing
zones and lines, and anything else that rewrites your config.

- `MESOPIC_RTSP_URL` — the ONVIF/RTSP stream to analyse. That's the only thing you *must* provide.
- `-p 8080:8080` — the local HUD dashboard + metrics API.
- `-v mesopic-data:/data` — persists the SQLite metric store and your `mesopic.yaml` config.

By default the engine samples ~2–5 effective FPS and runs an INT8-quantized YOLOX-nano model on CPU — sized
so an **Intel N100-class mini-PC (4 cores, no GPU)** handles a couple of cameras. A GPU, Coral TPU, or
NPU is an **optional** speed-up, never a requirement.

> For more than the single-camera demo, mount a config file (`-v ./mesopic.yaml:/data/mesopic.yaml`)
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
        │            MESOPIC ENGINE  (your box, MIT)           │
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

**The one thing to internalise:** inference never runs in Mesopic's cloud in v1. The cloud is a
metrics viewer. Cloud-side inference is not built, not on by default, and would need its own
decision record before it ever were — it would break both the cost model and the privacy posture.

---

## Supported cameras

**Any camera that speaks RTSP or ONVIF** — which is essentially every IP security camera made in the
last decade (Hikvision, Dahua, Reolink, Amcrest, Axis, Ubiquiti, generic ONVIF, and re-badges of all
of the above). Mesopic ingests the existing stream; it does not need its own sensor, unlike
FootfallCam or Milesight, which require dedicated hardware per door.

- Prefer a **sub-stream** (e.g. 640×480–1280×720) for counting — it's cheaper to decode and plenty for
  foot-point detection. Mesopic does not need your 4K main stream.
- Both **H.264** and **H.265/HEVC** are supported via FFmpeg.

---

## How Mesopic compares

Mesopic is not the first open-source project to point a model at a camera. It is aimed at a different
layer from most of them.

| | What it is | Where Mesopic differs |
| --- | --- | --- |
| **[Frigate](https://frigate.video/)** | An excellent open-source NVR — recording, review, and real-time detection, with accelerator support Mesopic does not try to match. | Frigate answers *"what happened, and do I have the clip?"* Mesopic answers *"how many, how long, and is that up on last week?"* A zone in an NVR tells you an object was present; that is not the same thing as a time-weighted metric series. Run both — Frigate is a first-class [integration](#integrations), not a competitor. |
| **[OpenDataCam](https://opendata.cam/)** | MIT, and the reference tool for *urban traffic* studies — modal split, turn counts, 50+ object classes across drawn counters. | Built for streets and pitched at city researchers, and it wants an NVIDIA GPU or a Jetson. Mesopic is CPU-first and shaped for premises: occupancy, dwell, queue length, and conversion are metrics OpenDataCam does not model. Line-crossing is where the two genuinely overlap. |
| **Perception libraries** — e.g. [Supervision](https://github.com/roboflow/supervision), [trio-retina](https://github.com/machinefi/trio-retina) | Well-built toolkits that turn detections into tracks and zone/line events. | They stop at the event stream, by design. Turning `zone.enter` into a defensible occupancy figure — sampling, Δt-weighting, minute buckets, a store, idempotent rollups — is most of the work, and it is the part Mesopic is. |
| **Commercial counters** — V-Count, RetailNext, FootfallCam, Verkada, Spot AI | Mature, accurate, supported, and the incumbents Mesopic is aimed at. | A dedicated sensor per door or a proprietary camera estate, an annual contract, and a sales call. Mesopic runs on the cameras already screwed to your ceiling, installs in one command, and the engine is MIT. |

Two things Mesopic does not claim: it is not more accurate than an audited commercial counter today,
and it does not do the security-camera job Frigate does well. It does the measurement layer, in the
open, on hardware you already own.

---

## Integrations

### Home Assistant (MQTT)

Turn on the MQTT exporter and Mesopic announces itself to Home Assistant — no YAML on the
HA side:

```yaml
exporters:
  mqtt:
    enabled: true
    broker: "192.168.1.10"
    port: 1883
    base_topic: "mesopic"
    discovery: true                # publish HA discovery configs
    discovery_prefix: "homeassistant"
```

Each minute's value is published **retained** to `mesopic/<camera_id>/<metric>`, with a
fourth segment for anything scoped to a zone or line
(`mesopic/entrance/dwell_seconds/waiting_area`). Retained is the point: a sensor shows its
last known value the moment HA restarts, instead of `unknown` until the next minute ticks.

Discovery configs go to `homeassistant/sensor/<unique_id>/config`, where `unique_id` is
`mesopic_<site>_<camera>_<metric>[_<scope>]`. Each entity is declared
`state_class: measurement`, so HA's statistics average it as a level rather than
differencing it as a counter.

Liveness rides on `mesopic/status` (`online` / `offline`), set as the MQTT will, and every
entity points its `availability_topic` at it. If the engine dies, its sensors go
unavailable in HA rather than freezing on a stale number that looks current.

> **Known limitation:** renaming or deleting a zone leaves its retained discovery config
> on the broker, so the old entity lingers in HA until you clear that topic by hand.

### Frigate

Already running [Frigate](https://frigate.video/)? Mesopic can read its object stream
instead of decoding the camera a second time — no second detector, no second decode
budget:

```yaml
frigate:                           # site-wide: one Frigate per site
  broker: "192.168.1.10"
  port: 1883
  username_env: "FRIGATE_MQTT_USER"     # by reference; never inline
  password_env: "FRIGATE_MQTT_PASSWORD"

cameras:
  - camera_id: entrance
    source:
      kind: frigate
      mqtt_topic: "frigate/events"      # default carries every camera on the box
      camera: "front_door"              # Frigate's name, when it differs
```

Geometry, metrics, and storage are identical to the RTSP path — the adapter produces the
same tracks, so nothing downstream can tell the difference. A `kind: frigate` camera with
no `frigate.broker` is rejected when the config loads, rather than starting up and
silently counting nothing.

### Exports, everywhere

| Channel        | Use it for                                        |
| -------------- | ------------------------------------------------- |
| **CSV**        | ad-hoc analysis, spreadsheets, BI import          |
| **Webhook**    | push events/rollups to your own backend           |
| **MQTT**       | Home Assistant, Node-RED, and the broader IoT bus |
| **Prometheus** | `/metrics` scrape endpoint for Grafana/alerting   |

Prometheus scrapes `:8080/metrics` with no extra configuration. Metric values arrive as
`mesopic_metric`, labelled by camera, metric, and scope; the engine's own health gauges
share the `mesopic_` prefix.

---

## Example config

`mesopic.yaml` — one camera, one counting line, one dwell zone. Coordinates are normalized `[0,1]`
image space, so they survive resolution changes.

```yaml
# mesopic.yaml — mount at /data/mesopic.yaml
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
  mqtt: { enabled: true, broker: "192.168.1.10", base_topic: "mesopic" }
  prometheus: { enabled: true }    # scrape at :8080/metrics

# Optional: push metrics-only to the hosted dashboard. Omit to stay fully local.
cloud_sync:
  enabled: false
  # site_token_env: "MESOPIC_SITE_TOKEN"   # by reference; never the token itself
```

A fuller worked example, with two cameras and every metric wired up, is in
[`examples/mesopic.yaml`](./examples/mesopic.yaml).

---

## Mesopic Cloud (optional, paid)

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

**Cloud is not built yet.** If you want to hear when it is,
[join the waitlist](https://mesopic.dev/?source=readme) — one email, when there is something real to
try. Nothing about the engine depends on it, and self-hosting stays free either way.

> **Pricing is not final.** Mesopic has no paying customers yet; these tiers are what we intend to
> charge, and we would rather say so than pretend otherwise. The self-hosted tier being free forever
> is the part that is not going to change.

---

## Privacy

Mesopic is built **UK/EU-first and GDPR-by-design.**

- **Footage never leaves the building.** All inference is local. Mesopic's cloud does not run inference
  in v1 and cannot receive video.
- **Frames are discarded immediately** after they're processed. Mesopic stores only foot-point
  coordinates and the metrics derived from them.
- **No face recognition, no biometric storage in v1.** Mesopic counts and tracks anonymous points, not
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

## Documentation

Architecture, algorithms, and the decision records are published alongside the docs site
as it lands (see the roadmap). The short version of the design lives in this README; the
things most people want next are:

| Question | Where |
|---|---|
| How do I run it? | [Quickstart](#quickstart) above |
| How do I configure cameras, lines, and zones? | [Example config](#example-config), and `examples/mesopic.yaml` |
| How do I connect Home Assistant or Frigate? | [Integrations](#integrations) above |
| What licence is the model under? | [License](#license) below, and `/healthz` on a running engine |
| How do I contribute? | [CONTRIBUTING.md](./CONTRIBUTING.md) |
| I found a security issue | [SECURITY.md](./SECURITY.md) |
| What does it store about people? | [Privacy](#privacy) above |

---

## Contributing

Mesopic is MIT-licensed and contributions are welcome — especially camera compatibility reports,
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

Mesopic's engine is released under the **[MIT License](./LICENSE)** — use it, fork it, ship it. The
hosted cloud service is a separate, optional, paid offering; running your own engine and dashboard
never requires it.

**The model is a separate artefact from the engine, on purpose.** No weights are baked
into the image; the detector is downloaded at first run, which keeps its licence its own
rather than something the image inherits. The default install is free of AGPL end to
end — MIT engine, MIT runtime, and **Apache-2.0** weights (YOLOX-nano) — so nothing here
creates a combined work for the AGPL's network clause to attach to. The licence of the
model you are actually running is reported at `/healthz`, because "which licence is on
the box" should be a question you can answer by curling it rather than by reading source.

If you want an Ultralytics YOLO model instead, it is available through the opt-in
`[ultralytics]` extra. That package is **AGPL-3.0**, installing it is your own deliberate
act, and the CLI says so the first time you use it. It is never pulled in by default.

<div align="center">
<sub>Mesopic · started 13 July 2026 · open core, video-intelligence for cameras you already own.</sub>
</div>
