## How Mesopic compares

Mesopic is not the first open-source project to point a model at a camera. It is aimed at a different
layer from most of them.

| | What it is | Where Mesopic differs |
| --- | --- | --- |
| **[Frigate](https://frigate.video/)** | An excellent open-source NVR — recording, review, and real-time detection, with accelerator support Mesopic does not try to match. | Frigate answers *"what happened, and do I have the clip?"* Mesopic answers *"how many, how long, and is that up on last week?"* A zone in an NVR tells you an object was present; that is not the same thing as a time-weighted metric series. Run both — Frigate is a first-class [integration](integrations.md), not a competitor. |
| **[OpenDataCam](https://opendata.cam/)** | MIT, and the reference tool for *urban traffic* studies — modal split, turn counts, 50+ object classes across drawn counters. | Built for streets and pitched at city researchers, and it wants an NVIDIA GPU or a Jetson. Mesopic is CPU-first and shaped for premises: occupancy, dwell, queue length, and conversion are metrics OpenDataCam does not model. Line-crossing is where the two genuinely overlap. |
| **Perception libraries** — e.g. [Supervision](https://github.com/roboflow/supervision), [trio-retina](https://github.com/machinefi/trio-retina) | Well-built toolkits that turn detections into tracks and zone/line events. | They stop at the event stream, by design. Turning `zone.enter` into a defensible occupancy figure — sampling, Δt-weighting, minute buckets, a store, idempotent rollups — is most of the work, and it is the part Mesopic is. |
| **Commercial counters** — V-Count, RetailNext, FootfallCam, Verkada, Spot AI | Mature, accurate, supported, and the incumbents Mesopic is aimed at. | A dedicated sensor per door or a proprietary camera estate, an annual contract, and a sales call. Mesopic runs on the cameras already screwed to your ceiling, installs in one command, and the engine is MIT. |

Two things Mesopic does not claim: it is not more accurate than an audited commercial counter today,
and it does not do the security-camera job Frigate does well. It does the measurement layer, in the
open, on hardware you already own.
