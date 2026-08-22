## Mesopic

Mesopic turns the RTSP/ONVIF security cameras you already have into business sensors —
footfall, queues, dwell, occupancy — without new hardware and without a sales call. The
computer vision runs on your box; **footage never leaves the building.**

This is the engine's documentation. The engine is MIT-licensed and free forever; the
hosted cloud is a separate, optional, paid layer that only ever receives metrics JSON.

### Start here

| If you want to | Read |
|---|---|
| Run it against a camera in one command | [Quickstart](quickstart.md) |
| Define cameras, lines, and zones | [Configuration](configuration.md) |
| Connect Home Assistant or Frigate | [Integrations](integrations.md) |
| Understand what runs where | [Architecture](architecture.md) |
| Check your camera will work | [Cameras](cameras.md) |
| See how this differs from Frigate or a commercial counter | [How it compares](comparison.md) |
| Know what is and is not planned | [Roadmap](roadmap.md) |

### What it measures

**The core six** — the metrics a shop, café, gym, clinic, or venue actually acts on:
footfall, live occupancy, queue length, dwell time, line-crossings, and conversion. Two
adjacencies fall out of the same pipeline nearly for free: zone heatmaps, and
staff-vs-customer filtering.

Not in v1, by design rather than by omission: loss-prevention and theft detection, face
recognition, pose estimation, and ANPR/parking.

### Privacy in one paragraph

Frames are discarded the instant they are processed — no frame, crop, or bounding-box
pixel is ever written to disk, a database, a log, or the network. Mesopic stores
anonymous foot-point coordinates and the metrics derived from them, nothing else. There
is no face recognition and no biometric storage. If you disable cloud sync, nothing
leaves your network at all.
