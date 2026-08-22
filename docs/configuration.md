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
[`examples/mesopic.yaml`](../examples/mesopic.yaml).
