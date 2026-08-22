## Supported cameras

**Any camera that speaks RTSP or ONVIF** — which is essentially every IP security camera made in the
last decade (Hikvision, Dahua, Reolink, Amcrest, Axis, Ubiquiti, generic ONVIF, and re-badges of all
of the above). Mesopic ingests the existing stream; it does not need its own sensor, unlike
FootfallCam or Milesight, which require dedicated hardware per door.

- Prefer a **sub-stream** (e.g. 640×480–1280×720) for counting — it's cheaper to decode and plenty for
  foot-point detection. Mesopic does not need your 4K main stream.
- Both **H.264** and **H.265/HEVC** are supported via FFmpeg.
