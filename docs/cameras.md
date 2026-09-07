## Supported cameras

**Any camera that speaks RTSP or ONVIF** — which is essentially every IP security camera made in the
last decade (Hikvision, Dahua, Reolink, Amcrest, Axis, Ubiquiti, generic ONVIF, and re-badges of all
of the above). Mesopic ingests the existing stream; it does not need its own sensor, unlike
FootfallCam or Milesight, which require dedicated hardware per door.

- Prefer a **sub-stream** (e.g. 640×480–1280×720) for counting — it's cheaper to decode and plenty for
  foot-point detection. Mesopic does not need your 4K main stream.
- Both **H.264** and **H.265/HEVC** are supported via FFmpeg.

## Check a camera before you configure it

`mesopic doctor` is the preflight. With no arguments it reports the box — cores, which
inference runtimes it can execute, and the detector model with its licence. Give it a URL
and it also opens the stream once and says what came back:

```bash
export MESOPIC_RTSP_URL="rtsp://user:pass@192.168.1.64:554/stream1"
mesopic doctor --rtsp-env MESOPIC_RTSP_URL
```

Or against the container, without installing anything:

```bash
docker run --rm -e MESOPIC_RTSP_URL \
  ghcr.io/mesopic-labs/mesopic-engine:latest doctor --rtsp-env MESOPIC_RTSP_URL
```

```text
camera
  target      rtsp://***@192.168.1.64:554/stream1
  opened      yes
  resolution  1280x720
  stream rate 15.0 fps
```

It exits non-zero when the stream does not open, so a setup script can gate on it. A
failure says whether anything answered on that port at all, which separates the two cases
FFmpeg reports identically — a wrong address or a blocked port, versus a camera that is
there and refused the credentials or the stream path.

Use `--rtsp-env` rather than `--rtsp` where you can: the URL is a credential, and the flag
form leaves it in your shell history and in `ps` output. `doctor` itself never prints it —
the target line shows the address and stream path with the username and password replaced,
because a typo in the path is the failure you want to see and the password is not.
