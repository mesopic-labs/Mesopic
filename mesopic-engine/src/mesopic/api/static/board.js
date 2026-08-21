/* The chart island: the one place this dashboard runs hand-written JavaScript.
 *
 * ADR-0007 keeps the page server-rendered; uPlot is the admitted exception, because a
 * canvas chart cannot be swapped in as HTML. Everything else on the page — the readings,
 * the exposure strip, the range controls — is Jinja and htmx.
 *
 * The charts live OUTSIDE the swapped fragment and are updated in place. htmx replaces
 * `#board` every 30 seconds; if the canvases were inside it, every poll would destroy
 * and rebuild them, throwing away the reader's cursor and costing a full re-layout for
 * data that mostly did not change.
 */
(() => {
  "use strict";

  const FALLBACK_WIDTH = 480;
  const FALLBACK_HEIGHT = 150;

  /** metric -> {plot, signature} */
  const charts = new Map();

  const token = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const UTC = "Etc/UTC";

  /* The site's zone, from the element that owns it. Re-read on every call rather than
   * captured once: `#board` is replaced wholesale by every poll, so a reference taken at
   * load would be to a detached node — the same reason nothing else here holds one.
   *
   * A zone the browser cannot resolve falls back rather than throwing. `site.timezone` is
   * validated against Python's tzdata at load, and browsers ship the same IANA database,
   * so this should be unreachable; what it prevents is a charts-wide render failure if it
   * ever is not. The exposure tick would then disagree with the axes, which is visible —
   * and better than a page of empty plates. */
  const siteZone = () => {
    const zone = document.getElementById("board")?.dataset.timezone;
    if (!zone) return UTC;
    try {
      Intl.DateTimeFormat(undefined, { timeZone: zone });
      return zone;
    } catch {
      return UTC;
    }
  };

  /* The server sends a slot per label, and the slot comes from the config rather than
   * from the label's position in this chart. That is what keeps one zone one colour
   * across every plate and its reading above — assigning by position would make
   * `shop-floor` blue on the occupancy plate and magenta on the dwell plate. */
  const strokeFor = (slot) => token(slot < 4 ? `--s${slot + 1}` : "--s-other");

  /* Counts are bars, levels are lines. The server says which, because it is the same
   * `COUNTING_METRICS` split the readings columns are built from and the page must not
   * make that call twice. A per-minute count drawn as a line is a sawtooth that implies
   * the floor emptied and refilled every minute. */
  const paths = (chart) =>
    chart.total ? uPlot.paths.bars({ align: 1, size: [1, 6] }) : undefined;

  /* Counts are whole people. Left to itself the axis offers 0.5 and 1.5, and half a
   * line crossing is not a thing that happened. */
  const WHOLE = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];

  const options = (chart, width, height) => {
    const axis = {
      stroke: token("--muted"),
      grid: { stroke: token("--line"), width: 1 },
      ticks: { stroke: token("--line"), width: 1 },
    };
    return {
      width,
      height,
      /* One page, one clock: the site's own, read from `#board` where the exposure
       * tick's zone also comes from (P3.9). An axis that stayed on UTC while the tick
       * localised would put two different times on the same screen and leave the reader
       * to notice. */
      tzDate: (ts) => uPlot.tzDate(new Date(ts * 1000), siteZone()),
      /* The legend is also the crosshair readout, so it is shown even for one series:
       * hovering names the scope and prints its value at that minute. Idle, it prints
       * the same em dash the readings do, and means the same thing by it. */
      legend: { show: true },
      cursor: { x: true, y: false, points: { size: 8 } },
      axes: [axis, chart.total ? { ...axis, incrs: WHOLE } : axis],
      series: [
        {},
        ...chart.labels.map((label, index) => ({
          label,
          stroke: strokeFor(chart.slots[index]),
          fill: chart.total ? strokeFor(chart.slots[index]) : undefined,
          width: 2,
          points: { show: false },
          paths: paths(chart),
          /* A missing bucket is a hole, not a straight line across it: the server sends
           * null there precisely so the gap stays visible. */
          spanGaps: false,
        })),
      ],
    };
  };

  /* Height comes from the plate's `--plot-h` token, not from the host element: uPlot
   * renders its legend inside the host, so the host's own height is the canvas plus the
   * legend rather than the canvas alone. Layout stays in the stylesheet either way. */
  const size = (host) => {
    const declared = parseInt(getComputedStyle(host).getPropertyValue("--plot-h"), 10);
    return {
      width: host.clientWidth || FALLBACK_WIDTH,
      height: Number.isFinite(declared) ? declared : FALLBACK_HEIGHT,
    };
  };

  const paint = () => {
    const island = document.getElementById("board-data");
    if (!island) return;

    for (const chart of JSON.parse(island.textContent)) {
      const figure = document.querySelector(`.plate[data-metric="${chart.metric}"]`);
      if (!figure) continue;

      const data = [chart.t, ...chart.v];
      const signature = JSON.stringify([chart.labels, chart.slots]);
      const existing = charts.get(chart.metric);

      if (existing && existing.signature === signature) {
        existing.plot.setData(data);
        continue;
      }

      /* The scope set changed — a config reload added a zone, say — so the plot's series
       * no longer describe the data and it has to be rebuilt rather than re-fed. */
      if (existing) existing.plot.destroy();
      const host = figure.querySelector(".plot");
      const box = size(host);
      host.replaceChildren();
      charts.set(chart.metric, {
        signature,
        plot: new uPlot(options(chart, box.width, box.height), data, host),
      });
    }
  };

  const resize = () => {
    for (const { plot } of charts.values()) {
      plot.setSize(size(plot.root.parentElement));
    }
  };

  /* ---------- the live durations ----------
   *
   * Uptime and each camera's frame age are durations the server renders once per swap.
   * Left alone they freeze: a row reading `0s ago` keeps reading `0s ago` for the whole
   * 30-second poll interval, so a camera that dies a second after a swap stays the
   * freshest thing on the page until the next one lands. That is the liveness indicator
   * being least honest exactly when it matters.
   *
   * So the server sends the number beside the text and the browser ages it locally. The
   * browser clock is used ONLY as a stopwatch — elapsed since the last swap — and never
   * as a calendar, so a skewed client clock cannot invent freshness. Every swap re-seeds
   * the baseline from the server, so the display cannot drift out of step either. */

  const TICK_MS = 1000;

  /* A faithful mirror of `human_duration` in `api/board.py` — the coarsest two units
   * that still say something. Kept identical on purpose: the server renders the first
   * frame of every swap and this renders every one after it, so a divergence would show
   * up as the number changing format for no reason a second after it appears.
   * `test_human_duration_boundaries_the_ticker_mirrors` pins the cases below. */
  const humanDuration = (seconds) => {
    const whole = Math.max(Math.trunc(seconds), 0);
    const hours = Math.floor(whole / 3600);
    const minutes = Math.floor((whole % 3600) / 60);
    const secs = whole % 60;
    if (hours) return `${hours}h ${minutes}m`;
    if (minutes) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  /** Elements being aged, with the value and the moment the server vouched for it. */
  let live = [];

  const reseed = () => {
    const at = performance.now();
    const uptime = document.querySelector("[data-uptime-s]");
    const ages = document.querySelectorAll(".age[data-age-s]");

    live = [];
    if (uptime) {
      live.push({ node: uptime, base: Number(uptime.dataset.uptimeS), at, suffix: "" });
    }
    for (const node of ages) {
      /* Only rows the server gave a number to. A camera that has never reported renders
       * an em dash and carries no `data-age-s`, so it is not in this list and cannot be
       * ticked into claiming a frame arrived. */
      live.push({ node, base: Number(node.dataset.ageS), at, suffix: " ago" });
    }
  };

  const tick = () => {
    const now = performance.now();
    for (const entry of live) {
      if (!Number.isFinite(entry.base)) continue;
      const seconds = entry.base + (now - entry.at) / 1000;
      entry.node.textContent = humanDuration(seconds) + entry.suffix;
    }
  };

  document.addEventListener("DOMContentLoaded", paint);
  document.addEventListener("htmx:afterSwap", paint);
  document.addEventListener("DOMContentLoaded", reseed);
  document.addEventListener("htmx:afterSwap", reseed);
  window.addEventListener("resize", resize);
  setInterval(tick, TICK_MS);
})();
