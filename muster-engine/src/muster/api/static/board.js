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
      /* One page, one clock. The exposure strip is labelled UTC because UTC is what is
       * stored; an axis that quietly localised would put two different times on the same
       * screen and leave the reader to notice. */
      tzDate: (ts) => uPlot.tzDate(new Date(ts * 1000), "Etc/UTC"),
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

  document.addEventListener("DOMContentLoaded", paint);
  document.addEventListener("htmx:afterSwap", paint);
  window.addEventListener("resize", resize);
})();
