/* The chart island: the one place this dashboard runs hand-written JavaScript.
 *
 * ADR-0007 keeps the page server-rendered; uPlot is the admitted exception, because a
 * canvas chart cannot be swapped in as HTML. Everything else on the page — the tiles,
 * the health strip, the range controls — is Jinja and htmx.
 *
 * The charts live OUTSIDE the swapped fragment and are updated in place. htmx replaces
 * `#board` every 30 seconds; if the canvases were inside it, every poll would destroy
 * and rebuild them, throwing away the reader's cursor and costing a full re-layout for
 * data that mostly did not change.
 */
(() => {
  "use strict";

  const HEIGHT = 160;
  const FALLBACK_WIDTH = 480;

  /** metric -> {plot, labels} */
  const charts = new Map();

  const token = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /* Series colours come from the HUD tokens rather than literals, so a chart re-themes
   * with the rest of the page instead of drifting into its own palette. */
  const strokes = () => [token("--signal"), token("--stream"), token("--muted")];

  const options = (labels, width) => {
    const colours = strokes();
    const axis = { stroke: token("--muted"), grid: { stroke: token("--line") } };
    return {
      width,
      height: HEIGHT,
      legend: { show: labels.length > 1 },
      axes: [axis, axis],
      series: [
        {},
        ...labels.map((label, index) => ({
          label,
          stroke: colours[index % colours.length],
          width: 1.5,
          /* A missing bucket is a hole, not a straight line across it: the server sends
           * null there precisely so the gap stays visible. */
          spanGaps: false,
        })),
      ],
    };
  };

  const paint = () => {
    const island = document.getElementById("board-data");
    if (!island) return;

    for (const chart of JSON.parse(island.textContent)) {
      const figure = document.querySelector(`.chart[data-metric="${chart.metric}"]`);
      if (!figure) continue;

      const data = [chart.t, ...chart.v];
      const signature = JSON.stringify(chart.labels);
      const existing = charts.get(chart.metric);

      if (existing && existing.signature === signature) {
        existing.plot.setData(data);
        continue;
      }

      /* The scope set changed — a config reload added a zone, say — so the plot's series
       * no longer describe the data and it has to be rebuilt rather than re-fed. */
      if (existing) existing.plot.destroy();
      const host = figure.querySelector(".plot");
      host.replaceChildren();
      charts.set(chart.metric, {
        signature,
        plot: new uPlot(
          options(chart.labels, host.clientWidth || FALLBACK_WIDTH),
          data,
          host,
        ),
      });
    }
  };

  const resize = () => {
    for (const { plot } of charts.values()) {
      const host = plot.root.parentElement;
      plot.setSize({ width: host.clientWidth || FALLBACK_WIDTH, height: HEIGHT });
    }
  };

  document.addEventListener("DOMContentLoaded", paint);
  document.addEventListener("htmx:afterSwap", paint);
  window.addEventListener("resize", resize);
})();
