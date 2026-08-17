/*
 * The calibration canvas: draw zones and lines over one still, save normalized geometry.
 *
 * The overlay's `viewBox` is 0 0 1 1, so every coordinate this file handles is already
 * the normalized coordinate that gets saved (§3). The only conversion anywhere is a
 * pointer position divided by the element's own box — there is no reference resolution
 * in play, which is exactly why the saved geometry survives a camera swapping stream
 * size later.
 *
 * The still itself is never read back, never drawn into a canvas, and never posted. The
 * only thing that leaves is JSON.
 *
 * P3.3.
 */
(function () {
  "use strict";

  const stage = document.querySelector(".darkroom");
  if (!stage) return;

  const SVG = "http://www.w3.org/2000/svg";
  const overlay = stage.querySelector(".overlay");
  const still = stage.querySelector(".still");
  const cameraId = stage.dataset.camera;
  const status = document.querySelector(".saved");

  /* Shapes already saved are not loaded back into the editor: a save replaces this
     camera's geometry wholesale, so what is on the canvas is what the camera will have.
     Starting empty makes that plain rather than letting a half-edited set look merged. */
  const shapes = [];
  let pending = [];
  let mode = "zone";

  const round = (n) => Math.round(n * 10000) / 10000;

  function pointFrom(event) {
    const box = overlay.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width;
    const y = (event.clientY - box.top) / box.height;
    return [round(Math.min(1, Math.max(0, x))), round(Math.min(1, Math.max(0, y)))];
  }

  function node(name, attrs, cls) {
    const element = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (cls) element.setAttribute("class", cls);
    return element;
  }

  function draw() {
    overlay.replaceChildren();
    for (const shape of shapes) drawShape(shape, false);
    if (pending.length) drawShape({ kind: mode, points: pending }, true);
  }

  function drawShape(shape, isPending) {
    const cls = (isPending ? "sketch" : "shape") + " is-" + shape.kind;
    if (shape.kind === "zone") {
      const points = shape.points.map((p) => p.join(",")).join(" ");
      overlay.appendChild(node(isPending ? "polyline" : "polygon", { points }, cls));
    } else if (shape.points.length === 2) {
      const [a, b] = shape.points;
      overlay.appendChild(node("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1] }, cls));
      /* Which way is "in". A directed line whose direction is invisible is a line whose
         sign nobody can check until the counts come out backwards. */
      const mx = (a[0] + b[0]) / 2;
      const my = (a[1] + b[1]) / 2;
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const length = Math.hypot(dx, dy) || 1;
      overlay.appendChild(
        node(
          "line",
          { x1: mx, y1: my, x2: mx - (dy / length) * 0.06, y2: my + (dx / length) * 0.06 },
          "normal",
        ),
      );
    }
    for (const [x, y] of shape.points) {
      overlay.appendChild(node("circle", { cx: x, cy: y, r: 0.008 }, "vertex"));
    }
  }

  function commitPending() {
    if (mode === "zone" && pending.length >= 3) {
      shapes.push({ kind: "zone", points: pending });
    } else if (mode === "line" && pending.length === 2) {
      shapes.push({ kind: "line", points: pending });
    } else {
      return false;
    }
    pending = [];
    draw();
    return true;
  }

  overlay.addEventListener("click", (event) => {
    pending.push(pointFrom(event));
    if (mode === "line" && pending.length === 2) commitPending();
    else draw();
  });

  overlay.addEventListener("dblclick", (event) => {
    event.preventDefault();
    commitPending();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter") commitPending();
    if (event.key === "Escape") {
      pending = [];
      draw();
    }
  });

  for (const button of document.querySelectorAll(".tool")) {
    button.addEventListener("click", () => {
      if (button.dataset.mode) {
        mode = button.dataset.mode;
        pending = [];
        for (const other of document.querySelectorAll(".tool[data-mode]")) {
          other.classList.toggle("is-current", other === button);
        }
        draw();
        return;
      }
      if (button.dataset.act === "undo") {
        if (pending.length) pending.pop();
        else shapes.pop();
        draw();
      }
      if (button.dataset.act === "clear") {
        shapes.length = 0;
        pending = [];
        draw();
      }
      if (button.dataset.act === "save") save(button);
    });
  }

  function body() {
    let zones = 0;
    let lines = 0;
    const edit = { zones: [], lines: [] };
    for (const shape of shapes) {
      if (shape.kind === "zone") {
        edit.zones.push({
          zone_id: "zone-" + ++zones,
          role: "area",
          polygon: shape.points,
          metrics: ["occupancy", "dwell_seconds"],
        });
      } else {
        edit.lines.push({
          line_id: "line-" + ++lines,
          a: shape.points[0],
          b: shape.points[1],
          positive_dir: "in",
          metrics: ["line_cross", "footfall"],
        });
      }
    }
    return edit;
  }

  async function save(button) {
    button.disabled = true;
    say("Saving…");
    try {
      const response = await fetch("/calibrate/" + encodeURIComponent(cameraId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body()),
      });
      if (response.ok) {
        say("Saved. Reloading the page to show the new geometry.");
        window.location.reload();
      } else {
        say("The engine refused that geometry (" + response.status + ").");
      }
    } catch (error) {
      say("Could not reach the engine.");
    } finally {
      button.disabled = false;
    }
  }

  function say(text) {
    if (!status) return;
    status.textContent = text;
    status.hidden = false;
  }

  still.addEventListener("error", () => {
    still.hidden = true;
    const note = stage.querySelector(".unavailable");
    if (note) note.hidden = false;
  });
})();
