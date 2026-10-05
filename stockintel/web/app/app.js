"use strict";
/* StockIntel web app. No framework: a tiny element builder (text always via
   textContent, never innerHTML with data), a hash router, and SVG charts. */

// ---------- helpers ----------
const $ = (s, r = document) => r.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
function s(tag, attrs) { const el = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v); return el; }
const inr = (v, d = 2) => v == null || isNaN(v) ? "—" : (v < 0 ? "-₹" : "₹") + Math.abs(Number(v)).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
const num = (v, d = 2) => v == null || isNaN(v) ? "—" : Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 2) => v == null || isNaN(v) ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const cls = (v) => v > 0 ? "up" : v < 0 ? "down" : "";
const compactInr = (v) => v == null ? "—" : "₹" + Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 2 }).format(v);
const initials = (sym) => (sym || "?").replace(/[^A-Z0-9]/gi, "").slice(0, 2).toUpperCase();
const skel = (hgt = 16, w = "100%") => h("div", { class: "skeleton", style: `height:${hgt}px;width:${w}` });
function toast(msg) { const t = h("div", { class: "toast" }, msg); document.body.append(t); setTimeout(() => t.remove(), 2600); }

// ---------- auth + api ----------
(function captureKey() {
  const m = location.hash.match(/key=([A-Za-z0-9]+)/);
  if (m) { localStorage.setItem("si-key", m[1]); history.replaceState(null, "", location.pathname + "#/home"); }
})();
async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { "X-API-Key": localStorage.getItem("si-key") || "", "Content-Type": "application/json", ...(opts.headers || {}) } });
  const body = await res.json().catch(() => ({}));
  if (res.status === 401) { askKey(); throw new Error("Please log in with your StockIntel key"); }
  if (!res.ok) throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  return body;
}
function askKey() {
  if ($(".modal-bg")) return;
  const input = h("input", { placeholder: "Paste the key printed by `stockintel app`", type: "password" });
  const bg = h("div", { class: "modal-bg" }, h("div", { class: "modal form" },
    h("h2", {}, "Log in"), h("p", { class: "muted" }, "Your key is printed in the terminal when you run `stockintel app`."), input,
    h("button", { class: "btn primary", onclick: () => { localStorage.setItem("si-key", input.value.trim()); bg.remove(); route(); } }, "Continue")));
  document.body.append(bg); input.focus();
}

// ---------- theme ----------
const savedTheme = localStorage.getItem("si-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$("#theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next; localStorage.setItem("si-theme", next); route();
});
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------- search ----------
(function search() {
  const q = $("#q"), box = $("#suggest"); let timer, items = [], sel = -1;
  const close = () => { box.classList.remove("open"); sel = -1; };
  const draw = () => {
    box.replaceChildren(...items.map((r, i) => h("a", { href: `#/stock/${encodeURIComponent(r.symbol)}`, class: i === sel ? "sel" : "", onclick: close },
      h("span", {}, h("span", { class: "sym" }, r.symbol), "  ", h("span", { class: "muted" }, r.name)),
      h("span", { class: "muted" }, r.nifty200 ? "Nifty 200" : r.sector || ""))));
    box.classList.toggle("open", items.length > 0);
  };
  q.addEventListener("input", () => {
    clearTimeout(timer);
    const v = q.value.trim(); if (!v) { items = []; return close(); }
    timer = setTimeout(async () => { try { items = await api("/ui/search?q=" + encodeURIComponent(v)); sel = -1; draw(); } catch { } }, 150);
  });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(items.length - 1, sel + 1); draw(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); }
    else if (e.key === "Enter") {
      const pick = items[sel >= 0 ? sel : 0]; const sym = pick ? pick.symbol : q.value.trim().toUpperCase();
      if (sym) { location.hash = `#/stock/${encodeURIComponent(sym)}`; q.value = ""; close(); q.blur(); }
    } else if (e.key === "Escape") close();
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) close(); });
})();

// ---------- charts ----------
// user-space bounds: a bounding-box filter drops perfectly flat lines (zero-height box)
function glow(svg, W, H, sd = 3) {
  const id = "glow" + Math.random().toString(36).slice(2), f = s("filter", { id, filterUnits: "userSpaceOnUse", x: 0, y: 0, width: W, height: H });
  const merge = s("feMerge"); merge.append(s("feMergeNode", { in: "b" }), s("feMergeNode", { in: "SourceGraphic" }));
  f.append(s("feGaussianBlur", { stdDeviation: sd, result: "b" }), merge); svg.append(f); return `url(#${id})`;
}
const ICONS = { home: "M3 11l9-8 9 8M5 10v10h14V10", compass: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18zM15.5 8.5l-2 5-5 2 2-5z",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17.5l9 5 9-5", sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z",
  pie: "M12 3v9h9a9 9 0 1 1-9-9zM15 3.5A9 9 0 0 1 20.5 9H15z", candles: "M7 3v18M17 3v18M5 7h4v9H5zM15 5h4v7h-4z",
  chat: "M4 5h16v11H9l-5 4z", chart: "M4 20V4M4 20h16M7 15l4-5 3 3 6-7" };
function icon(name) { const svg = s("svg", { viewBox: "0 0 24 24", width: 22, height: 22, fill: "none", stroke: "currentColor", "stroke-width": 1.7, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" });
  svg.append(s("path", { d: ICONS[name] })); return svg; }
function ring(frac, label, color = css("--brand"), size = 84) {
  const c = size / 2, r = c - 8, len = 2 * Math.PI * r, f = Math.max(0, Math.min(1, frac));
  const svg = s("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size, role: "img", "aria-label": label, class: "ring" });
  svg.append(s("circle", { cx: c, cy: c, r, fill: "none", stroke: css("--bg-3"), "stroke-width": 7 }),
    s("circle", { cx: c, cy: c, r, fill: "none", stroke: color, "stroke-width": 7, "stroke-linecap": f ? "round" : "butt", "stroke-dasharray": `${len * f} ${len}`, transform: `rotate(-90 ${c} ${c})`, style: `filter:drop-shadow(0 0 5px ${color})` }));
  const t = s("text", { x: c, y: c + 6, "text-anchor": "middle", "font-size": 17, "font-weight": 700, fill: css("--ink") }); t.textContent = label; svg.append(t);
  return svg;
}
function areaChart(container, dates, values, { height = 280, showAxis = true } = {}) {
  container.replaceChildren();
  if (!values.length) return;
  const W = container.clientWidth || 700, H = height, m = { l: showAxis ? 8 : 0, r: showAxis ? 58 : 0, t: 10, b: showAxis ? 24 : 2 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  let lo = Math.min(...values), hi = Math.max(...values); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const x = (i) => m.l + (values.length === 1 ? pw / 2 : (i / (values.length - 1)) * pw), y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const up = values[values.length - 1] >= values[0], color = up ? css("--up") : css("--down");
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Price chart, ${up ? "up" : "down"} over the period` });
  const gid = "g" + Math.random().toString(36).slice(2);
  const defs = s("defs"); const grad = s("linearGradient", { id: gid, x1: 0, y1: 0, x2: 0, y2: 1 });
  grad.append(s("stop", { offset: "0%", "stop-color": color, "stop-opacity": 0.32 }), s("stop", { offset: "100%", "stop-color": color, "stop-opacity": 0 }));
  defs.append(grad); svg.append(defs);
  if (showAxis) {
    const ticks = 4;
    for (let k = 0; k <= ticks; k++) {
      const v = lo + (hi - lo) * (k / ticks), yy = y(v);
      svg.append(s("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: css("--line"), "stroke-width": 1, "stroke-dasharray": "2 4" }));
      const t = s("text", { x: W - m.r + 6, y: yy + 4, fill: css("--muted"), "font-size": 11 }); t.textContent = num(v, v > 1000 ? 0 : 2); svg.append(t);
    }
    const every = Math.max(1, Math.floor(values.length / 5));
    for (let i = 0; i < values.length; i += every) {
      const t = s("text", { x: x(i), y: H - 6, fill: css("--muted"), "font-size": 11, "text-anchor": "middle" });
      t.textContent = new Date(dates[i]).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: values.length > 300 ? "2-digit" : undefined });
      svg.append(t);
    }
  }
  const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  svg.append(s("path", { d: `M${m.l},${m.t + ph} L${pts.join(" L")} L${x(values.length - 1)},${m.t + ph} Z`, fill: `url(#${gid})` }));
  svg.append(s("path", { d: `M${pts.join(" L")}`, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", filter: showAxis ? glow(svg, W, H) : "none" }));
  if (showAxis) {
    const li = values.length - 1;
    svg.append(s("circle", { cx: x(li), cy: y(values[li]), r: 9, fill: color, "fill-opacity": 0.18 }), s("circle", { cx: x(li), cy: y(values[li]), r: 4, fill: color, stroke: css("--card"), "stroke-width": 2 }));
    const cross = s("line", { y1: m.t, y2: m.t + ph, stroke: css("--muted"), "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" });
    const dot = s("circle", { r: 5, fill: color, stroke: css("--card"), "stroke-width": 2, visibility: "hidden" });
    const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
    svg.append(cross, dot, hit);
    const tip = h("div", { class: "tip" }); container.append(tip);
    hit.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect(); const mx = (e.clientX - r.left) * (W / r.width);
      const i = Math.max(0, Math.min(values.length - 1, Math.round(((mx - m.l) / pw) * (values.length - 1))));
      for (const el of [cross, dot]) el.setAttribute("visibility", "visible");
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(values[i]));
      const chg = (values[i] / values[0] - 1) * 100;
      tip.replaceChildren(h("b", { class: "num" }, inr(values[i])), " ", h("span", { class: cls(chg) }, pct(chg)), h("br"),
        h("span", { class: "muted" }, new Date(dates[i]).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })));
      tip.style.display = "block"; const px = (x(i) / W) * r.width;
      tip.style.left = Math.min(r.width - 150, Math.max(0, px - 60)) + "px"; tip.style.top = "0px";
    });
    hit.addEventListener("mouseleave", () => { tip.style.display = "none"; for (const el of [cross, dot]) el.setAttribute("visibility", "hidden"); });
  }
  container.prepend(svg);
}
function spark(values, w = 120, hgt = 36) {
  const el = h("div", { style: `width:${w}px;height:${hgt}px` });
  requestAnimationFrame(() => areaChart(el, values.map(() => ""), values, { height: hgt, showAxis: false }));
  return el;
}

// ---------- candlestick chart with pattern overlays ----------
function candleChart(container, t, layers, selected, onPick) {
  container.replaceChildren();
  const n = t.dates.length, ghost = t.ghost;
  // each forecast fan: [data, colour, name, label below (1) or above (-1) the band]
  const fans = [[t.fan, css("--ai"), "AI", 1], [t.alfa, css("--gold"), "Our model", -1]].filter(([f]) => f);
  const fut = Math.max(layers.cone ? t.cone.length : 0, ghost ? ghost.dates.length : 0, ...fans.map(([f]) => f.dates.length)), total = n + fut;
  const W = container.clientWidth || 760, H = 380, volH = 54, m = { l: 8, r: 64, t: 12, b: 24 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b - volH - 8, step = pw / total;
  const vals = [...t.low, ...t.high];
  if (layers.ma) vals.push(...t.sma50.filter((v) => v != null), ...t.sma200.filter((v) => v != null));
  if (layers.cone) vals.push(...t.cone.map((c) => c.lo80), ...t.cone.map((c) => c.hi80));
  if (layers.stop && t.stop) vals.push(t.stop);
  if (layers.patterns) for (const p of t.patterns) { if (p.target) vals.push(p.target); }
  for (const [f] of fans) vals.push(...f.q10, ...f.q90);
  if (ghost) vals.push(...ghost.low, ...ghost.high);
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
  const dateIdx = new Map(t.dates.map((d, i) => [d, i])); t.cone.forEach((c, k) => dateIdx.set(c.date, n + k));
  const X = (i) => m.l + step * (i + 0.5), Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const XD = (d) => X(dateIdx.has(d) ? dateIdx.get(d) : d < t.dates[0] ? 0 : n - 1);   // clamp dates outside the window
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Candlestick chart of ${t.symbol} with detected patterns` });
  const txt = (x, y, str, attrs = {}) => { const e = s("text", { x, y, "font-size": 11, fill: css("--muted"), ...attrs }); e.textContent = str; svg.append(e); return e; };
  for (let k = 1; k <= 3; k++) { const v = lo + (hi - lo) * k / 4; svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--line"), "stroke-dasharray": "2 4" })); txt(W - m.r + 6, Y(v) + 4, num(v, v > 1000 ? 0 : 2)); }
  const every = Math.max(1, Math.floor(total / 5));
  for (let i = 0; i < total; i += every) { const d = i < n ? t.dates[i] : t.cone[i - n].date; txt(X(i), H - 6, new Date(d).toLocaleDateString("en-IN", { day: "numeric", month: "short" }), { "text-anchor": "middle" }); }
  // forward range cone
  if (layers.stop && t.stop) {
    const sx = X(Math.max(0, n - Math.round(n * 0.35)));
    svg.append(s("line", { x1: sx, x2: X(total - 1), y1: Y(t.stop), y2: Y(t.stop), stroke: css("--down"), "stroke-width": 1.5 }));
    txt(sx + 2, Y(t.stop) + 13, `Stop-loss ${inr(t.stop, 0)} (−${t.stop_pct}%)`, { fill: css("--down"), "font-size": 10.5, "font-weight": 600 });
  }
  if (layers.cone) {
    const pts = (key) => t.cone.map((c, k) => `${X(n + k)},${Y(c[key])}`);
    const start = `${X(n - 1)},${Y(t.close[n - 1])}`;
    svg.append(s("path", { d: `M${start} L${pts("hi80").join(" L")} L${pts("lo80").reverse().join(" L")} Z`, fill: css("--cone"), "fill-opacity": 0.12 }));
    svg.append(s("path", { d: `M${start} L${pts("hi50").join(" L")} L${pts("lo50").reverse().join(" L")} Z`, fill: css("--cone"), "fill-opacity": 0.18 }));
    svg.append(s("line", { x1: X(n - 1), x2: X(total - 1), y1: Y(t.close[n - 1]), y2: Y(t.close[n - 1]), stroke: css("--cone"), "stroke-dasharray": "4 4" }));
    const last = t.cone[t.cone.length - 1];
    if (!fans.length) { txt(X(total - 1) - 4, Y(last.hi80) - 4, `80%: ${inr(last.hi80, 0)}`, { "text-anchor": "end" });
      txt(X(total - 1) - 4, Y(last.lo80) + 13, `80%: ${inr(last.lo80, 0)}`, { "text-anchor": "end" }); }
    svg.append(s("line", { x1: X(n - 0.5), x2: X(n - 0.5), y1: m.t, y2: m.t + ph, stroke: css("--line"), "stroke-width": 1 }));
    txt(X(n) + 2, m.t + 10, "next " + t.horizon + " days", { "font-size": 10 });
  }
  if (layers.sr) {
    for (const [lvl, lab] of [...t.support.map((v) => [v, "Support"]), ...t.resistance.map((v) => [v, "Resistance"])]) {
      if (lvl < lo || lvl > hi) continue;
      svg.append(s("line", { x1: m.l, x2: X(total - 1), y1: Y(lvl), y2: Y(lvl), stroke: lab === "Support" ? css("--up") : css("--down"), "stroke-width": 1, "stroke-dasharray": "6 4", opacity: 0.7 }));
      txt(m.l + 4, Y(lvl) - 3, `${lab} ${inr(lvl, 0)}`, { fill: css("--ink-2"), "font-size": 10.5 });
    }
  }
  // candles (or a line in "price" mode)
  const bw = Math.max(1, Math.min(9, step * 0.64));
  if (layers.line) {
    const up = t.close[n - 1] >= t.close[0], col = up ? css("--up") : css("--down");
    const pts = t.close.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
    svg.append(s("path", { d: `M${X(0)},${m.t + ph} L${pts.join(" L")} L${X(n - 1)},${m.t + ph} Z`, fill: col, "fill-opacity": 0.08 }));
    svg.append(s("path", { d: `M${pts.join(" L")}`, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  for (let i = 0; i < n && !layers.line; i++) {
    const up = t.close[i] >= t.open[i], col = up ? css("--up") : css("--down");
    svg.append(s("line", { x1: X(i), x2: X(i), y1: Y(t.high[i]), y2: Y(t.low[i]), stroke: col, "stroke-width": 1 }));
    const top = Y(Math.max(t.open[i], t.close[i])), hgt = Math.max(1, Math.abs(Y(t.open[i]) - Y(t.close[i])));
    svg.append(s("rect", { x: X(i) - bw / 2, y: top, width: bw, height: hgt, fill: col, stroke: col, "stroke-width": 1 }));
  }
  for (const [fan, ai, name, side] of fans) {
    const xs = fan.dates.map((d, k) => X(n + k)), start = `${X(n - 1)},${Y(t.close[n - 1])}`;
    const band = (a, b) => `M${start} L${xs.map((x, k) => `${x},${Y(fan[a][k])}`).join(" L")} L${xs.map((x, k) => `${x},${Y(fan[b][k])}`).reverse().join(" L")} Z`;
    svg.append(s("path", { d: band("q90", "q10"), fill: ai, "fill-opacity": 0.1 }), s("path", { d: band("q75", "q25"), fill: ai, "fill-opacity": 0.18 }));
    svg.append(s("path", { d: `M${start} L${xs.map((x, k) => `${x},${Y(fan.q50[k])}`).join(" L")}`, fill: "none", stroke: ai, "stroke-width": 2, "stroke-dasharray": "5 4", filter: glow(svg, W, H, 2.5) }));
    fan.q50.forEach((v, k) => { const dot = s("circle", { cx: xs[k], cy: Y(v), r: Math.min(3.2, step * 0.4), fill: ai, stroke: css("--card"), "stroke-width": 1.5, cursor: "help" });
      dot.addEventListener("mousemove", (ev) => showTip(ev, [h("b", {}, `${name} median · ${fan.dates[k]}`), `${inr(v)} (50% range ${inr(fan.q25[k], 0)}–${inr(fan.q75[k], 0)})`, `80% range ${inr(fan.q10[k], 0)}–${inr(fan.q90[k], 0)}`]));
      dot.addEventListener("mouseleave", () => hideTip()); svg.append(dot); });
    const last = fan.q50.length - 1;
    txt(xs[last], side > 0 ? Y(fan.q10[last]) + 14 : Y(fan.q90[last]) - 8, `${name} median ${inr(fan.q50[last], 0)}`, { fill: ai, "text-anchor": "middle", "font-weight": 700 });
  }
  if (ghost) ghost.dates.forEach((d, k) => { const x = X(n + k), up = ghost.close[k] >= ghost.open[k], col = up ? css("--up") : css("--down");
    svg.append(s("line", { x1: x, x2: x, y1: Y(ghost.high[k]), y2: Y(ghost.low[k]), stroke: col, opacity: 0.6 }));
    svg.append(s("rect", { x: x - bw / 2, y: Y(Math.max(ghost.open[k], ghost.close[k])), width: bw, height: Math.max(1, Math.abs(Y(ghost.open[k]) - Y(ghost.close[k]))), fill: "none", stroke: col, "stroke-dasharray": "2 2", opacity: 0.8 })); });
  // volume
  const vmax = Math.max(...t.volume) || 1, vy = H - m.b - volH;
  for (let i = 0; i < n; i++) svg.append(s("rect", { x: X(i) - bw / 2, y: vy + volH - (t.volume[i] / vmax) * volH, width: bw, height: (t.volume[i] / vmax) * volH, fill: css("--bg-3") }));
  if (layers.ma) for (const [key, col] of [["sma50", "--series-1"], ["sma200", "--series-2"]]) {
    let d = "", pen = false; t[key].forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(i) + " " + Y(v); pen = true; });
    if (d) svg.append(s("path", { d, fill: "none", stroke: css(col), "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  // patterns: shaded span, joined swing points, labelled lines, breakout + target, extended trendlines
  const tip = h("div", { class: "tip" });
  const showTip = (ev, lines) => { const r = svg.getBoundingClientRect(); tip.replaceChildren(...lines.flatMap((l, k) => k ? [h("br"), l] : [l]));
    tip.style.display = "block"; tip.style.maxWidth = "300px"; tip.style.whiteSpace = "normal";
    tip.style.left = Math.min(r.width - 300, Math.max(0, ev.clientX - r.left + 12)) + "px"; tip.style.top = Math.max(0, ev.clientY - r.top - 10) + "px"; };
  const hideTip = () => { tip.style.display = "none"; };
  if (layers.patterns) t.patterns.forEach((p, k) => {
    if (selected != null && selected !== k) return;
    const g = s("g", { cursor: "pointer" }); const pc = css("--pattern"); const geo = p.geometry || {};
    const x0 = XD(p.start_date), x1 = XD(p.end_date);
    g.append(s("rect", { x: x0, y: m.t, width: Math.max(2, x1 - x0), height: ph, fill: pc, "fill-opacity": 0.05 }));
    if (geo.points && geo.points.length > 1) g.append(s("polyline", { points: geo.points.map((q) => `${XD(q[0])},${Y(q[1])}`).join(" "), fill: "none", stroke: pc, "stroke-width": 2.2, "stroke-linejoin": "round" }));
    if (geo.curve) g.append(s("polyline", { points: geo.curve.map((q) => `${XD(q[0])},${Y(q[1])}`).join(" "), fill: "none", stroke: pc, "stroke-width": 2.2 }));
    for (const q of geo.touches || []) g.append(s("circle", { cx: XD(q[0]), cy: Y(q[1]), r: 3.5, fill: pc, stroke: css("--card"), "stroke-width": 1.5 }));
    for (const ln of geo.lines || []) {
      g.append(s("line", { x1: XD(ln.from[0]), y1: Y(ln.from[1]), x2: XD(ln.to[0]), y2: Y(ln.to[1]), stroke: pc, "stroke-width": 1.8, "stroke-dasharray": ln.style === "dashed" ? "5 4" : null }));
      const lt = s("text", { x: XD(ln.to[0]) - 4, y: Y(ln.to[1]) - 5, "font-size": 10.5, fill: pc, "text-anchor": "end", "font-weight": 600 }); lt.textContent = ln.label; g.append(lt);
    }
    for (const q of geo.points || []) {
      g.append(s("circle", { cx: XD(q[0]), cy: Y(q[1]), r: 4.5, fill: pc, stroke: css("--card"), "stroke-width": 2 }));
      if (q[2]) { const lt = s("text", { x: XD(q[0]), y: Y(q[1]) + (p.direction < 0 && /Top|Head|Shoulder|Pole top/.test(q[2]) ? -9 : 16), "font-size": 10.5, fill: css("--ink"), "text-anchor": "middle", "font-weight": 600 }); lt.textContent = q[2]; g.append(lt); }
    }
    const endX = layers.cone ? X(total - 1) : X(n - 1);
    if (p.breakout_level) { g.append(s("line", { x1: x1, x2: endX, y1: Y(p.breakout_level), y2: Y(p.breakout_level), stroke: pc, "stroke-dasharray": "3 3", "stroke-width": 1.4 }));
      const bt = s("text", { x: endX - 4, y: Y(p.breakout_level) + 13, "font-size": 10.5, fill: pc, "text-anchor": "end" }); bt.textContent = `Breakout ${inr(p.breakout_level, 0)}`; g.append(bt); }
    if (p.target) { g.append(s("line", { x1: x1, x2: endX, y1: Y(p.target), y2: Y(p.target), stroke: pc, "stroke-dasharray": "1 4", "stroke-width": 2 }));
      const tt = s("text", { x: endX - 4, y: Y(p.target) - 4, "font-size": 10.5, fill: pc, "text-anchor": "end", "font-weight": 700 }); tt.textContent = `Textbook target ${inr(p.target, 0)} (unproven)`; g.append(tt); }
    if (layers.cone) for (const pr of t.projections) if (dateIdx.has(pr.from[0]) && XD(pr.from[0]) >= x0 && XD(pr.from[0]) <= x1 + 1)
      g.append(s("line", { x1: XD(pr.from[0]), y1: Y(pr.from[1]), x2: XD(pr.to[0]), y2: Y(pr.to[1]), stroke: pc, "stroke-dasharray": "6 5", "stroke-width": 1.4, opacity: 0.8 }));
    const name = p.pattern.replace(/_/g, " ");
    g.addEventListener("mousemove", (ev) => showTip(ev, [h("b", {}, name), `${p.confirmation_status} · confidence ${num(p.confidence)}`, p.supporting_features.join(" · "), h("span", { class: p.record.has_edge ? "up" : "down" }, p.record.summary)]));
    g.addEventListener("mouseleave", hideTip); g.addEventListener("click", () => onPick && onPick(k));
    svg.append(g);
  });
  if (layers.candles) for (const mk of t.markers) {
    if (mk.direction === 0 && !layers.neutral) continue;
    const i = dateIdx.get(mk.date); if (i == null) continue;
    const bull = mk.direction > 0, col = mk.direction === 0 ? css("--muted") : bull ? css("--up") : css("--down");
    const y = bull || mk.direction === 0 ? Y(mk.low) + 12 : Y(mk.high) - 12;
    const shape = mk.direction === 0 ? s("circle", { cx: X(i), cy: y, r: 3.5, fill: col })
      : s("path", { d: bull ? `M${X(i)} ${y - 6} L${X(i) - 5} ${y + 3} L${X(i) + 5} ${y + 3} Z` : `M${X(i)} ${y + 6} L${X(i) - 5} ${y - 3} L${X(i) + 5} ${y - 3} Z`, fill: col });
    const hitc = s("circle", { cx: X(i), cy: y, r: 9, fill: "transparent", cursor: "help" });
    hitc.addEventListener("mousemove", (ev) => showTip(ev, [h("b", {}, `${mk.label} · ${mk.date}`), bull ? "Textbook reading: bullish" : mk.direction < 0 ? "Textbook reading: bearish" : "Indecision candle",
      h("span", { class: mk.record.has_edge ? "up" : "down" }, mk.record.summary)]));
    hitc.addEventListener("mouseleave", hideTip);
    svg.append(shape, hitc);
  }
  // crosshair over price area
  const cross = s("line", { y1: m.t, y2: m.t + ph, stroke: css("--muted"), "stroke-dasharray": "3 3", visibility: "hidden", "pointer-events": "none" });
  svg.append(cross);
  const area = s("rect", { x: m.l, y: m.t, width: step * n, height: ph, fill: "transparent" });
  svg.insertBefore(area, svg.firstChild.nextSibling);
  svg.addEventListener("mousemove", (ev) => {
    if (ev.target !== area && ev.target !== svg) return;
    const r = svg.getBoundingClientRect(); const mx = (ev.clientX - r.left) * (W / r.width); const i = Math.floor((mx - m.l) / step);
    if (i < 0 || i >= n) { cross.setAttribute("visibility", "hidden"); return hideTip(); }
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    const chg = i ? (t.close[i] / t.close[i - 1] - 1) * 100 : 0;
    showTip(ev, [h("b", {}, new Date(t.dates[i]).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })),
      `O ${num(t.open[i])}  H ${num(t.high[i])}  L ${num(t.low[i])}  C ${num(t.close[i])}`, h("span", { class: cls(chg) }, `${pct(chg)} on the day`)]);
  });
  svg.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  container.append(svg, tip);
}

function intradayPanel(sym) {
  const wrap = h("div"), plot = h("div", { class: "chart" }), note = h("div", { class: "chart-note" }); let rng = "5D";
  const seg = h("div", { class: "seg" }, ...["1D", "5D", "1M"].map((r) => h("button", { class: r === rng ? "active" : "", onclick: (e) => { rng = r; for (const b of seg.children) b.classList.remove("active"); e.target.classList.add("active"); load(); } }, r)));
  const draw = (d) => {
    plot.replaceChildren();
    const n = d.times.length, W = plot.clientWidth || 760, H = 360, volH = 50, m = { l: 8, r: 64, t: 12, b: 24 }, pw = W - m.l - m.r, ph = H - m.t - m.b - volH - 8, step = pw / n;
    const L = d.levels, lv = [["Prev close", L.prev_close, "--muted", "4 3"], ...L.support.map((v) => ["Support", v, "--up", "6 4"]), ...L.resistance.map((v) => ["Resistance", v, "--down", "6 4"]),
      ["5-day range high (80%)", L.range5_hi, "--cone", "2 3"], ["5-day range low (80%)", L.range5_lo, "--cone", "2 3"]].filter((x) => x[1] != null);
    const vals = [...d.low, ...d.high]; let lo = Math.min(...vals), hi = Math.max(...vals);
    for (const [, v] of lv) if (v > lo - (hi - lo) * 0.6 && v < hi + (hi - lo) * 0.6) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
    const X = (i) => m.l + step * (i + 0.5), Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Intraday chart of ${sym}` });
    const txt = (x, y, str, a = {}) => { const e = s("text", { x, y, "font-size": 11, fill: css("--muted"), ...a }); e.textContent = str; svg.append(e); };
    for (let k = 0; k <= 5; k++) { const v = lo + (hi - lo) * k / 5; svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--line") })); txt(W - m.r + 6, Y(v) + 4, num(v, v > 1000 ? 0 : 2)); }
    let lastDay = null;
    d.times.forEach((t, i) => { const day = t.slice(0, 10); if (day !== lastDay) { if (lastDay) svg.append(s("line", { x1: X(i) - step / 2, x2: X(i) - step / 2, y1: m.t, y2: H - m.b, stroke: css("--line") }));
      txt(X(i) + 2, H - 6, rng === "1D" ? t.slice(11) : new Date(day).toLocaleDateString("en-IN", { day: "numeric", month: "short" })); lastDay = day; }
      else if (rng === "1D" && i % 12 === 0) txt(X(i), H - 6, t.slice(11), { "text-anchor": "middle" }); });
    for (const [lab, v, col, dash] of lv) { if (v < lo || v > hi) continue;
      svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css(col), "stroke-dasharray": dash, "stroke-width": 1.3 }));
      txt(m.l + 4, Y(v) - 3, `${lab} ${inr(v, 0)}`, { fill: css("--ink-2"), "font-size": 10.5 }); }
    const bw = Math.max(1, Math.min(8, step * 0.62)), vmax = Math.max(...d.volume) || 1, vy = H - m.b - volH;
    for (let i = 0; i < n; i++) { const up = d.close[i] >= d.open[i], col = up ? css("--up") : css("--down");
      svg.append(s("line", { x1: X(i), x2: X(i), y1: Y(d.high[i]), y2: Y(d.low[i]), stroke: col }));
      svg.append(s("rect", { x: X(i) - bw / 2, y: Y(Math.max(d.open[i], d.close[i])), width: bw, height: Math.max(1, Math.abs(Y(d.open[i]) - Y(d.close[i]))), fill: col, stroke: col }));
      svg.append(s("rect", { x: X(i) - bw / 2, y: vy + volH - (d.volume[i] / vmax) * volH, width: bw, height: (d.volume[i] / vmax) * volH, fill: css("--bg-3") })); }
    let path = "", pen = false; d.vwap.forEach((v, i) => { if (v == null || (i && d.times[i].slice(0, 10) !== d.times[i - 1].slice(0, 10))) { pen = false; } if (v == null) return; path += (pen ? "L" : "M") + X(i) + " " + Y(v); pen = true; });
    svg.append(s("path", { d: path, fill: "none", stroke: css("--series-1"), "stroke-width": 1.8 }));
    const tip = h("div", { class: "tip" }), cross = s("line", { y1: m.t, y2: m.t + ph, stroke: css("--muted"), "stroke-dasharray": "3 3", visibility: "hidden" });
    const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }); svg.append(cross, hit);
    hit.addEventListener("mousemove", (ev) => { const r = svg.getBoundingClientRect(); const i = Math.max(0, Math.min(n - 1, Math.floor(((ev.clientX - r.left) * (W / r.width) - m.l) / step)));
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
      const vs = L.prev_close ? (d.close[i] / L.prev_close - 1) * 100 : null;
      tip.replaceChildren(h("b", {}, d.times[i]), h("br"), `O ${num(d.open[i])} H ${num(d.high[i])} L ${num(d.low[i])} C ${num(d.close[i])}`, h("br"), `VWAP ${num(d.vwap[i])}`, vs != null ? [h("br"), h("span", { class: cls(vs) }, `${pct(vs)} vs prev close`)] : null);
      tip.style.display = "block"; tip.style.left = Math.min(r.width - 230, (X(i) / W) * r.width + 12) + "px"; tip.style.top = "8px"; });
    hit.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
    plot.append(svg, tip);
    note.textContent = `${d.interval} candles · blue line = VWAP (average price paid today) · levels from the daily analysis · ${d.note}`;
  };
  let cached = null;
  const load = async () => { plot.replaceChildren(skel(360)); try { cached = await api(`/ui/stock/${encodeURIComponent(sym)}/intraday?range=${rng}`); draw(cached); } catch (e) { plot.replaceChildren(errorBox(e)); } };
  wrap.append(h("div", { class: "spread" }, h("div", { class: "legend" }, h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--series-1)" }), " VWAP"),
    h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--up);border-top-style:dashed" }), " support"), h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--down);border-top-style:dashed" }), " resistance"),
    h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--cone);border-top-style:dotted" }), " 5-day 80% range")), seg), plot, note);
  load(); window.addEventListener("resize", () => cached && draw(cached));
  return wrap;
}

function miniPanel(container, dates, series, { bands = null, bars = false, fmt = (v) => num(v, 1), zero = false, fill = null, height = 120 } = {}) {
  container.replaceChildren();
  const W = container.clientWidth || 300, H = height, m = { l: 4, r: 40, t: 6, b: 16 }, pw = W - m.l - m.r, ph = H - m.t - m.b, n = dates.length;
  const vals = series.flatMap((x) => x.values.filter((v) => v != null)).concat(bands ? bands.map((b) => b[0]) : []).concat(zero ? [0] : []);
  let lo = Math.min(...vals), hi = Math.max(...vals); if (hi === lo) hi = lo + 1;
  const X = (i) => m.l + (i / Math.max(1, n - 1)) * pw, Y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  for (const [v, lab, col] of bands || []) { svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), stroke: css(col || "--line"), "stroke-dasharray": "4 3" }));
    const t = s("text", { x: m.l + pw + 4, y: Y(v) + 4, "font-size": 10, fill: css("--muted") }); t.textContent = lab; svg.append(t); }
  if (zero) svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: Y(0), y2: Y(0), stroke: css("--ink-2"), "stroke-width": 0.8 }));
  for (const sr of series) {
    if (bars) { const bw = Math.max(1, pw / n * 0.7); sr.values.forEach((v, i) => { if (v == null) return; svg.append(s("rect", { x: X(i) - bw / 2, y: Math.min(Y(v), Y(0)), width: bw, height: Math.max(0.5, Math.abs(Y(v) - Y(0))), fill: v >= 0 ? css("--up") : css("--down"), opacity: 0.8 })); }); continue; }
    let d = "", pen = false; sr.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1); pen = true; });
    if (fill) svg.append(s("path", { d: `${d} L${X(n - 1)},${Y(0)} L${X(0)},${Y(0)} Z`, fill: css(fill), "fill-opacity": 0.15 }));
    svg.append(s("path", { d, fill: "none", stroke: css(sr.color), "stroke-width": 1.8 }));
  }
  const last = series[0].values[series[0].values.length - 1];
  const t = s("text", { x: m.l + pw + 4, y: Y(last) + 4, "font-size": 11, "font-weight": 700, fill: css("--ink") }); t.textContent = fmt(last); svg.append(t);
  for (const i of [0, n - 1]) { const e = s("text", { x: X(i), y: H - 3, "font-size": 10, fill: css("--muted"), "text-anchor": i ? "end" : "start" }); e.textContent = new Date(dates[i]).toLocaleDateString("en-IN", { day: "numeric", month: "short" }); svg.append(e); }
  container.append(svg);
}

function analyticsSection(sym) {
  const gaugeCard = h("div", { class: "card" }, h("div", { class: "panel-title" }, "Signal strength"), skel(80));
  const votesCard = h("div", { class: "card" }, h("div", { class: "panel-title" }, "How the engines vote"), skel(120));
  const ladderCard = h("div", { class: "card" }, h("div", { class: "panel-title" }, "Next 20 trading days — plausible range"), skel(120));
  const rsiCard = h("div", { class: "card" }, h("div", { class: "panel-title" }, "RSI (14) — momentum oscillator"), h("div", { class: "rsi" }));
  const macdCard = h("div", { class: "card" }, h("div", { class: "panel-title" }, "MACD histogram — trend acceleration"), h("div", { class: "macd" }));
  const ddCard = h("div", { class: "card" }, h("div", { class: "panel-title" }, "Drawdown — % below the previous peak"), h("div", { class: "dd" }));
  const root = h("div", { class: "grid" }, h("div", { class: "grid cols-3" }, gaugeCard, votesCard, ladderCard), h("div", { class: "grid cols-3" }, rsiCard, macdCard, ddCard));
  api(`/ui/stock/${encodeURIComponent(sym)}/indicators?bars=126`).then((k) => {
    miniPanel(rsiCard.querySelector(".rsi"), k.dates, [{ values: k.rsi, color: "--series-1" }], { bands: [[70, "70", "--down"], [30, "30", "--up"]], fmt: (v) => num(v, 0) });
    rsiCard.append(h("div", { class: "chart-note" }, k.rsi[k.rsi.length - 1] > 70 ? "Above 70: stretched after a strong run." : k.rsi[k.rsi.length - 1] < 30 ? "Below 30: stretched after a sharp fall." : "Between 30 and 70: no extreme."));
    miniPanel(macdCard.querySelector(".macd"), k.dates, [{ values: k.hist }], { bars: true, zero: true, fmt: (v) => num(v, 1) });
    macdCard.append(h("div", { class: "chart-note" }, k.hist[k.hist.length - 1] >= 0 ? "Green bars: short-term trend above its average." : "Red bars: short-term trend below its average."));
    miniPanel(ddCard.querySelector(".dd"), k.dates, [{ values: k.drawdown_pct, color: "--down" }], { fill: "--down", zero: true, fmt: (v) => num(v, 1) + "%" });
    ddCard.append(h("div", { class: "chart-note" }, `Deepest fall in the loaded 5 years: ${num(k.max_drawdown_5y_pct, 1)}%.`));
  }).catch(() => {});
  api(`/stocks/${encodeURIComponent(sym)}/analysis`).then((a) => {
    const st = a.decision.domain_states, names = { technical: "Technicals", candlestick: "Candles", pattern: "Patterns", fundamental: "Fundamentals", news_sentiment: "News", risk: "Risk", regime: "Regime", forecast: "Forecast", historical: "Analogues" };
    const host = h("div"); votesCard.replaceChildren(h("div", { class: "panel-title" }, "How the engines vote"), host,
      h("div", { class: "chart-note" }, "Research engines, −1 bearish to +1 bullish. Grey = no usable signal. Context only — the buy rule is the momentum rank."));
    requestAnimationFrame(() => barChart(host, Object.entries(names).filter(([k2]) => st[k2]).map(([k2, lab]) => ({ label: lab, value: st[k2].score == null || st[k2].confidence === 0 ? 0 : st[k2].score,
      color: st[k2].score == null || st[k2].confidence === 0 ? css("--muted") : st[k2].score >= 0 ? css("--up") : css("--down") })), { fmt: (v) => (v >= 0 ? "+" : "") + num(v, 2) }));
  }).catch((e) => votesCard.replaceChildren(h("div", { class: "panel-title" }, "How the engines vote"), h("p", { class: "muted" }, e.message)));
  api(`/ui/stock/${encodeURIComponent(sym)}/technical?bars=21`).then((t) => {
    const c = t.cone[t.cone.length - 1], last = t.close[t.close.length - 1];
    const rows = [["80% high", c.hi80, "--cone"], ["50% high", c.hi50, "--cone"], ["Today", last, "--ink"], ["50% low", c.lo50, "--cone"], ["80% low", c.lo80, "--cone"], [`Stop-loss (−${t.stop_pct}%)`, t.stop, "--down"]];
    const lo = Math.min(...rows.map((r) => r[1])), hi = Math.max(...rows.map((r) => r[1]));
    ladderCard.replaceChildren(h("div", { class: "panel-title" }, `Next ${t.horizon} trading days — plausible range`),
      ...rows.map(([lab, v, col]) => h("div", { class: "spread", style: "margin:6px 0;font-size:13px" }, h("span", { style: "width:120px" }, lab),
        h("div", { style: "flex:1;margin:0 10px;position:relative;height:8px;background:var(--bg-3);border-radius:4px" }, h("i", { style: `position:absolute;left:${(v - lo) / (hi - lo || 1) * 100}%;top:-3px;width:14px;height:14px;border-radius:50%;background:var(${col});transform:translateX(-50%)` })),
        h("b", { class: "num" }, inr(v, 0)))), h("div", { class: "chart-note" }, `From current volatility. ${t.held_note} Up after ${t.horizon} days ${Math.round(t.base_rate_up * 100)}% of the time.`));
  }).catch(() => {});
  root.setGauge = (v) => {
    if (!v.momentum_rank) { gaugeCard.replaceChildren(h("div", { class: "panel-title" }, "Signal strength"), h("p", { class: "muted" }, "Not ranked — outside the Nifty 200.")); return; }
    const N = v.universe_size, x = (r) => (1 - (r - 1) / (N - 1)) * 100;
    gaugeCard.replaceChildren(h("div", { class: "panel-title" }, "Signal strength — momentum rank"),
      h("div", { class: "row", style: "align-items:baseline" }, h("span", { class: "big-num" }, `#${v.momentum_rank}`), h("span", { class: "muted" }, `of ${N} in the Nifty 200`)),
      h("div", { class: "gauge-wrap" }, h("div", { class: "gauge" }, h("span", { style: `width:${100 - x(v.keep_rank)}%;background:var(--down-soft)` }), h("span", { style: `width:${x(v.keep_rank) - x(v.buy_rank)}%;background:var(--warn-soft)` }), h("span", { style: `width:${x(v.buy_rank)}%;background:var(--brand-soft)` })),
        h("div", { class: "gauge-marker", style: `left:${x(v.momentum_rank)}%` })),
      h("div", { class: "gauge-labels" }, h("span", {}, "weakest"), h("span", {}, `keep zone (top ${v.keep_rank})`), h("span", {}, `buy zone (top ${v.buy_rank})`)),
      h("div", { class: "chart-note" }, `Verdict: ${v.verdict}. The rank is re-checked at each rebalance.`));
  };
  return root;
}

// older candle markers crowd the chart and none has a proven edge, so only recent ones are drawn
const RECENT_MARKERS = 15;
let featuresP; const features = () => featuresP || (featuresP = api("/ui/features").catch((e) => { featuresP = null; throw e; }));
function technicalPanel(sym, opts = {}) {
  const wrap = h("div"), plot = h("div", { class: "chart" }), cards = h("div", { class: "pattern-cards" }), note = h("div", { class: "chart-note" });
  const layers = opts.line ? { line: true, ma: true, sr: true, patterns: false, candles: false, neutral: false, cone: true, stop: true }
    : { ma: true, sr: false, patterns: true, candles: true, neutral: false, cone: true, stop: false };
  let data = null, selected = null, bars = opts.bars || 63; const ai = {};
  const names = opts.line ? { cone: "Volatility range", stop: "Stop-loss level", sr: "Support / resistance", ma: "50/200-day averages", patterns: "Pattern target" }
    : { patterns: "Chart patterns", candles: "Candle signals", neutral: "Show indecision (doji)", ma: "50/200-day averages", sr: "Support / resistance", cone: "Volatility range" };
  const toggle = (k) => { const cb = h("input", { type: "checkbox" }); cb.checked = layers[k];
    cb.addEventListener("change", () => { layers[k] = cb.checked; loadAi(); draw(); drawCards(); }); return h("label", {}, cb, names[k]); };
  const toggles = h("div", { class: "overlay-toggles" }, ...Object.keys(names).map(toggle));
  features().then((f) => { if (f.alfa) { names.alfa = "Our return model (ALFA)"; layers.alfa = true; toggles.prepend(toggle("alfa")); }
    if (f.ml) { names.ai = "AI forecast (Chronos)"; layers.ai = false; toggles.append(toggle("ai")); }
    if (f.kronos) { names.kronos = "AI candles (Kronos, experimental)"; toggles.append(toggle("kronos")); } loadAi(); }).catch((e) => toast(e.message));
  const AI_LAYERS = [["alfa", "alfa"], ["ai", "chronos"], ["kronos", "kronos"]];
  const loadAi = () => { for (const [k, model] of AI_LAYERS) if (layers[k] && !(model in ai)) {
    ai[model] = null;
    api(`/ui/stock/${encodeURIComponent(sym)}/ai?model=${model}`).then((r) => { ai[model] = r; draw(); drawCards(); })
      .catch((e) => { delete ai[model]; toast(`${model}: ${e.message}`); }); } };
  const seg = h("div", { class: "seg" }, ...[["1M", 21], ["3M", 63], ["6M", 126], ["1Y", 252], ["2Y", 500]].map(([lab, b]) => h("button", { class: b === bars ? "active" : "", onclick: (e) => { bars = b; for (const x of seg.children) x.classList.remove("active"); e.target.classList.add("active"); load(); } }, lab)));
  const draw = () => { if (!data) return; candleChart(plot, { ...data, fan: layers.ai && ai.chronos, alfa: layers.alfa && ai.alfa, ghost: layers.kronos && ai.kronos }, layers, selected, (k) => { selected = selected === k ? null : k; draw(); drawCards(); }); };
  const drawCards = () => {
    if (!data) return;
    cards.replaceChildren(...(data.patterns.length > 1 ? [h("div", { class: "muted", style: "grid-column:1/-1;font-size:12px" },
      selected == null ? "Showing all patterns — click a card to focus on one." : "Showing the highlighted pattern — click another card to switch, or the same card to show all.")] : []),
      ...(data.patterns.length ? data.patterns.map((p, k) => h("div", { class: `pattern-card ${selected === k ? "sel" : ""}`, onclick: () => { selected = selected === k ? null : k; draw(); drawCards(); } },
      h("div", { class: "spread" }, h("span", { class: "nm" }, p.pattern.replace(/_/g, " ")), h("span", { class: `edge-badge ${p.record.has_edge ? "yes" : "no"}` }, p.record.has_edge ? "Proven edge" : "No proven edge")),
      h("div", { class: "muted", style: "font-size:12px" }, `${p.start_date} → ${p.end_date} · ${p.confirmation_status} · textbook ${p.direction > 0 ? "bullish" : p.direction < 0 ? "bearish" : "neutral"}`),
      h("div", { style: "font-size:12.5px;margin-top:6px" }, p.supporting_features.join(" · ")),
      h("div", { style: "font-size:12.5px;margin-top:6px" }, p.breakout_level ? `Breakout level ${inr(p.breakout_level)}` : null, p.target ? ` · textbook target ${inr(p.target)}` : null),
      h("div", { class: "muted", style: "font-size:12px;margin-top:6px" }, p.record.summary)))
      : [h("p", { class: "muted" }, "No chart pattern is forming in this window.")]));
    const counts = {}; for (const mk of data.markers) if (mk.direction !== 0) counts[mk.label] = (counts[mk.label] || 0) + 1;
    note.replaceChildren(data.cone_note, h("br"),
      Object.keys(counts).length ? `Candle signals in the last ${RECENT_MARKERS} sessions: ${Object.entries(counts).map(([k, v]) => `${k} ×${v}`).join(", ")}. Hover a marker to see that formation's tested record.` : "",
      ...AI_LAYERS.filter(([k, model]) => layers[k] && ai[model]).flatMap(([, model]) => [h("br"), h("span", { style: `color:var(${model === "alfa" ? "--gold" : "--ai"})` }, ai[model].note)]));
  };
  const load = async () => { plot.replaceChildren(skel(380)); try { data = await api(`/ui/stock/${encodeURIComponent(sym)}/technical?bars=${bars}`);
    const recent = new Set(data.dates.slice(-RECENT_MARKERS)); data.markers = data.markers.filter((mk) => recent.has(mk.date));
    // one pattern at a time reads clearly; start with the most confident, cards switch it
    selected = data.patterns.length ? data.patterns.reduce((best, p, k) => p.confidence > data.patterns[best].confidence ? k : best, 0) : null;
    draw(); drawCards(); } catch (e) { plot.replaceChildren(errorBox(e)); } };
  wrap.append(h("div", { class: "spread" }, h("div", { class: "row", style: "font-size:12px;color:var(--muted)" },
    h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--series-1)" }), " 50-day"), h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--series-2)" }), " 200-day"),
    h("span", {}, h("span", { class: "legend-line", style: "border-color:var(--pattern)" }), " pattern"), h("span", { class: "up" }, "▲"), "bullish candle", h("span", { class: "down" }, "▼"), "bearish candle"), seg), toggles, plot, note, cards);
  load();
  window.addEventListener("resize", draw);
  return wrap;
}

// ---------- generic charts (performance + portfolio) ----------
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4"];
const textW = (str, size = 12) => { const c = textW.c || (textW.c = document.createElement("canvas").getContext("2d")); c.font = `${size}px Inter, system-ui, sans-serif`; return c.measureText(str).width; };
function legend(items) { return h("div", { class: "legend" }, ...items.map(([name, col, dashed]) => h("span", {}, dashed ? h("span", { class: "legend-line", style: `border-color:${col};border-top-style:dashed;margin-right:6px` }) : h("span", { class: "sw", style: `background:${col}` }), name))); }

function lineChart(container, dates, series, { height = 300, fmt = (v) => num(v), logScale = false } = {}) {
  container.replaceChildren();
  const endW = Math.max(...series.map((sr) => textW(`${sr.short || sr.name} ${fmt(sr.values[sr.values.length - 1])}`, 11)));
  const W = container.clientWidth || 700, H = height, m = { l: 8, r: Math.min(W * 0.45, 70 + endW), t: 10, b: 24 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const all = series.flatMap((x) => x.values.filter((v) => v != null));
  const tf = (v) => logScale ? Math.log(v) : v;
  let lo = Math.min(...all.map(tf)), hi = Math.max(...all.map(tf)); const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
  const x = (i) => m.l + (i / Math.max(1, dates.length - 1)) * pw, y = (v) => m.t + ph - ((tf(v) - lo) / (hi - lo)) * ph;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  for (let k = 0; k <= 4; k++) { const tv = lo + (hi - lo) * k / 4, v = logScale ? Math.exp(tv) : tv, yy = m.t + ph - (k / 4) * ph;
    svg.append(s("line", { x1: m.l, x2: m.l + pw, y1: yy, y2: yy, stroke: css("--line"), "stroke-dasharray": "2 4" })); const t = s("text", { x: m.l + pw + 6, y: yy + 4, "font-size": 11, fill: css("--muted") }); t.textContent = fmt(v); svg.append(t); }
  const every = Math.max(1, Math.ceil(dates.length / Math.max(2, Math.min(6, Math.floor(pw / 80)))));
  for (let i = 0; i < dates.length; i += every) { const t = s("text", { x: x(i), y: H - 6, "font-size": 11, fill: css("--muted"), "text-anchor": i === 0 ? "start" : "middle" }); t.textContent = new Date(dates[i]).toLocaleDateString("en-IN", { month: "short", year: "2-digit" }); svg.append(t); }
  const ends = [];
  series.forEach((sr) => {
    let d = "", pen = false; sr.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); pen = true; });
    svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-dasharray": sr.dashed ? "6 4" : null, "stroke-linejoin": "round", filter: sr.dashed ? "none" : glow(svg, W, H, 2.5) }));
    const li = sr.values.length - 1; ends.push({ y: y(sr.values[li]), label: `${sr.short || sr.name} ${fmt(sr.values[li])}`, color: sr.color });
  });
  ends.sort((a, b) => a.y - b.y); for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 13) ends[k].y = ends[k - 1].y + 13;
  for (const e of ends) { svg.append(s("line", { x1: m.l + pw + 52, x2: m.l + pw + 60, y1: e.y, y2: e.y, stroke: e.color, "stroke-width": 2 }));
    const t = s("text", { x: m.l + pw + 63, y: e.y + 4, "font-size": 11, fill: css("--ink-2") }); t.textContent = e.label; svg.append(t); }
  const cross = s("line", { y1: m.t, y2: m.t + ph, stroke: css("--muted"), "stroke-dasharray": "3 3", visibility: "hidden" });
  const hit = s("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }); svg.append(cross, hit);
  const tip = h("div", { class: "tip" });
  hit.addEventListener("mousemove", (ev) => { const r = svg.getBoundingClientRect(); const i = Math.max(0, Math.min(dates.length - 1, Math.round(((ev.clientX - r.left) * (W / r.width) - m.l) / pw * (dates.length - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(h("b", {}, new Date(dates[i]).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })), ...series.map((sr) => h("div", {}, h("span", { class: "sw", style: `display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;background:${sr.color}` }), h("b", { class: "num" }, fmt(sr.values[i])), " ", sr.name)));
    tip.style.display = "block"; tip.style.left = Math.min(r.width - 260, (x(i) / W) * r.width + 12) + "px"; tip.style.top = "8px"; });
  hit.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
  container.append(svg, tip);
}

function forestPlot(container, rows, { threshold = null } = {}) {
  container.replaceChildren();
  const ok = rows.filter((r) => r.excess_pct != null && r.ci_lo != null);
  const W = container.clientWidth || 700, rowH = 22, m = { l: Math.min(W * 0.45, 16 + Math.max(...ok.map((r) => textW(r.pattern)))), r: 70, t: 22, b: 26 }, H = m.t + m.b + ok.length * rowH, pw = W - m.l - m.r;
  const lim = Math.max(1, ...ok.map((r) => Math.max(Math.abs(r.ci_lo), Math.abs(r.ci_hi)))) * 1.05;
  const x = (v) => m.l + ((v + lim) / (2 * lim)) * pw;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Excess return after each pattern with 95% interval" });
  for (const v of [-lim, -lim / 2, 0, lim / 2, lim]) { svg.append(s("line", { x1: x(v), x2: x(v), y1: m.t - 6, y2: H - m.b, stroke: v === 0 ? css("--ink-2") : css("--line"), "stroke-width": v === 0 ? 1.5 : 1 }));
    const t = s("text", { x: x(v), y: H - 8, "font-size": 11, fill: css("--muted"), "text-anchor": "middle" }); t.textContent = pct(v, 1); svg.append(t); }
  const hdr = s("text", { x: x(0), y: 12, "font-size": 11, fill: css("--muted"), "text-anchor": "middle" }); hdr.textContent = "← worse than normal   0 = no effect   better than normal →"; svg.append(hdr);
  const tip = h("div", { class: "tip" });
  ok.forEach((r, k) => {
    const yy = m.t + k * rowH + rowH / 2, sig = threshold != null && r.z != null && Math.abs(r.z) >= threshold;
    const col = sig ? (r.excess_pct > 0 ? css("--up") : css("--down")) : css("--muted");
    const lab = s("text", { x: m.l - 10, y: yy + 4, "font-size": 12, fill: css("--ink"), "text-anchor": "end" }); lab.textContent = r.pattern; svg.append(lab);
    svg.append(s("line", { x1: x(r.ci_lo), x2: x(r.ci_hi), y1: yy, y2: yy, stroke: col, "stroke-width": 2 }));
    svg.append(s("circle", { cx: x(r.excess_pct), cy: yy, r: 4.5, fill: col, stroke: css("--card"), "stroke-width": 2 }));
    const n = s("text", { x: m.l + pw + 8, y: yy + 4, "font-size": 11, fill: css("--muted") }); n.textContent = `${r.events} events`; svg.append(n);
    const hit = s("rect", { x: 0, y: yy - rowH / 2, width: W, height: rowH, fill: "transparent" });
    hit.addEventListener("mousemove", (ev) => { const b = svg.getBoundingClientRect(); tip.replaceChildren(h("b", {}, r.pattern), h("br"), `${r.events} events in ${r.stocks} stocks, next ${r.horizon} days`, h("br"),
      `average excess ${pct(r.excess_pct)} (95% range ${pct(r.ci_lo)} to ${pct(r.ci_hi)})`, h("br"), `z = ${num(r.z)} — ${r.verdict}`);
      tip.style.display = "block"; tip.style.left = Math.min(b.width - 300, ev.clientX - b.left + 12) + "px"; tip.style.top = (yy * b.height / H + 8) + "px"; });
    hit.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    svg.append(hit);
  });
  container.append(svg, tip);
}

function barChart(container, items, { fmt = (v) => pct(v, 1), ref = null, refLabel = "" } = {}) {
  container.replaceChildren();
  const W = container.clientWidth || 500, labW = Math.max(...items.map((i) => textW(i.label)));
  // long labels in a narrow card go on their own line above the bar instead of being clipped
  const stacked = labW + 16 > W * 0.42, rowH = stacked ? 46 : 34;
  const m = { l: stacked ? 4 : 16 + labW, r: 70, t: 8, b: 8 }, H = m.t + m.b + items.length * rowH, pw = W - m.l - m.r;
  const vals = items.map((i) => i.value).concat(ref != null ? [ref] : []);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals) * 1.08 || 1;
  const x = (v) => m.l + ((v - lo) / (hi - lo)) * pw;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H + (ref != null ? 16 : 0)}`, role: "img" });
  items.forEach((it, k) => {
    const yy = m.t + k * rowH, bh = stacked ? 16 : Math.min(22, rowH - 10), by = stacked ? yy + 22 : yy + 5;
    const x0 = x(Math.min(0, it.value)), x1 = x(Math.max(0, it.value));
    const lab = s("text", stacked ? { x: m.l, y: yy + 15, "font-size": 12, fill: css("--ink") } : { x: m.l - 10, y: by + bh / 2 + 4, "font-size": 12, fill: css("--ink"), "text-anchor": "end" });
    lab.textContent = it.label; svg.append(lab);
    svg.append(s("rect", { x: x0, y: by, width: Math.max(2, x1 - x0), height: bh, rx: 4, fill: it.color || css("--series-1"), "fill-opacity": it.faded ? 0.45 : 1 }));
    const v = s("text", { x: x1 + 6, y: by + bh / 2 + 4, "font-size": 12, fill: css("--ink-2"), "font-weight": 600 }); v.textContent = fmt(it.value); svg.append(v);
  });
  if (stacked) items.forEach((_, k) => { const by = m.t + k * rowH + 22; svg.append(s("line", { x1: x(0), x2: x(0), y1: by - 3, y2: by + 19, stroke: css("--ink-2") })); });
  else svg.append(s("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, stroke: css("--ink-2") }));
  if (ref != null) { svg.append(s("line", { x1: x(ref), x2: x(ref), y1: m.t, y2: H - m.b + 4, stroke: css("--down"), "stroke-dasharray": "4 3", "stroke-width": 1.5 }));
    const t = s("text", { x: x(ref), y: H + 12, "font-size": 11, fill: css("--down"), "text-anchor": "middle" }); t.textContent = refLabel; svg.append(t); }
  container.append(svg);
}

function calibrationPlot(container, pointsByH) {
  container.replaceChildren();
  const W = Math.min(container.clientWidth || 360, 420), H = W, m = { l: 44, r: 12, t: 12, b: 38 }, p = W - m.l - m.r;
  const lo = 0.25, hi = 0.75, x = (v) => m.l + ((v - lo) / (hi - lo)) * p, y = (v) => m.t + p - ((v - lo) / (hi - lo)) * p;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", style: `max-width:${W}px` });
  for (const v of [0.3, 0.4, 0.5, 0.6, 0.7]) { svg.append(s("line", { x1: x(v), x2: x(v), y1: m.t, y2: m.t + p, stroke: css("--line") }), s("line", { x1: m.l, x2: m.l + p, y1: y(v), y2: y(v), stroke: css("--line") }));
    const a = s("text", { x: x(v), y: m.t + p + 14, "font-size": 10.5, fill: css("--muted"), "text-anchor": "middle" }); a.textContent = `${v * 100}%`; svg.append(a);
    const b = s("text", { x: m.l - 6, y: y(v) + 4, "font-size": 10.5, fill: css("--muted"), "text-anchor": "end" }); b.textContent = `${v * 100}%`; svg.append(b); }
  svg.append(s("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), stroke: css("--ink-2"), "stroke-dasharray": "5 4" }));
  const lx = s("text", { x: m.l + p / 2, y: H - 4, "font-size": 11, fill: css("--muted"), "text-anchor": "middle" }); lx.textContent = "model said: chance of going up"; svg.append(lx);
  Object.entries(pointsByH).forEach(([hz, pts], k) => { const col = css(SERIES[k]); let d = "";
    pts.forEach((q, i) => { d += (i ? "L" : "M") + x(q.predicted) + " " + y(q.actual); });
    svg.append(s("path", { d, fill: "none", stroke: col, "stroke-width": 1.5, opacity: 0.6 }));
    for (const q of pts) svg.append(s("circle", { cx: x(q.predicted), cy: y(q.actual), r: 4.5, fill: col, stroke: css("--card"), "stroke-width": 2 })); });
  container.append(svg);
}

function donut(container, parts) {
  container.replaceChildren();
  const size = 180, r = 72, sw = 26, c = size / 2, total = parts.reduce((a, b) => a + b.value, 0) || 1;
  const palette = ["--series-1", "--series-2", "--series-3", "--series-4", "--pattern", "--up", "--down", "--muted"];
  const svg = s("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size, role: "img" });
  let a0 = -Math.PI / 2;
  parts.forEach((p, k) => { const a1 = a0 + (p.value / total) * Math.PI * 2 - 0.02;
    const large = a1 - a0 > Math.PI ? 1 : 0, P = (a) => `${c + r * Math.cos(a)} ${c + r * Math.sin(a)}`;
    svg.append(s("path", { d: `M${P(a0)} A${r} ${r} 0 ${large} 1 ${P(a1)}`, fill: "none", stroke: css(palette[k % palette.length]), "stroke-width": sw }));
    p.color = css(palette[k % palette.length]); a0 = a1 + 0.02; });
  const t = s("text", { x: c, y: c + 5, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: css("--ink") }); t.textContent = `${parts.length} sectors`; svg.append(t);
  container.append(h("div", { class: "row", style: "gap:22px;align-items:center" }, svg, h("div", { class: "grid", style: "gap:6px" },
    ...parts.map((p) => h("div", { style: "font-size:13px" }, h("span", { class: "sw", style: `display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;background:${p.color}` }), `${p.label} `, h("b", { class: "num" }, `${num(p.value, 1)}%`))))));
}

async function pagePerformance(view) {
  let d; try { d = await api("/ui/performance"); } catch (e) { view.append(h("div", { class: "card empty" }, h("div", { class: "big" }, "📊"), h("h2", {}, "Performance results not built yet"), h("p", {}, "Run `stockintel build-performance` once (about 30 seconds), then reload."))); return; }
  view.append(h("div", { class: "card" }, h("h2", {}, "How well does StockIntel actually work?"),
    h("p", { class: "muted", style: "margin-top:-6px" }, `Every number below is out of sample — tested on data the method never saw. Built ${new Date(d.built_at).toLocaleString("en-IN")}.`)));
  // 1. momentum
  const mom = d.momentum, live = mom.live_check;
  const curveCard = h("div", { class: "card" }, h("h3", {}, "1 · The buy/sell signal: momentum vs the market"),
    h("p", { class: "sub" }, `Growth of ₹1 since ${new Date(mom.dates[0]).toLocaleDateString("en-IN", { month: "short", year: "numeric" })}. Backtest lines are inflated by survivorship bias (they use today's Nifty 500 list), so compare momentum with the equal-weight line, not with the index.`));
  const plot1 = h("div", { class: "chart" });
  const names = Object.keys(mom.curves); const cols = [css("--series-1"), css("--series-3"), css("--series-2"), css("--series-4")];
  const series = names.map((n, k) => ({ name: n, short: ["Momentum", "Mom + overlay", "Equal wt", "Nifty 500"][k], values: mom.curves[n], color: cols[k], dashed: n.includes("real") }));
  curveCard.append(legend(series.map((x) => [x.name, x.color, x.dashed])), plot1);
  const liveBars = h("div"), yearBars = h("div");
  const liveCard = h("div", { class: "card" }, h("h3", {}, "Reality check: the live momentum ETF"), h("p", { class: "sub" }, `Real fund tracking the same NSE formula, ${live.window}. Real money, no survivorship bias.`), liveBars,
    h("h3", { style: "margin-top:14px" }, "Year by year"), yearBars,
    h("div", { class: "verdict-line" }, `Edge: ${pct(live.bars[0].cagr - live.bars[1].cagr, 1)} a year over the Nifty 500 — real but uneven. It lost to the market in ${live.yearly.filter((y) => y.etf < y.nifty500).map((y) => y.year).join(", ") || "no year"}.`));
  view.append(h("div", { class: "grid cols-2", style: "margin-top:18px" }, curveCard, liveCard));
  // 2. forecasts
  const acc = h("div"), cal = h("div"), fh = d.forecast.horizons, beats = fh.filter((x) => x.accuracy > x.always_up_accuracy);
  const fcCard = h("div", { class: "card" }, h("h3", {}, "2 · Price-direction forecasts (logistic model)"),
    h("p", { class: "sub" }, `${new Set(d.forecast.coverage.map((c) => c.symbol)).size} large caps, ${fh.map((x) => `${num(x.n, 0)} (${x.horizon}-day)`).join(", ")} out-of-sample predictions. Dashed line: accuracy of simply guessing "up" every time.`), acc,
    h("div", { class: "verdict-line" }, beats.length ? `Verdict: beat always guessing “up” at ${beats.map((x) => `${x.horizon}-day (${num(x.accuracy * 100, 1)}% vs ${num(x.always_up_accuracy * 100, 1)}%)`).join(", ")}.`
      : `Verdict: no better than always guessing “up” at any of the ${fh.length} horizons. That's why the app never shows a predicted direction.`));
  const calCard = h("div", { class: "card" }, h("h3", {}, "Calibration: when the model said X%, how often did it go up?"), h("p", { class: "sub" }, "A useful model's dots sit on the dashed diagonal."),
    legend(Object.keys(d.forecast.calibration).map((hz, k) => [`${hz}-day`, css(SERIES[k])])), cal);
  view.append(h("div", { class: "grid cols-2", style: "margin-top:18px" }, fcCard, calCard));
  // 3. ranges
  // provisional: within 3 points of the 80% target counts as calibrated
  const cov = h("div"), cv = d.forecast.coverage.filter((c) => c.horizon === 5).map((c) => c.coverage * 100).sort((a, b) => a - b), cvMid = cv[Math.floor(cv.length / 2)];
  const covCard = h("div", { class: "card" }, h("h3", {}, "3 · Price ranges (the grey band on charts)"), h("p", { class: "sub" }, "Share of real outcomes that landed inside the 80% band. Target: 80%."), cov,
    h("div", { class: "verdict-line" }, `Verdict: a median ${num(cvMid, 0)}% of outcomes landed inside (${num(cv[0], 0)}–${num(cv[cv.length - 1], 0)}% across stocks) — `
      + (Math.abs(cvMid - 80) <= 3 ? "close to target, so the band is usable for sizing risk and stops." : "off target, so treat the band with caution.")));
  // 4. insider
  const ins = h("div");
  const insCard = h("div", { class: "card" }, h("h3", {}, "4 · Promoter buying"), h("p", { class: "sub" }, "Average return over the next 20 trading days versus all other stocks. The placebo uses random dates for the same stocks."), ins,
    h("div", { class: "verdict-line" }, `Verdict: ${d.insider.slice(0, 3).map((r) => `${pct(r.excess_20d, 2)} ${r.test.toLowerCase()}`).join("; ")}. Shown as a flag, never a buy reason on its own.`));
  view.append(h("div", { class: "grid cols-2", style: "margin-top:18px" }, covCard, insCard));
  // 5. patterns
  const candles = d.patterns.filter((r) => r.family === "candlestick"), chartsP = d.patterns.filter((r) => r.family === "chart");
  const f1 = h("div", { class: "chart" }), f2 = h("div", { class: "chart" });
  view.append(h("div", { class: "card", style: "margin-top:18px" }, h("h3", {}, `5 · Candlestick patterns — do they predict the next ${candles[0].horizon} days?`),
    h("p", { class: "sub" }, `Each dot is the average return after the pattern versus normal; the line is the 95% range. A pattern works only if its whole line clears zero with |z| ≥ ${d.bonferroni_z} (strict because ${d.patterns.length} patterns were tested). Grey = no proven edge.`), f1,
    h("div", { class: "verdict-line" }, `Verdict: ${candles.filter((r) => Math.abs(r.z || 0) >= d.bonferroni_z).length} of ${candles.length} candle patterns have an edge.`)),
    h("div", { class: "card", style: "margin-top:18px" }, h("h3", {}, `6 · Chart patterns — do they predict the next ${chartsP[0].horizon} days?`), h("p", { class: "sub" }, `Same test for triangles, double tops, head-and-shoulders and the rest, replayed point-in-time across up to ${Math.max(...chartsP.map((r) => r.stocks || 0))} NSE stocks.`), f2,
      h("div", { class: "verdict-line" }, `Verdict: ${chartsP.filter((r) => Math.abs(r.z || 0) >= d.bonferroni_z).length} of ${chartsP.length} chart patterns have an edge. The app still draws them so you can see them — with this record attached.`)));
  const om = d.open_models, omBars = h("div");
  if (om) { const ch = om.chronos.horizons, kr = om.kronos;
    view.append(h("div", { class: "card", style: "margin-top:18px" }, h("h3", {}, "7 · Open-source AI models: Chronos (price range) and Kronos (candles)"),
      h("p", { class: "sub" }, `Walk-forward on ${ch[0].stocks} large caps: each forecast saw only the prices before its date. Direction accuracy against the base rate (guessing the stock's usual direction).`), omBars,
      h("div", { class: "kv", style: "margin-top:12px" }, ...ch.flatMap((x) => [h("span", { class: "k" }, `Chronos ${x.horizon}-day 80% range held`), h("span", { class: "num" }, `${num(x.chronos_cover80 * 100, 1)}% (plain volatility band ${num(x.ewma_cover80 * 100, 1)}%)`)]),
        ...(kr ? [h("span", { class: "k" }, `Kronos ${kr.horizon}-day price error`), h("span", { class: `num ${kr.mae_pct > kr.no_change_mae_pct ? "down" : "up"}` }, `${num(kr.mae_pct, 1)}% (assuming no change: ${num(kr.no_change_mae_pct, 1)}%)`)] : [])),
      h("div", { class: "verdict-line" }, `Verdict: Chronos — ${om.verdicts.chronos}; it is shown on charts with this record.` + (kr ? ` Kronos — ${om.verdicts.kronos}; it is off by default and marked experimental.` : ""))));
  }
  const drawAll = () => {
    lineChart(plot1, mom.dates, series, { fmt: (v) => "₹" + num(v, 2), logScale: true });
    barChart(liveBars, live.bars.map((b, k) => ({ label: b.label, value: b.cagr, color: css(["--series-1", "--series-4", "--series-1", "--series-2"][k]), faded: !b.real })), { fmt: (v) => pct(v, 1) + "/yr" });
    barChart(yearBars, live.yearly.flatMap((y) => [{ label: `${y.year} momentum ETF`, value: y.etf, color: css("--series-1") }, { label: `${y.year} Nifty 500`, value: y.nifty500, color: css("--series-4") }]));
    barChart(acc, d.forecast.horizons.map((x) => ({ label: `${x.horizon}-day accuracy`, value: x.accuracy * 100, color: css("--series-1") })), { fmt: (v) => num(v, 1) + "%", ref: Math.max(...d.forecast.horizons.map((x) => x.always_up_accuracy)) * 100, refLabel: "always-up guess" });
    calibrationPlot(cal, d.forecast.calibration);
    barChart(cov, d.forecast.coverage.filter((c) => c.horizon === 5).map((c) => ({ label: `${c.symbol} (5-day)`, value: c.coverage * 100, color: css("--series-3") })), { fmt: (v) => num(v, 0) + "%", ref: 80, refLabel: "target 80%" });
    barChart(ins, d.insider.map((r, k) => ({ label: r.test, value: r.excess_20d, color: css(k === 1 ? "--muted" : "--series-1") })), { fmt: (v) => pct(v, 2) });
    forestPlot(f1, candles, { threshold: d.bonferroni_z }); forestPlot(f2, chartsP, { threshold: d.bonferroni_z });
    if (om) barChart(omBars, [...om.chronos.horizons.flatMap((x) => [{ label: `Chronos ${x.horizon}-day`, value: x.chronos_direction_acc * 100, color: css("--ai") }, { label: `Base rate ${x.horizon}-day`, value: x.base_rate_acc * 100, color: css("--muted") }]),
      ...(om.kronos ? [{ label: `Kronos ${om.kronos.horizon}-day`, value: om.kronos.direction_acc * 100, color: css("--ai") }, { label: "Base rate (Kronos sample)", value: om.kronos.base_rate_acc * 100, color: css("--muted") }] : [])], { fmt: (v) => num(v, 1) + "%" });
  };
  requestAnimationFrame(drawAll);
}

// ---------- shared UI ----------
const stockRow = (r, right) => h("a", { href: `#/stock/${encodeURIComponent(r.symbol)}` },
  h("div", { class: "stack" }, h("div", { class: "avatar" }, initials(r.symbol)),
    h("div", { style: "min-width:0" }, h("div", { class: "t" }, r.name || r.symbol), h("div", { class: "s" }, r.symbol))),
  h("div", { class: "right" }, right));
const priceChange = (price, chgPct) => [h("div", { class: "num" }, inr(price)), h("div", { class: `num ${cls(chgPct)}`, style: "font-size:12.5px" }, pct(chgPct))];
function errorBox(e) { return h("div", { class: "card empty" }, h("div", { class: "big" }, "⚠"), h("p", {}, e.message || String(e))); }

// ---------- pages ----------
async function pageExplore(view) {
  const hour = new Date().getHours();
  view.append(h("div", { class: "welcome" }, h("div", {}, h("div", { class: "welcome-kicker" }, hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening"),
      h("h1", {}, "Invest with evidence, not hunches"),
      h("p", {}, "Every call here comes from a method that was tested on real NSE data — and every chart says how well it held up.")),
    h("div", { class: "row" }, h("a", { class: "btn primary", href: "#/signals" }, "Build my plan"), h("a", { class: "btn", href: "#/performance" }, "See the track record"))));
  view.append(h("div", { class: "grid cols-4", id: "idx" }, [1, 2, 3, 4].map(() => h("div", { class: "card" }, skel(14, "50%"), skel(22, "70%")))));
  const body = h("div", { class: "grid cols-2", style: "margin-top:18px" }, h("div", { class: "card" }, skel(200)), h("div", { class: "card" }, skel(200)));
  view.append(body);
  let d; try { d = await api("/ui/explore"); } catch (e) { view.replaceChildren(errorBox(e)); return; }
  $("#idx").replaceChildren(...d.indices.map((ix) => h("div", { class: "card index-card" },
    h("div", { class: "spread" }, h("span", { class: "name" }, ix.name), h("span", { class: "muted", style: "font-size:11px" }, `${ix.spark.length} sessions`)), h("div", { class: "spread" }, h("div", {}, h("div", { class: "val num" }, num(ix.value)),
      h("div", { class: `num ${cls(ix.change)}` }, `${ix.change > 0 ? "+" : ""}${num(ix.change)} (${pct(ix.change_pct)})`)), spark(ix.spark)))));
  const m = d.market, good = m.trend === "up";
  const banner = h("div", { class: `banner ${good ? "good" : "bad"}`, style: "margin-top:18px" },
    h("div", { class: "big" }, good ? "📈" : "📉"),
    h("div", {}, h("b", {}, good ? "Market trend is UP" : "Market trend is DOWN"),
      h("div", { class: "muted" }, `Nifty month-end ${num(m.month_end_close, 0)} vs 10-month average ${num(m.sma_10m, 0)} · volatility ${m.ewma_vol_pct}% · 24-month return ${pct(m.return_24m_pct, 1)}` +
        (m.momentum_crash_risk ? " · momentum crash-risk regime: size positions at half" : ""))));
  const tabsHost = h("div");
  const moversCard = h("div", { class: "card" }, h("div", { class: "spread" }, h("h3", {}, "Top movers · Nifty 200"), tabsHost), h("div", { class: "list", id: "movers" }));
  const showMovers = (which) => { $("#movers").replaceChildren(...d[which].map((r) => stockRow(r, priceChange(r.price, r.change_pct)))); };
  const seg = h("div", { class: "seg" }, ...["gainers", "losers"].map((w, i) => h("button", { class: i === 0 ? "active" : "", onclick: (e) => { for (const b of seg.children) b.classList.remove("active"); e.target.classList.add("active"); showMovers(w); } }, w === "gainers" ? "Gainers" : "Losers")));
  tabsHost.append(seg);
  const momCard = h("div", { class: "card" }, h("div", { class: "spread" }, h("h3", {}, "Top momentum picks"), h("a", { class: "btn ghost small", href: "#/signals" }, "Build my plan →")),
    h("p", { class: "sub" }, "NSE Nifty200 Momentum 30 method — the one signal with a measured edge"),
    h("div", { class: "list" }, ...d.momentum_top.map((r) => stockRow(r, [h("div", { class: "num" }, inr(r.price)), h("div", { class: "muted", style: "font-size:12px" }, `#${r.rank} · 6M ${pct(r.r6_pct, 0)}`)]))));
  const insCard = h("div", { class: "card" }, h("h3", {}, "Promoters buying recently"), h("p", { class: "sub" }, `Market purchases ≥ ₹${num(d.insider_rule.min_value_cr, 0)} cr disclosed to NSE in the last ${d.insider_rule.days} days · a weak signal on its own`),
    d.insider_buys.length ? h("div", { class: "list" }, ...d.insider_buys.map((r) => stockRow({ symbol: r.symbol, name: r.name }, [h("div", { class: "num" }, `₹${num(r.value_cr)} cr`), h("div", { class: "muted", style: "font-size:12px" }, r.filings > 1 ? `${r.filings} filings · latest ${r.date}` : r.date)])))
      : h("p", { class: "muted" }, "No data yet — run `stockintel insider-fetch`."));
  const learnCard = h("div", { class: "card" }, h("h3", {}, "New to investing?"), h("p", { class: "muted" }, "Learn what RSI, P/E, momentum and drawdown mean — and how this app decides."),
    h("div", { class: "chips" }, ...["What is momentum investing?", "What is a P/E ratio?", "What is a stop-loss?", "What is maximum drawdown?"].map((q) => h("a", { class: "chip", href: `#/chat?q=${encodeURIComponent(q)}` }, q))));
  body.replaceChildren(h("div", { class: "grid" }, momCard, insCard), h("div", { class: "grid" }, moversCard, learnCard));
  view.insertBefore(banner, body);
  showMovers("gainers");
  view.append(h("p", { class: "footer-note" }, `Prices as of ${d.as_of}. Decision support, not investment advice.`));
}

async function pageStock(view, sym) {
  const head = h("div", { class: "card hero" }, skel(26, "40%"), skel(34, "25%"));
  const chartCard = h("div", { class: "card" }, skel(380));
  const verdictCard = h("div", { class: "card verdict" }, h("h3", {}, "Should I buy?"), h("p", { class: "muted" }, "Running every engine on this stock…"), skel(18), skel(18, "80%"), skel(18, "90%"));
  const tabsCard = h("div", { class: "card" }, skel(120));
  const analytics = analyticsSection(sym);
  view.append(h("div", { class: "grid cols-2" }, h("div", { class: "grid" }, head, chartCard), h("div", { class: "grid", style: "align-content:start" }, verdictCard, askCard(sym))),
    h("div", { style: "margin-top:18px" }, analytics), h("div", { style: "margin-top:18px" }, tabsCard));
  let o; try { o = await api(`/ui/stock/${encodeURIComponent(sym)}`); } catch (e) { view.replaceChildren(errorBox(e)); return; }
  const chip = (k, v, c = "") => h("div", { class: "stat-chip" }, h("div", { class: "k" }, k), h("div", { class: `v num ${c}` }, v));
  head.replaceChildren(
    h("div", { class: "spread stock-head" }, h("div", { class: "stack" }, h("div", { class: "avatar", style: "width:48px;height:48px;font-size:15px" }, initials(o.symbol)),
      h("div", {}, h("h1", {}, o.name), h("div", { class: "muted" }, [o.symbol, o.sector, o.in_nifty200 ? "Nifty 200" : null].filter(Boolean).join(" · ")))),
      h("div", { class: "row" }, h("button", { class: "btn", onclick: () => addTxModal(o.symbol, o.price) }, "+ Add to portfolio"),
        h("button", { class: "btn", onclick: async (e) => { e.target.disabled = true; e.target.textContent = "Writing…"; try { const r = await api(`/reports/${encodeURIComponent(o.symbol)}`, { method: "POST" }); location.hash = `#/report/${r.id}`; } catch (err) { toast(err.message); e.target.disabled = false; e.target.textContent = "Research report"; } } }, "Research report"))),
    h("div", { class: "row", style: "margin-top:14px;align-items:baseline" }, h("div", { class: "price num" }, inr(o.price)),
      h("div", { class: `num ${cls(o.change)}`, style: "font-size:16px;font-weight:600" }, `${o.change > 0 ? "+" : ""}${num(o.change)} (${pct(o.change_pct)}) today`)),
    h("div", { class: "chips-row" }, chip("1M", pct(o.returns["1M"], 1), cls(o.returns["1M"])), chip("1Y", pct(o.returns["1Y"], 1), cls(o.returns["1Y"])),
      chip("52W low", inr(o.low_52w, 0)), chip("52W high", inr(o.high_52w, 0)), chip("P/E", num(o.stats["P/E (TTM)"], 1)), chip("Mkt cap", compactInr(o.stats["Market cap"]))),
    h("div", { class: "muted", style: "font-size:12px;margin-top:8px" }, `Daily close ${o.as_of} · NSE`));
  const modes = { "Technical": () => technicalPanel(sym, { bars: 63 }), "Intraday": () => intradayPanel(sym), "Price & projection": () => technicalPanel(sym, { line: true, bars: 126 }) };
  const built = {}; const tabsEl = h("div", { class: "mode-tabs" });
  const show = (k) => { for (const b of tabsEl.children) b.classList.toggle("active", b.textContent === k); built[k] = built[k] || modes[k](); chartCard.replaceChildren(tabsEl, built[k]); };
  tabsEl.append(...Object.keys(modes).map((k) => h("button", { onclick: () => show(k) }, k)));
  show("Technical");
  const panes = { Overview: () => overviewPane(o), Research: () => researchPane(sym), News: () => newsPane(sym), Insider: () => insiderPane(sym) };
  const tabs = h("div", { class: "tabs" }); const pane = h("div");
  const showPane = (k) => { for (const b of tabs.children) b.classList.toggle("active", b.textContent === k); pane.replaceChildren(skel(100)); Promise.resolve(panes[k]()).then((el) => pane.replaceChildren(el)).catch((e) => pane.replaceChildren(errorBox(e))); };
  tabs.append(...Object.keys(panes).map((k) => h("button", { onclick: () => showPane(k) }, k)));
  tabsCard.replaceChildren(tabs, pane); showPane("Overview");
  try { const v = await api(`/ui/stock/${encodeURIComponent(sym)}/verdict`); renderVerdict(verdictCard, v); analytics.setGauge(v); }
  catch (e) { verdictCard.replaceChildren(h("h3", {}, "Should I buy?"), h("p", { class: "muted" }, e.message)); }
}

function renderVerdict(card, v) {
  card.className = `card verdict ${v.tone}`;
  const colour = v.tone === "positive" ? "green" : v.tone === "negative" ? "red" : "amber";
  card.replaceChildren(
    h("div", { class: "spread" }, h("h3", { style: "margin:0" }, "Should I buy?"), h("span", { class: `pill ${colour}` }, v.held ? "You own this" : "Not in your portfolio")),
    h("div", { class: `verdict-tag ${colour === "green" ? "up" : colour === "red" ? "down" : ""}`, style: colour === "amber" ? "color:var(--warn)" : "" }, v.verdict),
    h("p", { style: "margin:4px 0 14px" }, v.headline),
    h("div", {}, v.why_buy.length ? [h("b", {}, "Why you might buy"), h("ul", { class: "checks pro" }, ...v.why_buy.map((t) => h("li", {}, t)))] : null),
    h("div", {}, v.why_not.length ? [h("b", {}, "Why you might not"), h("ul", { class: "checks con" }, ...v.why_not.map((t) => h("li", {}, t)))] : null),
    h("div", { class: "kv", style: "margin-bottom:12px" },
      h("span", { class: "k" }, "Momentum rank"), h("span", {}, v.momentum_rank ? `${v.momentum_rank} of ${v.universe_size} (Nifty 200)` : "not ranked (outside Nifty 200)"),
      h("span", { class: "k" }, "Research view"), h("span", {}, `${v.research_label} · score ${num(v.research_score)} · confidence ${num(v.research_confidence)}${v.conflict ? " · engines disagree" : ""}`),
      v.plan.stop_loss ? [h("span", { class: "k" }, "Stop-loss"), h("span", {}, `${inr(v.plan.stop_loss)} (10% below today)`)] : null,
      v.plan.position_size ? [h("span", { class: "k" }, "Position size"), h("span", {}, v.plan.position_size)] : null,
      h("span", { class: "k" }, "Next review"), h("span", {}, v.plan.next_review)),
    h("div", { class: "note" }, v.evidence_note),
    h("div", { class: "muted", style: "font-size:11.5px;margin-top:8px" }, `Data as of ${v.data_as_of} · computed ${new Date(v.analysis_timestamp).toLocaleString("en-IN")}`));
}

function overviewPane(o) {
  const pos = o.high_52w > o.low_52w ? ((o.price - o.low_52w) / (o.high_52w - o.low_52w)) * 100 : 50;
  const st = o.stats;
  const fmt = { "Market cap": compactInr, "P/E (TTM)": (v) => num(v), "P/B": (v) => num(v), "Dividend yield": (v) => v == null ? "—" : num(v) + "%",
    "ROE": (v) => v == null ? "—" : num(v * 100) + "%", "Debt to equity": (v) => v == null ? "—" : num(v / 100), "Beta": (v) => num(v), "Book value": (v) => inr(v) };
  return h("div", {},
    h("div", { class: "row", style: "gap:22px;margin-bottom:16px" }, ...Object.entries(o.returns).map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: `v num ${cls(v)}` }, pct(v, 1))))),
    h("div", { class: "spread muted", style: "font-size:12px" }, h("span", {}, "52W low ", h("b", { class: "num" }, inr(o.low_52w))), h("span", {}, "52W high ", h("b", { class: "num" }, inr(o.high_52w)))),
    h("div", { class: "range-bar" }, h("i", { style: `left:${Math.max(0, Math.min(100, pos))}%` })),
    h("h3", { style: "margin-top:18px" }, "Fundamentals"),
    h("div", { class: "stats" }, ...Object.entries(st).map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: "v num" }, (fmt[k] || num)(v))))),
    h("p", { class: "muted", style: "font-size:11.5px" }, "Snapshot figures reported by the data provider. Ratios the engines compute themselves are in the Research tab."),
    o.about ? [h("h3", { style: "margin-top:18px" }, "About"), h("p", { style: "color:var(--ink-2)" }, o.about.length > 700 ? o.about.slice(0, 700) + "…" : o.about)] : null);
}

async function researchPane(sym) {
  const a = await api(`/stocks/${encodeURIComponent(sym)}/analysis`);
  const names = { technical: "Technicals", candlestick: "Candlesticks", pattern: "Chart patterns", fundamental: "Fundamentals", news_sentiment: "News & sentiment", risk: "Risk", regime: "Market regime", forecast: "Forecast", historical: "Similar past setups" };
  const wrap = h("div");
  wrap.append(h("p", { class: "muted" }, "Every engine's evidence. ✓ bullish · ✕ bearish · • neutral. FACT = measured; MODEL = a model's output."));
  for (const [k, label] of Object.entries(names)) {
    const r = a.results[k]; if (!r) continue;
    const tone = r.score == null ? "" : r.score > 0.1 ? "green" : r.score < -0.1 ? "red" : "amber";
    const det = h("details", {}, h("summary", {}, h("span", { class: "spread", style: "display:inline-flex;width:calc(100% - 20px)" }, h("span", {}, label), h("span", { class: `pill ${tone}` }, r.state))));
    if (r.error) det.append(h("p", {}, `Data unavailable: ${r.error}`));
    for (const e of r.evidence.slice(0, 8)) det.append(h("p", {}, `${e.direction > 0 ? "✓" : e.direction < 0 ? "✕" : "•"} `, h("span", { class: "muted", style: "font-size:11px" }, e.is_model_output ? "MODEL " : "FACT "), e.claim));
    wrap.append(det);
  }
  return wrap;
}

async function newsPane(sym) {
  const d = await api(`/stocks/${encodeURIComponent(sym)}/news`);
  const ev = (d.result.details || {}).events || [];
  if (!ev.length) return h("p", { class: "muted" }, d.result.error ? `News unavailable: ${d.result.error}` : "No recent news found.");
  return h("div", { class: "list" }, ...ev.slice(0, 15).map((e) => h(e.url ? "a" : "div", { class: "item", href: e.url || null, target: "_blank", rel: "noopener" },
    h("div", { style: "min-width:0" }, h("div", { class: "t", style: "font-weight:600" }, e.headline), h("div", { class: "muted", style: "font-size:12px" }, `${e.source} · ${(e.timestamp || "").slice(0, 10)} · ${e.event_type.replace("_", " ")}`)),
    h("span", { class: `pill ${e.sentiment > 0.2 ? "green" : e.sentiment < -0.2 ? "red" : ""}` }, e.sentiment > 0.2 ? "Positive" : e.sentiment < -0.2 ? "Negative" : "Neutral"))));
}

async function insiderPane(sym) {
  const rows = await api(`/ui/stock/${encodeURIComponent(sym)}/insider`);
  if (!rows.length) return h("p", { class: "muted" }, "No promoter or director market trades disclosed in the last year.");
  return h("table", {}, h("tr", {}, h("th", {}, "Date"), h("th", {}, "Who"), h("th", {}, "Action"), h("th", { class: "r" }, "Value")),
    ...rows.map((r) => h("tr", {}, h("td", {}, r.date), h("td", {}, r.category), h("td", { class: r.action === "Bought" ? "up" : "down" }, r.action), h("td", { class: "r num" }, `₹${num(r.value_cr)} cr`))));
}

function askCard(sym) {
  const log = h("div", { class: "grid", style: "gap:8px;margin:8px 0" });
  const input = h("input", { placeholder: `Ask about ${sym}…`, style: "flex:1" });
  const send = async (q) => {
    q = q || input.value.trim(); if (!q) return; input.value = "";
    const msg = new RegExp(`\\b${sym.replace(/[^A-Z0-9]/g, "")}\\b`, "i").test(q) ? q : `${q} (${sym})`;
    log.append(h("div", { class: "bubble me", style: "max-width:100%" }, q));
    const bot = h("div", { class: "bubble bot", style: "max-width:100%" }, "Thinking…"); log.append(bot);
    try { const r = await api("/chat", { method: "POST", body: JSON.stringify({ message: msg, session_id: "stock-" + sym.replace(/[^A-Za-z0-9]/g, "") }) }); bot.textContent = r.text; }
    catch (e) { bot.textContent = e.message; }
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  return h("div", { class: "card" }, h("h3", {}, `Ask about ${sym}`),
    h("div", { class: "chips" }, ...["Why did it move this week?", "How are the fundamentals?", "What are the risks?", "What is the current RSI?"].map((q) => h("button", { class: "chip", onclick: () => send(q) }, q))),
    log, h("div", { class: "row" }, input, h("button", { class: "btn primary", onclick: () => send() }, "Ask")));
}

function addTxModal(sym, price) {
  const f = { symbol: h("input", { value: sym || "", placeholder: "Symbol, e.g. TCS" }), side: h("select", {}, h("option", {}, "BUY"), h("option", {}, "SELL")),
    quantity: h("input", { type: "number", min: "1", value: "1" }), price: h("input", { type: "number", step: "0.05", value: price ? price.toFixed(2) : "" }),
    trade_date: h("input", { type: "date", value: new Date().toISOString().slice(0, 10) }) };
  const bg = h("div", { class: "modal-bg", onclick: (e) => { if (e.target === bg) bg.remove(); } });
  const save = async () => {
    try { await api("/portfolio/transactions", { method: "POST", body: JSON.stringify({ symbol: f.symbol.value.trim().toUpperCase(), side: f.side.value, quantity: +f.quantity.value, price: +f.price.value, trade_date: f.trade_date.value }) });
      bg.remove(); toast("Saved to portfolio"); if (location.hash.startsWith("#/portfolio")) route(); } catch (e) { toast(e.message); }
  };
  bg.append(h("div", { class: "modal form" }, h("h2", {}, "Add transaction"),
    h("label", {}, "Stock", f.symbol), h("div", { class: "grid cols-2", style: "grid-template-columns:1fr 1fr" }, h("label", {}, "Type", f.side), h("label", {}, "Date", f.trade_date)),
    h("div", { class: "grid", style: "grid-template-columns:1fr 1fr" }, h("label", {}, "Quantity", f.quantity), h("label", {}, "Price per share (₹)", f.price)),
    h("div", { class: "row", style: "justify-content:flex-end" }, h("button", { class: "btn", onclick: () => bg.remove() }, "Cancel"), h("button", { class: "btn primary", onclick: save }, "Save"))));
  document.body.append(bg); f.quantity.focus();
}

async function pagePortfolio(view) {
  view.append(h("div", { class: "card" }, skel(60)));
  let p, rep = null;
  try { p = await api("/portfolio"); } catch (e) { p = null; }
  if (!p || !p.transactions.length) {
    view.replaceChildren(h("div", { class: "card empty" }, h("div", { class: "big" }, "💼"), h("h2", {}, "Your portfolio is empty"),
      h("p", {}, "Add the stocks you own (quantity, average price, date) to track P&L, risk and get sell signals."),
      h("button", { class: "btn primary", onclick: () => addTxModal("", null) }, "+ Add your first stock")));
    return;
  }
  try { rep = await api("/portfolio/analysis"); } catch (e) { view.replaceChildren(errorBox(e)); return; }
  const upl = rep.unrealized_pl;
  const summary = h("div", { class: "card" }, h("div", { class: "spread" }, h("div", {}, h("div", { class: "muted" }, "Current value"), h("div", { class: "price num" }, inr(rep.total_value)),
    h("div", { class: `num ${cls(upl)}`, style: "font-weight:600" }, `${upl > 0 ? "+" : ""}${inr(upl)} (${pct(rep.unrealized_pl_pct)}) total returns`)),
    h("div", { class: "row" }, h("button", { class: "btn", onclick: () => addTxModal("", null) }, "+ Add transaction"), h("a", { class: "btn primary", href: "#/signals" }, "What should I do?"))),
    h("div", { class: "grid cols-4", style: "margin-top:16px" }, ...[["Invested", inr(rep.cost_basis)], ["Cash", inr(rep.cash)], ["Realised P&L", inr(rep.realized_pl)], ["Last 5 sessions", `${inr(rep.week_change.total_change_value)} (${pct(rep.week_change.total_change_pct)})`]]
      .map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: "v num" }, v)))));
  const holdings = h("div", { class: "card" }, h("h3", {}, `Holdings (${rep.holdings.length})`), h("table", {},
    h("tr", {}, h("th", {}, "Stock"), h("th", { class: "r" }, "Qty"), h("th", { class: "r" }, "Avg / LTP"), h("th", { class: "r" }, "Value"), h("th", { class: "r" }, "Returns")),
    ...rep.holdings.map((x) => h("tr", { class: "link", onclick: () => location.hash = `#/stock/${encodeURIComponent(x.symbol)}` },
      h("td", {}, h("div", { class: "stack" }, h("div", { class: "avatar" }, initials(x.symbol)), h("div", {}, h("div", { class: "t" }, x.symbol), h("div", { class: "s" }, `${x.sector} · ${num(x.weight_pct, 1)}%`)))),
      h("td", { class: "r num" }, num(x.quantity, 0)), h("td", { class: "r num" }, h("div", {}, inr(x.avg_cost)), h("div", { class: "muted" }, inr(x.price))),
      h("td", { class: "r num" }, inr(x.value, 0)), h("td", { class: `r num ${cls(x.unrealized_pl)}` }, h("div", {}, inr(x.unrealized_pl, 0)), h("div", {}, pct(x.unrealized_pl_pct)))))));
  const donutHost = h("div");
  const alloc = h("div", { class: "card" }, h("h3", {}, "Sector allocation"), h("p", { class: "sub" }, "Share of total value, cash included"), donutHost);
  const histHost = h("div", { class: "chart" });
  const histCard = h("div", { class: "card" }, h("h3", {}, "Value over time"), h("p", { class: "sub" }, "Your holdings as they actually were on each day (cash excluded), against the money you put in"), histHost);
  requestAnimationFrame(() => donut(donutHost, Object.entries(rep.sector_exposure_pct).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: k, value: v }))));
  api("/ui/portfolio/history").then((ph) => { if (!ph.dates.length) { histHost.replaceChildren(h("p", { class: "muted" }, "No history yet.")); return; }
    lineChart(histHost, ph.dates, [{ name: "Holdings value", short: "Value", values: ph.value, color: css("--series-1") }, { name: "Money invested", short: "Invested", values: ph.invested, color: css("--series-4"), dashed: true }], { fmt: (v) => compactInr(v), height: 260 });
    histHost.prepend(legend([["Holdings value", css("--series-1")], ["Money invested", css("--series-4"), true]])); }).catch((e) => histHost.replaceChildren(errorBox(e)));
  const r = rep.risk || {};
  const risk = h("div", { class: "card" }, h("h3", {}, "Risk"),
    r.portfolio_vol_pct != null ? h("div", { class: "kv" }, h("span", { class: "k" }, "Volatility"), h("span", { class: "num" }, `${num(r.portfolio_vol_pct, 1)}% a year`),
      h("span", { class: "k" }, "Worst fall (1y backcast)"), h("span", { class: "num down" }, `${num(r.max_drawdown_pct, 1)}%`),
      h("span", { class: "k" }, "Bad-day loss (95% VaR)"), h("span", { class: "num" }, `${inr(r.var_95_1d_value, 0)} (${num(r.var_95_1d_pct)}%)`),
      h("span", { class: "k" }, "Beta to Nifty"), h("span", { class: "num" }, num(r.beta)),
      h("span", { class: "k" }, "Effective holdings"), h("span", { class: "num" }, num(rep.concentration.effective_holdings, 1)),
      h("span", { class: "k" }, "Biggest risk source"), h("span", {}, Object.entries(r.risk_contribution_pct || {}).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, v]) => `${k} ${num(v, 0)}%`).join(", "))) : h("p", { class: "muted" }, r.note || "Not enough history yet."),
    rep.flags.length ? h("ul", { class: "checks con", style: "margin-top:12px" }, ...rep.flags.map((f) => h("li", {}, f))) : h("p", { class: "up" }, "No concentration or volatility flags."));
  const txs = h("div", { class: "card" }, h("h3", {}, "Transactions"), h("table", {},
    h("tr", {}, h("th", {}, "Date"), h("th", {}, "Stock"), h("th", {}, "Type"), h("th", { class: "r" }, "Qty"), h("th", { class: "r" }, "Price"), h("th", {})),
    ...p.transactions.map((t, i) => h("tr", {}, h("td", {}, t.trade_date), h("td", {}, t.symbol), h("td", { class: t.side === "BUY" ? "up" : "down" }, t.side), h("td", { class: "r num" }, num(t.quantity, 0)), h("td", { class: "r num" }, inr(t.price)),
      h("td", { class: "r" }, h("button", { class: "btn small", onclick: async () => { if (!confirm(`Delete ${t.side} ${t.quantity} ${t.symbol}?`)) return; try { await api(`/portfolio/transactions/${i}`, { method: "DELETE" }); route(); } catch (e) { toast(e.message); } } }, "Delete"))))));
  view.replaceChildren(summary, h("div", { style: "margin-top:18px" }, histCard), h("div", { class: "grid cols-2", style: "margin-top:18px" }, h("div", { class: "grid" }, holdings, txs), h("div", { class: "grid", style: "align-content:start" }, alloc, risk)),
    h("p", { class: "footer-note" }, `As of ${rep.as_of}. ${rep.assumptions.join(" ")}`));
}

async function pageSignals(view) {
  const cap = h("input", { type: "number", min: "0", value: localStorage.getItem("si-cap") || "100000", style: "width:150px" });
  const strat = h("select", {}, h("option", { value: "momentum" }, "Momentum (NSE method)"), h("option", { value: "lowvol" }, "Low volatility"), h("option", { value: "blend" }, "Blend 50/50"));
  const risk = h("select", {}, h("option", { value: "balanced" }, "Balanced"), h("option", { value: "full" }, "Full exposure"), h("option", { value: "defensive" }, "Defensive (trend + vol)"));
  const out = h("div", { class: "grid", style: "margin-top:18px" });
  const run = async () => {
    localStorage.setItem("si-cap", cap.value);
    out.replaceChildren(h("div", { class: "card" }, h("p", { class: "muted" }, "Ranking the Nifty 200… (the first run downloads every stock's history, so it takes longer)"), skel(200)));
    let p; try { p = await api(`/signals?capital=${+cap.value || 0}&strategy=${strat.value}&risk=${risk.value}`); } catch (e) { out.replaceChildren(errorBox(e)); return; }
    const m = p.market;
    const groups = { SELL: "Sell", TRIM: "Trim", BUY: "Buy", ADD: "Add", HOLD: "Hold" };
    const cards = [];
    cards.push(h("div", { class: "card" }, h("div", { class: "grid cols-4" }, ...[["Portfolio value", inr(p.capital, 0)], ["Invest now", `${Math.round(p.invest_fraction * 100)}%`], ["Positions", p.positions], ["Cash left", inr(p.cash_left, 0)]]
      .map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: "v num", style: "font-size:18px" }, v)))),
      h("div", { class: `banner ${m.trend === "up" ? "good" : "bad"}`, style: "margin-top:14px" }, h("div", {}, h("b", {}, `Market trend ${m.trend.toUpperCase()}`),
        h("div", { class: "muted" }, `Nifty ${num(m.month_end_close, 0)} vs 10-month avg ${num(m.sma_10m, 0)} · volatility ${m.ewma_vol_pct}% · crash-risk regime: ${m.momentum_crash_risk ? "yes" : "no"} · next rebalance: ${p.next_rebalance}`)))));
    for (const [k, label] of Object.entries(groups)) {
      const acts = p.actions.filter((a) => a.action === k); if (!acts.length) continue;
      cards.push(h("div", { class: "card" }, h("h3", {}, `${label} (${acts.length})`), h("table", {},
        h("tr", {}, h("th", {}, "Stock"), h("th", { class: "r" }, "Shares"), h("th", { class: "r" }, "Price"), h("th", { class: "r" }, "Amount"), h("th", { class: "r" }, "Stop-loss"), h("th", {}, "Why")),
        ...acts.map((a) => h("tr", { class: "link", onclick: () => location.hash = `#/stock/${encodeURIComponent(a.symbol)}` },
          h("td", {}, h("b", {}, a.symbol), a.rank ? h("div", { class: "muted", style: "font-size:12px" }, `rank #${a.rank}`) : null),
          h("td", { class: "r num" }, a.shares || "—"), h("td", { class: "r num" }, inr(a.price)), h("td", { class: "r num" }, a.value ? inr(a.value, 0) : "—"),
          h("td", { class: "r num" }, a.stop_level ? inr(a.stop_level) : "—"),
          h("td", { style: "font-size:12.5px;color:var(--ink-2)" }, a.reasons.join(" · "), a.tax_note ? h("div", { class: "muted" }, "Tax: " + a.tax_note) : null))))));
    }
    if (p.warnings.length) cards.push(h("div", { class: "card" }, h("h3", {}, "Heads-up"), h("ul", { class: "checks con" }, ...p.warnings.map((w) => h("li", {}, w)))));
    cards.push(h("div", { class: "card" }, h("h3", {}, "How much to trust this"), h("ul", {}, ...p.evidence.map((e) => h("li", { style: "color:var(--ink-2);margin-bottom:6px" }, e)))));
    out.replaceChildren(...cards);
  };
  view.append(h("div", { class: "card" }, h("h2", {}, "Your signal plan"), h("p", { class: "muted", style: "margin-top:-6px" }, "Combines your portfolio with new money and tells you what to sell, buy and hold — with share counts, stops and tax notes."),
    h("div", { class: "row" }, h("label", { class: "row" }, "New money ₹", cap), strat, risk, h("button", { class: "btn primary", onclick: run }, "Build plan"))), out);
}

async function pageChat(view, params) {
  const log = h("div", { class: "log" }); const input = h("input", { placeholder: "Ask anything: 'Should I buy TCS?', 'What is RSI?', 'Compare INFY and HCLTECH'" });
  const sid = sessionStorage.getItem("si-chat") || ("web-" + Math.random().toString(36).slice(2, 10)); sessionStorage.setItem("si-chat", sid);
  const history = JSON.parse(sessionStorage.getItem("si-chat-log") || "[]");
  const bubble = (who, text, meta) => { const b = h("div", { class: `bubble ${who}` }, text, meta ? h("span", { class: "meta" }, meta) : null); log.append(b); log.scrollTop = log.scrollHeight; return b; };
  for (const m of history) bubble(m.who, m.text, m.meta);
  const useAgent = h("input", { type: "checkbox" }); useAgent.checked = localStorage.getItem("si-agent") === "1";
  useAgent.addEventListener("change", () => localStorage.setItem("si-agent", useAgent.checked ? "1" : "0"));
  const agentRow = h("label", { class: "agent-toggle", hidden: true }, useAgent, "Also ask my self-learning agent (a second opinion that stays quiet when unsure)");
  api("/ui/agent/status").then((st) => { agentRow.hidden = !st.available; }).catch((e) => toast(e.message));
  const send = async (q) => {
    q = (q || input.value).trim(); if (!q) return; input.value = "";
    bubble("me", q); history.push({ who: "me", text: q });
    const b = bubble("bot", "Thinking…");
    try { const r = await api("/chat", { method: "POST", body: JSON.stringify({ message: q, session_id: sid }) });
      b.replaceChildren(r.text, h("span", { class: "meta" }, `${r.intent}${r.tools_used.length ? " · " + r.tools_used.join(", ") : ""}`)); history.push({ who: "bot", text: r.text, meta: r.intent }); }
    catch (e) { b.textContent = e.message; }
    if (useAgent.checked && !agentRow.hidden) { const a = bubble("bot agent", "Asking the self-learning agent…");
      try { const t = await api("/ui/agent/ask", { method: "POST", body: JSON.stringify({ message: q, session_id: sid }) });
        const meta = `self-learning agent · data as of ${t.as_of} · ` + (t.spoke ? `confidence ${num(t.confidence)}` : `held back: ${t.because}`);
        a.replaceChildren(t.served, h("span", { class: "meta" }, meta)); history.push({ who: "bot agent", text: t.served, meta }); }
      catch (e) { a.textContent = `Self-learning agent: ${e.message}`; } }
    sessionStorage.setItem("si-chat-log", JSON.stringify(history.slice(-40))); log.scrollTop = log.scrollHeight;
  };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  const chips = h("div", { class: "chips" }, ...["What should I buy with ₹1 lakh?", "Is the market in a downtrend?", "Analyze RELIANCE", "Why?", "Compare it with TCS", "How is my portfolio performing?", "What is a Sharpe ratio?", "Check my stops"]
    .map((q) => h("button", { class: "chip", onclick: () => send(q) }, q)));
  view.append(h("div", { class: "card chat" }, h("div", { class: "spread" }, h("h2", { style: "margin:0" }, "Ask StockIntel"),
    h("button", { class: "btn small", onclick: () => { sessionStorage.removeItem("si-chat-log"); sessionStorage.removeItem("si-chat"); route(); } }, "New chat")), chips, agentRow, log,
    h("div", { class: "composer" }, input, h("button", { class: "btn primary", onclick: () => send() }, "Send"))));
  if (!history.length) bubble("bot", "Hi! I can analyse any NSE stock, tell you whether it qualifies as a buy and why, explain concepts, check your portfolio, and build a buy/sell plan. Every number comes from real data — I'll say so when something is unavailable.");
  if (params.get("q")) send(params.get("q")); else input.focus();
}

async function pageLearn(view) {
  const q = h("input", { placeholder: "Search lessons — e.g. RSI, P/E, drawdown, LTCG", style: "flex:1" });
  const results = h("div");
  const doSearch = async () => {
    if (!q.value.trim()) { results.replaceChildren(); return; }
    try { const r = await api("/knowledge?q=" + encodeURIComponent(q.value)); results.replaceChildren(h("div", { class: "card", style: "margin-top:14px" }, h("h3", {}, "Results"),
      r.results.length ? r.results.map((x) => h("details", { open: true }, h("summary", {}, x.title), h("p", {}, x.text))) : h("p", { class: "muted" }, "No lesson found. Try asking in Ask AI."))); } catch (e) { results.replaceChildren(errorBox(e)); }
  };
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  view.append(h("div", { class: "card" }, h("h2", {}, "Learn"), h("p", { class: "muted", style: "margin-top:-6px" }, "Plain-English lessons on how markets, ratios and risk work — and exactly how this app makes its calls."),
    h("div", { class: "row" }, q, h("button", { class: "btn primary", onclick: doSearch }, "Search"))), results);
  const topics = await api("/ui/learn");
  view.append(h("div", { class: "grid cols-3", style: "margin-top:18px" }, ...topics.map((t) => h("div", { class: "card" }, h("h3", {}, t.topic), h("p", { class: "sub" }, `${t.lessons.length} lessons`),
    ...t.lessons.map((l) => h("details", {}, h("summary", {}, l.title), h("p", {}, l.text), h("a", { class: "btn ghost small", href: `#/chat?q=${encodeURIComponent("Explain " + l.title + " with an example")}` }, "Ask a follow-up →")))))));
}

async function pageReports(view) {
  const list = await api("/reports");
  view.append(h("div", { class: "card" }, h("h2", {}, "Research reports"), h("p", { class: "muted", style: "margin-top:-6px" }, "18-section reports generated from the engines. Open any stock and press “Research report” to create one."),
    list.length ? h("div", { class: "list" }, ...list.map((r) => h("a", { href: `#/report/${r.id}` }, h("div", { class: "stack" }, h("div", { class: "avatar" }, initials(r.symbol)), h("div", {}, h("div", { class: "t" }, r.symbol), h("div", { class: "s" }, new Date(r.created_at).toLocaleString("en-IN")))), h("span", { class: "muted" }, "Open →"))))
      : h("div", { class: "empty" }, h("div", { class: "big" }, "📄"), h("p", {}, "No reports yet."))));
}

function mdToNodes(md) {
  const out = []; let list = null;
  const inline = (text) => { const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|_[^_]+_)/g); return parts.map((p) => /^\*\*.*\*\*$/.test(p) ? h("b", {}, p.slice(2, -2)) : /^(\*|_).*(\*|_)$/.test(p) && p.length > 2 ? h("i", {}, p.slice(1, -1)) : p); };
  for (const raw of md.split("\n")) {
    const line = raw.trimEnd();
    if (/^#{1,3} /.test(line)) { list = null; const lvl = line.match(/^#+/)[0].length; out.push(h("h" + lvl, {}, inline(line.replace(/^#+ /, "")))); }
    else if (/^\s*[-+·] /.test(line)) { if (!list) { list = h("ul"); out.push(list); } list.append(h("li", {}, inline(line.replace(/^\s*[-+·] /, "")))); }
    else if (line.trim() === "") list = null;
    else { list = null; out.push(h("p", {}, inline(line))); }
  }
  return out;
}
async function pageReport(view, id) {
  const list = await api("/reports"); const r = list.find((x) => String(x.id) === String(id));
  if (!r) { view.append(errorBox(new Error("Report not found"))); return; }
  view.append(h("div", { class: "card md" }, h("div", { class: "row", style: "justify-content:space-between" }, h("a", { class: "btn small", href: "#/reports" }, "← All reports"),
    h("div", { class: "row" }, h("a", { class: "btn small", href: `#/stock/${encodeURIComponent(r.symbol)}` }, `Open ${r.symbol}`), h("button", { class: "btn small", onclick: () => window.print() }, "Print / PDF"))), ...mdToNodes(r.body)));
}

// ---------- home + builder ----------
function isoArt() {
  // decorative only: a row of glass cards with a gold one in the middle, no data
  const W = 540, H = 380, svg = s("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true", class: "iso-art" });
  const defs = s("defs"), grad = (id, stops) => { const g = s("linearGradient", { id, x1: 0, y1: 0, x2: 1, y2: 1 });
    for (const [o, c, a] of stops) g.append(s("stop", { offset: o, "stop-color": c, "stop-opacity": a ?? 1 })); defs.append(g); };
  grad("iso-glass", [[0, "#3b434e"], [0.45, "#151a21"], [1, "#07090c"]]);
  grad("iso-gold", [[0, "#fbe7ad"], [0.4, "#d6ad4f"], [1, "#6b4c14"]]);
  grad("iso-edge", [[0, "#ffffff", 0.55], [1, "#ffffff", 0.04]]);
  const halo = s("radialGradient", { id: "iso-halo" }); halo.append(s("stop", { offset: 0, "stop-color": "#e8c46a", "stop-opacity": 0.5 }), s("stop", { offset: 1, "stop-color": "#e8c46a", "stop-opacity": 0 }));
  defs.append(halo); svg.append(defs, s("ellipse", { cx: 300, cy: 300, rx: 210, ry: 60, fill: "url(#iso-halo)" }));
  const marks = {
    card: (g, w, hh, ink) => g.append(s("rect", { x: w * 0.18, y: hh * 0.3, width: w * 0.3, height: hh * 0.13, rx: 3, fill: "none", stroke: ink, "stroke-width": 2 }), s("path", { d: `M${w * 0.18} ${hh * 0.62}h${w * 0.6}M${w * 0.18} ${hh * 0.72}h${w * 0.4}`, stroke: ink, "stroke-width": 2 })),
    shield: (g, w, hh, ink) => g.append(s("path", { d: `M${w / 2} ${hh * 0.3}l${w * 0.24} ${hh * 0.08}v${hh * 0.16}c0 ${hh * 0.14}-${w * 0.12} ${hh * 0.22}-${w * 0.24} ${hh * 0.27}c-${w * 0.12}-${hh * 0.05}-${w * 0.24}-${hh * 0.13}-${w * 0.24}-${hh * 0.27}v-${hh * 0.16}z`, fill: "none", stroke: ink, "stroke-width": 2 })),
    candles: (g, w, hh, ink) => [[0.3, 0.32, 0.5, 0.62], [0.5, 0.42, 0.56, 0.72], [0.7, 0.26, 0.36, 0.52]].forEach(([cx, t, b0, b1]) => g.append(
      s("line", { x1: w * cx, x2: w * cx, y1: hh * t, y2: hh * (b1 + 0.08), stroke: ink, "stroke-width": 2 }), s("rect", { x: w * cx - 5, y: hh * b0 - hh * 0.1, width: 10, height: hh * (b1 - b0 + 0.1), rx: 2, fill: ink }))),
    rupee: (g, w, hh, ink) => { const t = s("text", { x: w / 2, y: hh * 0.62, "text-anchor": "middle", "font-size": hh * 0.42, "font-family": "Georgia, serif", fill: ink }); t.textContent = "₹";
      g.append(t, s("path", { d: `M${w * 0.2} ${hh * 0.86}l${w * 0.18}-${hh * 0.06} ${w * 0.16} ${hh * 0.03} ${w * 0.26}-${hh * 0.1}`, fill: "none", stroke: ink, "stroke-width": 2.4, "stroke-linecap": "round" })); },
    pie: (g, w, hh, ink) => g.append(s("circle", { cx: w / 2, cy: hh / 2, r: w * 0.24, fill: "none", stroke: ink, "stroke-width": 2 }), s("path", { d: `M${w / 2} ${hh / 2}V${hh / 2 - w * 0.24}A${w * 0.24} ${w * 0.24} 0 0 1 ${w / 2 + w * 0.24} ${hh / 2}Z`, fill: ink })),
    bars: (g, w, hh, ink) => [0.35, 0.55, 0.45, 0.7].forEach((v, k) => g.append(s("rect", { x: w * (0.2 + k * 0.16), y: hh * (0.75 - v * 0.5), width: w * 0.1, height: hh * v * 0.5, rx: 2, fill: ink }))),
  };
  // x, top, width, height, kind, mark; drawn back (right) to front, gold last so it sits on top
  const slabs = [[455, 92, 62, 128, "glass", "bars"], [380, 102, 78, 156, "glass", "pie"], [118, 168, 78, 150, "glass", "shield"], [40, 200, 70, 128, "glass", "card"],
    [190, 128, 88, 172, "glass", "candles"], [282, 70, 104, 206, "gold", "rupee"]];
  for (const [x, y, w, hh, kind, mark] of slabs) {
    const g = s("g", { transform: `matrix(1 -0.36 0 1 ${x} ${y + w * 0.36})` });
    g.append(s("rect", { x: 10, y: 7, width: w, height: hh, rx: 12, fill: kind === "gold" ? "#4b3510" : "#030405", stroke: "#ffffff", "stroke-opacity": 0.1 }),
      s("rect", { width: w, height: hh, rx: 12, fill: `url(#iso-${kind})`, stroke: "url(#iso-edge)", "stroke-width": 1.3 }));
    marks[mark](g, w, hh, kind === "gold" ? "#5a3f0e" : "#8f99a6"); svg.append(g);
  }
  return svg;
}

async function pageHome(view) {
  const keyInput = h("input", { type: "password", placeholder: "Paste your StockIntel key", autocomplete: "current-password" });
  const login = () => { const k = keyInput.value.trim(); if (!k) return toast("Paste the key first"); localStorage.setItem("si-key", k); route(); };
  keyInput.addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
  const hasKey = !!localStorage.getItem("si-key");
  const floatA = h("div", { class: "float-card a", hidden: true }), floatB = h("div", { class: "float-card b", hidden: true });
  view.append(h("section", { class: "hero-dark" },
    h("div", {}, h("span", { class: "hero-kicker" }, h("i"), "Your personal NSE stock assistant"),
      h("h1", { class: "display" }, "Invest with evidence, built for ", h("em", {}, "Indian markets.")),
      h("p", {}, "Search any NSE stock and get a plain buy / wait / don't-buy call with the reasons on both sides. Build a portfolio in seconds, see patterns and AI forecast ranges on the chart, and ask questions in plain English. Every method shows its tested track record."),
      hasKey ? h("div", { class: "row", style: "margin-top:22px" }, h("a", { class: "btn primary", href: "#/builder" }, "Get started"), h("a", { class: "btn", href: "#/explore" }, "Explore the market"))
        : h("div", { class: "login-box" }, h("b", {}, "Log in"), h("p", {}, "Your key is printed in the terminal by `stockintel app`, or is STOCKINTEL_API_KEY in your server's settings."),
          h("div", { class: "row" }, keyInput, h("button", { class: "btn primary", onclick: login }, "Continue")))),
    h("div", { class: "hero-visual" }, isoArt(), floatA, floatB)));
  const stats = h("div");
  view.append(stats);
  const feature = (ic, title, text, href) => h("a", { class: "card feature", href }, h("div", { class: "feature-icon" }, icon(ic)), h("h3", {}, title), h("p", { class: "muted" }, text), h("span", { class: "feature-go" }, "Open →"));
  view.append(h("h2", { class: "section-title" }, "What you can do"), h("div", { class: "grid cols-3" },
    feature("compass", "Should I buy?", "Open any stock for a BUY / WAIT / DON'T BUY call, why you might and why you might not, a stop-loss and a position size.", "#/explore"),
    feature("layers", "Portfolio builder", "Enter an amount, pick a style and risk level, and get a ready basket with share counts, stops, risk and a sector split.", "#/builder"),
    feature("candles", "Pattern charts", "Candlesticks with each pattern's swing points, trendlines, breakout and textbook target drawn — and its tested record.", "#/stock/RELIANCE"),
    feature("sparkle", "Our forecast models", "Our ALFA return model, plus Chronos and Kronos fine-tuned on NSE prices, draw the likely range for the coming days, each next to its tested record.", "#/stock/TCS"),
    feature("chat", "Ask AI", "Chat with memory: “Should I buy ITC?”, “Why?”, “Compare it with HUL”. Optionally add your self-learning agent's view.", "#/chat"),
    feature("chart", "Track record", "How well each method did on data it never saw — including the ones that don't work.", "#/performance")));
  view.append(h("p", { class: "footer-note" }, "Decision support, not investment advice. Verify figures against NSE/BSE filings before acting."));
  if (!hasKey) return;
  let d; try { d = await api("/ui/performance"); } catch (e) { stats.replaceChildren(h("p", { class: "muted", style: "margin-top:14px" }, "Run `stockintel build-performance` once to show the measured track record here.")); return; }
  const live = d.momentum.live_check, cov5 = d.forecast.coverage.filter((c) => c.horizon === 5), cov = cov5.reduce((a, c) => a + c.coverage, 0) / cov5.length;
  const edge = d.patterns.filter((r) => Math.abs(r.z || 0) >= d.bonferroni_z).length, ch5 = d.open_models && d.open_models.chronos.horizons.find((x) => x.horizon === 5);
  const gap = live.bars[0].cagr - live.bars[1].cagr, gapText = `${gap > 0 ? "+" : ""}${num(gap, 1)} pts/yr`;
  floatA.replaceChildren(h("b", { class: cls(gap) }, gapText), "momentum fund vs Nifty 500"); floatB.replaceChildren(h("b", {}, `${num(cov * 100, 0)}%`), "of outcomes inside the 80% range");
  floatA.hidden = floatB.hidden = false;
  const tile = (visual, k, sub) => h("div", { class: "card ring-tile" }, visual, h("div", {}, h("div", { class: "k" }, k), h("div", { class: "muted", style: "font-size:12px" }, sub)));
  stats.replaceChildren(h("h2", { class: "section-title" }, "Measured, not promised"), h("div", { class: "grid cols-4" },
    tile(h("div", { class: `big-num ${cls(gap)}` }, gapText), "Momentum fund vs Nifty 500", `live ETF, ${live.window}`),
    tile(ring(cov, `${num(cov * 100, 0)}%`, css("--up")), "Outcomes inside the 80% range", "5-day band, out of sample — the target is 80%"),
    tile(ring(edge / d.patterns.length, `${edge}/${d.patterns.length}`, css("--down")), "Patterns with a proven edge", "so patterns are shown, never traded"),
    ch5 ? tile(ring(ch5.chronos_direction_acc, `${num(ch5.chronos_direction_acc * 100, 0)}%`, css("--ai")), `AI (Chronos) ${ch5.horizon}-day direction`, `vs ${num(ch5.base_rate_acc * 100, 1)}% for the base rate, over ${num(ch5.n, 0)} forecasts`)
      : tile(h("div", { class: "big-num" }, "—"), "AI model record", "run `stockintel evaluate-models`")));
}

async function pageBuilder(view) {
  const state = { capital: 100000, strategy: "momentum", risk: "balanced", n: 10, exclude: [], ...JSON.parse(localStorage.getItem("si-builder") || "{}") };
  let timer, seq = 0;
  const out = h("div", { class: "grid", style: "margin-top:18px" });
  const set = (patch) => { Object.assign(state, patch); localStorage.setItem("si-builder", JSON.stringify(state)); clearTimeout(timer); timer = setTimeout(build, 450); };
  const cap = h("input", { type: "number", min: "1000", step: "1000", value: state.capital, class: "big-input", "aria-label": "Amount to invest" });
  cap.addEventListener("input", () => set({ capital: +cap.value }));
  const capChips = h("div", { class: "chips" }, ...[25000, 50000, 100000, 500000].map((v) => h("button", { class: "chip", onclick: () => { cap.value = v; set({ capital: v }); } }, compactInr(v))));
  const seg = (key, opts) => { const el = h("div", { class: "seg" }, ...opts.map(([v, lab]) => h("button", { class: state[key] === v ? "active" : "",
    onclick: (e) => { for (const b of el.children) b.classList.remove("active"); e.target.classList.add("active"); set({ [key]: v }); } }, lab))); return el; };
  const nOut = h("b", { class: "num" }, state.n), nRange = h("input", { type: "range", min: 3, max: 25, value: state.n, "aria-label": "Number of stocks" });
  nRange.addEventListener("input", () => { nOut.textContent = nRange.value; set({ n: +nRange.value }); });
  const sectors = h("div", { class: "chips" });
  api("/ui/sectors").then((list) => sectors.replaceChildren(...list.map((sec) => { const b = h("button", { class: `chip ${state.exclude.includes(sec) ? "off" : ""}` }, sec);
    b.addEventListener("click", () => { b.classList.toggle("off"); set({ exclude: state.exclude.includes(sec) ? state.exclude.filter((x) => x !== sec) : [...state.exclude, sec] }); }); return b; })))
    .catch((e) => sectors.replaceChildren(errorBox(e)));
  view.append(h("div", { class: "card" }, h("h2", {}, "Portfolio builder"), h("p", { class: "muted", style: "margin-top:-6px" }, "Pick an amount and a style; the basket rebuilds as you change anything. Stocks come from the Nifty 200."),
    h("div", { class: "grid cols-2 builder-controls" },
      h("div", { class: "grid", style: "align-content:start" }, h("label", { class: "ctl" }, h("span", { class: "k" }, "Amount to invest (₹)"), cap), capChips,
        h("div", { class: "ctl" }, h("span", { class: "k" }, "Style"), seg("strategy", [["momentum", "Momentum"], ["lowvol", "Low volatility"], ["blend", "Blend"]])),
        h("div", { class: "ctl" }, h("span", { class: "k" }, "Risk"), seg("risk", [["full", "Always invested"], ["balanced", "Balanced"], ["defensive", "Defensive"]]))),
      h("div", { class: "grid", style: "align-content:start" }, h("label", { class: "ctl" }, h("span", { class: "k" }, "Number of stocks: ", nOut), nRange),
        h("div", { class: "ctl" }, h("span", { class: "k" }, "Tap a sector to leave it out"), sectors),
        h("p", { class: "muted", style: "font-size:12.5px;margin:0" }, "Momentum uses NSE's Nifty200 Momentum 30 method, the one signal here with a measured edge. Balanced cuts the position in crash-risk markets; Defensive also holds cash when the market trend is down.")))), out);
  const build = async () => {
    if (!(state.capital >= 1000)) { out.replaceChildren(h("div", { class: "card" }, h("p", { class: "muted" }, "Enter at least ₹1,000."))); return; }
    const my = ++seq;
    out.replaceChildren(h("div", { class: "card" }, h("p", { class: "muted" }, "Picking stocks and sizing positions… (the first run downloads every stock's history, so it takes longer)"), skel(220)));
    let b; try { b = await api(`/ui/builder?capital=${state.capital}&strategy=${state.strategy}&risk=${state.risk}&n=${state.n}&exclude=${encodeURIComponent(state.exclude.join(","))}`); }
    catch (e) { if (my === seq) out.replaceChildren(errorBox(e)); return; }
    if (my === seq) renderBuild(b);
  };
  const renderBuild = (b) => {
    const warn = b.warnings.length ? h("div", { class: "card" }, h("h3", {}, "Heads-up"), h("ul", { class: "checks con" }, ...b.warnings.map((w) => h("li", {}, w)))) : null;
    if (!b.holdings.length) { out.replaceChildren(h("div", { class: "card empty" }, h("div", { class: "big" }, "🧱"), h("p", {}, "No stock to buy with these settings.")), warn || ""); return; }
    const r = b.risk, maxW = Math.max(...b.holdings.map((x) => x.weight_pct));
    const apply = async () => {
      if (!confirm(`Add ${b.holdings.length} BUY transactions dated today to your portfolio? Do this after you have actually bought them.`)) return;
      try { await api("/ui/builder/apply", { method: "POST", body: JSON.stringify({ holdings: b.holdings.map((x) => ({ symbol: x.symbol, shares: x.shares, price: x.price })) }) });
        toast("Added to your portfolio"); location.hash = "#/portfolio"; } catch (e) { toast(e.message); } };
    const kpi = (k, v, sub, c = "") => h("div", { class: "stat kpi" }, h("div", { class: "k" }, k), h("div", { class: `v num ${c}` }, v), h("div", { class: "muted", style: "font-size:12px" }, sub));
    const kpis = h("div", { class: "card" }, h("div", { class: "grid cols-4" }, kpi("Invested", inr(b.capital - b.cash_left, 0), `${b.holdings.length} stocks`), kpi("Cash left", inr(b.cash_left, 0), `plan invests ${Math.round(b.invest_fraction * 100)}% now`),
      kpi("Volatility", `${num(r.portfolio_vol_pct, 1)}%`, `a year · beta ${num(r.beta)}`), kpi("Worst fall, past year", `${num(r.max_drawdown_pct, 1)}%`, `bad day (95%): ${num(r.var_95_1d_pct, 1)}%`, "down")),
      h("p", { class: "muted", style: "font-size:12.5px;margin:12px 0 0" }, `Next review: ${b.next_rebalance}. Market trend ${b.market.trend}.`));
    const table = h("div", { class: "card" }, h("div", { class: "spread" }, h("h3", { style: "margin:0" }, "Your basket"), h("button", { class: "btn primary", onclick: apply }, "Add all to my portfolio")),
      h("div", { class: "table-wrap" }, h("table", { class: "basket" }, h("tr", {}, h("th", {}, "Stock"), h("th", { class: "r" }, "Shares"), h("th", { class: "r" }, "Price"), h("th", { class: "r" }, "Amount"), h("th", {}, "Weight"), h("th", { class: "r" }, "Stop-loss")),
        ...b.holdings.map((x) => h("tr", { class: "link", title: x.why, onclick: () => location.hash = `#/stock/${encodeURIComponent(x.symbol)}` },
          h("td", {}, h("div", { class: "stack" }, h("div", { class: "avatar" }, initials(x.symbol)), h("div", {}, h("div", { class: "t" }, x.name), h("div", { class: "s" }, `#${x.rank} · ${x.sector}`)))),
          h("td", { class: "r num" }, x.shares), h("td", { class: "r num" }, inr(x.price)), h("td", { class: "r num" }, inr(x.value, 0)),
          h("td", {}, h("div", { class: "wbar" }, h("span", { style: `width:${(x.weight_pct / maxW) * 100}%` }), h("em", { class: "num" }, `${num(x.weight_pct, 1)}%`))),
          h("td", { class: "r num" }, x.stop ? inr(x.stop, 0) : "—"))))));
    const donutHost = h("div"), curveHost = h("div", { class: "chart" });
    out.replaceChildren(kpis, table, h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h3", {}, "Sector split"), h("p", { class: "sub" }, "Share of the amount, cash included"), donutHost),
      h("div", { class: "card" }, h("h3", {}, "This basket over the past year"), h("p", { class: "sub" }, b.note), legend([["This basket", css("--series-1")], ["Nifty 50", css("--series-4"), true]]), curveHost)), warn || "");
    requestAnimationFrame(() => { donut(donutHost, Object.entries(r.sector_exposure_pct).sort((a, c) => c[1] - a[1]).map(([k, v]) => ({ label: k, value: v })));
      lineChart(curveHost, b.backcast.dates, [{ name: "This basket", short: "Basket", values: b.backcast.basket, color: css("--series-1") }, { name: "Nifty 50", short: "Nifty", values: b.backcast.nifty, color: css("--series-4"), dashed: true }], { fmt: (v) => compactInr(v), height: 240 }); });
  };
  build();
}

function loadTicker() {
  const bar = $("#ticker"); if (bar.childElementCount || !localStorage.getItem("si-key")) return;
  bar.append(h("span", { class: "ticker-item" }, "Loading prices…"));
  api("/ui/ticker").then((items) => { const row = () => items.map((x) => h(x.index ? "span" : "a", { class: "ticker-item", href: x.index ? null : `#/stock/${encodeURIComponent(x.symbol)}` },
    h("b", {}, x.symbol), num(x.price), h("span", { class: cls(x.change_pct) }, pct(x.change_pct))));
    bar.replaceChildren(h("div", { class: "ticker-track" }, ...row(), ...row())); })
    .catch((e) => bar.replaceChildren(h("span", { class: "ticker-item" }, `Ticker unavailable: ${e.message}`)));
}
for (const el of document.querySelectorAll("[data-icon]")) el.append(icon(el.dataset.icon));

// ---------- router ----------
async function route() {
  loadTicker();
  const hash = location.hash || "#/home";
  const [path, query] = hash.slice(2).split("?"); const parts = path.split("/"); const params = new URLSearchParams(query || "");
  for (const a of document.querySelectorAll("#nav a, #bnav a")) a.classList.toggle("active", a.dataset.r === parts[0] || (parts[0] === "report" && a.dataset.r === "reports"));
  const view = $("#view"); view.replaceChildren(); window.scrollTo(0, 0);
  const pages = { home: () => pageHome(view), builder: () => pageBuilder(view), explore: () => pageExplore(view), stock: () => pageStock(view, decodeURIComponent(parts[1] || "").toUpperCase()), portfolio: () => pagePortfolio(view),
    signals: () => pageSignals(view), chat: () => pageChat(view, params), performance: () => pagePerformance(view), learn: () => pageLearn(view), reports: () => pageReports(view), report: () => pageReport(view, parts[1]) };
  try { await (pages[parts[0]] || pages.home)(); } catch (e) { view.replaceChildren(errorBox(e)); }
}
window.addEventListener("hashchange", route);
route();
