/* BRUMA · tablero (vanilla JS, sin build) */
const $ = (s) => document.querySelector(s);
const state = { meta: null, date: null, station: 29509, mapH: 1, shapH: 1, fc: null };
const tip = $("#tooltip");
const fmt = (x, d = 0) => (x == null ? "—" : Number(x).toLocaleString("es-CO", { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (p) => `${Math.round(p * 100)} %`;
const fmtDate = (s, opts = { weekday: "short", day: "numeric", month: "short" }) =>
  new Date(s + "T12:00:00").toLocaleDateString("es-CO", opts);

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px";
  tip.style.top = Math.max(8, y - h - 12) + "px";
}
const hideTip = () => tip.classList.remove("on");

/* ---------- arranque ---------- */
async function init() {
  const meta = await api("/api/meta");
  state.meta = meta;
  const qs = new URLSearchParams(location.search);
  state.date = qs.get("fecha") || meta.fechas.sugerida;
  if (meta.estaciones.some((s) => s.station === +qs.get("estacion"))) state.station = +qs.get("estacion");
  const sel = $("#station");
  const sorted = [...meta.estaciones].sort((a, b) => a.municipio.localeCompare(b.municipio) || a.nombre.localeCompare(b.nombre));
  sel.innerHTML = sorted.map((s) => `<option value="${s.station}">${s.nombre} · ${s.municipio}</option>`).join("");
  sel.value = state.station;
  const di = $("#date");
  di.min = meta.fechas.min; di.max = meta.fechas.max; di.value = state.date;

  di.addEventListener("change", () => { if (di.value) { state.date = di.value; refresh(); } });
  sel.addEventListener("change", () => { state.station = +sel.value; refreshStation(); });
  document.querySelectorAll(".episodes button").forEach((b) =>
    b.addEventListener("click", () => { state.date = b.dataset.date; di.value = state.date; refresh(); }));
  tabs("#mapTabs", (h) => { state.mapH = h; drawMap(); });
  tabs("#shapTabs", (h) => { state.shapH = h; drawShap(); });

  $("#icaLegend").innerHTML = meta.bandas_ica.slice(0, 5).map((b) =>
    `<span><i style="background:${b.color}"></i>${b.nombre} (${b.min}–${b.max})</span>`).join("");
  drawMetrics(meta.metricas);
  initMap();
  refresh();
  window.addEventListener("resize", () => state.fc && drawChart());
}

function tabs(sel, cb) {
  const root = $(sel);
  root.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    root.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    cb(+b.dataset.h);
  });
}

async function refresh() {
  await Promise.all([refreshStation(), drawMap(), drawBulletin()]);
}

function syncUrl() {
  history.replaceState(null, "", `?fecha=${state.date}&estacion=${state.station}`);
}

async function refreshStation() {
  try {
    state.fc = await api(`/api/forecast?station=${state.station}&date=${state.date}`);
  } catch (e) {
    $("#answer").textContent = "Sin datos";
    $("#heroSub").textContent = e.message;
    return;
  }
  if (state.fc.fecha_emision !== state.date) { state.date = state.fc.fecha_emision; $("#date").value = state.date; }
  syncUrl();
  drawHero(); drawHorizons(); drawChart(); drawShap();
}

/* ---------- héroe: ¿habrá alerta mañana? ---------- */
async function drawHero() {
  const h1 = state.fc.horizontes.find((h) => h.h === 1);
  const thr = state.meta.umbral_alerta[1];
  const a = $("#answer");
  if (!h1) { a.textContent = "—"; return; }
  let txt, color;
  if (h1.alerta) { txt = "Sí, es probable"; color = "#fb923c"; }
  else if (h1.p_naranja >= thr / 2) { txt = "Posible"; color = "#facc15"; }
  else { txt = "No se espera"; color = "#4ade80"; }
  a.textContent = txt; a.style.color = color;
  $("#heroSub").innerHTML = `En <b>${state.fc.estacion.nombre}</b> el ${fmtDate(h1.fecha, { weekday: "long", day: "numeric", month: "long" })}:
    probabilidad de superar 37 µg/m³ (nivel naranja) de <b>${pct(h1.p_naranja)}</b>.`;
  $("#gaugeFill").style.width = Math.max(2, h1.p_naranja * 100) + "%";
  $("#gaugeFill").style.background = `linear-gradient(90deg, #4ade80, ${color})`;
  $("#gaugeMark").style.left = thr * 100 + "%";
  $("#gaugeLabel").textContent = `aviso desde ${pct(thr)}`;
  try {
    const v = await api(`/api/valley?date=${state.date}&h=1`);
    const n = v.estaciones_en_alerta.length;
    $("#valleyLine").innerHTML = `Valle completo: <b>${n}</b> de ${v.estaciones.length} estaciones con aviso ·
      promedio esperado <b>${fmt(v.promedio_valle)} µg/m³</b>` +
      (v.observado_valle != null ? ` · observado después: <b>${fmt(v.observado_valle)}</b>` : "");
  } catch { /* opcional */ }
}

function drawHorizons() {
  $("#horizons").innerHTML = state.fc.horizontes.map((h) => {
    const hit = h.observado != null && h.observado >= h.q10 && h.observado <= h.q90;
    return `<div class="hz" style="--c:${h.ica.color}">
      <div class="lbl">${h.etiqueta}</div>
      <div class="date">${fmtDate(h.fecha)}</div>
      <div class="val">${fmt(h.q50, 1)} <small>µg/m³</small></div>
      <div class="rng">80 %: ${fmt(h.q10)} – ${fmt(h.q90)}</div>
      <div class="cat"><i></i>${h.ica.nombre}</div>
      <div class="prob">P(naranja) ${pct(h.p_naranja)} ${h.alerta ? '<span class="pill alert">⚠ aviso</span>' : '<span class="pill ok">✓ sin aviso</span>'}</div>
      <div class="truth">Pasó: <b>${h.observado == null ? "sin dato" : fmt(h.observado, 1) + " µg/m³"}</b>
        ${h.observado == null ? "" : hit ? " · dentro de la banda ✓" : " · fuera de la banda ✗"}</div>
    </div>`;
  }).join("");
}

/* ---------- mapa ---------- */
let map, layer;
function initMap() {
  map = L.map("map", { zoomControl: true, scrollWheelZoom: false, attributionControl: true }).setView([6.25, -75.57], 11);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { className: "dark-tiles",
    attribution: "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a>", maxZoom: 17,
  }).addTo(map);
  layer = L.layerGroup().addTo(map);
}

async function drawMap() {
  if (!map) return;
  const v = await api(`/api/valley?date=${state.date}&h=${state.mapH}`);
  layer.clearLayers();
  const pts = [];
  v.estaciones.forEach((s) => {
    pts.push([s.lat, s.lon]);
    const r = 7 + s.p_naranja * 14;
    if (s.alerta) L.circleMarker([s.lat, s.lon], { radius: r + 7, color: s.ica.color, weight: 1, opacity: .5, fillOpacity: .08 }).addTo(layer);
    const m = L.circleMarker([s.lat, s.lon], {
      radius: r, color: s.station === state.station ? "#ffffff" : "#0b1030", weight: s.station === state.station ? 3 : 2,
      fillColor: s.ica.color, fillOpacity: .9,
    }).addTo(layer);
    m.bindTooltip(`<b>${s.nombre}</b><br>${fmtDate(v.fecha_objetivo)}: ${fmt(s.q50)} µg/m³ (${fmt(s.q10)}–${fmt(s.q90)})
      <br>P(naranja): ${pct(s.p_naranja)}${s.observado != null ? `<br>Observado: ${fmt(s.observado, 1)}` : ""}`);
    m.on("click", () => { state.station = s.station; $("#station").value = s.station; refreshStation(); drawMap(); });
  });
  if (pts.length && !map._fitted) { map.invalidateSize(); map.fitBounds(pts, { padding: [28, 28] }); map._fitted = true; }
}

/* ---------- gráfica de pronóstico (SVG) ---------- */
function drawChart() {
  const el = $("#chart");
  const W = el.clientWidth, H = el.clientHeight;
  const m = { l: 40, r: 34, t: 12, b: 28 };
  const hist = state.fc.historia;
  const issue = state.fc.fecha_emision;
  const past = hist.filter((d) => !d.futuro);
  const future = hist.filter((d) => d.futuro);
  const hz = state.fc.horizontes;
  $("#chartTitle").textContent = `Pronóstico con incertidumbre · ${state.fc.estacion.nombre}`;

  const days = [...past.map((d) => d.fecha), ...hz.map((h) => h.fecha)];
  const t = (s) => new Date(s + "T00:00:00").getTime();
  const x0 = t(days[0]), x1 = t(days[days.length - 1]);
  const ymax = Math.max(45, ...past.map((d) => d.pm25 || 0), ...hz.map((h) => h.q90), ...future.map((d) => d.pm25 || 0)) * 1.08;
  const X = (s) => m.l + ((t(s) - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);

  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Serie observada y pronóstico con banda del 80 %">`;
  // franjas ICA muy tenues
  state.meta.bandas_ica.forEach((b) => {
    if (b.min > ymax) return;
    svg += `<rect x="${m.l}" width="${W - m.l - m.r}" y="${Y(Math.min(b.max, ymax))}" height="${Y(b.min) - Y(Math.min(b.max, ymax))}" fill="${b.color}" opacity="0.05"/>`;
  });
  // rejilla
  for (let v = 0; v <= ymax; v += 10) {
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/>
      <text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v}</text>`;
  }
  const step = Math.ceil(days.length / Math.max(3, Math.floor((W - 60) / 70)));
  days.forEach((d, i) => {
    if (i % step) return;
    svg += `<text x="${X(d)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="var(--muted)">${fmtDate(d, { day: "numeric", month: "short" })}</text>`;
  });
  // umbral naranja
  svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(37)}" y2="${Y(37)}" stroke="#f97316" stroke-dasharray="5 4" stroke-width="1.5"/>
    <text x="${m.l + 6}" y="${Y(37) - 6}" font-size="11" fill="#fdba74">naranja 37</text>`;
  // línea de emisión
  svg += `<line x1="${X(issue)}" x2="${X(issue)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--ink-2)" stroke-dasharray="2 4" opacity=".6"/>
    <text x="${X(issue) - 6}" y="${m.t + 12}" text-anchor="end" font-size="11" fill="var(--ink-2)">hoy (emisión)</text>`;
  // observado
  const anchor = past.length && past[past.length - 1].pm25 != null ? past[past.length - 1] : null;
  let path = "", pen = false;
  past.forEach((d) => {
    if (d.pm25 == null) { pen = false; return; }
    path += `${pen ? "L" : "M"}${X(d.fecha)},${Y(d.pm25)}`; pen = true;
  });
  svg += `<path d="${path}" fill="none" stroke="var(--ink-2)" stroke-width="2" stroke-linejoin="round"/>`;
  // banda + mediana
  const bandPts = (anchor ? [[anchor.fecha, anchor.pm25, anchor.pm25]] : []).concat(hz.map((h) => [h.fecha, h.q10, h.q90]));
  svg += `<path d="M${bandPts.map((p) => `${X(p[0])},${Y(p[2])}`).join("L")}L${bandPts.slice().reverse().map((p) => `${X(p[0])},${Y(p[1])}`).join("L")}Z"
    fill="rgba(142,162,255,0.25)" stroke="none"/>`;
  const med = (anchor ? [[anchor.fecha, anchor.pm25]] : []).concat(hz.map((h) => [h.fecha, h.q50]));
  svg += `<path d="M${med.map((p) => `${X(p[0])},${Y(p[1])}`).join("L")}" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="6 3"/>`;
  hz.forEach((h) => { svg += `<circle cx="${X(h.fecha)}" cy="${Y(h.q50)}" r="4.5" fill="var(--accent)" stroke="#0b1030" stroke-width="2"/>`; });
  // lo que pasó realmente
  future.forEach((d) => {
    if (d.pm25 == null) return;
    svg += `<circle cx="${X(d.fecha)}" cy="${Y(d.pm25)}" r="5" fill="var(--accent-2)" stroke="#0b1030" stroke-width="2"/>`;
  });
  svg += `<line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="var(--ink)" opacity="0" />`;
  svg += `<rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
  el.innerHTML = svg;

  // capa de hover: cruz + tooltip
  const byDate = {};
  past.forEach((d) => (byDate[d.fecha] = { obs: d.pm25 }));
  hz.forEach((h) => (byDate[h.fecha] = { ...(byDate[h.fecha] || {}), h }));
  future.forEach((d) => (byDate[d.fecha] = { ...(byDate[d.fecha] || {}), real: d.pm25 }));
  const hit = el.querySelector("#hit"), xh = el.querySelector("#xh");
  hit.addEventListener("mousemove", (e) => {
    const r = el.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = days[0], bd = Infinity;
    days.forEach((d) => { const dd = Math.abs(X(d) - px); if (dd < bd) { bd = dd; best = d; } });
    xh.setAttribute("x1", X(best)); xh.setAttribute("x2", X(best)); xh.setAttribute("opacity", .35);
    const v = byDate[best] || {};
    let html = `<b>${fmtDate(best, { weekday: "long", day: "numeric", month: "short" })}</b>`;
    if (v.obs != null) html += `<br>Observado: ${fmt(v.obs, 1)} µg/m³`;
    if (v.h) html += `<br>Pronóstico: ${fmt(v.h.q50, 1)} µg/m³<br>Banda 80 %: ${fmt(v.h.q10, 1)} – ${fmt(v.h.q90, 1)}<br>P(naranja): ${pct(v.h.p_naranja)}`;
    if (v.real != null) html += `<br>Lo que pasó: <b>${fmt(v.real, 1)}</b> µg/m³`;
    showTip(html, e.clientX, e.clientY);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); xh.setAttribute("opacity", 0); });
}

/* ---------- SHAP ---------- */
function drawShap() {
  const h = state.fc.horizontes.find((x) => x.h === state.shapH) || state.fc.horizontes[0];
  const ex = h.explicacion;
  const rows = [...ex.impulsores, { etiqueta: "Resto de variables", valor: "", aporte: ex.otros }];
  const maxAbs = Math.max(1, ...rows.map((r) => Math.abs(r.aporte)));
  $("#shap").innerHTML = `
    <div class="shap-total"><span>Nivel típico (valor base del modelo)</span><b>${fmt(ex.base, 1)} µg/m³</b></div>
    ${rows.map((r) => {
      const w = (Math.abs(r.aporte) / maxAbs) * 50;
      const up = r.aporte >= 0;
      return `<div class="shap-row">
        <div class="name">${r.etiqueta}${r.valor ? `<small>${r.valor}</small>` : ""}</div>
        <div class="shap-bar" aria-hidden="true"><div class="axis"></div>
          <div class="b" style="left:${up ? 50 : 50 - w}%;width:${w}%;background:${up ? "var(--up)" : "var(--down)"}"></div></div>
        <div class="num">${up ? "+" : "−"}${fmt(Math.abs(r.aporte), 1)}</div>
      </div>`;
    }).join("")}
    <div class="shap-total"><span>Pronóstico (mediana) para ${fmtDate(h.fecha)}</span><b>${fmt(ex.prediccion, 1)} µg/m³</b></div>
    <div class="shap-total"><span><i class="sw" style="background:var(--up)"></i>sube el pronóstico &nbsp; <i class="sw" style="background:var(--down)"></i>lo baja</span></div>`;
}

/* ---------- boletín ---------- */
async function drawBulletin() {
  const box = $("#bulletin");
  box.innerHTML = '<p class="hint">Redactando…</p>';
  try {
    const b = await api(`/api/bulletin?date=${state.date}`);
    const html = b.texto
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\s)_(.+?)_(?=\s|$)/g, "$1<em>$2</em>")
      .split(/\n\s*\n/).map((p) => `<p>${p}</p>`).join("");
    box.innerHTML = html;
    const src = $("#bulletinSource");
    src.textContent = b.fuente === "gemini" ? "✦ Redactado por Gemini con salidas del modelo" : "Plantilla determinista";
    src.className = "badge" + (b.fuente === "gemini" ? " gemini" : "");
  } catch (e) {
    box.innerHTML = `<p class="hint">${e.message}</p>`;
  }
}

/* ---------- métricas ---------- */
function drawMetrics(ms) {
  const names = { h1: "24 h", h2: "48 h", h3: "72 h" };
  const maxMae = Math.max(...Object.values(ms).map((m) => m.mae_persistencia));
  const cards = Object.entries(ms).map(([k, m]) => `
    <div class="metric">
      <div class="k">Horizonte ${names[k]} · MAE</div>
      <div class="v">${fmt(m.mae, 2)} <small style="font-size:14px">µg/m³</small></div>
      <div class="s">${fmt(m.skill * 100)} % menos error que la persistencia · R² ${fmt(m.r2, 2)}</div>
      <div class="bars">
        <div><span>BRUMA</span><span class="t" style="width:${(m.mae / maxMae) * 100}%"></span><span>${fmt(m.mae, 2)}</span></div>
        <div><span>Persistencia</span><span class="t ref" style="width:${(m.mae_persistencia / maxMae) * 100}%"></span><span>${fmt(m.mae_persistencia, 2)}</span></div>
      </div>
      <div class="s" style="margin-top:8px">Cobertura banda 80 %: <b>${fmt(m.cobertura_80 * 100)} %</b> · AUC alerta ${fmt(m.roc_auc, 2)}</div>
    </div>`).join("");
  $("#metrics").innerHTML = cards + `
    <div class="metric"><div class="k">Datos de evaluación</div>
      <div class="v">${fmt(ms.h1.n)}</div>
      <div class="s">pronósticos estación-día por horizonte (2024-2025), ${ms.h1.dias_naranja} de ellos en nivel naranja</div></div>`;
}

init().catch((e) => { $("#answer").textContent = "Error"; $("#heroSub").textContent = e.message; });
