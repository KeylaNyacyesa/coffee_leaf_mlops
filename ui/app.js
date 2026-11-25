const CLASS_COLORS = { Cercospora: "#8c6d31", Healthy: "#2ca02c", Leaf_rust: "#e07b10", Miner: "#8a63b8", Phoma: "#c0392b" };
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "–" : (100 * x).toFixed(1) + "%");
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.color = "#6b7366";

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

/* ---------------- tabs ---------------- */
document.querySelectorAll(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
    if (b.dataset.tab === "insights") loadInsights();
    if (b.dataset.tab === "retrain") refreshRetrain();
  })
);

/* ---------------- uptime monitoring ---------------- */
let uptimeBase = null, uptimeAt = null, pings = 0, okPings = 0;
const uptimeChart = new Chart($("uptimeChart"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "response time (ms)", data: [], borderColor: "#2f7d32", backgroundColor: "rgba(47,125,50,.12)", fill: true, tension: .3, pointRadius: 2 }] },
  options: { animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: "ms" } } } },
});

function fmtDuration(s) {
  s = Math.floor(s);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (d ? d + "d " : "") + String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
}

async function ping() {
  const t0 = performance.now();
  pings++;
  let ok = false;
  try {
    const h = await api("/health");
    ok = h.status === "ok";
    uptimeBase = h.uptime_seconds; uptimeAt = Date.now();
    $("cStatus").textContent = ok ? "Online" : "No model";
    $("cContainer").textContent = "container " + h.container;
    $("cVersion").textContent = h.model_version || "–";
    $("cLoaded").textContent = h.model_loaded_at ? "loaded " + new Date(h.model_loaded_at * 1000).toLocaleString() : "";
    $("statusText").textContent = h.retraining ? "online · retraining…" : "online";
  } catch (e) {
    $("cStatus").textContent = "Offline";
    $("statusText").textContent = "offline";
  }
  if (ok) okPings++;
  document.querySelector("#statusPill .dot").className = "dot " + (ok ? "ok" : "bad");
  $("cAvail").textContent = `availability ${((100 * okPings) / pings).toFixed(1)}% (${pings} checks)`;
  const ms = performance.now() - t0;
  const ds = uptimeChart.data;
  ds.labels.push(new Date().toLocaleTimeString()); ds.datasets[0].data.push(ok ? +ms.toFixed(1) : null);
  if (ds.labels.length > 60) { ds.labels.shift(); ds.datasets[0].data.shift(); }
  uptimeChart.update();
  try {
    const s = await api("/stats");
    $("cPreds").textContent = s.predictions;
    $("cLat").textContent = s.latency_ms.avg ? `avg ${s.latency_ms.avg} ms · p95 ${s.latency_ms.p95} ms` : "no predictions yet";
  } catch (e) {}
}
setInterval(() => { if (uptimeBase !== null) $("cUptime").textContent = fmtDuration(uptimeBase + (Date.now() - uptimeAt) / 1000); }, 1000);
setInterval(ping, 5000);
ping();

/* ---------------- model metrics ---------------- */
let historyChart;
async function loadModel() {
  let m;
  try { m = await api("/model"); } catch (e) { $("metricsNote").textContent = "No model metrics yet."; return; }
  const t = m.test, b = m.baseline_test || {};
  $("cAcc").textContent = pct(t.accuracy);
  $("cF1").textContent = "macro F1 " + fmt(t.f1_macro);
  const rows = [["Accuracy", "accuracy", 1], ["Loss (cross-entropy)", "loss", -1], ["Precision (macro)", "precision_macro", 1],
    ["Recall (macro)", "recall_macro", 1], ["F1 (macro)", "f1_macro", 1], ["F1 (weighted)", "f1_weighted", 1], ["ROC-AUC (OvR macro)", "roc_auc_ovr_macro", 1]];
  let html = "<tr><th>Metric</th><th>Baseline CNN</th><th>Production model</th></tr>";
  for (const [name, k, dir] of rows) {
    const better = b[k] !== undefined && (dir > 0 ? t[k] > b[k] : t[k] < b[k]);
    html += `<tr><td>${name}</td><td class="num">${fmt(b[k])}</td><td class="num ${better ? "better" : ""}">${fmt(t[k])}</td></tr>`;
  }
  $("metricsTable").innerHTML = html;
  $("metricsNote").textContent = `${t.n_samples} test images (leaves never seen in training, in any rotation). ` +
    (m.retrained_at ? `Last retrained ${new Date(m.retrained_at * 1000).toLocaleString()} (previous accuracy ${pct(m.previous_test_accuracy)}).` : `Trained ${new Date(m.trained_at * 1000).toLocaleString()}.`);

  // confusion matrix
  const cm = t.confusion_matrix, names = m.class_names || Object.keys(CLASS_COLORS);
  const max = Math.max(...cm.map((r) => Math.max(...r)));
  let g = `<div class="cm" style="grid-template-columns: 90px repeat(${names.length}, 1fr)"><div class="h"></div>`;
  names.forEach((n) => (g += `<div class="h">${n.replace("_", " ")}</div>`));
  cm.forEach((row, i) => {
    g += `<div class="h" style="text-align:right">${names[i].replace("_", " ")}</div>`;
    row.forEach((v, j) => {
      const a = v / max, diag = i === j;
      const bg = diag ? `rgba(47,125,50,${0.12 + 0.8 * a})` : v ? `rgba(179,54,31,${0.15 + 0.7 * a})` : "#f3f0e8";
      g += `<div style="background:${bg};color:${a > .5 ? "#fff" : "#333"}">${v}</div>`;
    });
  });
  $("confusion").innerHTML = g + "</div>";

  // training curves
  const h = m.retrain_history || m.history;
  if (h) {
    const labels = h.loss.map((_, i) => i + 1);
    historyChart && historyChart.destroy();
    const split = m.retrain_history ? null : m.phase1_epochs;
    $("historyNote").textContent = m.retrain_history ? "(latest retraining run)" : "(dashed line = start of fine-tuning)";
    historyChart = new Chart($("historyChart"), {
      type: "line",
      data: { labels, datasets: [
        { label: "train acc", data: h.accuracy, borderColor: "#2f7d32", yAxisID: "y", pointRadius: 1 },
        { label: "val acc", data: h.val_accuracy, borderColor: "#8fd17a", borderDash: [4, 3], yAxisID: "y", pointRadius: 1 },
        { label: "train loss", data: h.loss, borderColor: "#5b3a29", yAxisID: "y1", pointRadius: 1 },
        { label: "val loss", data: h.val_loss, borderColor: "#c7741a", borderDash: [4, 3], yAxisID: "y1", pointRadius: 1 },
      ] },
      options: { scales: { x: { title: { display: true, text: "epoch" } }, y: { min: 0, max: 1, title: { display: true, text: "accuracy" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "loss" } } } },
      plugins: [{ id: "split", afterDraw(c) { if (!split) return; const x = c.scales.x.getPixelForValue(split - 0.5); const ctx = c.ctx; ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = "#999"; ctx.beginPath(); ctx.moveTo(x, c.chartArea.top); ctx.lineTo(x, c.chartArea.bottom); ctx.stroke(); ctx.restore(); } }],
    });
  }
}
loadModel();

/* ---------------- prediction ---------------- */
let predFile = null;
const drop = $("drop");
function setPredFile(f) {
  predFile = f;
  $("predPreview").src = URL.createObjectURL(f);
  $("predPreview").style.display = "block";
  $("dropText").style.display = "none";
  $("predBtn").disabled = false;
}
$("predFile").addEventListener("change", (e) => e.target.files[0] && setPredFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setPredFile(e.dataTransfer.files[0]));

$("predBtn").addEventListener("click", async () => {
  if (!predFile) return;
  $("predBtn").disabled = true; $("predBtn").textContent = "Predicting…";
  const fd = new FormData(); fd.append("file", predFile);
  try {
    const r = await api("/predict", { method: "POST", body: fd });
    $("predEmpty").hidden = true; $("predResult").hidden = false;
    $("predLabel").textContent = r.prediction.replace("_", " ");
    $("predLabel").style.color = CLASS_COLORS[r.prediction] || "";
    $("predConf").textContent = `confidence ${(100 * r.confidence).toFixed(1)}%`;
    $("predDesc").textContent = r.description;
    const sorted = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);
    $("probBars").innerHTML = sorted.map(([c, p], i) =>
      `<div class="prob ${i === 0 ? "top" : ""}"><span>${c.replace("_", " ")}</span><div class="track"><div class="fill" style="width:${(100 * p).toFixed(1)}%"></div></div><span class="mono">${(100 * p).toFixed(1)}%</span></div>`).join("");
    $("predMeta").textContent = `${r.filename} · ${r.latency_ms} ms · model ${r.model_version} · container ${r.container}`;
  } catch (e) {
    alert("Prediction failed: " + e.message);
  } finally {
    $("predBtn").disabled = false; $("predBtn").textContent = "Predict";
    ping();
  }
});

/* ---------------- insights ---------------- */
let insightsLoaded = false;
const STORIES = {
  green_index: (s) => `<b>Story.</b> Healthy leaves are by far the greenest (mean ${s.Healthy}); every disease lowers greenness because green tissue is replaced by yellow, orange or brown tissue. Leaf rust is the least green (${s.Leaf_rust}), matching its pale yellow-orange look. Greenness alone already separates Healthy from diseased leaves.`,
  lesion_fraction: (s) => `<b>Story.</b> A rough "disease severity" score: Healthy leaves have almost no discoloured leaf area (${s.Healthy}), while Leaf rust (${s.Leaf_rust}), Cercospora (${s.Cercospora}) and Phoma (${s.Phoma}) have the most. Miner damage forms separate brown patches on a green leaf (${s.Miner}). The histograms overlap, so colour alone cannot tell the diseases apart. The CNN has to learn lesion shape and texture.`,
  texture: (s) => `<b>Story.</b> Miner (${s.Miner}) and Phoma (${s.Phoma}) photos are the most textured: sharp lesion borders, veins and background foliage. Leaf rust photos are the smoothest (${s.Leaf_rust}) because they are blurred close-ups. That is partly a <i>dataset artefact</i>: the model could learn "blurry means rust". Retraining on new field photos uploaded by users is how we remove that shortcut.`,
};
async function loadInsights() {
  if (insightsLoaded) return;
  let d;
  try { d = await api("/insights"); } catch (e) { return; }
  insightsLoaded = true;
  const cls = d.classes;
  new Chart($("countChart"), {
    type: "bar",
    data: { labels: cls.map((c) => c.replace("_", " ")), datasets: [
      { label: "files downloaded", data: cls.map((c) => d.raw_counts[c] || 0), backgroundColor: "#d8cfbb" },
      { label: "unique images used", data: cls.map((c) => d.counts[c]), backgroundColor: cls.map((c) => CLASS_COLORS[c]) },
    ] },
    options: { scales: { y: { type: "logarithmic", title: { display: true, text: "images (log scale)" } } } },
  });
  for (const f of ["green_index", "lesion_fraction", "texture"]) {
    const F = d.features[f];
    const centers = F.bins.slice(0, -1).map((b, i) => ((b + F.bins[i + 1]) / 2).toFixed(f === "texture" ? 1 : 2));
    new Chart($("f_" + f), {
      type: "line",
      data: { labels: centers, datasets: cls.filter((c) => F.per_class[c]).map((c) => ({
        label: `${c.replace("_", " ")} (μ=${F.per_class[c].mean.toFixed(2)})`, data: F.per_class[c].hist.map((v) => +(100 * v).toFixed(1)),
        borderColor: CLASS_COLORS[c], backgroundColor: CLASS_COLORS[c] + "22", fill: true, tension: .35, pointRadius: 0 })) },
      options: { scales: { y: { title: { display: true, text: "% of class images" } }, x: { title: { display: true, text: f.replace("_", " ") } } } },
    });
    const means = Object.fromEntries(cls.map((c) => [c, F.per_class[c] ? F.per_class[c].mean.toFixed(2) : "–"]));
    $("s_" + f).innerHTML = STORIES[f](means);
  }
  $("swatches").innerHTML = cls.map((c) => {
    const [r, g, b] = d.mean_color[c].map(Math.round);
    return `<div class="swatch"><div style="background:rgb(${r},${g},${b})"></div>${c.replace("_", " ")}<br><span class="muted mono">${r},${g},${b}</span></div>`;
  }).join("");
}

/* ---------------- upload + retrain ---------------- */
(async () => {
  const cls = await api("/classes").catch(() => []);
  cls.forEach((c) => $("upLabel").insertAdjacentHTML("beforeend", `<option>${c}</option>`));
})();

$("upBtn").addEventListener("click", async () => {
  const files = $("upFiles").files;
  if (!files.length) return alert("Choose images or a zip first");
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  if ($("upLabel").value) fd.append("label", $("upLabel").value);
  $("upBtn").disabled = true; $("upResult").textContent = `Uploading ${files.length} file(s)…`;
  try {
    const r = await api("/upload", { method: "POST", body: fd });
    const reasons = {};
    r.details.filter((x) => x.status !== "saved").forEach((x) => (reasons[x.reason] = (reasons[x.reason] || 0) + 1));
    $("upResult").innerHTML = `<b>${r.saved}</b> saved, ${r.rejected} rejected ${Object.keys(reasons).length ? "(" + Object.entries(reasons).map(([k, v]) => `${v} ${k}`).join(", ") + ")" : ""}. ` +
      `${r.pending_total} images waiting for retraining.` + (r.auto_retrain_started ? " <b>Threshold reached: automatic retraining started.</b>" : "");
    $("upFiles").value = "";
  } catch (e) {
    $("upResult").textContent = "Upload failed: " + e.message;
  } finally { $("upBtn").disabled = false; refreshRetrain(); }
});

$("retrainBtn").addEventListener("click", async () => {
  try {
    await api(`/retrain?epochs=${$("epochs").value}`, { method: "POST" });
    refreshRetrain();
  } catch (e) { alert(e.message); }
});

let pollTimer = null;
let wasRunning = false;      // to notice the moment a run finishes
async function refreshRetrain() {
  try {
    const u = await api("/uploads");
    const by = Object.entries(u.by_label).map(([k, v]) => `${k}: ${v.pending || 0}`).join(" · ");
    $("pendingInfo").innerHTML = `<b>${u.pending}</b> new image(s) waiting for retraining ${by ? "(" + by + ")" : ""}`;
    $("thresholdBar").style.width = Math.min(100, (100 * u.pending) / u.retrain_threshold) + "%";
    $("thresholdText").textContent = u.auto_retrain
      ? `Automatic trigger: retraining starts on its own when ${u.retrain_threshold} new images are waiting (${u.pending}/${u.retrain_threshold}).`
      : "Automatic trigger disabled. Use the button.";
    const s = await api("/retrain/status");
    const running = s.state === "running";
    $("retrainBtn").disabled = running || u.pending === 0;
    $("retrainBtn").textContent = running ? "Retraining…" : "Retrain model";
    if (s.log.length) $("retrainLog").textContent = s.log.join("\n");
    $("retrainLog").scrollTop = 1e9;
    let html = "<tr><th>#</th><th>Started</th><th>Trigger</th><th>New imgs</th><th>Epochs</th><th>Old acc</th><th>New acc</th><th>Old F1</th><th>New F1</th><th>Result</th></tr>";
    for (const r of s.runs) {
      const badge = r.status === "running" ? `<span class="badge run">running</span>` : r.promoted ? `<span class="badge yes">promoted</span>`
        : r.status === "completed" ? `<span class="badge no">kept old model</span>` : `<span class="badge no">${r.status}</span>`;
      html += `<tr><td>${r.id}</td><td>${new Date(r.started_at * 1000).toLocaleString()}</td><td>${r.trigger}</td><td>${r.n_new_images ?? ""}</td><td>${r.epochs_run ?? ""}</td>` +
        `<td class="num">${pct(r.old_accuracy)}</td><td class="num">${pct(r.new_accuracy)}</td><td class="num">${fmt(r.old_f1)}</td><td class="num">${fmt(r.new_f1)}</td><td>${badge}</td></tr>`;
    }
    $("runsTable").innerHTML = s.runs.length ? html : "<tr><td class='muted'>No retraining runs yet.</td></tr>";
    clearTimeout(pollTimer);
    if (running) pollTimer = setTimeout(refreshRetrain, 2000);
    else if (wasRunning) { loadModel(); insightsLoaded = false; ping(); }   // a run just finished: refresh metrics
    wasRunning = running;
  } catch (e) { console.error(e); }
}
