// Vanilla JS — Upscaler frontend wired to FastAPI backend (image + video).
const $ = (id) => document.getElementById(id);

const FAMILY_GUESS = (key) => {
  if (key.startsWith("realesrgan") || key === "ultrasharp" || key === "remacri") return "ESRGAN";
  if (key === "drct-l") return "DRCT";
  if (key === "nomos-atd-jpg") return "ATD";
  if (key.startsWith("seedvr2")) return "SEEDVR2";
  if (key.startsWith("flashvsr")) return "FLASHVSR";
  return "SR";
};

const MAX_VIDEO_SECONDS = 60;
const MAX_VIDEO_MB = 500;

const state = {
  media: "image",        // "image" | "video"
  models: { image: [], video: [] },
  selectedModel: { image: null, video: null },
  scale: 4,
  fp16: true,
  batchSize: 25,
  file: null,
  inputUrl: null,
  inputSize: null,
  inputDuration: null,
  jobId: null,
  logSeen: 0,
  status: "idle",
  outputUrl: null,
  outputSize: null,
  zoom: 1,
  split: 50,
  history: [],
};

const fmtPx = (w, h) => `${w.toLocaleString()} × ${h.toLocaleString()}`;
const now = () => new Date().toLocaleTimeString("en-US", { hour12: false });
const isVideo = () => state.media === "video";

// ── Models ─────────────────────────────────────────────────────────────
async function loadModels() {
  const [imgRes, vidRes] = await Promise.all([
    fetch("/api/models").then((r) => r.json()),
    fetch("/api/video-models").then((r) => r.json()).catch(() => []),
  ]);
  state.models.image = imgRes;
  state.models.video = vidRes;
  state.selectedModel.image = imgRes[0]?.key ?? null;
  state.selectedModel.video = vidRes[0]?.key ?? null;
  renderModels();
  syncInfo();
}

function currentModels() { return state.models[state.media]; }
function currentModelKey() { return state.selectedModel[state.media]; }

function renderModels() {
  const list = $("model-list");
  list.innerHTML = "";
  const models = currentModels();
  if (!models.length) {
    list.innerHTML = `<div class="mono small dim">no ${state.media} models registered</div>`;
    return;
  }
  for (const m of models) {
    const card = document.createElement("button");
    card.className = "model-card" + (m.key === currentModelKey() ? " is-active" : "");
    const extra = isVideo() ? `<span class="mono dim">~${m.vram_gb}GB</span>` : `<span class="mono dim">${m.key}</span>`;
    card.innerHTML = `
      <div class="model-card-head">
        <span class="model-name">${m.name}</span>
        <span class="model-tag">${FAMILY_GUESS(m.key)}</span>
      </div>
      <div class="model-card-body">${m.best_for}</div>
      <div class="model-card-foot">
        <span class="mono">${m.scale}×</span>
        ${extra}
      </div>`;
    card.addEventListener("click", () => {
      state.selectedModel[state.media] = m.key;
      renderModels();
      syncInfo();
    });
    list.appendChild(card);
  }
}

// ── Media segmented (image/video) ──────────────────────────────────────
$("media-seg").addEventListener("click", (e) => {
  const btn = e.target.closest(".seg-btn");
  if (!btn) return;
  const next = btn.dataset.media;
  if (next === state.media) return;
  state.media = next;
  document.querySelectorAll("#media-seg .seg-btn").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.media === state.media);
  });
  // Reset stage
  showDropzone();
  renderModels();
  applyDropzoneCopy();
  applyAccept();
  syncInfo();
});

function applyBatchVisibility() {
  $("batch-section").hidden = !isVideo();
}

function applyDropzoneCopy() {
  applyBatchVisibility();
  if (isVideo()) {
    $("dz-title").textContent = "Drop a video to upscale";
    $("dz-sub").innerHTML = `MP4 · MOV · WebM &nbsp;·&nbsp; ≤${MAX_VIDEO_SECONDS}s &nbsp;·&nbsp; ≤720p input &nbsp;·&nbsp; up to ${MAX_VIDEO_MB} MB`;
    $("dz-meta").textContent = "POST /api/upscale-video · multipart/form-data";
  } else {
    $("dz-title").textContent = "Drop an image to upscale";
    $("dz-sub").innerHTML = "JPG · PNG · WebP &nbsp;·&nbsp; up to 50 MB";
    $("dz-meta").textContent = "POST /api/upscale · multipart/form-data";
  }
}

function applyAccept() {
  $("file-input").accept = isVideo()
    ? "video/mp4,video/quicktime,video/webm"
    : "image/png,image/jpeg,image/webp";
}

// ── Scale segmented ────────────────────────────────────────────────────
$("scale-seg").addEventListener("click", (e) => {
  const btn = e.target.closest(".seg-btn");
  if (!btn) return;
  state.scale = Number(btn.dataset.scale);
  document.querySelectorAll("#scale-seg .seg-btn").forEach((b) => {
    b.classList.toggle("is-active", Number(b.dataset.scale) === state.scale);
  });
  $("upscale-btn").textContent = `Upscale ${state.scale}×`;
  $("label-after").textContent = `AFTER · ${state.scale}×`;
  syncInfo();
});

document.querySelectorAll("[data-toggle]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.toggle;
    const sw = btn.querySelector(".toggle-switch");
    const on = !sw.classList.contains("is-on");
    sw.classList.toggle("is-on", on);
    btn.setAttribute("aria-pressed", on);
    if (key === "fp16") state.fp16 = on;
  });
});

// ── Drop zone & file input ─────────────────────────────────────────────
const dropzone = $("dropzone");
const fileInput = $("file-input");
$("choose-file").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("is-drag"); })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (!f) return;
  if (isVideo() && f.type.startsWith("video/")) handleFile(f);
  else if (!isVideo() && f.type.startsWith("image/")) handleFile(f);
});

function handleFile(f) {
  if (isVideo()) return handleVideoFile(f);
  return handleImageFile(f);
}

function handleImageFile(f) {
  state.file = f;
  state.inputUrl = URL.createObjectURL(f);
  state.outputUrl = null;
  state.outputSize = null;
  state.status = "idle";
  state.jobId = null;
  const img = new Image();
  img.onload = () => {
    state.inputSize = [img.naturalWidth, img.naturalHeight];
    showImageViewer();
    syncInfo();
  };
  img.src = state.inputUrl;
}

function handleVideoFile(f) {
  if (f.size > MAX_VIDEO_MB * 1024 * 1024) {
    alert(`Video exceeds ${MAX_VIDEO_MB} MB limit`);
    return;
  }
  state.file = f;
  state.inputUrl = URL.createObjectURL(f);
  state.outputUrl = null;
  state.outputSize = null;
  state.status = "idle";
  state.jobId = null;
  const v = document.createElement("video");
  v.preload = "metadata";
  v.onloadedmetadata = () => {
    if (v.duration > MAX_VIDEO_SECONDS + 0.5) {
      alert(`Video is ${v.duration.toFixed(1)}s — max ${MAX_VIDEO_SECONDS}s.`);
      return;
    }
    if (v.videoHeight > 720) {
      alert(`Video is ${v.videoHeight}p — max 720p input.`);
      return;
    }
    state.inputSize = [v.videoWidth, v.videoHeight];
    state.inputDuration = v.duration;
    showVideoViewer();
    syncInfo();
  };
  v.src = state.inputUrl;
}

function showImageViewer() {
  $("dropzone").hidden = true;
  $("viewer").hidden = false;
  $("img-before").hidden = false;
  $("img-before").src = state.inputUrl;
  $("img-after").hidden = true;
  $("img-after").removeAttribute("src");
  $("vid-before").hidden = true;
  $("vid-after").hidden = true;
  $("split").hidden = true;
  $("label-after").hidden = true;
  $("download-btn").hidden = true;
  $("upscale-btn").hidden = false;
  $("upscale-btn").disabled = false;
  $("upscale-btn").textContent = `Upscale ${state.scale}×`;
  applyZoom();
}

function showVideoViewer() {
  $("dropzone").hidden = true;
  $("viewer").hidden = false;
  $("img-before").hidden = true;
  $("img-after").hidden = true;
  const vb = $("vid-before");
  vb.hidden = false;
  vb.src = state.inputUrl;
  $("vid-after").hidden = true;
  $("vid-after").removeAttribute("src");
  $("split").hidden = true;
  $("label-after").hidden = true;
  $("download-btn").hidden = true;
  $("upscale-btn").hidden = false;
  $("upscale-btn").disabled = false;
  const m = currentModels().find((x) => x.key === currentModelKey());
  $("upscale-btn").textContent = m ? `Upscale ${m.scale}×` : "Upscale";
  applyZoom();
}

function showDropzone() {
  $("dropzone").hidden = false;
  $("viewer").hidden = true;
  state.file = null;
  state.inputUrl = null;
  state.inputSize = null;
  state.inputDuration = null;
  state.outputUrl = null;
  state.outputSize = null;
  state.status = "idle";
  state.jobId = null;
  $("progress-overlay").hidden = true;
}

$("new-image").addEventListener("click", showDropzone);
$("cancel-btn").addEventListener("click", async () => {
  const inFlight = state.jobId && (state.status === "processing" || state.status === "queued");
  if (inFlight) {
    try {
      await fetch(`/api/job/${state.jobId}/cancel`, { method: "POST" });
      $("prog-status").textContent = "CANCELLING";
    } catch (e) {
      console.warn("cancel failed", e);
    }
    return;
  }
  showDropzone();
});

// ── Zoom ───────────────────────────────────────────────────────────────
$("zoom-in").addEventListener("click", () => { state.zoom = Math.min(4, state.zoom + 0.25); applyZoom(); });
$("zoom-out").addEventListener("click", () => { state.zoom = Math.max(0.25, state.zoom - 0.25); applyZoom(); });
function applyZoom() {
  $("compare-inner").style.transform = `scale(${state.zoom})`;
  $("zoom-readout").textContent = `${Math.round(state.zoom * 100)}%`;
}

// ── Split drag (image only) ────────────────────────────────────────────
const compare = $("compare");
let dragging = false;
compare.addEventListener("pointerdown", (e) => {
  if (isVideo()) return;
  if ($("img-after").hidden) return;
  dragging = true;
  compare.setPointerCapture(e.pointerId);
  movesplit(e);
});
compare.addEventListener("pointermove", (e) => { if (dragging) movesplit(e); });
compare.addEventListener("pointerup", (e) => { dragging = false; try { compare.releasePointerCapture(e.pointerId); } catch {} });
function movesplit(e) {
  const r = compare.getBoundingClientRect();
  const pct = ((e.clientX - r.left) / r.width) * 100;
  state.split = Math.max(0, Math.min(100, pct));
  $("split").style.left = `${state.split}%`;
  $("img-after").style.clipPath = `inset(0 0 0 ${state.split}%)`;
}

// ── Upscale ────────────────────────────────────────────────────────────
$("upscale-btn").addEventListener("click", upscale);

async function upscale() {
  if (!state.file) return;
  state.status = "queued";
  state.outputUrl = null;
  state.logSeen = 0;
  $("upscale-btn").disabled = true;
  $("progress-overlay").hidden = false;
  document.querySelector(".progress-card")?.classList.remove("is-error");
  $("prog-dismiss").hidden = true;
  $("prog-status").textContent = "QUEUED";
  $("prog-pct").textContent = "0%";
  $("prog-fill").style.width = "0%";
  resetPhases();

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("model", currentModelKey());
  if (!isVideo()) fd.append("scale", String(state.scale));
  if (isVideo()) fd.append("batch_size", String(state.batchSize));

  const url = isVideo() ? "/api/upscale-video" : "/api/upscale";
  let res;
  try {
    res = await fetch(url, { method: "POST", body: fd });
  } catch (e) {
    return fail(`network: ${e.message}`);
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    return fail(detail.detail || `HTTP ${res.status}`);
  }
  const { job_id } = await res.json();
  state.jobId = job_id;
  pollJob(job_id);
}

async function pollJob(id) {
  while (true) {
    await new Promise((r) => setTimeout(r, 500));
    const res = await fetch(`/api/job/${id}?since=${state.logSeen}`);
    if (!res.ok) return fail(`job lookup HTTP ${res.status}`);
    const j = await res.json();
    state.status = j.status;
    $("prog-status").textContent = j.status.toUpperCase();
    $("prog-pct").textContent = `${j.progress}%`;
    $("prog-fill").style.width = `${j.progress}%`;
    if (Array.isArray(j.logs) && j.logs.length) {
      for (const ln of j.logs) classifyLog(ln);
      state.logSeen = j.log_seq ?? state.logSeen;
    }
    if (j.kind === "video" && j.total_frames) {
      // Server-side authoritative frame counter feeds whichever phase is active.
      const target = phases.decode.active  || phases.decode.done  > 0 ? phases.decode
                   : phases.upscale.active || phases.upscale.done > 0 ? phases.upscale
                   : phases.frames;
      target.done = Math.max(target.done, j.frame ?? 0);
      target.total = Math.max(target.total, j.total_frames);
      target.active = true;
    }
    renderPhases();

    if (j.status === "done") {
      state.outputUrl = j.output_url;
      state.outputSize = j.output_size;
      finishJob(j);
      return;
    }
    if (j.status === "error") return fail(j.error || "unknown error");
    if (j.status === "cancelled") {
      $("prog-status").textContent = "CANCELLED";
      $("prog-pct").textContent = "—";
      $("prog-fill").style.width = "0%";
      $("upscale-btn").disabled = false;
      $("prog-dismiss").hidden = false;
      return;
    }
  }
}

function finishJob(j) {
  $("progress-overlay").hidden = true;
  if (isVideo()) {
    const va = $("vid-after");
    va.src = j.output_url;
    va.hidden = false;
    $("split").hidden = true;
  } else {
    $("img-after").src = j.output_url;
    $("img-after").hidden = false;
    $("img-after").style.clipPath = `inset(0 0 0 ${state.split}%)`;
    $("split").hidden = false;
    $("split").style.left = `${state.split}%`;
  }
  const scale = j.input_size && j.output_size ? Math.round(j.output_size[0] / j.input_size[0]) : state.scale;
  $("label-after").textContent = `AFTER · ${scale}×`;
  $("label-after").hidden = false;
  $("upscale-btn").hidden = true;
  $("download-btn").hidden = false;
  $("download-btn").href = j.output_url;
  const ext = isVideo() ? "mp4" : "png";
  const base = (state.file?.name || state.media).replace(/\.[^.]+$/, "");
  $("download-btn").download = `${base}-${currentModelKey()}-${scale}x.${ext}`;
  $("download-btn").querySelector("svg")?.nextSibling?.remove?.();
  $("zoom-in").disabled = false;
  $("zoom-out").disabled = false;
  syncInfo();

  state.history.unshift({
    id: state.jobId,
    media: state.media,
    input: state.inputUrl,
    output: j.output_url,
    model: currentModelKey(),
    scale,
    elapsedMs: j.elapsed_ms,
  });
  if (state.history.length > 12) state.history.length = 12;
  renderHistory();
}

function fail(msg) {
  state.status = "error";
  document.querySelector(".progress-card")?.classList.add("is-error");
  $("prog-status").textContent = "ERROR";
  $("prog-pct").textContent = "—";
  $("prog-fill").style.width = "100%";
  $("prog-batches").textContent = msg.slice(0, 60);
  $("upscale-btn").disabled = false;
  $("prog-dismiss").hidden = false;
}

// ── Phase progress (frames + upscale) ──────────────────────────────────
const phases = {
  frames:  { done: 0, total: 0, active: false, finished: false },
  upscale: { done: 0, total: 0, active: false, finished: false },
  decode:  { done: 0, total: 0, active: false, finished: false },
};
const PHASE_KEYS = ["frames", "upscale", "decode"];
let aggregateCount = 0;

// Order matters: decode checked first (matches "VAE decoding"), then upscale, then frames.
const DECODE_RE  = /(VAE decod|decoding latent|writing video|encoding output|finalizing|\bmux\b|saving)/i;
const UPSCALE_RE = /(euler|sampl|denois|diffus|\bDiT\b|step\s*\d+\s*\/|batch\s*\d+\s*\/|upscal)/i;
const FRAME_RE   = /(extract|reading frames|input frames|VAE encoding|frame\s*\d+\s*\/)/i;
const COUNT_RE   = /(\d+)\s*\/\s*(\d+)/;

function resetPhases() {
  for (const k of PHASE_KEYS) phases[k] = { done: 0, total: 0, active: false, finished: false };
  aggregateCount = 0;
  $("prog-batches").textContent = "stage —";
  $("prog-aggregate").textContent = "+0 events aggregated";
  for (const k of PHASE_KEYS) {
    $(`phase-${k}-count`).textContent = "— / —";
    $(`phase-${k}-fill`).style.width = "0%";
    const el = $(`phase-${k}`);
    el.classList.remove("is-active", "is-done");
    el.classList.add("is-idle");
  }
}

function classifyLog(line) {
  const txt = String(line || "");
  let routed = null;
  if (DECODE_RE.test(txt))       routed = "decode";
  else if (UPSCALE_RE.test(txt)) routed = "upscale";
  else if (FRAME_RE.test(txt))   routed = "frames";
  if (!routed) { aggregateCount++; return; }
  bumpPhase(routed, txt);
  // Auto-finalize earlier phases when a later one starts.
  const idx = PHASE_KEYS.indexOf(routed);
  for (let i = 0; i < idx; i++) {
    const ph = phases[PHASE_KEYS[i]];
    if (!ph.finished) {
      ph.finished = true; ph.active = false;
      if (ph.total) ph.done = ph.total;
    }
  }
}

function bumpPhase(key, txt) {
  const ph = phases[key];
  ph.active = true;
  const m = txt.match(COUNT_RE);
  if (m) {
    const d = parseInt(m[1], 10), t = parseInt(m[2], 10);
    if (t > 0) { ph.done = Math.max(ph.done, d); ph.total = Math.max(ph.total, t); }
  }
}

function renderPhases() {
  for (const k of PHASE_KEYS) {
    const ph = phases[k];
    const el = $(`phase-${k}`);
    const pct = ph.total ? Math.min(100, (ph.done / ph.total) * 100) : (ph.active ? 5 : 0);
    $(`phase-${k}-fill`).style.width = `${pct}%`;
    $(`phase-${k}-count`).textContent = ph.total ? `${ph.done} / ${ph.total}` : (ph.active ? "…" : "— / —");
    el.classList.toggle("is-active", ph.active && !ph.finished);
    el.classList.toggle("is-done", ph.finished || (ph.total && ph.done >= ph.total));
    el.classList.toggle("is-idle", !ph.active && !ph.finished);
  }
  // Top stage indicator
  const totalStages = isVideo() ? 3 : 1;
  let cur = 1;
  if (phases.upscale.active || phases.upscale.done > 0) cur = 2;
  if (phases.decode.active  || phases.decode.done  > 0) cur = 3;
  $("prog-batches").textContent = `stage ${Math.min(cur, totalStages)}/${totalStages}`;
  $("prog-aggregate").textContent = `+${aggregateCount} events aggregated`;
}

// ── Info bar ───────────────────────────────────────────────────────────
function syncInfo() {
  $("info-input").textContent = state.inputSize ? fmtPx(...state.inputSize) : "—";
  const pendingScale = isVideo()
    ? (currentModels().find((x) => x.key === currentModelKey())?.scale ?? state.scale)
    : state.scale;
  $("info-output").textContent = state.outputSize
    ? fmtPx(...state.outputSize)
    : (state.inputSize ? `${pendingScale}× pending` : "—");
  const m = currentModels().find((x) => x.key === currentModelKey());
  $("info-model").textContent = m ? m.name : "—";
}

// ── History ────────────────────────────────────────────────────────────
function renderHistory() {
  const row = $("history-row");
  $("history-count").textContent = `· ${state.history.length}`;
  $("clear-history").hidden = state.history.length === 0;
  row.innerHTML = "";
  if (!state.history.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty mono small dim";
    empty.textContent = "No jobs yet · upscaled images appear here";
    row.appendChild(empty);
    return;
  }
  for (const h of state.history) {
    const item = document.createElement("button");
    item.className = "history-item";
    const thumb = h.media === "video"
      ? `<video src="${h.output}" muted></video>`
      : `<img src="${h.output}" alt="" />`;
    item.innerHTML = `
      ${thumb}
      <div class="history-meta">
        <span class="mono small">${h.scale}× · ${h.model}</span>
        <span class="mono small dim">${h.elapsedMs}ms</span>
      </div>`;
    item.addEventListener("click", () => window.open(h.output, "_blank"));
    row.appendChild(item);
  }
}
$("batch-select").addEventListener("change", (e) => {
  state.batchSize = Number(e.target.value);
});
$("clear-history").addEventListener("click", () => { state.history = []; renderHistory(); });
$("prog-dismiss").addEventListener("click", () => { $("progress-overlay").hidden = true; });

// ── GPU stats poll ─────────────────────────────────────────────────────
async function pollGpu() {
  try {
    const r = await fetch("/api/gpu");
    if (!r.ok) return;
    const g = await r.json();
    $("stat-gpu").textContent = g.name.replace(/^NVIDIA\s+/i, "");
    $("stat-vram").textContent = `${g.vram_used_gb.toFixed(1)} / ${g.vram_total_gb.toFixed(1)} GB`;
    const pct = g.vram_total_gb ? (g.vram_used_gb / g.vram_total_gb) * 100 : 0;
    $("stat-vram-bar").style.width = `${Math.min(100, pct)}%`;
    $("stat-jobs").textContent = String(g.jobs_done).padStart(3, "0");
  } catch {}
}
setInterval(pollGpu, 2000);

// ── Init ───────────────────────────────────────────────────────────────
loadModels();
applyDropzoneCopy();
applyAccept();
pollGpu();
