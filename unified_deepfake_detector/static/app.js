/* ════════════════════════════════════════════════
   SENTINEL — Cyber Forensics JS
   ════════════════════════════════════════════════ */

const tabs = document.querySelectorAll(".mode-btn");
const modeSwitch = document.querySelector(".mode-switch");
const imageForm = document.getElementById("image-form");
const videoForm = document.getElementById("video-form");
const imageInput = document.getElementById("image-input");
const videoInput = document.getElementById("video-input");
const imagePreview = document.getElementById("image-preview");
const videoPreview = document.getElementById("video-preview");
const results = document.getElementById("results");
const statusBox = document.getElementById("system-status");

let mode = "image";

/* ── Mode Switching ── */
function setMode(m) {
    mode = m;
    modeSwitch.dataset.active = m;
    tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === m));
    imageForm.classList.toggle("active", m === "image");
    videoForm.classList.toggle("active", m === "video");
    results.innerHTML = idleHTML();
}

tabs.forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));

/* ── Idle Screen ── */
function idleHTML() {
    return `<div class="idle-screen">
        <div class="idle-radar">
            <div class="radar-ring r1"></div>
            <div class="radar-ring r2"></div>
            <div class="radar-ring r3"></div>
            <div class="radar-sweep"></div>
            <div class="radar-dot"></div>
        </div>
        <p class="idle-text">AWAITING TARGET</p>
        <p class="idle-sub">Upload media to begin forensic analysis</p>
    </div>`;
}

/* ── Preview ── */
function attachPreview(input, preview) {
    input.addEventListener("change", () => {
        const f = input.files[0];
        if (!f) return;
        preview.src = URL.createObjectURL(f);
        preview.classList.add("ready");
    });
}
attachPreview(imageInput, imagePreview);
attachPreview(videoInput, videoPreview);

/* ── Health ── */
async function refreshHealth() {
    try {
        const r = await fetch("/api/health");
        const d = await r.json();
        statusBox.innerHTML = d.video_model_exists
            ? '<span class="sys-led"></span><span>SYSTEMS ONLINE</span>'
            : '<span class="sys-led"></span><span>VIDEO MODEL MISSING</span>';
    } catch {
        statusBox.innerHTML = '<span class="sys-led" style="background:var(--red);box-shadow:0 0 6px var(--red)"></span><span>OFFLINE</span>';
    }
}

/* ── Confetti ── */
function fireConfetti(color) {
    const canvas = document.getElementById("confetti-canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const colors = color === "green"
        ? ["#00ff88", "#6ee7b7", "#34d399", "#a7f3d0", "#fff"]
        : ["#ff3355", "#fb923c", "#ff006e", "#fca5a5", "#fff"];
    const P = [];
    for (let i = 0; i < 90; i++) {
        P.push({
            x: Math.random() * canvas.width,
            y: -10 - Math.random() * 250,
            w: 4 + Math.random() * 6, h: 3 + Math.random() * 5,
            vx: (Math.random() - 0.5) * 4, vy: 2.5 + Math.random() * 4,
            rot: Math.random() * 360, rv: (Math.random() - 0.5) * 12,
            c: colors[Math.floor(Math.random() * colors.length)],
            life: 1
        });
    }
    let f = 0;
    (function go() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let alive = false;
        P.forEach(p => {
            if (p.life <= 0) return;
            alive = true;
            p.x += p.vx; p.y += p.vy; p.vy += 0.1;
            p.rot += p.rv; p.life -= 0.007;
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot * Math.PI / 180);
            ctx.globalAlpha = Math.max(0, p.life);
            ctx.fillStyle = p.c;
            ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
            ctx.restore();
        });
        f++;
        if (alive && f < 250) requestAnimationFrame(go);
        else ctx.clearRect(0, 0, canvas.width, canvas.height);
    })();
}

/* ── Gauge SVG ── */
function gaugeHTML(pct, cls) {
    const offset = 283 - (283 * Math.min(pct, 100) / 100);
    return `<div class="gauge-wrap">
        <svg class="gauge-svg" viewBox="0 0 100 100">
            <circle class="gauge-bg" cx="50" cy="50" r="45"/>
            <circle class="gauge-fill" cx="50" cy="50" r="45" style="stroke-dashoffset:${offset}"/>
        </svg>
        <div class="gauge-pct">${Math.round(pct)}%</div>
    </div>`;
}

/* ── Verdict class ── */
function vc(pred) {
    const p = (pred || "").toUpperCase();
    if (p === "FAKE" || p === "AI-GENERATED") return "v-fake";
    if (p === "REAL" || p === "REAL IMAGE") return "v-real";
    return "v-uncertain";
}

/* ── Data Cell ── */
function cell(key, val, pct, barCls) {
    const bar = typeof pct === "number" && isFinite(pct)
        ? `<div class="data-bar"><div class="data-bar-fill ${barCls || 'bar-cyan'}" style="width:${Math.max(0, Math.min(100, pct))}%"></div></div>`
        : "";
    return `<div class="data-cell">
        <span class="data-key">${key}</span>
        <span class="data-val">${val}</span>
        ${bar}
    </div>`;
}

/* ── Download ── */
function dlFrame(b64, name) {
    const a = document.createElement("a");
    a.href = "data:image/jpeg;base64," + b64;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
window.dlFrame = dlFrame;

/* ── Error ── */
function showError(msg) {
    results.innerHTML = `<div class="error-box">⚠ ${msg}</div>`;
}

/* ── Image Result ── */
function renderImageResult(data) {
    const a = data.analysis, fr = a.final_result, s = fr.component_scores;
    const v = vc(fr.classification);
    const pct = fr.final_score;
    fireConfetti(v === "v-real" ? "green" : "red");

    results.innerHTML = `
        <div class="verdict-card ${v}">
            <div class="verdict-info">
                <div class="verdict-tag">CLASSIFICATION RESULT</div>
                <div class="verdict-label">${fr.classification}</div>
                <div class="verdict-band">${fr.confidence_level} confidence</div>
            </div>
            ${gaugeHTML(pct, v)}
        </div>
        <div class="data-grid">
            ${cell("Model Prediction", a.model_prediction.class, s.model, s.model > 50 ? "bar-green" : "bar-red")}
            ${cell("Model Confidence", `${a.model_prediction.confidence.toFixed(1)}%`, a.model_prediction.confidence, "bar-cyan")}
            ${cell("Frequency", `${s.frequency}%`, s.frequency, "bar-cyan")}
            ${cell("Noise Pattern", `${s.noise}%`, s.noise, "bar-cyan")}
            ${cell("Pixel Stats", `${s.pixel}%`, s.pixel, "bar-cyan")}
            ${cell("Metadata", `${s.metadata}%`, s.metadata, "bar-cyan")}
            ${cell("ELA", `${(s.ela ?? 0)}%`, s.ela ?? 0, "bar-cyan")}
            ${cell("Resolution", `${data.image_info.width} × ${data.image_info.height}`, null)}
        </div>`;
}

/* ── Video Result ── */
function renderVideoResult(data) {
    const conf = Number(data.confidence || 0);
    const pred = data.prediction || "UNKNOWN";
    const v = vc(pred);
    fireConfetti(v === "v-real" ? "green" : "red");

    const topList = (data.top_frames || [])
        .map(f => `F${f.frame_index}:${Math.round(f.score * 100)}%`).join("  ") || "—";
    const segList = (data.suspicious_segments || [])
        .map(s => `${s.start_frame}–${s.end_frame}`).join(", ") || "None";

    // Timeline bars
    const timeline = Array.isArray(data.timeline) ? data.timeline : [];
    let timelineHTML = "";
    if (timeline.length > 0) {
        const bars = timeline.map((s, i) => {
            const pct = Math.round(s * 100);
            const h = Math.max(3, pct * 0.6);
            const hue = s > 0.6 ? "var(--red)" : s > 0.3 ? "var(--yellow)" : "var(--green)";
            return `<div class="t-bar" style="height:${h}px;background:${hue}" data-tip="F${i}: ${pct}%"></div>`;
        }).join("");
        timelineHTML = `
            <div class="timeline-section">
                <div class="timeline-title">⫸ FRAME-BY-FRAME ANALYSIS</div>
                <div class="timeline-bars">${bars}</div>
            </div>`;
    }

    // Evidence cards
    const imgs = data.top_frame_images || [];
    let evidenceHTML = "";
    if (imgs.length > 0) {
        evidenceHTML = `
            <div class="evidence-section">
                <div class="evidence-title">⫸ EVIDENCE — TOP SUSPICIOUS FRAMES</div>
                <div class="evidence-grid">
                    ${imgs.map(f => `
                        <div class="evidence-card">
                            <div class="evidence-badge">${Math.round(f.score * 100)}%</div>
                            <img src="data:image/jpeg;base64,${f.image_base64}" alt="Frame ${f.frame_index}" class="evidence-img">
                            <div class="evidence-meta">
                                <div class="evidence-name">FRAME ${f.frame_index}</div>
                                <div class="evidence-score">${Math.round(f.score * 100)}% anomaly detected</div>
                            </div>
                            <button class="evidence-dl" onclick="dlFrame('${f.image_base64}','evidence_frame_${f.frame_index}.jpg')">
                                ↓ DOWNLOAD EVIDENCE
                            </button>
                        </div>
                    `).join("")}
                </div>
            </div>`;
    }

    results.innerHTML = `
        <div class="verdict-card ${v}">
            <div class="verdict-info">
                <div class="verdict-tag">FORENSIC VERDICT</div>
                <div class="verdict-label">${pred}</div>
                <div class="verdict-band">Decision band: ${data.decision_band}</div>
            </div>
            ${gaugeHTML(conf * 100, v)}
        </div>
        <div class="data-grid">
            ${cell("Fake Probability", `${(conf * 100).toFixed(1)}%`, conf * 100, "bar-red")}
            ${cell("Real Probability", `${((data.prob_real || 0) * 100).toFixed(1)}%`, (data.prob_real || 0) * 100, "bar-green")}
            ${cell("Faces Scanned", data.num_detected_faces ?? 0, null)}
            ${cell("Coverage", `${Math.round((data.frame_coverage_ratio || 0) * 100)}%`, (data.frame_coverage_ratio || 0) * 100, "bar-cyan")}
            ${cell("Hot Frames", topList, null)}
            ${cell("Suspicious Zones", segList, null)}
            ${cell("Alerts", (data.warnings || []).join(", ") || "Clear", null)}
            ${cell("Samples", timeline.length, null)}
        </div>
        ${timelineHTML}
        ${evidenceHTML}`;
}

/* ── Submit ── */
async function submitFile(form, input, url, field, renderer) {
    const file = input.files[0];
    if (!file) { showError("SELECT A TARGET FILE"); return; }

    const btn = form.querySelector("button");
    btn.disabled = true;
    results.innerHTML = `
        <div class="scan-active">
            <div class="scan-ring"></div>
            <div class="scan-text">SCANNING FOR ANOMALIES...</div>
        </div>`;

    const fd = new FormData();
    fd.append(field, file);

    try {
        const res = await fetch(url, { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok || data.error) { showError(data.error || data.detail || "SCAN FAILED"); return; }
        renderer(data);
    } catch (e) { showError(`CONNECTION ERROR: ${e.message}`); }
    finally { btn.disabled = false; }
}

imageForm.addEventListener("submit", e => {
    e.preventDefault();
    submitFile(imageForm, imageInput, "/image-service/api/analyze", "image", renderImageResult);
});

videoForm.addEventListener("submit", e => {
    e.preventDefault();
    submitFile(videoForm, videoInput, "/api/video/analyze", "video", renderVideoResult);
});

refreshHealth();
