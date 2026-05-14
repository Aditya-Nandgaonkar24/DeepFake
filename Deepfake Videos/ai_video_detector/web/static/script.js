document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const laser = document.getElementById('laser');
    const loaderContainer = document.getElementById('loader-container');
    const statusText = document.getElementById('status-text');
    const resultsPanel = document.getElementById('results');
    const dropText = document.getElementById('drop-text');
    const progressRing = document.getElementById('progress-ring');
    const ringValue = document.getElementById('ring-value');
    const ringLabel = document.getElementById('ring-label');
    const thresholdNote = document.getElementById('threshold-note');
    const decisionBandNote = document.getElementById('decision-band');
    const coverageNote = document.getElementById('coverage-note');
    const timelineBars = document.getElementById('timeline-bars');
    const topFrames = document.getElementById('top-frames');
    const segments = document.getElementById('segments');
    const warnings = document.getElementById('warnings');
    const downloadBtn = document.getElementById('download-btn');
    let lastAnalysis = null;

    const CIRCUMFERENCE = 408;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
        dropzone.addEventListener(e, preventDefaults, false);
    });

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(e => {
        dropzone.addEventListener(e, () => dropzone.classList.add('active'), false);
    });

    ['dragleave', 'drop'].forEach(e => {
        dropzone.addEventListener(e, () => dropzone.classList.remove('active'), false);
    });

    dropzone.addEventListener('drop', e => handleFiles(e.dataTransfer.files), false);
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => { if (e.target.files.length) handleFiles(e.target.files); });

    function handleFiles(files) {
        if (!files[0].type.startsWith('video/')) return alert('Target must be standard video encode.');
        dropText.innerHTML = `Scanning Vector: <strong style="color:var(--neon-blue)">${files[0].name}</strong>`;
        uploadEngine(files[0]);
    }

    async function uploadEngine(file) {
        resultsPanel.style.display = 'none';
        loaderContainer.style.display = 'block';
        laser.classList.add('scanning');
        timelineBars.innerHTML = '';
        topFrames.innerText = 'Waiting for analysis...';
        segments.innerText = 'Waiting for analysis...';
        warnings.innerText = 'Waiting for analysis...';
        progressRing.style.strokeDashoffset = CIRCUMFERENCE;

        const texts = [
            "Isolating Neural Signatures...",
            "Mapping Temporal Anomalies...",
            "Validating Optical Flow...",
            "Computing Forgery Matrix..."
        ];
        let tIdx = 0;
        const textInterval = setInterval(() => {
            tIdx = (tIdx + 1) % texts.length;
            statusText.innerText = texts[tIdx];
        }, 1500);

        const formData = new FormData();
        formData.append('video', file);

        try {
            const res = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await res.json();

            clearInterval(textInterval);
            loaderContainer.style.display = 'none';
            laser.classList.remove('scanning');

            if (data.error) return alert(`API Rejection: ${data.error}`);
            if (data.detail) return alert(`API Rejection: ${data.detail}`);
            triggerUIState(data);
        } catch (err) {
            clearInterval(textInterval);
            console.error(err);
            alert("FastAPI core offline. Spin up backend via 'uvicorn app:app'");
            loaderContainer.style.display = 'none';
            laser.classList.remove('scanning');
        }
    }

    function triggerUIState(data) {
        resultsPanel.style.display = 'block';

        const isFake = data.prediction === "FAKE";
        const isUncertain = data.prediction === "UNCERTAIN";
        const percent = Math.round(data.confidence * 100);
        const threshold = typeof data.threshold === 'number' ? data.threshold : 0.33;

        ringValue.innerText = `${percent}%`;
        ringLabel.innerText = isUncertain ? "REVIEW NEEDED" : (isFake ? "FORGERY CLASSIFIED" : "SOURCE AUTHENTIC");
        thresholdNote.innerText = `Decision threshold: ${threshold.toFixed(2)}`;
        decisionBandNote.innerText = `Decision band: ${String(data.decision_band || 'unknown').replaceAll('_', ' ')}`;
        coverageNote.innerText = `Frame coverage: ${Math.round((data.frame_coverage_ratio || 0) * 100)}% (${data.num_detected_faces || 0} faces)`;

        const targetColor = isUncertain ? "var(--neon-amber)" : (isFake ? "var(--neon-red)" : "var(--neon-green)");
        ringLabel.style.color = targetColor;
        progressRing.style.stroke = targetColor;

        setTimeout(() => {
            progressRing.style.strokeDashoffset = CIRCUMFERENCE - (data.confidence * CIRCUMFERENCE);
        }, 100);

        data.timeline.forEach((intensity, idx) => {
            const h = Math.max(8, intensity * 100);
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.height = `${h}%`;
            bar.style.animationDelay = `${idx * 0.05}s`;

            if (intensity > 0.65) bar.style.background = "var(--neon-red)";
            else if (intensity > 0.4) bar.style.background = "var(--neon-amber)";
            else bar.style.background = "var(--neon-green)";

            bar.style.color = bar.style.background;
            timelineBars.appendChild(bar);
        });

        if (Array.isArray(data.top_frames) && data.top_frames.length) {
            topFrames.innerHTML = data.top_frames
                .map(frame => `Frame ${frame.frame_index + 1}: ${Math.round(frame.score * 100)}%`)
                .join('<br>');
        } else {
            topFrames.innerText = 'No standout frames.';
        }

        if (Array.isArray(data.suspicious_segments) && data.suspicious_segments.length) {
            segments.innerHTML = data.suspicious_segments
                .map(seg => `Frames ${seg.start_frame + 1}-${seg.end_frame + 1} | peak ${Math.round(seg.peak_score * 100)}%`)
                .join('<br>');
        } else {
            segments.innerText = 'No strong suspicious span detected.';
        }

        if (Array.isArray(data.warnings) && data.warnings.length) {
            warnings.innerHTML = data.warnings.map(item => item.replaceAll('_', ' ')).join('<br>');
        } else {
            warnings.innerText = 'No warnings.';
        }

        lastAnalysis = {
            prediction: data.prediction,
            decision_band: data.decision_band,
            confidence: data.confidence,
            prob_real: data.prob_real,
            prob_fake: data.prob_fake,
            threshold: threshold,
            uncertainty_margin: data.uncertainty_margin,
            frame_coverage_ratio: data.frame_coverage_ratio,
            num_detected_faces: data.num_detected_faces,
            timeline: data.timeline,
            top_frames: data.top_frames,
            suspicious_segments: data.suspicious_segments,
            warnings: data.warnings,
            timestamp: new Date().toISOString()
        };
        downloadBtn.style.display = 'block';
    }

    downloadBtn.addEventListener('click', () => {
        if (!lastAnalysis) return;
        const blob = new Blob([JSON.stringify(lastAnalysis, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `deepfake_report_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    });
});
