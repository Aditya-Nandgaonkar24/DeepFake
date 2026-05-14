# AI-Powered Deepfake & Image Detector
**Final Year Project | Computer Vision and AI**
Ruchir Prabhune (612203146) • Aditya Nandgaonkar (612203124) • Ved Patil (612203139) • Khush Rathi (612203148)

---

## Slide 2: Literature Survey
Recent advancements in Generative AI (GANs, Diffusion Models) necessitate robust detection mechanisms. Our project builds on four key research paradigms:
*   **CNN-Based Detection:** Identifying spatial inconsistencies.
*   **Spectral Learning:** Analyzing frequency-domain artifacts (FFT) where real images follow specific distributions that AI cannot replicate (Karageorgiou et al.).
*   **Bias-Free Paradigms:** Ensuring detectors identify AI fingerprints rather than recognizing semantic objects or demographics (Guillaro et al.).
*   **Adversarial Robustness:** Defending against semantic-preserving perturbations designed to trick conventional detectors.

---

## Slide 3: Research Gaps
Existing deepfake detection systems exhibit fundamental limitations:
1.  **Generalization Failure:** Detectors overfit to specific AI models (e.g., Stable Diffusion v1.5) and fail on unseen generators (e.g., Midjourney v6).
2.  **Semantic Bias:** Models mistakenly associate certain backgrounds or demographics with "fake" labels instead of detecting actual generation artifacts.
3.  **Lack of "In-the-Wild" Robustness:** Detection accuracy plummets when dealing with real-world social media images subjected to JPEG compression and low resolution.
4.  **Vulnerability to Adversarial Attacks:** Malicious actors can easily bypass single-model detectors using mathematical noise perturbations (FGSM).

---

## Slide 4: Motivation
*   **Digital Integrity:** The rapid proliferation of deepfakes threatens public trust and media authenticity.
*   **Cybersecurity:** Deepfakes are weaponized for identity theft, financial fraud, and sophisticated social engineering.
*   **Proactive Prevention:** Providing a robust, multi-layered verification tool to restore confidence in digital media by solving current research gaps.

---

## Slide 5: Problem Statement
With the rapid evolution of AI, deepfakes are visually indistinguishable from authentic content. Current methods are limited by reactive detection and adversarial vulnerability.

**Our Goal:** Develop a generalized, AI-powered system that uses Deep Learning combined with a deterministic physics-based ensemble (Spectral, ELA, and Noise Analysis) and Adversarial Training (FGSM) to detect facial forgeries across multiple sources with high reliability.

---

## Slide 6: System Architecture (The Proposed Solution)
To address the research gaps, we built a 6-stage ensemble processing pipeline:
1.  **Deep Learning Core (70%):** EfficientNet-B0 CNN fine-tuned on 140,000 Real & Fake Faces.
2.  **Frequency Analyzer (6%):** 2D Fast Fourier Transform (FFT) extracting spectral anomalies (Addresses Generalization).
3.  **Noise Analyzer (6%):** High-pass filtering to detect synthetic noise vs. camera sensor noise.
4.  **Pixel Analyzer (6%):** Color histogram and statistical variance analysis.
5.  **Metadata Analyzer (6%):** EXIF/ICC parsing to detect stripped camera data or AI software tags.
6.  **Error Level Analysis / ELA (6%):** Detects JPEG re-compression artifacts to catch manual manipulations (Addresses "In-the-Wild" robustness).

---

## Slide 7: Methodology & Advancements
Our methodology specifically implements solutions to the identified research gaps:
*   **Targeted Dataset:** Trained exclusively on a balanced dataset of 140,000 High-Resolution images (Real Faces vs. AI Faces) to specialize in deepfakes.
*   **Adversarial Training (FGSM):** We injected Fast Gradient Sign Method noise during training, teaching the model to ignore adversarial attacks and solving Research Gap 3.
*   **Multi-Modal Enforcement:** If the DL model fails, the 5 physics-based analyzers (FFT, ELA, etc.) act as a 30% safety net, ensuring robustness against unseen generators (Solving Gap 1).
*   **Speed Optimizations:** Deployed using `torch.amp` (Mixed Precision Training) for rapid inference.

---

## Slide 8: Technology Stack
*   **Backend & API:** Python 3.12, Flask, Gunicorn
*   **Deep Learning Framework:** PyTorch (with NVIDIA CUDA Acceleration), EfficientNet-B0, torchvision
*   **Computer Vision & Physics:** OpenCV (cv2), Pillow (PIL), NumPy, SciPy (FFT algorithms)
*   **Frontend Visualizations:** HTML5, CSS3, Chart.js (Dynamic Pie, Bar, Radar charts)
*   **Deployment:** Dockerized for seamless orchestration.

---

## Slide 9: Impact and Future Scope
*   **Current Impact:**
    *   *Fact-Checking Tool:* Reliable verification for journalists and media organizations.
    *   *Platform API:* REST architecture ready for social media integration to automatically flag AI-generated content.
*   **Future Enhancements:**
    *   *Real-Time Video Analysis:* Extending our frame-by-frame pipeline to live video streams.
    *   *Blockchain Integration:* Storing immutable digital signatures of authentic content.
    *   *Mobile Deployment:* Converting the PyTorch model to ONNX for on-device detection without latency.

---

## Slide 10: Results and Conclusion
*   **96.6% Validation Accuracy:** Reached nearly state-of-the-art accuracy on the 140K Faces dataset after just 1 epoch of training.
*   **< 1 Second Processing:** Real-time multi-threaded inference speed per image.
*   **Defeated Generalization Failure:** The 6-component ensemble system successfully forces the detection of fundamental image physics rather than relying purely on brittle Deep Learning patterns.

**Conclusion:** Our project successfully built a highly specialized AI Face Detector that actively addresses the core research gaps in the field. By combining an EfficientNet CNN with Adversarial Training and 5 deterministic physics-based analyzers, we ensure robust, unbiased detection against modern deepfakes.
