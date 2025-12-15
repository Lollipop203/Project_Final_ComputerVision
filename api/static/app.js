// Global state
let currentMode = 'image';
let webcamStream = null;
let webcamInterval = null;
let fpsCounter = { frames: 0, lastTime: Date.now() };
let currentImageFile = null;

// API Configuration
const API_BASE = window.location.origin;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupModeSwitch();
    setupImageUpload();
    setupVideoUpload();
    setupWebcam();
    setupSettings();
}

// Mode Switching
function setupModeSwitch() {
    const modeButtons = document.querySelectorAll('.mode-btn');
    const modeContents = document.querySelectorAll('.mode-content');

    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;

            // Update active states
            modeButtons.forEach(b => b.classList.remove('active'));
            modeContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${mode}-mode`).classList.add('active');

            currentMode = mode;

            // Stop webcam if switching away
            if (mode !== 'webcam' && webcamStream) {
                stopWebcam();
            }

            // Hide results when switching modes
            hideResults();
        });
    });
}

// Settings
function setupSettings() {
    const confSlider = document.getElementById('confidence');
    const iouSlider = document.getElementById('iou');
    const confValue = document.getElementById('conf-value');
    const iouValue = document.getElementById('iou-value');

    confSlider.addEventListener('input', (e) => {
        confValue.textContent = parseFloat(e.target.value).toFixed(2);
    });

    iouSlider.addEventListener('input', (e) => {
        iouValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
}

function getSettings() {
    return {
        conf: parseFloat(document.getElementById('confidence').value),
        iou: parseFloat(document.getElementById('iou').value)
    };
}

// Image Upload
function setupImageUpload() {
    const uploadArea = document.getElementById('image-upload-area');
    const fileInput = document.getElementById('image-input');
    const previewArea = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const detectBtn = document.getElementById('detect-image');
    const removeBtn = document.getElementById('remove-image');

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleImageFile(file);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file);
        }
    });

    // Detect button
    detectBtn.addEventListener('click', () => detectImage());

    // Remove button
    removeBtn.addEventListener('click', () => {
        uploadArea.style.display = 'flex';
        previewArea.style.display = 'none';
        detectBtn.style.display = 'none';
        detectBtn.style.display = 'none';
        fileInput.value = '';
        currentImageFile = null;
        hideResults();
    });

    // Paste from clipboard
    document.addEventListener('paste', (e) => {
        // Only handle paste when in image mode
        if (currentMode !== 'image') return;

        const items = e.clipboardData?.items;
        if (!items) return;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                if (blob) {
                    handleImageFile(blob);
                    showStatus('Đã dán ảnh từ clipboard!', 'success');
                }
                break;
            }
        }
    });
}

function handleImageFile(file) {
    currentImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = document.getElementById('preview-img');
        const uploadArea = document.getElementById('image-upload-area');
        const previewArea = document.getElementById('image-preview');
        const detectBtn = document.getElementById('detect-image');

        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        detectBtn.style.display = 'flex';
        hideResults();
    };
    reader.readAsDataURL(file);
}

async function detectImage() {
    if (!currentImageFile) return;

    const file = currentImageFile;

    const detectBtn = document.getElementById('detect-image');
    const btnText = detectBtn.querySelector('.btn-text');
    const btnLoader = detectBtn.querySelector('.btn-loader');

    try {
        // Show loading state
        detectBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        const settings = getSettings();
        const url = `${API_BASE}/detect?conf=${settings.conf}&iou=${settings.iou}&save_image=true`;

        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Detection failed');
        }

        const result = await response.json();
        displayResults(result, 'image');
        showStatus('Phát hiện thành công!', 'success');

    } catch (error) {
        console.error('Detection error:', error);
        showStatus('Lỗi: ' + error.message, 'error');
    } finally {
        // Reset button state
        detectBtn.disabled = false;
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
    }
}

// Video Upload
function setupVideoUpload() {
    const uploadArea = document.getElementById('video-upload-area');
    const fileInput = document.getElementById('video-input');
    const previewArea = document.getElementById('video-preview');
    const previewVideo = document.getElementById('preview-video');
    const detectBtn = document.getElementById('detect-video');
    const removeBtn = document.getElementById('remove-video');

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleVideoFile(file);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            handleVideoFile(file);
        }
    });

    // Detect button
    detectBtn.addEventListener('click', () => detectVideo());

    // Remove button
    removeBtn.addEventListener('click', () => {
        uploadArea.style.display = 'flex';
        previewArea.style.display = 'none';
        detectBtn.style.display = 'none';
        fileInput.value = '';
        previewVideo.src = '';
        hideResults();
    });
}

function handleVideoFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewVideo = document.getElementById('preview-video');
        const uploadArea = document.getElementById('video-upload-area');
        const previewArea = document.getElementById('video-preview');
        const detectBtn = document.getElementById('detect-video');

        previewVideo.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        detectBtn.style.display = 'flex';
        hideResults();
    };
    reader.readAsDataURL(file);
}

async function detectVideo() {
    const fileInput = document.getElementById('video-input');
    const file = fileInput.files[0];
    if (!file) return;

    const detectBtn = document.getElementById('detect-video');
    const btnText = detectBtn.querySelector('.btn-text');
    const btnLoader = detectBtn.querySelector('.btn-loader');
    const progressBar = document.getElementById('video-progress');
    const progressFill = progressBar.querySelector('.progress-fill');
    const progressText = progressBar.querySelector('.progress-text');

    try {
        // Show loading state
        detectBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
        progressBar.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Đang tải lên...';

        const formData = new FormData();
        formData.append('file', file);

        const settings = getSettings();
        const url = `${API_BASE}/detect-video?conf=${settings.conf}&iou=${settings.iou}`;

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
            progressText.textContent = `Đang xử lý... ${Math.round(progress)}%`;
        }, 500);

        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Hoàn thành!';

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Video processing failed');
        }

        const result = await response.json();
        displayVideoResults(result);
        showStatus('Xử lý video thành công!', 'success');

    } catch (error) {
        console.error('Video processing error:', error);
        showStatus('Lỗi: ' + error.message, 'error');
    } finally {
        // Reset button state
        detectBtn.disabled = false;
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';

        setTimeout(() => {
            progressBar.style.display = 'none';
        }, 2000);
    }
}

// Webcam
function setupWebcam() {
    const startBtn = document.getElementById('start-webcam');
    const stopBtn = document.getElementById('stop-webcam');

    startBtn.addEventListener('click', startWebcam);
    stopBtn.addEventListener('click', stopWebcam);
}

async function startWebcam() {
    try {
        const video = document.getElementById('webcam-video');
        const canvas = document.getElementById('webcam-canvas');
        const startBtn = document.getElementById('start-webcam');
        const stopBtn = document.getElementById('stop-webcam');
        const fpsDisplay = document.getElementById('fps-counter');

        // Request webcam access
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });

        video.srcObject = webcamStream;

        // Setup canvas
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;

        // Update UI
        startBtn.style.display = 'none';
        stopBtn.style.display = 'flex';
        fpsDisplay.style.display = 'block';

        // Start detection loop
        fpsCounter = { frames: 0, lastTime: Date.now() };
        webcamInterval = setInterval(() => detectWebcamFrame(), 100); // 10 FPS

        showStatus('Webcam đã bắt đầu', 'success');

    } catch (error) {
        console.error('Webcam error:', error);
        showStatus('Không thể truy cập webcam: ' + error.message, 'error');
    }
}

function stopWebcam() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const ctx = canvas.getContext('2d');
    const startBtn = document.getElementById('start-webcam');
    const stopBtn = document.getElementById('stop-webcam');
    const fpsDisplay = document.getElementById('fps-counter');

    // Stop stream
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    // Stop detection
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update UI
    startBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    fpsDisplay.style.display = 'none';

    showStatus('Webcam đã dừng', 'info');
}

async function detectWebcamFrame() {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const ctx = canvas.getContext('2d');

    if (!video.videoWidth) return;

    // Update canvas size if needed
    if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    try {
        // Capture frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);

        // Convert to blob
        const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));

        // Send to API
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const settings = getSettings();
        const url = `${API_BASE}/detect?conf=${settings.conf}&iou=${settings.iou}&save_image=false`;

        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            drawDetections(ctx, result, canvas.width, canvas.height);
            updateFPS();
        }

    } catch (error) {
        console.error('Frame detection error:', error);
    }
}

function drawDetections(ctx, result, width, height) {
    // Clear previous drawings
    ctx.clearRect(0, 0, width, height);

    if (!result.detections || result.detections.length === 0) return;

    // Draw each detection
    result.detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const w = x2 - x1;
        const h = y2 - y1;

        // Choose color based on class
        const colors = {
            'red': '#ef4444',
            'yellow': '#f59e0b',
            'green': '#10b981',
            'off': '#6b7280'
        };
        const color = colors[det.class.toLowerCase()] || '#6366f1';

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, w, h);

        // Draw label background
        const label = `${det.class} ${(det.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 16px Inter';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x1 + 5, y1 - 7);
    });
}

function updateFPS() {
    fpsCounter.frames++;
    const now = Date.now();
    const elapsed = now - fpsCounter.lastTime;

    if (elapsed >= 1000) {
        const fps = Math.round((fpsCounter.frames * 1000) / elapsed);
        document.querySelector('#fps-counter span').textContent = fps;
        fpsCounter.frames = 0;
        fpsCounter.lastTime = now;
    }
}

// Results Display
function displayResults(response, type) { // renamed result to response for clarity
    const resultsArea = document.getElementById('results-area');
    const resultImage = document.getElementById('result-image');
    const resultVideo = document.getElementById('result-video');
    const detectionCount = document.getElementById('detection-count');
    const detectionsList = document.getElementById('detections-list');

    // Show results area
    resultsArea.style.display = 'block';

    // Extract results object if nested (new API structure)
    // New API: { success: true, results: { count, detections, output_image } }
    // Old API (legacy support): { num_detections, detections, image_url }
    const resultData = response.results || response;

    // Display image
    // New API uses output_image, Old uses image_url
    const imgUrl = resultData.output_image || response.image_url;

    if (type === 'image' && imgUrl) {
        // Handle if URL is already absolute or relative
        resultImage.src = imgUrl.startsWith('http') ? imgUrl : (API_BASE + imgUrl);
        resultImage.style.display = 'block';
        resultVideo.style.display = 'none';

        // Add error handler for broken images
        resultImage.onerror = function () {
            this.style.display = 'none';
            showStatus('Lỗi hiển thị ảnh kết quả', 'error');
        };
    }

    // Update detection count
    // New API uses count, Old uses num_detections
    detectionCount.textContent = resultData.count !== undefined ? resultData.count : (resultData.num_detections || 0);

    // Display detections list
    detectionsList.innerHTML = '';
    const detections = resultData.detections || [];

    if (detections.length > 0) {
        detections.forEach((det, index) => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            item.innerHTML = `
                <div class="detection-class">#${index + 1} - ${det.class}</div>
                <div class="detection-conf">Độ tin cậy: <span>${(det.confidence * 100).toFixed(2)}%</span></div>
            `;
            detectionsList.appendChild(item);
        });
    } else {
        detectionsList.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 1rem;">Không phát hiện đèn giao thông</p>';
    }

    // Scroll to results
    resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayVideoResults(result) {
    const resultsArea = document.getElementById('results-area');
    const resultImage = document.getElementById('result-image');
    const resultVideo = document.getElementById('result-video');
    const detectionCount = document.getElementById('detection-count');
    const detectionsList = document.getElementById('detections-list');

    // Show results area
    resultsArea.style.display = 'block';

    // Display video
    if (result.video_url) {
        resultVideo.src = API_BASE + result.video_url;
        resultVideo.style.display = 'block';
        resultImage.style.display = 'none';
    }

    // Update stats
    detectionCount.textContent = result.summary.frames_with_detections;

    // Display summary
    detectionsList.innerHTML = `
        <div class="detection-item">
            <div class="detection-class">Tổng số khung hình</div>
            <div class="detection-conf"><span>${result.total_frames}</span></div>
        </div>
        <div class="detection-item">
            <div class="detection-class">Khung hình có phát hiện</div>
            <div class="detection-conf"><span>${result.summary.frames_with_detections}</span></div>
        </div>
    `;

    // Scroll to results
    resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    const resultsArea = document.getElementById('results-area');
    resultsArea.style.display = 'none';
}

// Status Messages
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('status-message');
    statusEl.textContent = message;
    statusEl.className = `status-message ${type} show`;

    setTimeout(() => {
        statusEl.classList.remove('show');
    }, 3000);
}
