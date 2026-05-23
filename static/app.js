// Core variables and state
let selectedModel = 'wav2vec2'; // Default to state-of-the-art Wav2Vec2 model
let currentAudioFile = null;
let mediaRecorder = null;
let audioChunks = [];
let liveChunks = []; // Sliding window of real-time audio chunks
let isStreamingMode = false;
let recordingTimer = null;
let recordingStartTime = 0;
let historyData = [];

// Emotion styling & meta map
const emotionMap = {
    neutral: { emoji: '😐', colorClass: 'result-neutral', colorHex: '#8e9bb0' },
    calm: { emoji: '😌', colorClass: 'result-calm', colorHex: '#00d2d3' },
    happy: { emoji: '😄', colorClass: 'result-happy', colorHex: '#fbc531' },
    sad: { emoji: '😢', colorClass: 'result-sad', colorHex: '#3867d6' },
    angry: { emoji: '😡', colorClass: 'result-angry', colorHex: '#ff3f34' },
    fearful: { emoji: '😨', colorClass: 'result-fearful', colorHex: '#9c88ff' },
    disgust: { emoji: '🤢', colorClass: 'result-disgust', colorHex: '#26de81' },
    surprised: { emoji: '😲', colorClass: 'result-surprised', colorHex: '#fd79a8' }
};

// Initialize UI
document.addEventListener('DOMContentLoaded', () => {
    initModelSelector();
    initUploadZone();
    initRecordingZone();
    initHistory();
    checkBackendHealth();
});

// Show floating notification
function showToast(message, type = 'error') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = type === 'success' ? 'fa-circle-check' : 'fa-circle-exclamation';
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
    
    container.appendChild(toast);
    
    // Auto dismiss
    setTimeout(() => {
        toast.style.transform = 'translateX(120%)';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Check Backend Connection & API capabilities
async function checkBackendHealth() {
    try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('API Health Check failed');
        const data = await response.json();
        console.log('Backend connected:', data);
        
        // Disable model buttons if model weights didn't load on backend
        const loadedModels = data.models_loaded || [];
        ['wav2vec2', 'lstm', 'cnn', 'mlp'].forEach(m => {
            const btn = document.getElementById(`btn-model-${m}`);
            if (btn && !loadedModels.includes(m)) {
                btn.style.opacity = '0.4';
                btn.style.pointerEvents = 'none';
                btn.title = 'Model weights not found on server';
            }
        });
    } catch (err) {
        showToast('Unable to connect to backend engine. Run "python server.py" first.', 'error');
    }
}

// 1. Model Selector pills
function initModelSelector() {
    const selector = document.getElementById('model-selector');
    if (!selector) return;
    selector.addEventListener('click', (e) => {
        const button = e.target.closest('.model-pill');
        if (!button) return;
        
        // Remove active class from all pills
        selector.querySelectorAll('.model-pill').forEach(btn => btn.classList.remove('active'));
        
        // Add active to current pill
        button.classList.add('active');
        selectedModel = button.dataset.model;
    });
}

// Tab Switching (Upload / Record)
function switchTab(tabId) {
    document.querySelectorAll('.tab-header').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabId}-header`).classList.add('active');
    
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`tab-${tabId}`).classList.add('active');
    
    if (!currentAudioFile) {
        resetAudioPreview();
    }
}

// 2. Upload Area Event Handlers
function initUploadZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    if (!dropZone || !fileInput) return;
    
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });
}

function handleSelectedFile(file) {
    const validFormats = ['.wav', '.mp3', '.ogg', '.flac'];
    const filename = file.name.toLowerCase();
    const isValid = validFormats.some(ext => filename.endsWith(ext));
    
    if (!isValid) {
        showToast('Invalid file format. Please upload WAV, MP3, OGG, or FLAC.', 'error');
        return;
    }
    
    currentAudioFile = file;
    setupAudioPreview(file);
}

function setupAudioPreview(fileOrBlob, filename = null) {
    const container = document.getElementById('preview-container');
    const filenameEl = document.getElementById('preview-filename');
    const player = document.getElementById('audio-player');
    if (!container || !filenameEl || !player) return;
    
    filenameEl.textContent = filename || fileOrBlob.name || 'recorded_speech.wav';
    
    const audioURL = URL.createObjectURL(fileOrBlob);
    player.src = audioURL;
    player.load();
    
    container.classList.remove('hidden');
    
    document.getElementById('btn-clear-audio').onclick = resetAudioPreview;
    document.getElementById('btn-analyze').onclick = startAnalysis;
}

function resetAudioPreview() {
    const container = document.getElementById('preview-container');
    const player = document.getElementById('audio-player');
    if (!container || !player) return;
    
    player.pause();
    player.src = '';
    currentAudioFile = null;
    container.classList.add('hidden');
    
    showResultView('idle');
}

// 3. Microphone Recording & Streaming Module
function initRecordingZone() {
    const recordBtn = document.getElementById('btn-record-toggle');
    if (!recordBtn) return;
    recordBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });
}

async function startRecording() {
    audioChunks = [];
    liveChunks = [];
    const micCircle = document.getElementById('mic-status-circle');
    const statusText = document.getElementById('record-status-text');
    const recordBtn = document.getElementById('btn-record-toggle');
    const realtimeStreamCheckbox = document.getElementById('chk-realtime-stream');
    
    isStreamingMode = realtimeStreamCheckbox ? realtimeStreamCheckbox.checked : false;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Pick optimal container format
        let options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options = { mimeType: 'audio/ogg' };
        }
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options = { mimeType: '' };
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        
        // Real-time chart configuration
        if (isStreamingMode) {
            document.getElementById('timeline-chart').innerHTML = '';
            document.getElementById('timeline-box').classList.remove('hidden');
            document.getElementById('waveform-box').classList.add('hidden');
            showResultView('loading');
            document.getElementById('loading-status-text').textContent = 'Tracking voice emotions in real-time...';
        } else {
            document.getElementById('timeline-box').classList.add('hidden');
            document.getElementById('waveform-box').classList.remove('hidden');
        }
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                
                if (isStreamingMode) {
                    liveChunks.push(event.data);
                    // Keep sliding window of last 3 seconds (3 ticks of 1000ms timeslices)
                    if (liveChunks.length > 3) {
                        liveChunks.shift();
                    }
                    const streamBlob = new Blob(liveChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                    sendStreamChunk(streamBlob);
                }
            }
        };
        
        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(track => track.stop());
            
            // Build final complete file
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/wav' });
            const file = new File([audioBlob], 'live_recording.wav', { type: audioBlob.type });
            currentAudioFile = file;
            setupAudioPreview(file, isStreamingMode ? 'Live Tracker Recording' : 'Microphone Recording');
            
            // Reset state
            micCircle.classList.remove('recording');
            statusText.textContent = isStreamingMode ? 'Live Session Finished' : 'Recording Saved';
            recordBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Recording';
            liveChunks = [];
            isStreamingMode = false;
        };
        
        // Start recorder. If streaming, slice data every 1000ms
        if (isStreamingMode) {
            mediaRecorder.start(1000);
        } else {
            mediaRecorder.start();
        }
        
        recordingStartTime = Date.now();
        updateTimer();
        recordingTimer = setInterval(updateTimer, 1000);
        
        micCircle.classList.add('recording');
        statusText.textContent = isStreamingMode ? 'Streaming Live...' : 'Recording...';
        recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Session';
        
    } catch (err) {
        console.error('Error starting mic:', err);
        showToast('Microphone access denied or unavailable.', 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        clearInterval(recordingTimer);
    }
}

function updateTimer() {
    const elapsedMs = Date.now() - recordingStartTime;
    const totalSecs = Math.floor(elapsedMs / 1000);
    const mins = String(Math.floor(totalSecs / 60)).padStart(2, '0');
    const secs = String(totalSecs % 60).padStart(2, '0');
    const timerEl = document.getElementById('record-timer');
    if (timerEl) timerEl.textContent = `${mins}:${secs}`;
}

// 4. API Requests
async function sendStreamChunk(blob) {
    if (!mediaRecorder || mediaRecorder.state !== 'recording' || !isStreamingMode) return;
    
    const formData = new FormData();
    const file = new File([blob], 'stream_chunk.wav', { type: blob.type });
    formData.append('file', file);
    formData.append('model', selectedModel);
    
    try {
        const response = await fetch('/api/predict_stream', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        // Verify state hasn't changed during fetch
        if (result.success && isStreamingMode && mediaRecorder.state === 'recording') {
            displayResults(result);
            addTimelineTick(result.emotion, result.confidence);
        }
    } catch (err) {
        console.error('Error sending stream chunk:', err);
    }
}

async function startAnalysis() {
    if (!currentAudioFile) {
        showToast('No audio loaded for analysis.', 'error');
        return;
    }
    
    showResultView('loading');
    document.getElementById('loading-status-text').textContent = 'Running inference on selected neural architecture';
    
    const formData = new FormData();
    formData.append('file', currentAudioFile);
    formData.append('model', selectedModel);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Server error occurred');
        }
        
        displayResults(result);
        saveToHistory(result, currentAudioFile.name);
        showToast('Analysis completed successfully!', 'success');
        
    } catch (err) {
        console.error('Prediction error:', err);
        showToast(err.message || 'Analysis failed. Check backend console.', 'error');
        showResultView('idle');
    }
}

function showResultView(state) {
    const resultCard = document.getElementById('result-card');
    const idleView = document.getElementById('result-idle-view');
    const loadingView = document.getElementById('result-loading-view');
    const contentView = document.getElementById('result-content-view');
    if (!resultCard || !idleView || !loadingView || !contentView) return;
    
    resultCard.className = 'glass-card card-result';
    
    if (state === 'idle') {
        resultCard.classList.add('empty');
        idleView.classList.remove('hidden');
        loadingView.classList.add('hidden');
        contentView.classList.add('hidden');
    } else if (state === 'loading') {
        resultCard.classList.add('empty');
        idleView.classList.add('hidden');
        loadingView.classList.remove('hidden');
        contentView.classList.add('hidden');
    } else {
        idleView.classList.add('hidden');
        loadingView.classList.add('hidden');
        contentView.classList.remove('hidden');
    }
}

function displayResults(data) {
    showResultView('content');
    
    const resultCard = document.getElementById('result-card');
    const primaryEmoji = document.getElementById('primary-emoji');
    const primaryEmotion = document.getElementById('primary-emotion');
    const primaryConfidence = document.getElementById('primary-confidence');
    const metaModel = document.getElementById('meta-model-used');
    const metaTime = document.getElementById('meta-time');
    if (!resultCard || !primaryEmoji || !primaryEmotion || !primaryConfidence) return;
    
    const emotion = data.emotion.toLowerCase();
    const styleMeta = emotionMap[emotion] || emotionMap['neutral'];
    
    // 1. Styling
    resultCard.classList.add(styleMeta.colorClass);
    primaryEmoji.textContent = styleMeta.emoji;
    primaryEmotion.textContent = data.emotion;
    primaryEmotion.style.color = styleMeta.colorHex;
    primaryConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% Confidence`;
    
    // 2. Metadata
    if (metaModel) metaModel.textContent = data.model_used.toUpperCase();
    if (metaTime) metaTime.textContent = new Date().toLocaleTimeString();
    
    // 3. Audio signature waveform (if present in payload)
    if (data.waveform) {
        drawWaveform(data.waveform, styleMeta.colorHex);
    }
    
    // 4. Fill Probability distribution chart
    const barsContainer = document.getElementById('emotion-bars-list');
    if (!barsContainer) return;
    barsContainer.innerHTML = '';
    
    const sortedEmotions = Object.entries(data.all_emotions)
        .sort((a, b) => b[1] - a[1]);
        
    sortedEmotions.forEach(([em, score]) => {
        const row = document.createElement('div');
        row.className = 'bar-row';
        
        const emColor = emotionMap[em.toLowerCase()]?.colorHex || '#8e9bb0';
        const percent = (score * 100).toFixed(1);
        
        row.innerHTML = `
            <div class="bar-labels">
                <span class="bar-name">${em}</span>
                <span class="bar-value">${percent}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="background-color: ${emColor}; width: 0%"></div>
            </div>
        `;
        
        barsContainer.appendChild(row);
        
        requestAnimationFrame(() => {
            setTimeout(() => {
                const fill = row.querySelector('.bar-fill');
                if (fill) fill.style.width = `${percent}%`;
            }, 50);
        });
    });
}

// Canvas Waveform Drawing
function drawWaveform(waveData, colorHex) {
    const canvas = document.getElementById('waveform-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    
    ctx.clearRect(0, 0, w, h);
    
    // Background grids
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for(let i = 0; i < w; i += 20) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, h);
        ctx.stroke();
    }
    
    // Wave paths
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    
    const numPoints = waveData.length;
    const step = w / numPoints;
    const midY = h / 2;
    
    ctx.strokeStyle = colorHex;
    
    ctx.beginPath();
    for (let i = 0; i < numPoints; i++) {
        const x = i * step;
        const amplitude = waveData[i] * (h / 2.2);
        
        ctx.moveTo(x, midY - amplitude);
        ctx.lineTo(x, midY + amplitude);
    }
    ctx.stroke();
}

// Real-time timeline ticks addition
function addTimelineTick(emotion, confidence) {
    const chart = document.getElementById('timeline-chart');
    if (!chart) return;
    
    const tick = document.createElement('div');
    tick.className = 'timeline-tick';
    
    const style = emotionMap[emotion.toLowerCase()] || emotionMap['neutral'];
    tick.style.backgroundColor = style.colorHex;
    
    // Set tick height representing confidence score (min 20% height for visual balance)
    const heightPercent = Math.max(20, Math.round(confidence * 100));
    tick.style.height = `${heightPercent}%`;
    
    const timeStr = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    tick.setAttribute('data-tooltip', `${emotion} (${(confidence * 100).toFixed(0)}%) at ${timeStr}`);
    
    chart.appendChild(tick);
    
    // Maintain maximum of 35 ticks in the horizontal scroll window
    if (chart.children.length > 35) {
        chart.removeChild(chart.firstChild);
    }
    
    // Auto-scroll timeline to the right
    chart.scrollLeft = chart.scrollWidth;
}

// 5. Session History Functions
function initHistory() {
    const saved = localStorage.getItem('voxsense_history');
    if (saved) {
        try {
            historyData = JSON.parse(saved);
            updateHistoryUI();
        } catch (e) {
            console.error('Error parsing session history:', e);
        }
    }
    
    const clearBtn = document.getElementById('btn-clear-history');
    if (clearBtn) {
        clearBtn.onclick = () => {
            historyData = [];
            localStorage.removeItem('voxsense_history');
            updateHistoryUI();
            showToast('History database cleared', 'success');
        };
    }
}

function saveToHistory(data, filename) {
    const entry = {
        emotion: data.emotion,
        confidence: data.confidence,
        model: data.model_used,
        file: filename,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        waveform: data.waveform,
        all_emotions: data.all_emotions
    };
    
    historyData.unshift(entry);
    if (historyData.length > 10) {
        historyData.pop();
    }
    
    localStorage.setItem('voxsense_history', JSON.stringify(historyData));
    updateHistoryUI();
}

function updateHistoryUI() {
    const list = document.getElementById('history-list');
    const emptyText = document.getElementById('history-empty-text');
    if (!list) return;
    
    list.innerHTML = '';
    
    if (historyData.length === 0) {
        if (emptyText) emptyText.classList.remove('hidden');
        list.classList.add('hidden');
        return;
    }
    
    if (emptyText) emptyText.classList.add('hidden');
    list.classList.remove('hidden');
    
    historyData.forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'history-item';
        
        const emLower = item.emotion.toLowerCase();
        const style = emotionMap[emLower] || emotionMap['neutral'];
        
        div.innerHTML = `
            <div class="history-emoji">${style.emoji}</div>
            <div class="history-meta">
                <span class="history-emotion" style="color: ${style.colorHex}">${item.emotion}</span>
                <span class="history-conf">${(item.confidence * 100).toFixed(0)}% (${item.model.toUpperCase()})</span>
                <span class="history-file" title="${item.file}">${item.file}</span>
            </div>
        `;
        
        div.onclick = () => {
            // Restore timeline elements state
            document.getElementById('timeline-box').classList.add('hidden');
            document.getElementById('waveform-box').classList.remove('hidden');
            displayResults(item);
        };
        
        list.appendChild(div);
    });
}
