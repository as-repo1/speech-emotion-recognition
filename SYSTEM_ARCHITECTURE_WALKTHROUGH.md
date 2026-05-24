# System Architecture Walkthrough: VoxSense Codebase & Android App

## 1. System Overview

VoxSense is a modular, multi-platform ecosystem designed for Speech Emotion Recognition (SER). The repository contains two distinct applications:

1. **Web Application (Flask API & HTML5/JS Frontend)**: Containerized with Docker, hosting static file and real-time streaming audio analysis endpoints.
2. **Android Application (Kotlin & Jetpack Compose)**: A mobile application that runs raw audio recording, feature extraction, and TFLite model execution entirely offline.

```
                  ┌──────────────────────────────────────────────┐
                  │              VOXSENSE PROJECT                │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │    WEB APPLICATION    │                       │  ANDROID APPLICATION  │
     │      (server.py)      │                       │     (android-app/)    │
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │                                               │
        ┌────────┴────────┐                             ┌────────┴────────┐
        ▼                 ▼                             ▼                 ▼
   [Flask API]     [static/ HTML/JS]              [Jetpack Compose]   [TFLite Local]
```

---

## 2. Directory Structure & File Inventory

The root codebase is organized as follows:

```text
speech-emotion-recognition/
├── android-app/                   # Kotlin Jetpack Compose Android Project
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── assets/            # Embedded TFLite Model
│   │   │   ├── java/com/example/voxsense/
│   │   │   │   ├── data/          # Recording, signal processing, and classification
│   │   │   │   ├── theme/         # App custom typography and themes
│   │   │   │   ├── ui/main/       # ViewModels and Compose screens
│   │   │   │   ├── MainActivity.kt # Main application entry point
│   │   │   │   └── Navigation.kt  # Compose navigation configurations
│   │   │   └── AndroidManifest.xml # Permissions declarations
│   │   └── build.gradle.kts       # App module configuration & dependencies
│   ├── gradle/libs.versions.toml  # Centralized dependency catalog
│   ├── gradle.properties          # Bypasses AAR namespace conflicts
│   └── build.gradle.kts           # Root gradle configuration
├── legacy/                        # Legacy/Deprecated training scripts
├── models/                        # Serialized models and label encoders
├── notebooks/                     # Training notebooks for model development
├── static/                        # Web Frontend single page application
│   ├── app.js                     # Web recorder and dynamic canvas rendering
│   ├── index.html                 # UI layout dashboard
│   └── style.css                  # Custom styling and keyframe animations
├── server.py                      # Flask REST API server
├── Dockerfile                     # Multi-stage Docker packaging
├── docker-compose.yml             # Local container orchestration
└── requirements.txt               # Python package dependencies
```

---

## 3. Web Application Architecture

### 3.1. The Backend REST API (`server.py`)

The Flask server manages model loading and classification routing:

- **Initialization**: On startup, the server scans the `models/` directory. It loads the Keras CNN and LSTM models, Scikit-learn MLP model, and the Hugging Face `Wav2Vec2` pipeline (`harshit345/xlsr-wav2vec-speech-emotion-recognition`).
- **Endpoint `GET /api/health`**: Returns the health status and lists which models were successfully loaded on startup.
- **Endpoint `POST /api/predict`**: Accepts a static audio file. It handles file saving, uses `pydub` to convert non-WAV formats to standard WAV, downsamples the audio to 2000Hz to generate a 500-point normalized waveform array for UI display, extracts 40 mean MFCCs (for CNN/LSTM/MLP) or feeds raw audio (for Wav2Vec2), and outputs classification confidence values.
- **Endpoint `POST /api/predict_stream`**: Optimized for real-time streaming, accepting short 3-second segments. To maintain low latency, it skips waveform visualization downsampling and returns only the prediction values.

### 3.2. Web Frontend Single-Page App (`static/`)

- **Structure (`index.html`)**: A responsive, glassmorphic layout. The left column contains the upload area, a dynamic recording control button, and an audio player. The right column displays the primary emotion card, animated probability progress bars, a canvas waveform, and a scrolling timeline chart.
- **Style (`style.css`)**: Implements design variables for emotion colors (e.g., `--angry: #e74c3c`), background gradients, frosted-glass filters, and fade-in keyframe animations.
- **Behavior (`app.js`)**:
  - Handles drag-and-drop file operations.
  - Recording: Accesses the mic using `navigator.mediaDevices.getUserMedia`. For static recording, it captures the full audio. For real-time streaming, it triggers `mediaRecorder.start(1000)` to capture 1-second chunks, storing them in a 3-second rolling queue.
  - Animation: Uses requestAnimationFrame to draw frequency sweeps on the canvas and staggered width adjustments on the confidence bars.

---

## 4. Android Application Architecture

The Android app is written in **Kotlin** and built using **Jetpack Compose** and **Coroutines**, utilizing local components for recording, signal processing, and neural network inference.

```
       [MainActivity] (Requests Mic Permissions)
             │
             ▼
      [MainNavigation]
             │
             ▼
       [MainScreen] ◄──────────────────────────────┐
             │                                     │
    (Start/Stop Record)                     (Flows States)
             │                                     │
             ▼                                     │
  [MainScreenViewModel]                            │
             │                                     │
     (Orchestrates)                                │
             │                                     │
             ▼                                     │
     [DataRepository] ───► [EmotionPrediction] ────┘
       ├── AudioRecorder (AudioRecord 48kHz PCM)
       ├── MfccExtractor (Kotlin FFT, Mel Filter, DCT-II)
       └── EmotionClassifier (TFLite CNN Model)
```

### 4.1. Audio Capture Component (`AudioRecorder.kt`)

Using the standard Android `AudioRecord` API, this class records raw audio data directly from the microphone:

- **Sampling Frequency**: $48000\text{ Hz}$ mono, 16-bit PCM configuration.
- **Buffer Stream**: As audio is recorded on a background thread, the recorder writes samples into a `ByteArrayOutputStream` in little-endian format. It concurrently calculates peak amplitude values to update the UI visualizer:
  $$
  \text{amplitude} = \frac{\max(|x(n)|)}{32768.0}
  $$
- **Data Conversion**: On stop, the raw byte stream is converted to a normalized float array ($[-1.0, 1.0]$) by merging every two bytes back into a 16-bit short value.

### 4.2. Feature Extraction Component (`MfccExtractor.kt`)

An optimized, on-device digital signal processing (DSP) pipeline written in Kotlin:

- **Pre-calculation**: On instantiation, it precomputes the periodic Hanning window array and a 128-band Slaney-normalized triangular Mel filterbank matrix to minimize runtime memory allocations.
- **Fast Fourier Transform (FFT)**: Implements an in-place Cooley-Tukey Radix-2 DIT FFT. If the input frame size is $N = 2048$, the complexity is $O(N \log N)$, taking less than 0.1ms per frame on modern mobile CPUs.
- **Mel Filtering & dB Conversion**: Multiplies the power spectrum by the Mel filterbank matrix, applies log-10 scaling, and clips values to an 80 dB range below the peak energy.
- **Cepstral Representation**: Computes the orthogonal Discrete Cosine Transform (DCT-II) to extract 40 coefficients per frame, and averages them across all frames.

### 4.3. Inference Component (`EmotionClassifier.kt`)

Wraps the TensorFlow Lite interpreter:

- **Model Loading**: Maps the file `speech_emotion_recognition_cnn_model.tflite` directly from assets using a read-only `MappedByteBuffer` from a `FileChannel`.
- **TFLite Execution**: Wraps the 40 extracted MFCC features into a 3D float array with dimensions `[1, 40, 1]` to match the CNN input layer. The interpreter outputs a `[1, 8]` float array containing probabilities for the 8 emotion classes.
- **Label Alignment**: Matches the output probabilities to the alphabetically sorted labels: `['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']`.

### 4.4. State & UI Components (`MainScreenViewModel.kt` & `MainScreen.kt`)

- **State Management**: The UI state is modeled as a Kotlin sealed interface `MainScreenUiState` with a `Success` state that encapsulates:
  - `latestPrediction`: The current prediction result (primary emotion, confidence, breakdown).
  - `history`: A list of past predictions.
  - `isRecording`: Indicates if the microphone is active.
  - `isProcessing`: Indicates if the MFCC extractor or TFLite engine is running on a background thread.
- **Coroutines & Threads**: Long-running DSP operations (MFCC extraction) and TFLite runs are executed on `Dispatchers.Default` or `Dispatchers.IO` using Kotlin coroutines to prevent UI blocking or frame drops.
- **Interactive UI**:
  - Displays a title header with a dynamic color gradient.
  - Waveform visualizer: A custom Compose Canvas that draws intersecting sine waves. When recording is active, a coroutine animates the wave phase, and its vertical scaling is driven by real-time microphone amplitude updates.
  - Horizontal progress bars: Renders custom-colored progress indicators for all 8 emotions.
  - Dynamic permission launcher: Prompts the user for `Manifest.permission.RECORD_AUDIO` when they tap the record button if permission has not yet been granted.

---

## 5. End-to-End Mobile Data Flow

The following diagram illustrates how raw audio recorded from the microphone travels through our Android components to produce local emotion predictions in the UI:

```
[User Speaks]
      │
      ▼
┌──────────────┐
│ AudioRecord  │ ◄─── Captures raw PCM bytes at 48000 Hz
└──────┬───────┘
       │ [Raw Byte Stream]
       ▼
┌──────────────┐
│AudioRecorder │ ◄─── Converts PCM bytes to normalized float array [-1.0, 1.0]
└──────┬───────┘
       │ [FloatArray of samples]
       ▼
┌──────────────┐
│MfccExtractor │ ◄─── Framing, Hanning Window, Cooley-Tukey FFT, Mel filter, DCT-II
└──────┬───────┘
       │ [Mean MFCC Vector: FloatArray(40)]
       ▼
┌──────────────┐
│  Classifier  │ ◄─── Reshapes to [1, 40, 1] & runs TFLite interpreter
└──────┬───────┘
       │ [Probabilities: FloatArray(8)]
       ▼
┌──────────────┐
│  Repository  │ ◄─── Creates EmotionPrediction, updates history and latest flows
└──────┬───────┘
       │ [StateFlow updates]
       ▼
┌──────────────┐
│  MainScreen  │ ◄─── Renders primary emoji, animates progress bars, saves to history
└──────────────┘
```

This self-contained on-device pipeline ensures zero network overhead, protects user privacy by keeping recordings local, and delivers real-time paralinguistic classification.
