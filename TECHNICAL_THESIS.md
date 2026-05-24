# Technical Thesis: Foundations of Speech Emotion Recognition (SER) in VoxSense

## Abstract
Speech Emotion Recognition (SER) is a key challenge in Human-Computer Interaction (HCI), requiring the extraction of nuanced psychological states from acoustic waveforms. This paper details the mathematical, signal processing, and machine learning foundations of the VoxSense SER Suite. We explore two distinct engineering paradigms: a high-capacity, server-side pipeline utilizing a fine-tuned Wav2Vec 2.0 transformer model, and a highly optimized, resource-constrained edge pipeline deployed on Android using local Mel-Frequency Cepstral Coefficients (MFCC) extraction and TensorFlow Lite (TFLite) 1D Convolutional Neural Network (CNN) inference. We present the complete mathematical formulation of the feature extraction pipeline, model architectures, and comparative trade-offs in accuracy, memory footprint, and inference latency.

---

## 1. Introduction
Human speech contains two parallel streams of information: *linguistic content* (what is said) and *paralinguistic content* (how it is said). While speech-to-text models have matured significantly, decoding paralinguistic cues—specifically emotion—remains difficult due to high variability in speakers, languages, background noise, and recording equipment.

VoxSense address this problem by providing a dual-pipeline architecture:
1. **Cloud/Server Pipeline**: Optimized for accuracy, deploying a state-of-the-art transformer model (Wav2Vec2) that captures contextual and semantic representations directly from raw waveforms.
2. **Local Edge Pipeline**: Optimized for privacy, latency, and offline capability, executing local audio recording, signal windowing, Fast Fourier Transforms (FFT), Mel filterbanks, Discrete Cosine Transforms (DCT), and convolutional neural network inference entirely on-device on Android.

---

## 2. Acoustic Feature Extraction: Mathematical Formulation
Traditional neural networks (CNN, LSTM, MLP) cannot process high-rate raw audio signals directly without a massive parameter cost. Thus, we compress raw waveforms into a compact spectral representation: Mel-Frequency Cepstral Coefficients (MFCCs).

```
   Raw Audio [s(n)] 
         │
         ▼
     Framing (N = 2048, H = 512)
         │
         ▼
   Windowing (Hanning Window)
         │
         ▼
     FFT (2048-point DFT)
         │
         ▼
   Power Spectrum [|X(k)|^2]
         │
         ▼
   Mel Filterbank (128 Slaney Bins)
         │
         ▼
  Logarithmic Compression (dB)
         │
         ▼
   DCT-II (Ortho Normalization)
         │
         ▼
  Temporal Mean Pooling (40 MFCCs)
```

The mathematical pipeline is defined as follows:

### 2.1. Framing and Windowing
A speech signal $s(n)$ is non-stationary, but can be assumed to be quasi-stationary over short intervals ($10\text{ ms}$ to $50\text{ ms}$). We segment the signal into overlapping frames $x_t(m)$ of length $N = 2048$ samples, with a hop size of $H = 512$ samples. At a sample rate of $f_s = 48000\text{ Hz}$, this corresponds to a frame duration of $42.6\text{ ms}$ and a frame rate of $10.7\text{ ms}$.

To prevent spectral leakage caused by boundary discontinuities of rectangular framing, we apply a periodic **Hanning window** $w(n)$ to each frame:
$$w(n) = 0.5 \cdot \left( 1 - \cos\left(\frac{2\pi n}{N}\right) \right), \quad 0 \le n < N$$

The windowed frame $y_t(n)$ is:
$$y_t(n) = x_t(n) \cdot w(n)$$

### 2.2. Discrete Fourier Transform (DFT)
We convert each windowed time-domain frame $y_t(n)$ into the frequency domain using an $N$-point Discrete Fourier Transform (DFT):
$$X_t(k) = \sum_{n=0}^{N-1} y_t(n) \cdot e^{-j \frac{2\pi k n}{N}}, \quad 0 \le k < N$$

The power spectrum $P_t(k)$ represents the energy distribution across discrete frequency bins:
$$P_t(k) = \frac{1}{N} |X_t(k)|^2$$
Due to conjugate symmetry for real inputs, we discard the negative frequencies, retaining the first $N/2 + 1 = 1025$ bins.

### 2.3. Mel Filterbank Integration
The human ear does not perceive pitch linearly. It is highly sensitive to frequency variations below $1000\text{ Hz}$ and logarithmically sensitive above. To model this, we warp the linear frequency scale (Hertz) to the **Mel Scale** using Slaney's formulation. 

Slaney's Mel conversion is defined as:
- **Linear Region ($f < 1000\text{ Hz}$)**:
  $$m = \frac{f}{f_{\text{sp}}}$$
  where $f_{\text{sp}} = \frac{200}{3} \approx 66.67\text{ Hz/Mel}$.
- **Logarithmic Region ($f \ge 1000\text{ Hz}$)**:
  $$m = 15.0 + \frac{\ln(f / 1000.0)}{\ln(6.4)/27.0}$$

The inverse mapping ($\text{Mel} \to \text{Hz}$) is:
- **Linear Region ($m < 15.0$)**:
  $$f = f_{\text{sp}} \cdot m$$
- **Logarithmic Region ($m \ge 15.0$)**:
  $$f = 1000.0 \cdot e^{\frac{\ln(6.4)}{27.0} \cdot (m - 15.0)}$$

We construct $M = 128$ triangular filter channels. The filterbank weights $H_m(k)$ for the $m$-th Mel band are defined over the discrete FFT bin frequencies $f(k) = k \cdot \frac{f_s}{N}$:
$$H_m(k) = \begin{cases} 
0 & f(k) < f_m \\
\frac{f(k) - f_m}{f_{m+1} - f_m} & f_m \le f(k) < f_{m+1} \\
\frac{f_{m+2} - f(k)}{f_{m+2} - f_{m+1}} & f_{m+1} \le f(k) < f_{m+2} \\
0 & f(k) \ge f_{m+2}
\end{cases}$$
where $f_m, f_{m+1}, f_{m+2}$ are the physical center frequencies of the Mel bands.

To maintain spectral energy density consistency, we apply **Slaney Normalization**, dividing the weights of each filter by its bandwidth in Hertz:
$$H'_m(k) = H_m(k) \cdot \frac{2}{f_{m+2} - f_m}$$

The Mel spectrogram energy $S_t(m)$ for frame $t$ is:
$$S_t(m) = \sum_{k=0}^{N/2} P_t(k) \cdot H'_m(k), \quad 0 \le m < M$$

### 2.4. Logarithmic Compression
To approximate human perception of loudness (which is logarithmic), we apply a power-to-decibel conversion:
$$D_t(m) = 10 \cdot \log_{10}(\max(S_t(m), 10^{-10}))$$
To mimic the dynamic range limits of hearing, we threshold the values to a range of $80.0\text{ dB}$ below the peak value across the entire audio file:
$$D'_t(m) = \max(D_t(m), D_{\text{max}} - 80.0)$$

### 2.5. Discrete Cosine Transform (DCT-II)
Because Mel bands are highly correlated, we apply the Discrete Cosine Transform (DCT-II) to decorrelate the Mel spectral vectors and project them into a lower-dimensional cepstral space:
$$C_t(i) = w(i) \cdot \sum_{m=0}^{M-1} D'_t(m) \cdot \cos\left( \frac{\pi \cdot i \cdot (2m + 1)}{2M} \right)$$
where $0 \le i < K$ (with $K = 40$ output coefficients), and $w(i)$ is the orthogonal normalization factor:
$$w(i) = \begin{cases} 
\sqrt{\frac{1}{M}} & i = 0 \\
\sqrt{\frac{2}{M}} & i > 0
\end{cases}$$

### 2.6. Temporal Mean Pooling
To construct a single static input vector for our classifiers, we average the cepstral coefficients $C_t(i)$ over the total number of frames $T$:
$$\bar{C}(i) = \frac{1}{T} \sum_{t=0}^{T-1} C_t(i)$$
The final feature vector $\bar{C}$ is of size $40$.

---

## 3. Machine Learning & Deep Learning Architectures

### 3.1. Wav2Vec 2.0 (Self-Supervised Transformer)
Wav2Vec2 represents a major paradigm shift in audio processing. Rather than relying on hand-crafted features like MFCCs, it learns representations directly from raw audio waveforms.

```
  Raw Audio (16kHz) ────► [ Temporal CNN Encoder ] ────► Latent Features (z_t)
                                 │
                                 ▼
                          [ Transformer Block ] ────► Context Tensors (c_t)
                                 │
                                 ▼
                          [ Linear Head / Softmax ] ──► Probability Map (8 classes)
```

- **Feature Encoder**: A multi-layer temporal convolutional network that downsamples raw $16\text{ kHz}$ audio into latent representations $z_t$ every $20\text{ ms}$.
- **Context Network**: A stack of Transformer blocks that applies multi-head self-attention over the latents to build contextualized representations $c_t$ containing semantic and phonetic characteristics.
- **Classification Head**: Average pooling followed by a linear projection layer mapping the context tensors to our 8 emotion classes.

### 3.2. 1D Convolutional Neural Network (CNN)
The 1D CNN is designed to recognize spatial features (such as formant shifts, pitch rises, and timbral changes) across the 40-dimensional MFCC vector.
- **Input**: Shape `[Batch, 40, 1]` (channel dimension is 1).
- **Architecture**:
  - `Conv1D` (64 filters, kernel size 3, ReLU activation)
  - `MaxPooling1D` (pool size 2)
  - `Dropout` (0.2)
  - `Conv1D` (128 filters, kernel size 3, ReLU activation)
  - `MaxPooling1D` (pool size 2)
  - `Dropout` (0.2)
  - `Flatten`
  - `Dense` (128 nodes, ReLU activation)
  - `Dropout` (0.5)
  - `Dense` (8 output nodes, Softmax activation)

### 3.3. Long Short-Term Memory Network (LSTM)
LSTMs are recurrent architectures suited for sequence modelling. In the baseline code, the LSTM is fed the same 40-dimensional feature vector reshaped to `(Batch, 40, 1)`, where the 40 features are processed sequentially as "time steps."
- **Input**: Shape `[Batch, 40, 1]`.
- **Architecture**:
  - Two stacked LSTM layers (64 units each) to capture sequential feature dependencies.
  - Dropout layer (0.2) to prevent overfitting.
  - Dense layer (8 output nodes, Softmax activation).

### 3.4. Multi-Layer Perceptron (MLP)
The baseline model is a standard feedforward neural network trained using Scikit-Learn. It accepts the flattened 40-dimensional feature vector.
- **Input**: Shape `[Batch, 40]`.
- **Architecture**:
  - A single hidden layer of 100 units.
  - Adam optimizer, learning rate 0.001.
  - Logistic/Softmax classification head.

---

## 4. Evaluation and Class Alignment
In the RAVDESS dataset, emotion classes are mapped numerically. However, Scikit-Learn's `LabelEncoder` fits target labels alphabetically. To avoid index mismatch errors between Python training and Android TFLite Kotlin execution, we use the alphabetical index structure:

| Index | Emotion Class | RAVDESS ID | Emoji | Color (Hex) | Color Name |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | angry | 05 | 😡 | `#E74C3C` | Crimson |
| **1** | calm | 02 | 😌 | `#4ECDC4` | Teal |
| **2** | disgust | 07 | 🤢 | `#26DE81` | Emerald |
| **3** | fearful | 06 | 😨 | `#8854D0` | Violet |
| **4** | happy | 03 | 😄 | `#F7B731` | Amber |
| **5** | neutral | 01 | 😐 | `#8892A4` | Silver |
| **6** | sad | 04 | 😢 | `#4A69BD` | Deep Blue |
| **7** | surprised | 08 | 😲 | `#FD79A8` | Pink |

---

## 5. Comparative Trade-offs & Inference Latency

Deploying SER models requires balancing accuracy against resource utilization:

1. **Wav2Vec2 (Transformer)**:
   - *Accuracy*: **SOTA (~85%+)**
   - *Model Size*: **~360 MB**
   - *Requirements*: High CPU/RAM. Not suitable for local mobile devices without massive quantization (which degrades accuracy). Excellent for server environments.
2. **1D CNN**:
   - *Accuracy*: **Moderate (~50-60%)**
   - *Model Size*: **~635 KB (TFLite format)**
   - *Requirements*: Extremely lightweight. Operates natively on standard TFLite delegates without custom operators. Highly suited for local edge deployment on mobile.
3. **LSTM**:
   - *Accuracy*: **Moderate-High (~55-65%)**
   - *Model Size*: **~2.5 MB**
   - *Requirements*: Requires `FlexDelegate` / `Select Ops` in TFLite to process complex recurrent cells. This bloats the TFLite runtime library size in the Android APK, making CNNs a better choice for small download packages.

---

## 6. Conclusion
The VoxSense suite successfully demonstrates the viability of both cloud-based and on-device Speech Emotion Recognition. By combining a mathematically matching Kotlin-based MFCC signal extractor with a lightweight 1D CNN TFLite model, the local Android app achieves real-time paralinguistic classification offline, preserving user privacy. Meanwhile, the server-side Wav2Vec2 pipeline provides a high-fidelity classification benchmark for complex, cloud-connected platforms.
