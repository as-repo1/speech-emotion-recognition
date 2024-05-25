import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load the pre-trained model
model = load_model("speech_emotion_recognition_lstm_model.h5")

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Define emotion labels
emotion_labels = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

# Function to extract features from audio files
def extract_features(file_path):
    try:
        with sf.SoundFile(file_path) as sound_file:
            audio_data = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            stft = np.abs(librosa.stft(audio_data))
            mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sample_rate).T, axis=0)
            return np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Streamlit UI
st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type="wav")

if uploaded_file is not None:
    # Display audio player
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

    # Extract features
    features = extract_features(uploaded_file)
    if features is not None:
        # Make prediction
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        st.success(f"Predicted Emotion: {predicted_label[0]}")
