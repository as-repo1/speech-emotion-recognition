import os
import io
import joblib
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input
import subprocess

# Define base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Emotion Mapping to standardize TESS and RAVDESS categories
emotion_mapping = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fearful',
    'fearful': 'fearful',
    'happy': 'happy',
    'neutral': 'neutral',
    'pleasant surprise': 'surprised',
    'surprised': 'surprised',
    'sad': 'sad',
    'sadness': 'sad',
    'calm': 'calm'
}

# Standard targets
standard_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_mfcc_from_bytes(audio_bytes):
    """Extract mean 40 MFCC features directly from in-memory WAV bytes."""
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as sound_file:
            audio_data = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
                
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def download_and_preprocess():
    print("Sourcing datasets...")
    features = []
    labels = []
    
    # 1. Load TESS Parquet from HuggingFace
    tess_url = 'https://huggingface.co/datasets/AbstractTTS/TESS/resolve/main/data/train-00000-of-00001.parquet'
    print(f"Loading TESS Parquet from {tess_url}...")
    try:
        tess_df = pd.read_parquet(tess_url)
        print(f"Loaded TESS. Processing {len(tess_df)} rows...")
        for idx, row in tess_df.iterrows():
            emotion_raw = row['emotion']
            mapped_emotion = emotion_mapping.get(emotion_raw.lower())
            if mapped_emotion:
                audio_data = row['audio']
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    mfcc = extract_mfcc_from_bytes(audio_data['bytes'])
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(mapped_emotion)
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1} / {len(tess_df)} TESS samples...")
    except Exception as e:
        print(f"Error loading/processing TESS: {e}")

    # 2. Load RAVDESS Parquet shards from HuggingFace (xbgoose/ravdess)
    ravdess_shards = [
        'https://huggingface.co/datasets/xbgoose/ravdess/resolve/main/data/train-00000-of-00002-94d632c9f1f51bbe.parquet',
        'https://huggingface.co/datasets/xbgoose/ravdess/resolve/main/data/train-00001-of-00002-bcaf733d4b46d6b2.parquet'
    ]
    
    for shard_idx, url in enumerate(ravdess_shards):
        print(f"Loading RAVDESS Parquet Shard {shard_idx + 1} from {url}...")
        try:
            rav_df = pd.read_parquet(url)
            print(f"Loaded RAVDESS Shard {shard_idx + 1}. Processing {len(rav_df)} rows...")
            for idx, row in rav_df.iterrows():
                emotion_raw = row['emotion']
                mapped_emotion = emotion_mapping.get(emotion_raw.lower())
                if mapped_emotion:
                    audio_data = row['audio']
                    if isinstance(audio_data, dict) and 'bytes' in audio_data:
                        mfcc = extract_mfcc_from_bytes(audio_data['bytes'])
                        if mfcc is not None:
                            features.append(mfcc)
                            labels.append(mapped_emotion)
                if (idx + 1) % 400 == 0:
                    print(f"  Processed {idx + 1} / {len(rav_df)} RAVDESS samples...")
        except Exception as e:
            print(f"Error loading/processing RAVDESS Shard {shard_idx + 1}: {e}")

    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Preprocessing completed. Total samples: {len(features)}, features shape: {features.shape}")
    return features, labels

def main():
    X, y = download_and_preprocess()
    
    # Save the label encoder
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(standard_classes)
    y_encoded = label_encoder.transform(y)
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"Label Encoder saved with classes: {label_encoder.classes_}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 1. Train MLP Model
    print("\n--- Training MLP Classifier ---")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, learning_rate_init=0.001, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"MLP Test Accuracy: {mlp_acc * 100:.2f}%")
    joblib.dump(mlp, os.path.join(MODELS_DIR, "speech_emotion_recognition_model.pkl"))
    print("MLP Model saved successfully.")

    # Reshape for Keras models (1D CNN, LSTM, CRNN)
    X_train_reshaped = np.expand_dims(X_train, axis=2)
    X_test_reshaped = np.expand_dims(X_test, axis=2)
    
    # 2. Train 1D CNN Model
    print("\n--- Training 1D CNN Classifier ---")
    cnn = Sequential([
        Input(shape=(40, 1)),
        Conv1D(128, 5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(128, 5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train_reshaped, y_train, epochs=40, batch_size=64, validation_split=0.1, verbose=1)
    cnn_loss, cnn_acc = cnn.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"CNN Test Accuracy: {cnn_acc * 100:.2f}%")
    cnn_h5_path = os.path.join(MODELS_DIR, "speech_emotion_recognition_cnn_model.h5")
    cnn.save(cnn_h5_path)
    print("CNN Keras model saved successfully.")

    # 3. Train LSTM Model
    print("\n--- Training LSTM Classifier ---")
    lstm = Sequential([
        Input(shape=(40, 1)),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(8, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_reshaped, y_train, epochs=40, batch_size=64, validation_split=0.1, verbose=1)
    lstm_loss, lstm_acc = lstm.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"LSTM Test Accuracy: {lstm_acc * 100:.2f}%")
    lstm_h5_path = os.path.join(MODELS_DIR, "speech_emotion_recognition_lstm_model.h5")
    lstm.save(lstm_h5_path)
    print("LSTM Keras model saved successfully.")

    # 4. Train CRNN Model
    print("\n--- Training CRNN Classifier ---")
    crnn = Sequential([
        Input(shape=(40, 1)),
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(8, activation='softmax')
    ])
    crnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    crnn.fit(X_train_reshaped, y_train, epochs=40, batch_size=64, validation_split=0.1, verbose=1)
    crnn_loss, crnn_acc = crnn.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"CRNN Test Accuracy: {crnn_acc * 100:.2f}%")
    crnn_h5_path = os.path.join(MODELS_DIR, "speech_emotion_recognition_crnn_model.h5")
    crnn.save(crnn_h5_path)
    print("CRNN Keras model saved successfully.")

    # Convert to TFLite
    print("\n--- Converting Keras models to TFLite ---")
    for name, path in [("cnn", cnn_h5_path), ("lstm", lstm_h5_path), ("crnn", crnn_h5_path)]:
        try:
            print(f"Converting {name} model...")
            model = tf.keras.models.load_model(path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # Enable Select TF ops if needed (LSTM sometimes needs it, though standard LSTM is supported in newer TF versions)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter._experimental_lower_tensor_list_ops = False
            tflite_model = converter.convert()
            tflite_path = os.path.join(MODELS_DIR, f"speech_emotion_recognition_{name}_model.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            print(f"Saved {name} TFLite model to {tflite_path}")
        except Exception as e:
            print(f"Error converting {name} to TFLite: {e}")

    # Convert to ONNX
    print("\n--- Converting Keras models to ONNX ---")
    for name, path in [("cnn", cnn_h5_path), ("lstm", lstm_h5_path), ("crnn", crnn_h5_path)]:
        try:
            print(f"Converting {name} model to ONNX...")
            onnx_path = os.path.join(MODELS_DIR, f"speech_emotion_recognition_{name}_model.onnx")
            cmd = f"python3 -m tf2onnx.convert --keras-model {path} --output {onnx_path}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"Saved {name} ONNX model to {onnx_path}")
        except Exception as e:
            print(f"Error converting {name} to ONNX: {e}")

    print("\nAll models trained and exported successfully!")

if __name__ == "__main__":
    main()
