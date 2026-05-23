import os
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Directory where the dataset is located (relative to the Jupyter notebook)
DATASET_DIR = "data/"

# Function to extract features from audio files
def extract_features(file_path):
    try:
        with soundfile.SoundFile(file_path) as sound_file:
            audio_data = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Create lists to hold the features and labels
features = []
labels = []

# Emotion labels in the RAVDESS dataset
emotion_labels = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Traverse through the dataset directory and extract features
file_count = 0
for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion = emotion_labels.get(file.split("-")[2])
            if emotion:
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
                file_count += 1

print(f"Processed {file_count} files.")
print(f"Extracted {len(features)} feature sets.")

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Check if features and labels are not empty
if features.size == 0 or labels.size == 0:
    raise ValueError("No features or labels extracted. Please check the dataset path and file processing logic.")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Initialize and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=500)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model to a file
joblib.dump(model, "speech_emotion_recognition_model.pkl")

# Load the model from the file (optional, for later use)
model = joblib.load("speech_emotion_recognition_model.pkl")
