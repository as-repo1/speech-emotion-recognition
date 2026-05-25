import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import onnxruntime as ort
from pydub import AudioSegment
from transformers import pipeline
import torch

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Define base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Label mapper to map transformer predictions to standard RAVDESS categories
label_mapper = {
    'neu': 'neutral', 'neutral': 'neutral',
    'cal': 'calm', 'calm': 'calm',
    'hap': 'happy', 'happy': 'happy',
    'sad': 'sad',
    'ang': 'angry', 'angry': 'angry',
    'fea': 'fearful', 'fear': 'fearful', 'fearful': 'fearful',
    'dis': 'disgust', 'disgust': 'disgust',
    'sur': 'surprised', 'surprise': 'surprised', 'surprised': 'surprised'
}

import subprocess
import requests

# Load models and label encoder
models = {}
label_encoder = None

def get_github_remote_info():
    try:
        res = subprocess.check_output(["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL)
        url = res.decode("utf-8").strip()
        if "github.com" in url:
            parts = url.split("github.com/")[-1].replace(".git", "").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
    except Exception:
        pass
    return "as-repo1", "speech-emotion-recognition"

username, repo = get_github_remote_info()

AVAILABLE_MODELS = {
    "cnn": {
        "id": "cnn",
        "name": "CNN (RAVDESS + TESS)",
        "description": "1D Convolutional Neural Network. Trained on combined dataset. Fast and balanced.",
        "size": "975 KB",
        "filename": "speech_emotion_recognition_cnn_model.h5",
        "url": f"https://raw.githubusercontent.com/{username}/{repo}/main/models/speech_emotion_recognition_cnn_model.h5"
    },
    "lstm": {
        "id": "lstm",
        "name": "LSTM (RAVDESS + TESS)",
        "description": "Recurrent LSTM model. Trained on combined dataset. Good temporal analysis.",
        "size": "491 KB",
        "filename": "speech_emotion_recognition_lstm_model.h5",
        "url": f"https://raw.githubusercontent.com/{username}/{repo}/main/models/speech_emotion_recognition_lstm_model.h5"
    },
    "crnn": {
        "id": "crnn",
        "name": "CRNN (RAVDESS + TESS)",
        "description": "Convolutional Recurrent Neural Network. Combines spatial and sequential features. Most sophisticated.",
        "size": "323 KB",
        "filename": "speech_emotion_recognition_crnn_model.h5",
        "url": f"https://raw.githubusercontent.com/{username}/{repo}/main/models/speech_emotion_recognition_crnn_model.h5"
    },
    "mlp": {
        "id": "mlp",
        "name": "MLP (RAVDESS + TESS)",
        "description": "Multi-Layer Perceptron baseline. Very lightweight and quick.",
        "size": "703 KB",
        "filename": "speech_emotion_recognition_model.pkl",
        "url": f"https://raw.githubusercontent.com/{username}/{repo}/main/models/speech_emotion_recognition_model.pkl"
    },
    "wav2vec2": {
        "id": "wav2vec2",
        "name": "Wav2Vec2 (Hugging Face)",
        "description": "State-of-the-art transformer model loaded dynamically from HuggingFace.",
        "size": "380 MB",
        "filename": "harshit345/xlsr-wav2vec-speech-emotion-recognition",
        "url": ""
    }
}

def load_model_on_demand(model_name):
    model_name = model_name.lower()
    if model_name in models:
        return True
        
    if model_name == 'wav2vec2':
        try:
            print("Loading state-of-the-art Wav2Vec2 transformer model...")
            # Auto-detect CUDA GPU support
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU (cuda)' if device == 0 else 'CPU'}")
            models["wav2vec2"] = pipeline(
                "audio-classification", 
                model="harshit345/xlsr-wav2vec-speech-emotion-recognition",
                device=device
            )
            print("Loaded Wav2Vec2 successfully.")
            return True
        except Exception as e:
            print(f"Error loading Wav2Vec2: {e}")
            return False
            
    if model_name == 'cnn':
        cnn_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_cnn_model.onnx")
        # Fallback to .h5 if .onnx is missing
        if not os.path.exists(cnn_path):
            cnn_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_cnn_model.h5")
        
        if os.path.exists(cnn_path):
            try:
                if cnn_path.endswith('.onnx'):
                    models["cnn"] = ort.InferenceSession(cnn_path)
                    print("Loaded CNN model successfully using ONNX Runtime.")
                else:
                    from tensorflow.keras.models import load_model
                    models["cnn"] = load_model(cnn_path)
                    print("Loaded CNN model successfully using Keras.")
                return True
            except Exception as e:
                print(f"Error loading CNN model: {e}")
                
    if model_name == 'lstm':
        lstm_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_lstm_model.onnx")
        if not os.path.exists(lstm_path):
            lstm_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_lstm_model.h5")
            
        if os.path.exists(lstm_path):
            try:
                if lstm_path.endswith('.onnx'):
                    models["lstm"] = ort.InferenceSession(lstm_path)
                    print("Loaded LSTM model successfully using ONNX Runtime.")
                else:
                    from tensorflow.keras.models import load_model
                    models["lstm"] = load_model(lstm_path)
                    print("Loaded LSTM model successfully using Keras.")
                return True
            except Exception as e:
                print(f"Error loading LSTM model: {e}")
                
    if model_name == 'crnn':
        crnn_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_crnn_model.onnx")
        if not os.path.exists(crnn_path):
            crnn_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_crnn_model.h5")
            
        if os.path.exists(crnn_path):
            try:
                if crnn_path.endswith('.onnx'):
                    models["crnn"] = ort.InferenceSession(crnn_path)
                    print("Loaded CRNN model successfully using ONNX Runtime.")
                else:
                    from tensorflow.keras.models import load_model
                    models["crnn"] = load_model(crnn_path)
                    print("Loaded CRNN model successfully using Keras.")
                return True
            except Exception as e:
                print(f"Error loading CRNN model: {e}")
                
    if model_name == 'mlp':
        mlp_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_model.pkl")
        if os.path.exists(mlp_path):
            try:
                models["mlp"] = joblib.load(mlp_path)
                print("Loaded MLP model successfully.")
                return True
            except Exception as e:
                print(f"Error loading MLP model: {e}")
                
    return False

# Load Label Encoder
label_encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
if os.path.exists(label_encoder_path):
    try:
        label_encoder = joblib.load(label_encoder_path)
        print("Loaded Label Encoder successfully.")
    except Exception as e:
        print(f"Error loading Label Encoder: {e}")

# Fallback classes if label encoder is missing
fallback_classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
classes = list(label_encoder.classes_) if label_encoder is not None else fallback_classes

# Load standard models at startup if they exist
for m in ['cnn', 'lstm', 'crnn', 'mlp']:
    load_model_on_demand(m)



def convert_to_wav(input_path, output_path):
    """Convert audio file of any format to WAV format using pydub."""
    try:
        ext = os.path.splitext(input_path)[1].lower().replace('.', '')
        if ext == 'wav':
            os.replace(input_path, output_path)
            return True
        
        audio = AudioSegment.from_file(input_path, format=ext)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_path} to WAV: {e}")
        return False


def extract_features(file_path):
    """Extract 40 MFCC features from the audio file."""
    try:
        with sf.SoundFile(file_path) as sound_file:
            audio_data = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
                
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error using soundfile for {file_path}, falling back to librosa.load: {e}")
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            return np.mean(mfccs.T, axis=0)
        except Exception as e2:
            print(f"Librosa fallback failed: {e2}")
            return None


def get_waveform_data(file_path, num_points=500):
    """Get downsampled amplitude waveform data for UI visualization."""
    try:
        audio_data, _ = librosa.load(file_path, sr=2000, mono=True)
        if len(audio_data) == 0:
            return [0.0] * num_points
            
        indices = np.linspace(0, len(audio_data) - 1, num_points, dtype=int)
        waveform = audio_data[indices].tolist()
        
        max_val = max(abs(np.min(waveform)), abs(np.max(waveform)))
        if max_val > 0:
            waveform = [float(x / max_val) for x in waveform]
        return waveform
    except Exception as e:
        print(f"Error getting waveform: {e}")
        return [0.0] * num_points


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(models.keys()),
        "label_encoder_loaded": label_encoder is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"}), 400
        
    model_name = request.form.get('model', 'wav2vec2').lower()
    load_model_on_demand(model_name)
    if model_name not in models:
        # Fallback to any loaded model
        if models:
            model_name = list(models.keys())[0]
        else:
            return jsonify({"success": False, "error": f"Model {model_name} is not loaded on the server"}), 500
            
    temp_dir = tempfile.mkdtemp()
    original_ext = os.path.splitext(uploaded_file.filename)[1]
    temp_input_path = os.path.join(temp_dir, f"input{original_ext}")
    temp_wav_path = os.path.join(temp_dir, "input.wav")
    
    try:
        uploaded_file.save(temp_input_path)
        if not convert_to_wav(temp_input_path, temp_wav_path):
            return jsonify({"success": False, "error": "Failed to process audio format"}), 400
            
        # Get waveform data for UI
        waveform = get_waveform_data(temp_wav_path)

        if model_name == 'wav2vec2':
            # Run inference using Hugging Face pipeline
            pipe = models['wav2vec2']
            prediction = pipe(temp_wav_path)
            
            # Map predictions to scores
            all_emotions = {cls: 0.0 for cls in classes}
            primary_label = 'neutral'
            max_score = 0.0
            
            for item in prediction:
                label_raw = item['label'].lower()
                mapped_label = label_mapper.get(label_raw, label_raw)
                score = float(item['score'])
                
                if mapped_label in all_emotions:
                    all_emotions[mapped_label] = score
                    
                if score > max_score:
                    max_score = score
                    primary_label = mapped_label
            
            return jsonify({
                "success": True,
                "emotion": primary_label,
                "confidence": max_score,
                "all_emotions": all_emotions,
                "model_used": "wav2vec2",
                "waveform": waveform
            })
            
        else:
            # Extract features for traditional models
            features = extract_features(temp_wav_path)
            if features is None:
                return jsonify({"success": False, "error": "Failed to extract audio features"}), 400
                
            model = models[model_name]
            
            # Reshape features depending on the model architecture
            if model_name in ['lstm', 'cnn', 'crnn']:
                model_input = np.expand_dims(np.expand_dims(features, axis=0), axis=2).astype(np.float32)
                if isinstance(model, ort.InferenceSession):
                    input_name = model.get_inputs()[0].name
                    prediction = model.run(None, {input_name: model_input})[0][0]
                else:
                    prediction = model.predict(model_input)[0]
            else:  # mlp
                model_input = features.reshape(1, -1)
                prediction = model.predict_proba(model_input)[0]
                
            # Get predicted label
            pred_idx = np.argmax(prediction)
            predicted_emotion = label_encoder.inverse_transform([pred_idx])[0] if label_encoder else classes[pred_idx]
            confidence = float(prediction[pred_idx])
            
            # Map all classes
            all_emotions = {}
            for idx, cls in enumerate(classes):
                if idx < len(prediction):
                    all_emotions[cls] = float(prediction[idx])
            
            return jsonify({
                "success": True,
                "emotion": predicted_emotion,
                "confidence": confidence,
                "all_emotions": all_emotions,
                "model_used": model_name,
                "waveform": waveform
            })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            os.rmdir(temp_dir)
        except Exception as cleanup_error:
            print(f"Error cleaning up temp files: {cleanup_error}")


@app.route('/api/predict_stream', methods=['POST'])
def predict_stream():
    """Endpoint optimized for real-time streaming classification of short chunks."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"}), 400
        
    model_name = request.form.get('model', 'wav2vec2').lower()
    load_model_on_demand(model_name)
    if model_name not in models:
        # Fallback
        if 'wav2vec2' in models:
            model_name = 'wav2vec2'
        elif models:
            model_name = list(models.keys())[0]
        else:
            return jsonify({"success": False, "error": "No models loaded on server"}), 500
            
    temp_dir = tempfile.mkdtemp()
    original_ext = os.path.splitext(uploaded_file.filename)[1] or '.webm'
    temp_input_path = os.path.join(temp_dir, f"input{original_ext}")
    temp_wav_path = os.path.join(temp_dir, "input.wav")
    
    try:
        uploaded_file.save(temp_input_path)
        if not convert_to_wav(temp_input_path, temp_wav_path):
            return jsonify({"success": False, "error": "Failed to process audio format"}), 400
            
        if model_name == 'wav2vec2':
            pipe = models['wav2vec2']
            prediction = pipe(temp_wav_path)
            
            all_emotions = {cls: 0.0 for cls in classes}
            primary_label = 'neutral'
            max_score = 0.0
            
            for item in prediction:
                label_raw = item['label'].lower()
                mapped_label = label_mapper.get(label_raw, label_raw)
                score = float(item['score'])
                
                if mapped_label in all_emotions:
                    all_emotions[mapped_label] = score
                    
                if score > max_score:
                    max_score = score
                    primary_label = mapped_label
            
            return jsonify({
                "success": True,
                "emotion": primary_label,
                "confidence": max_score,
                "all_emotions": all_emotions,
                "model_used": "wav2vec2"
            })
        else:
            # Fallback to local CNN/LSTM/MLP
            features = extract_features(temp_wav_path)
            if features is None:
                return jsonify({"success": False, "error": "Failed to extract features"}), 400
                
            model = models[model_name]
            if model_name in ['lstm', 'cnn', 'crnn']:
                model_input = np.expand_dims(np.expand_dims(features, axis=0), axis=2).astype(np.float32)
                if isinstance(model, ort.InferenceSession):
                    input_name = model.get_inputs()[0].name
                    prediction = model.run(None, {input_name: model_input})[0][0]
                else:
                    prediction = model.predict(model_input)[0]
            else:
                model_input = features.reshape(1, -1)
                prediction = model.predict_proba(model_input)[0]
                
            pred_idx = np.argmax(prediction)
            predicted_emotion = label_encoder.inverse_transform([pred_idx])[0] if label_encoder else classes[pred_idx]
            confidence = float(prediction[pred_idx])
            
            all_emotions = {}
            for idx, cls in enumerate(classes):
                if idx < len(prediction):
                    all_emotions[cls] = float(prediction[idx])
                    
            return jsonify({
                "success": True,
                "emotion": predicted_emotion,
                "confidence": confidence,
                "all_emotions": all_emotions,
                "model_used": model_name
            })
            
    except Exception as e:
        print(f"Error in predict_stream: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

@app.route('/api/models_status', methods=['GET'])
def models_status():
    status = []
    for model_id, meta in AVAILABLE_MODELS.items():
        if model_id == 'wav2vec2':
            is_downloaded = True
        else:
            path = os.path.join(BASE_DIR, "models", meta["filename"])
            is_downloaded = os.path.exists(path)
            
        status.append({
            "id": model_id,
            "name": meta["name"],
            "description": meta["description"],
            "size": meta["size"],
            "url": meta["url"],
            "downloaded": is_downloaded,
            "active": model_id in models
        })
    return jsonify({"success": True, "models": status})

@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(os.path.join(BASE_DIR, "models"), filename)

@app.route('/api/download_model', methods=['POST'])
def download_model():
    data = request.json or {}
    model_id = data.get("model", "").lower()
    
    if model_id not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "Invalid model ID"}), 400
        
    meta = AVAILABLE_MODELS[model_id]
    if model_id == 'wav2vec2':
        success = load_model_on_demand('wav2vec2')
        if success:
            return jsonify({"success": True, "message": "Wav2Vec2 model loaded successfully"})
        else:
            return jsonify({"success": False, "error": "Failed to load Wav2Vec2"}), 500
            
    filename = meta["filename"]
    dest_path = os.path.join(BASE_DIR, "models", filename)
    
    # Short-circuit if file already exists locally
    if os.path.exists(dest_path):
        print(f"Model file {filename} already exists locally. Loading...")
        success = load_model_on_demand(model_id)
        if success:
            return jsonify({"success": True, "message": f"{meta['name']} loaded successfully from local storage"})
        return jsonify({"success": False, "error": "Model files exists but failed to load"}), 500

    url = meta["url"]
    
    try:
        print(f"Downloading model {model_id} from {url}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return jsonify({"success": False, "error": f"Failed to download. HTTP status {response.status_code}"}), 500
            
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Download complete. Loading model {model_id}...")
        success = load_model_on_demand(model_id)
        if success:
            return jsonify({"success": True, "message": f"{meta['name']} downloaded and loaded successfully"})
            
        return jsonify({"success": False, "error": "Model downloaded but failed to load"}), 500
        
    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
