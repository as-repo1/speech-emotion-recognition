import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from transformers import pipeline

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

# Load models and label encoder
models = {}
label_encoder = None

# 1. Load LSTM
lstm_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_lstm_model.h5")
if os.path.exists(lstm_path):
    try:
        models["lstm"] = load_model(lstm_path)
        print("Loaded LSTM model successfully.")
    except Exception as e:
        print(f"Error loading LSTM model: {e}")

# 2. Load CNN
cnn_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_cnn_model.h5")
if os.path.exists(cnn_path):
    try:
        models["cnn"] = load_model(cnn_path)
        print("Loaded CNN model successfully.")
    except Exception as e:
        print(f"Error loading CNN model: {e}")

# 3. Load MLP (scikit-learn)
mlp_path = os.path.join(BASE_DIR, "models", "speech_emotion_recognition_model.pkl")
if os.path.exists(mlp_path):
    try:
        models["mlp"] = joblib.load(mlp_path)
        print("Loaded MLP model successfully.")
    except Exception as e:
        print(f"Error loading MLP model: {e}")

# 4. Load Wav2Vec2 (Hugging Face)
try:
    print("Loading state-of-the-art Wav2Vec2 transformer model...")
    # Loading harshit345/xlsr-wav2vec-speech-emotion-recognition (fine-tuned on RAVDESS)
    models["wav2vec2"] = pipeline(
        "audio-classification", 
        model="harshit345/xlsr-wav2vec-speech-emotion-recognition"
    )
    print("Loaded Wav2Vec2 transformer model successfully.")
except Exception as e:
    print(f"Error loading Wav2Vec2 model: {e}")

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
    if model_name not in models:
        # Fallback to any loaded model
        if models:
            model_name = list(models.keys())[0]
        else:
            return jsonify({"success": False, "error": "No models are loaded on the server"}), 500
            
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
            if model_name in ['lstm', 'cnn']:
                model_input = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
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
            if model_name in ['lstm', 'cnn']:
                model_input = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
