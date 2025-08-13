import joblib
import numpy as np
import librosa
import os

# Load model and encoder
VOICE_MODEL = joblib.load(os.path.join("models", "voice_model.pkl"))
LABEL_ENCODER = joblib.load(os.path.join("models", "label_encoder.pkl"))

def extract_voice_features(file_path, sr=22050, n_mfcc=40):
    """
    Extract MFCC features from an audio file (supports wav & mp3).
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast', mono=True)
        # Optional: trim to first 3 seconds for consistency
        if len(audio) > sr * 3:
            audio = audio[:sr * 3]
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        features = np.mean(mfccs.T, axis=0)
        return features
    except Exception as e:
        print("librosa extraction error:", e)
        return None

def predict_voice_emotion(file_path):
    """
    Predict emotion from audio file path.
    """
    feats = extract_voice_features(file_path)
    if feats is None:
        return "neutral"
    
    try:
        pred = VOICE_MODEL.predict([feats])[0]
    except Exception:
        proba = VOICE_MODEL.predict_proba([feats])
        pred = VOICE_MODEL.classes_[int(np.argmax(proba))]
    
    try:
        label = LABEL_ENCODER.inverse_transform([pred])[0]
    except Exception:
        label = pred if isinstance(pred, str) else str(pred)
    
    return label.lower()


