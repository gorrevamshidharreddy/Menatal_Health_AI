# utils/text_utils.py
import os
import joblib
import numpy as np

# --- load artifacts from your models/ folder (matches your upload) ---
BASE = os.path.join(os.path.dirname(__file__), "..", "models")
TOKENIZER_PATH = os.path.abspath(os.path.join(BASE, "text_tokenizer.pkl"))
MODEL_PATH = os.path.abspath(os.path.join(BASE, "text_model_tfidf.pkl"))
LABEL_ENCODER_PATH = os.path.abspath(os.path.join(BASE, "label_encoder.pkl"))

# Load once at import time
try:
    TFIDF = joblib.load(TOKENIZER_PATH)
    TEXT_MODEL = joblib.load(MODEL_PATH)
    LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    # If loading fails, raise a clear error so you can fix file names/paths
    raise RuntimeError(f"Failed to load text model artifacts: {e}")

def predict_text_emotion(text: str) -> str:
    """
    Predict emotion label from text using a TF-IDF vectorizer + sklearn model.
    Ensures preprocessing is consistent with training and handles unseen words better.
    """
    try:
        if not text or not str(text).strip():
            return "neutral"

        # Step 1: lowercase and strip spaces (same as in training preprocessing)
        processed_text = str(text).lower().strip()

        # Step 2: TF-IDF transform
        X = TFIDF.transform([processed_text])

        # Step 3: Predict using loaded model
        pred = TEXT_MODEL.predict(X)

        # Step 4: Decode label
        if hasattr(LABEL_ENCODER, "inverse_transform"):
            label = LABEL_ENCODER.inverse_transform(pred)[0]
        else:
            label = pred[0] if isinstance(pred, (list, np.ndarray)) else str(pred)

        # Step 5: Return as lowercase string
        return str(label).lower()

    except Exception as e:
        print("predict_text_emotion error:", e)
        return "neutral"

