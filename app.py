import streamlit as st
import joblib
import tempfile
import os
from keras.models import load_model
from utils.text_utils import predict_text_emotion
from utils.voice_utils import predict_voice_emotion
from utils.face_utils import detect_face_emotion_from_bytes

# ---------- Page config & CSS ---------- #
st.set_page_config(page_title="Mental Health-AI", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color: #0f1115; color: #e6eef3; }
      .card { background: #0f1419; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
      h1, h2, h3 { color: #eaf6ff }
      .stButton>button {
        background: linear-gradient(90deg,#6DD56D, #1fa23a);
        color: #021004;
        padding: .55rem 1rem;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        transition: transform .12s ease, filter .12s ease;
      }
      .stButton>button:hover {
        filter: brightness(1.08);
        transform: translateY(-2px) scale(1.02);
        cursor: pointer;
      }
      /* smaller gap for the analyze/suggest pair */
      .tight-gap > div { display:inline-block; margin-right:8px; }
      textarea, input { background:#0b0d0f !important; color:#e6eef3 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load models (joblib / keras) ---------- #
try:
    TFIDF = joblib.load(os.path.join("models", "text_tokenizer.pkl"))
    TEXT_MODEL = joblib.load(os.path.join("models", "text_model_tfidf.pkl"))
    LABEL_ENCODER = joblib.load(os.path.join("models", "label_encoder.pkl"))
    VOICE_MODEL = joblib.load(os.path.join("models", "voice_model.pkl"))
    FACE_MODEL = load_model(os.path.join("models", "emotion_model.hdf5"), compile=False)
except Exception as e:
    st.error(f"Error loading models. Check models/ folder. Details: {e}")
    st.stop()

# suggestions mapping (lowercase keys)
SUGGESTIONS = {
    "happy": "Keep up the positivity! Consider spreading it by talking to a friend.",
    "sad": "It's okay to feel sad. Try journaling or taking a walk in nature.",
    "angry": "Take deep breaths. A short break or light exercise might help.",
    "neutral": "You're stable now. Maintain balance with a hobby or light reading.",
    "fear": "Try grounding techniques. Talk to someone you trust.",
    "disgust": "Disengage from the situation. Reflect and understand your feeling.",
    "surprise": "Channel your surprise into curiosity. Explore more about it."
}

st.title("🧠 Mental Health - AI Emotion Detection")

# ---------------- TEXT SECTION ---------------- #
# TEXT SECTION
st.subheader("1️⃣ Detect Emotion from Text")
user_text = st.text_area("Enter how you're feeling (text):", value="", height=100)

cols = st.columns([0.6, 0.4])
with cols[0]:
    analyze_text = st.button("Analyze Text", key="analyze_text")
with cols[1]:
    # Suggest button only visible after analyze
    suggest_text = st.button("Get Suggestion (Text)", key="suggest_text") if st.session_state.get("text_analyzed", False) else False

if analyze_text:
    if user_text.strip():
        # predict and store normalized label
        pred = predict_text_emotion(user_text)
        pred_label = str(pred).lower() if pred is not None else "neutral"
        st.session_state["text_predicted"] = pred_label
        st.session_state["text_analyzed"] = True
    else:
        st.warning("Please enter some text before analyzing.")

if suggest_text:
    emotion = st.session_state.get("text_predicted", "neutral")
    st.info(SUGGESTIONS.get(emotion, "Take care of your well-being."))


# ---------------- FACE SECTION ---------------- #
# -------------------- FACE EMOTION DETECTION --------------------
import streamlit as st
from utils.face_utils import detect_face_emotion_from_bytes

st.subheader("2️⃣ Detect Emotion from Face")

# Capture photo input (persists in Streamlit state)
img_bytes = st.camera_input("Capture your face")

if img_bytes is not None:
    # Run the emotion detection
    emotion = detect_face_emotion_from_bytes(img_bytes)
  # make sure .getvalue() if expecting bytes
    if emotion == "no_face":
        st.warning("No face detected. Please try again.")
    else:
        st.success(f"Detected Face Emotion: **{emotion.capitalize()}**")
        st.info(SUGGESTIONS.get(emotion.lower(), "Take care of your well-being."))

# ---------------- VOICE SECTION ---------------- #
st.subheader("3️⃣ Detect Emotion from Voice")
st.markdown('<div class="card">', unsafe_allow_html=True)

voice_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

if voice_file is not None:
    # Save uploaded file to temp path
    suffix = ".wav" if voice_file.name.lower().endswith(".wav") else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(voice_file.read())
        tmp_path = tmp.name

    try:
        emotion = predict_voice_emotion(tmp_path)
        st.success(f"Detected Voice Emotion: **{emotion.capitalize()}**")
        st.info(SUGGESTIONS.get(emotion.lower(), "Take care of your well-being."))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
else:
    st.info("Please upload a WAV or MP3 audio file to analyze.")

st.markdown('</div>', unsafe_allow_html=True)


