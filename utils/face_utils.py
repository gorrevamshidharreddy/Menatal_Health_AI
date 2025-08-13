import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Find Haarcascade file in models folder or subfolders
CASCADE_PATH = None
for root, dirs, files in os.walk(MODELS_DIR):
    for file in files:
        if "haarcascade_frontalface" in file.lower() and file.lower().endswith(".xml"):
            CASCADE_PATH = os.path.join(root, file)
            break
    if CASCADE_PATH:
        break

if CASCADE_PATH is None:
    raise FileNotFoundError(
        f"Could not find any haarcascade_frontalface*.xml file in {MODELS_DIR} or subfolders."
    )

print(f"[DEBUG] Using Haarcascade file: {CASCADE_PATH}")  # This will run only in console

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Failed to load Haarcascade from: {CASCADE_PATH}")

# Load face model
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.hdf5")
if not os.path.exists(FACE_MODEL_PATH):
    raise FileNotFoundError(f"Face model not found at: {FACE_MODEL_PATH}")
face_model = load_model(FACE_MODEL_PATH)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_face_emotion_from_bytes(img_bytes):
    file_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "no_face"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        preds = face_model.predict(roi_gray, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        return emotion

    return "no_face"





