# 🧠 Mental Health-AI: Multi-Modal Emotion Detection & Well-Being Suggestions

**TriSense AI: Multi-Input Emotion Recognition System** is an interactive, multi-modal application that detects emotional states from **text**, **voice**, and **facial expressions** in real time, then provides **personalized well-being suggestions** to support mental health.  
Built with **Streamlit** and powered by **machine learning models**, the app seamlessly integrates multiple AI pipelines into a single, user-friendly interface.

---

## ✨ Features
- **📝 Text Emotion Detection** — Uses **TF-IDF vectorization** with a machine learning classifier to predict emotions from typed input.  
- **🎤 Voice Emotion Recognition** — Extracts **MFCC audio features** and classifies them into emotions; supports `.wav` and `.mp3`.  
- **📷 Facial Expression Analysis** — Detects faces using Haarcascade, processes grayscale image regions, and predicts emotions using a deep learning model.  
- **💡 Instant Well-Being Suggestions** — Offers supportive recommendations tailored to the detected emotion.  
- **📡 Real-Time Camera & Audio Input** — Capture live webcam images or microphone recordings directly from the app.  
- **🔀 Multi-Modal Integration** — Text, voice, and face emotion detection all in one platform.

---

## 🛠 Tech Stack
- **Frontend**: Streamlit  
- **ML/NLP**: Scikit-Learn, TensorFlow/Keras, TF-IDF, MFCC  
- **Audio Processing**: Librosa  
- **Computer Vision**: OpenCV  
- **Data Handling**: Pandas, NumPy, Joblib  
- **Deployment**: Streamlit Cloud / Local Execution

---

