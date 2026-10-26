import os
import numpy as np, streamlit as st, tensorflow as tf
from PIL import Image

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# load the model once; the path is relative to this file, so it works locally and on Streamlit Cloud
if "model" not in st.session_state:
    st.session_state["model"] = tf.keras.models.load_model(
        os.path.join(os.path.dirname(__file__), "model2.keras"))
model = st.session_state["model"]

st.title("Facial Emotion Recognition")
st.caption("Use a close-up photo where the face fills the image.")
file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, width=300)

    gray = np.array(img.convert("L").resize((48, 48)), dtype="float32") / 255.0

    p = model.predict(gray.reshape(1, 48, 48, 1), verbose=0)[0]
    st.subheader(f"Prediction: {CLASSES[p.argmax()]} ({p.max():.0%})")