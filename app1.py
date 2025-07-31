# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os
import gdown
from PIL import Image, ImageOps
from utils.preprocessing import preprocess_image_for_model
from streamlit_drawable_canvas import st_canvas

# -----------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🔢 Handwritten Digit Recognition")
st.markdown("Select model and input method to predict digits.")

# -----------------------------
# Model Download Links (Google Drive shareable links converted)
# -----------------------------
MODEL_LINKS = {
    "English": {
        "CNN": "https://drive.google.com/file/d/1EDq5MO2_T9UN_n_N_PwW5tKIA25z4Li6/view?usp=drive_link",
        "ANN": "https://drive.google.com/file/d/1YDdAsYiTHwXtYNoxREd2365w7cnAzlgR/view?usp=drive_link",
        "RF": "https://drive.google.com/file/d/14caVZGrLCvocRujCpKB53H2EvpkRDQ3w/view?usp=drive_link"
    },
    "Hindi": {
        "CNN": "https://drive.google.com/file/d/15TNldBSc2Z7P8VVQ6QfopNx4Z6PcIfC3/view?usp=drive_link",
        "ANN": "https://drive.google.com/file/d/1a_0Y3dNu0MC7nr6EQCTDUWAxvFNcKvCW/view?usp=drive_link",
        "RF": "https://drive.google.com/file/d/1aq02W1RXxKCaeDxwJQ2RnG1OREN8jyYn/view?usp=drive_link"
    },
    "Kannada": {
        "CNN": "https://drive.google.com/file/d/1DYbOBjtT2SljAKG1bgraSlFjfBAojbEJ/view?usp=drive_link",
        "ANN": "https://drive.google.com/file/d/10SbZ1c4E05UdkPbGy8c97hOhAK1pSDOs/view?usp=drive_link",
        "RF": "https://drive.google.com/file/d/1kT7lrJe5jqsO71vzUpeOS-p1VClaIQRf/view?usp=drive_link"
    },
    "Roman": {
        "CNN": "https://drive.google.com/file/d/182axr0KB5PEEDCdJKlzYnMoQuC6OuZDU/view?usp=drive_link",
        "ANN": "https://drive.google.com/file/d/12YJpZcE3bWOp75WbxBcHbVItioWQd4y-/view?usp=drive_link",
        "RF": "https://drive.google.com/file/d/1Dy5KemK8Pcqk_F9Y-NBdCygb-DIGhVb7/view?usp=drive_link"
    }
}

MODEL_PATHS = {
    "English": {
        "CNN": "models/english_cnn_model.h5",
        "ANN": "models/english_ann_model.h5",
        "RF": "models/english_rf_model.pkl"
    },
    "Hindi": {
        "CNN": "models/hindi_cnn_model.h5",
        "ANN": "models/hindi_ann_model.h5",
        "RF": "models/hindi_rf_model.pkl"
    },
    "Kannada": {
        "CNN": "models/kannada_cnn_model.h5",
        "ANN": "models/kannada_ann_model.h5",
        "RF": "models/kannada_rf_model.pkl"
    },
    "Roman": {
        "CNN": "models/roman_cnn_model.h5",
        "ANN": "models/roman_ann_model.h5",
        "RF": "models/roman_rf_model.pkl"
    }
}

# -----------------------------
# Ensure Model Exists or Download
# -----------------------------
def ensure_model_exists(language, model_type):
    os.makedirs("models", exist_ok=True)
    model_path = MODEL_PATHS[language][model_type]
    if not os.path.exists(model_path):
        st.info(f"📥 Downloading {language} {model_type} model...")
        gdown.download(MODEL_LINKS[language][model_type], model_path, quiet=False)
    return model_path

# -----------------------------
# UI Inputs
# -----------------------------
language = st.selectbox("🌐 Select Language", list(MODEL_PATHS.keys()))
model_type = st.selectbox("🧠 Select Model Type", ["CNN", "ANN", "RF"])
model_path = ensure_model_exists(language, model_type)

input_method = st.radio("✍️ Input Method", ["Draw Digit", "Upload Image"])

img = None
if input_method == "Draw Digit":
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if canvas_result.image_data is not None:
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
else:
    uploaded_img = st.file_uploader("📁 Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("L")
        img = ImageOps.invert(img)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    if img is not None:
        input_img = preprocess_image_for_model(img, model_type, language)

        try:
            if model_type in ["CNN", "ANN"]:
                model = tf.keras.models.load_model(model_path)
                prediction = model.predict(input_img)[0]
                label = np.argmax(prediction)
            else:
                model = joblib.load(model_path)
                label = model.predict(input_img.reshape(1, -1))[0]

            st.success(f"✅ Predicted Digit: **{label}**")

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    else:
        st.warning("Please draw or upload a digit image.")
