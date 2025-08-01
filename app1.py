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
# Page Config and Custom Styling
# -----------------------------
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

st.markdown("""
<style>
/* Entire app container */
html, body, .stApp {
    height: 100%;
    background: linear-gradient(to bottom right, #eef2f3, #ffffff);
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.05rem;
    color: #222;
}

/* Main title */
h1 {
    font-size: 2.6rem !important;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 1rem;
}

/* Markdown text */
.stMarkdown {
    font-size: 1.1rem;
    line-height: 1.6;
    color: #333;
}

/* Input elements */
.stSelectbox div, .stRadio div, .stFileUploader, .stButton {
    font-size: 1.05rem !important;
}

/* Sidebar, if used */
section[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}

/* Hide model downloading messages */
.stInfo, .stSuccess, .stWarning {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Title & Info
# -----------------------------
st.title("Handwritten Digit Recognition")

with st.expander("About this App", expanded=True):
    st.markdown("""
    Welcome to the Handwritten Digit Recognition Web App.  
    This tool allows you to recognize handwritten digits using pre-trained machine learning models across multiple scripts.

    ---
    #### What is Handwritten Digit Recognition?
    It's a machine learning application that enables computers to interpret digits written by hand.  
    Commonly used in banking, forms processing, postal services, and more.

    ---
    #### Languages Supported
    - English: Standard digits (0–9)  
    - Hindi: Devanagari numerals  
    - Kannada: Regional script digits  
    - Roman Numerals: I to X

    ---
    #### Models Available
    - **CNN (Convolutional Neural Network)**: Best for image tasks, it learns features like edges and shapes.
    - **ANN (Artificial Neural Network)**: A simpler model useful for basic pattern recognition.
    - **Random Forest**: Traditional machine learning using decision trees. Requires the image to be flattened into features.

    ---
    Select a language and model below to try it out. You can draw a digit or upload an image.
    """)

# -----------------------------
# Google Drive Model Links
# -----------------------------
MODEL_LINKS = {
    "English": {
        "CNN": "https://drive.google.com/uc?id=1EDq5MO2_T9UN_n_N_PwW5tKIA25z4Li6",
        "ANN": "https://drive.google.com/uc?id=1YDdAsYiTHwXtYNoxREd2365w7cnAzlgR",
        "RF": "https://drive.google.com/uc?id=14caVZGrLCvocRujCpKB53H2EvpkRDQ3w"
    },
    "Hindi": {
        "CNN": "https://drive.google.com/uc?id=15TNldBSc2Z7P8VVQ6QfopNx4Z6PcIfC3",
        "ANN": "https://drive.google.com/uc?id=1a_0Y3dNu0MC7nr6EQCTDUWAxvFNcKvCW",
        "RF": "https://drive.google.com/uc?id=1aq02W1RXxKCaeDxwJQ2RnG1OREN8jyYn"
    },
    "Kannada": {
        "CNN": "https://drive.google.com/uc?id=1DYbOBjtT2SljAKG1bgraSlFjfBAojbEJ",
        "ANN": "https://drive.google.com/uc?id=10SbZ1c4E05UdkPbGy8c97hOhAK1pSDOs",
        "RF": "https://drive.google.com/uc?id=1kT7lrJe5jqsO71vzUpeOS-p1VClaIQRf"
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
    }
}

# -----------------------------
# Download model only once silently
# -----------------------------
def ensure_model_exists(language, model_type):
    os.makedirs("models", exist_ok=True)
    model_path = MODEL_PATHS[language][model_type]
    if not os.path.exists(model_path):
        gdown.download(MODEL_LINKS[language][model_type], model_path, quiet=True, fuzzy=True)
    return model_path

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Select Language", list(MODEL_PATHS.keys()))
with col2:
    model_type = st.selectbox("Select Model", ["CNN", "ANN", "RF"])

model_path = ensure_model_exists(language, model_type)

# -----------------------------
# Input method
# -----------------------------
input_method = st.radio("Choose Input Method", ["Draw Digit", "Upload Image"])

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
    uploaded_img = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("L")
        img = ImageOps.invert(img)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
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

            st.success(f"Predicted Digit: **{label}**")

        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("Please draw or upload a digit image.")
