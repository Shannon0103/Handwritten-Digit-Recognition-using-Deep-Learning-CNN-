import numpy as np
from PIL import Image

def preprocess_image_for_model(img: Image.Image, model_type: str, language: str):
    img = img.resize((28, 28)).convert("L")
    img_array = np.array(img).astype("float32") / 255.0

    if model_type == "CNN":
        return img_array.reshape(1, 28, 28, 1)
    elif model_type == "ANN":
        # English ANN expects (28, 28), others expect flat (784,)
        if language == "English":
            return img_array.reshape(1, 28, 28)
        else:
            return img_array.reshape(1, -1)
    else:
        return img_array.reshape(1, -1)
