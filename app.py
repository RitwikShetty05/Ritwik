import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 image of a digit (0–9).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and preprocess image
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    st.image(image, caption="Uploaded Digit", width=150)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_class}")
