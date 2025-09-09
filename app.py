import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="centered")

st.title("🔢 MNIST Digit Classifier")
st.write("Upload a digit (0–9) or draw it below and the model will predict!")

# Tabs for upload or draw
tab1, tab2 = st.tabs(["📂 Upload Image", "✍️ Draw Digit"])

# --- Upload Image Tab ---
with tab1:
    uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        image = ImageOps.invert(image)  # Invert in case background is black
        image = image.resize((28, 28))

        st.image(image, caption="Uploaded Digit", width=150)

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.success(f"### ✅ Predicted Digit: {predicted_class}")

# --- Draw Digit Tab ---
with tab2:
    st.write("Draw a digit in the box below 👇")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")

        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        if np.sum(img_array) > 0:  # Avoid predicting empty canvas
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            st.success(f"### ✅ Predicted Digit: {predicted_class}")
