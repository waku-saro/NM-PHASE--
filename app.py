import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load pre-trained model (you should train & save it first as mnist_cnn.h5)
model = tf.keras.models.load_model('mnist_cnn.h5')

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image of a single digit (0–9) and the AI will predict it.")

uploaded_file = st.file_uploader("Choose a digit image (28x28 px or larger)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((28, 28))  # Resize to MNIST format
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors (if needed)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimension

    # Prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"🔢 Predicted Digit: {predicted_digit}")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")