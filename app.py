import streamlit as st
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    h1, h2, h3, p {
        color: white;
    }

    button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.expander("ℹ️ Model information"):
    st.write("""
    • Model: MLPClassifier (Scikit-learn)  
    • Hidden layers: 128, 64  
    • Activation: ReLU  
    • Accuracy: ~96%  
    • Dataset: MNIST  
    """)

import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib

st.set_page_config(page_title="Digit AI", page_icon="KHAN$$")

st.title("Digit AI")
st.write("Draw a number and the neural network will try to guess it.")

model = joblib.load("digit_model.pkl")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]
    img = Image.fromarray(img).resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, -1)

    prediction = model.predict(img)[0]

    st.subheader(f"Recognized digit: {prediction}")

    st.divider()
st.subheader("📊 Examples from training dataset")

import pandas as pd
import random

df = pd.read_csv("mnist_test.csv")

idx = random.randint(0, len(df) - 1)
row = df.iloc[idx]

label = row["label"]
pixels = row.drop("label").values.reshape(28, 28)

st.write(f"True label from dataset: {label}")
st.image(pixels, width=150)

if st.button("Clear canvas"):
    st.rerun()
st.divider()
st.subheader("How it works")

st.write("""
1. User draws a digit
2. Image is resized to 28x28
3. Pixel values are normalized
4. Neural network predicts the digit
""")

