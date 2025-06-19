import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
import os
from PIL import Image

# Load model and labels
model = tf.keras.models.load_model("models/pulmonitor_model.h5")
with open("models/labels.json", "r") as f:
    labels = json.load(f)

label_map = {int(k): v for k, v in labels.items()}

# Title & UI
st.set_page_config(page_title="Pulmonitor AI", layout="centered")
st.title("🫁 Pulmonitor AI")
st.markdown("Upload a **Chest X-ray** to detect signs of pulmonary disease like Pneumonia.")
st.markdown("🔍 Powered by deep learning • Trained on Kaggle dataset")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Image preview
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Predict Button
    if st.button("🔎 Predict"):
        with st.spinner("Analyzing..."):
            # Preprocess image
            img_array = np.array(img.resize((224, 224))) / 255.0
            img_array = img_array.reshape(1, 224, 224, 3)

            # Predict
            preds = model.predict(img_array)
            class_idx = np.argmax(preds)
            label = label_map[class_idx]
            confidence = np.max(preds)

        st.success(f"✅ **Prediction**: {label}")
        st.info(f"📊 **Confidence**: {confidence:.2f}")

        if label != "Normal":
            st.warning("⚠️ This X-ray shows signs of disease. Please consult a doctor.")
            st.markdown("[🧭 Find nearby hospitals](https://www.google.com/maps/search/hospitals+near+me/)")

# Footer
st.markdown("---")
st.caption("🧠 Developed by Samartha ")
