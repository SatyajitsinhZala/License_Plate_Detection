import streamlit as st
import tensorflow as st_tf
from PIL import Image
import numpy as np
import cv2
import os

# ==========================================
# Task 7: Backend Integration & Deployment
# Streamlit App specifically built for Render.com
# ==========================================

# Set up the page config
st.set_page_config(
    page_title="Deep Learning Vehicle Plate Detector",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle License Plate Detector")
st.write("Upload an image of a vehicle to detect the presence of a license plate using our Custom Deep Learning Model.")

# 1. Connect the Frontend with the Saved Model
@st.cache_resource
def load_deep_learning_model():
    """
    Loads the trained model file. It checks for the 'best_model.h5' generated
    in the final Transfer Learning step.
    """
    model_path = 'best_model.h5'
    
    # Render.com path checking
    if os.path.exists(model_path):
        try:
            model = st_tf.keras.models.load_model(model_path)
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, False
    else:
        # Fallback dummy for UI testing before model is actually uploaded to Render
        return "Dummy", False

model, model_loaded_successfully = load_deep_learning_model()

if not model_loaded_successfully:
    st.warning("⚠️ Note for Deployment: 'best_model.h5' was not found in the directory. "
               "The app is running in UI-Only Simulation Mode. Please ensure you upload the .h5 "
               "file to your GitHub/Render repository when deploying!")

# 2. Handle Real-Time Prediction Requests
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing image in real-time...")
    
    # Pre-process User Input
    img_array = np.array(image.convert("RGB"))
    resized_img = cv2.resize(img_array, (224, 224))
    normalized_img = resized_img.astype('float32') / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=0)
    
    # Make Prediction Button
    if st.button("Detect License Plate!"):
        with st.spinner('Running Inference Pipeline...'):
            
            if model_loaded_successfully:
                # --- REAL INFERENCE CODE ---
                predictions = model.predict(input_tensor)
                class_idx = np.argmax(predictions, axis=1)[0]
                confidence = float(np.max(predictions))
                
                # Assuming Class 0 is Background/No Plate, Class 1 is Plate Detected
                class_names = ["No Plate Detected", "License Plate Detected"]
                result = class_names[class_idx]
            else:
                # --- DUMMY INFERENCE ---
                # Simulated delay for realism if running without model file
                import time
                time.sleep(1)
                result = "License Plate Detected (Dummy Output)"
                confidence = 0.9452 
            
            # 3. Output standard UI format
            st.success("✅ Prediction Complete!")
            st.markdown(f"### **Result:** {result}")
            st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            
            st.write("---")
            st.write("*What the Neural Network actually saw (224x224)*")
            st.image(normalized_img, width=150)
