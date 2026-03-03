import streamlit as st
import tensorflow as st_tf
from PIL import Image
import numpy as np
import cv2

# ==========================================
# Task 5: Application Interface (Frontend)
# Deep Learning Inference Pipeline using Streamlit
# ==========================================

# Set up the page config
st.set_page_config(
    page_title="Deep Learning Vehicle Plate Detector",
    page_icon="🚗",
    layout="centered"
)

# 1. Setup Title and Description
st.title("🚗 Vehicle License Plate Detector")
st.write("Upload an image of a vehicle to detect the presence of a license plate using our Custom Deep Learning Model.")

# 2. Function to Load Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    # In a real scenario, this would load your saved .h5 model file
    # Example: return st_tf.keras.models.load_model('license_plate_model.h5')
    
    # For demonstration without a trained file:
    return "Dummy Model Loaded"

model = load_model()

# 3. Create a Web Form to Accept User Input (Image Upload)
uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "png", "jpeg"])

# Ensure an image was uploaded before continuing
if uploaded_file is not None:
    
    # Read the file and convert it into a PIL Image for display
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing image...")
    
    # 4. Pre-process User Input to match Model Requirements before Prediction
    # We must match the exactly same transformations done in Task 2!
    
    # Convert PIL Image back to NumPy Array (as OpenCV uses)
    img_array = np.array(image.convert("RGB"))
    
    # Step A: Resize Image to 224x224 (as defined in our CNN architecture)
    resized_img = cv2.resize(img_array, (224, 224))
    
    # Step B: Normalize/Rescale pixel values between 0.0 and 1.0
    normalized_img = resized_img.astype('float32') / 255.0
    
    # Step C: Add Batch Dimension. 
    # Models expect shape (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
    # Our single image is just (224, 224, 3), so we make it (1, 224, 224, 3)
    input_tensor = np.expand_dims(normalized_img, axis=0)
    
    # 5. Make the Prediction!
    if st.button("Detect License Plate!"):
        with st.spinner('Running Inference Pipeline...'):
            
            # --- REAL INFERENCE CODE (Commented out until model exists) ---
            # predictions = model.predict(input_tensor)
            # class_idx = np.argmax(predictions, axis=1)[0]
            # confidence = np.max(predictions)
            # class_names = ["No Plate Detected", "License Plate Detected"]
            # result = class_names[class_idx]
            
            # --- DUMMY INFERENCE (For Previewing the Frontend UI) ---
            result = "License Plate Detected (Dummy Output)"
            confidence = 0.9452 
            
            st.success("✅ Prediction Complete!")
            
            # Display beautifully formatted results
            st.markdown(f"### **Result:** {result}")
            st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            
            # Show the pre-processed version the model actually "saw"
            st.write("---")
            st.write("*What the Neural Network actually saw (224x224 Normalized):*")
            st.image(normalized_img, width=150)
