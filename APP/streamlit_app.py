import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -------------------------------
# PAGE CONFIGURATION
st.set_page_config(page_title="Accident Detection", layout="centered", page_icon="🚗")

# -------------------------------
# PATH HELPER
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

PROJECT_ROOT = get_project_root()
MODEL_PATH = os.path.join(PROJECT_ROOT, "notebook", "Rawane_wassim_fixed.keras")

# -------------------------------
# CACHE AND LOAD MODEL
@st.cache_resource
def load_model():
    # loading the MobileNetV3Large model.
    # compiling is not strictly necessary for inference, 
    # but avoiding compile warnings by compiling=False
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image):
    # Model expects 300x300 RGB images
    image = image.resize((300, 300))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# -------------------------------
# MAIN APP
def main():
    st.title("🚗 Traffic Accident Detection 🚨")
    st.markdown("Upload an image of a scene to determine if there has been an accident.")

    with st.spinner("Loading AI model..."):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model from: {MODEL_PATH}\nDetails: {e}")
            return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Ensure it's in RGB formatting (remove alpha channels if any)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            st.image(image, caption="Uploaded Image", use_container_width=True)

            st.write("Analyzing...")
            
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0][0]
            
            # Based on dataset directory structure: 'Accident' (0) and 'Non Accident' (1)
            # Closer to 0 means Accident, Closer to 1 means Non Accident.
            if prediction < 0.5:
                confidence = (1 - prediction) * 100
                st.error(f"🚨 **Accident Detected!** (Confidence: {confidence:.2f}%)")
            else:
                confidence = prediction * 100
                st.success(f"✅ **No Accident Detected.** (Confidence: {confidence:.2f}%)")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()
