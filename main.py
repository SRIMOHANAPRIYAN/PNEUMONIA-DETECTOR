import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Config (Browser Title)
st.set_page_config(page_title="PneumoDetect AI", page_icon="🫁")

# 2. Load the Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    # Since main.py and the model are in the same folder, we just use the filename
    model = tf.keras.models.load_model('pneumonia_model.h5')
    return model

# Load it right away
with st.spinner('Loading AI Model...'):
    model = load_model()

# 3. UI Design
st.title("🫁 Pneumonia Detection AI")
st.markdown("""
This AI model analyzes chest X-Rays to detect signs of Pneumonia.
* **Accuracy:** 92%
* **Recall:** 96% (High Sensitivity)
""")

# 4. File Uploader
uploaded_file = st.file_uploader("Upload a Chest X-Ray (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray', use_column_width=True)
    
    # 5. Preprocessing (MUST match training data exactly)
    # Convert to grayscale (if not already) and resize to 224x224
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Normalize (0-1) - Crucial!
    img_array = img_array / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 6. Prediction
    if st.button("Analyze Image"):
        prediction = model.predict(img_array)
        
        # The output is a probability (0 to 1)
        probability = prediction[0][0]
        
        # 0 = Normal, 1 = Pneumonia
        if probability > 0.5:
            st.error(f"⚠️ PNEUMONIA DETECTED (Confidence: {probability:.2%})")
            st.warning("Recommendation: Please consult a Pulmonologist immediately.")
        else:
            st.success(f"✅ NORMAL (Confidence: {(1-probability):.2%})")
            st.info("No signs of pneumonia detected.")