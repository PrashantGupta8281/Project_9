import streamlit as st
import tensorflow as tf
import keras
import requests
import os
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Identifier Hub",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CAPTIVATING NEON DARK THEME (CUSTOM CSS) ---
st.markdown("""
    <style>
    /* Main Background & Text Color */
    .stApp {
        background: linear-gradient(135deg, #090a0f 0%, #151622 100%);
        color: #f0f2f6;
    }
    
    /* Header styling with a vivid electric gradient */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #00f2fe, #4facfe, #7f00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Glassmorphism containers for file upload and elements */
    div[data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(0, 242, 254, 0.25) !important;
        border-radius: 16px !important;
    }
    
    /* Custom High-Contrast Card Style for Results */
    .prediction-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 242, 254, 0.3);
        padding: 24px;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-top: 2rem;
    }
    
    /* Glowing Interactive Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0d0e15 !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5);
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL DOWNLOAD ---
MODEL_URL = "https://huggingface.co/Prassh/identifier_model/resolve/main/my_model.keras"
MODEL_PATH = "my_model.keras"

@st.cache_resource
def load_hf_model():
    """Downloads the Keras model directly from the public Hugging Face URL."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⚡ Fetching your public model from Hugging Face Hub..."):
            try:
                # No token or authentication headers needed since repo is public
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Failed to fetch model from Hugging Face: {e}")
                return None
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error parsing Keras file architecture: {e}")
        return None

# Compile step
model = load_hf_model()

# --- STUNNING PRESENTATION LAYER ---
st.markdown('<h1 class="main-title">AI Identifier Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">High-fidelity classification engine via Prassh/identifier_model</p>', unsafe_allow_html=True)

if model is not None:
    uploaded_file = st.file_uploader("Upload an target image for identification analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Target Scan", use_container_width=True)
        
        if st.button("Execute Identification Pass"):
            with st.spinner("Analyzing neural features..."):
                img_resized = image.resize((224, 224)) 
                img_array = np.array(img_resized) / 255.0
                input_tensor = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(input_tensor)
                
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: #00f2fe; margin-top: 0; font-weight:700;">Evaluation Complete</h3>
                    <p style="font-size: 1.05rem; color: #cbd5e1; margin-bottom: 12px;">Model Vector Readout Array:</p>
                    <code style="background: rgba(0,0,0,0.4); padding: 12px; display: block; border-radius: 8px; color: #22d3ee; font-family: monospace;">
                        {str(predictions[0])}
                    </code>
                </div>
            """, unsafe_allow_html=True)
