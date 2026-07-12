import streamlit as st
import tensorflow as tf
import keras
import requests
import os
from PIL import Image
import numpy as np

# --- ADVANCED PAGE MANAGEMENT ---
st.set_page_config(
    page_title="NEURAL INSIGHT LABS",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- LUXURY ULTRA-MODERN DARK CYBERPUNK THEME ---
st.markdown("""
    <style>
    /* Absolute background glow animation */
    .stApp {
        background-color: #06070d;
        background-image: 
            radial-gradient(at 0% 0%, rgba(127, 0, 255, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(0, 242, 254, 0.12) 0px, transparent 50%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Clean, gorgeous main text gradient branding */
    .hero-container {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #7f00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px !important;
        margin-bottom: 0px !important;
        text-transform: uppercase;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.15rem;
        font-weight: 400;
        letter-spacing: 1px;
        margin-top: 5px !important;
    }

    /* Completely overhaul the file uploader dropzone */
    div[data-testid="stFileUploadDropzone"] {
        background: rgba(13, 16, 27, 0.7) !important;
        border: 1px solid rgba(0, 242, 254, 0.2) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: inset 0 0 20px rgba(0, 242, 254, 0.05) !important;
        transition: all 0.4s ease !important;
    }
    
    div[data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(0, 242, 254, 0.8) !important;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.15) !important;
    }
    
    /* Modify default labels to make them look slicker */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* REVOLUTIONARY INTUITIVE BUTTON STYLING */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #7f00ff 0%, #4facfe 100%) !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        border: none !important;
        padding: 16px 32px !important;
        border-radius: 14px !important;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 20px rgba(127, 0, 255, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01);
        background: linear-gradient(135deg, #8f1fff 0%, #00f2fe 100%) !important;
        box-shadow: 0 12px 30px rgba(0, 242, 254, 0.4), 0 0 15px rgba(127, 0, 255, 0.4);
        color: #ffffff !important;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.99);
    }

    /* Cinematic glass card matrix display box */
    .prediction-card {
        background: rgba(10, 11, 18, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.06);
        position: relative;
        overflow: hidden;
        padding: 30px;
        border-radius: 22px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        margin-top: 2.5rem;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #7f00ff, #00f2fe);
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .code-container {
        background: #030407 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 16px !important;
        border-radius: 12px !important;
        color: #00f2fe !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 1.05rem !important;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.8);
    }
    
    /* Clean adjustments for default elements */
    .stSpinner > div > div {
        border-top-color: #00f2fe !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL WORKFLOW (PUBLIC DIRECT DOWNLOAD) ---
MODEL_URL = "https://huggingface.co/Prassh/identifier_model/resolve/main/my_model.keras"
MODEL_PATH = "my_model.keras"

@st.cache_resource
def load_hf_model():
    """Directly fetches network weights since repo visibility is public."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⚡ Initiating hyper-speed neural link download..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Failed to pull asset matrix: {e}")
                return None
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Parsing architecture error: {e}")
        return None

# Compile Model
model = load_hf_model()

# --- HERO PRESENTATION BRANDING ---
st.markdown("""
    <div class="hero-container">
        <h1 class="main-title">Neural Engine</h1>
        <p class="subtitle">AUTOMATED RECOGNITION STUDIO</p>
    </div>
""", unsafe_allow_html=True)

# --- APPLICATION INTERACTION PIPELINE ---
if model is not None:
    uploaded_file = st.file_uploader("Drop target asset for feature decoding matrix", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Layout splitting into gorgeous columns
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file)
        
        with col1:
            st.markdown("<div style='padding-top:10px;'></div>", unsafe_allow_html=True)
            st.image(image, caption="Source Feed Matrix", use_container_width=True)
            
        with col2:
            st.markdown("<div style='padding-top:25px;'></div>", unsafe_allow_html=True)
            if st.button("Analyze Target Matrix"):
                with st.spinner("Decoding tensor mappings..."):
                   # --- AFTER (FIXED VERSION) ---
# 1. Automatically extract the exact shape your model expects
# Most Keras models have input_shape like (None, Height, Width, Channels)
try:
    model_input_shape = model.input_shape
    # Extract height and width from the shape matrix
    target_height = model_input_shape[1] if model_input_shape[1] is not None else 224
    target_width = model_input_shape[2] if model_input_shape[2] is not None else 224
    target_channels = model_input_shape[3] if len(model_input_shape) > 3 else 3
except Exception:
    # Fallback if shape inspection fails
    target_height, target_width, target_channels = 224, 224, 3

# 2. Resize the image dynamically to match your exact model requirements
img_resized = image.resize((target_width, target_height))

# 3. Convert to grayscale if your model only expects 1 channel (Black & White)
if target_channels == 1:
    img_resized = img_resized.convert("L")

img_array = np.array(img_resized)

# 4. Handle normalization scaling
# (If your data array isn't scaled between 0-1, you can remove the "/ 255.0")
if img_array.max() > 1.0:
    img_array = img_array / 255.0

# 5. Reshape to match standard batch format
if target_channels == 1:
    img_array = np.expand_dims(img_array, axis=-1) # Add channel index if grayscale

input_tensor = np.expand_dims(img_array, axis=0)
                    
                    # Core inference pass
                    predictions = model.predict(input_tensor)
                    
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="card-title">🔮 Inference Matrix Computed</div>
                        <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">Array Probabilities Vector Output:</p>
                        <div class="code-container">
                            {str(predictions[0])}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
