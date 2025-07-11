import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
from typing import Tuple, Optional, Dict, Any
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "model.savedmodel"
CLASS_NAMES = ["ON Helmet", "NO Helmet"]
TARGET_SIZE = (224, 224)
TIMEZONE = "Asia/Kolkata"
MAX_HISTORY_ENTRIES = 10
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

# Page configuration
st.set_page_config(
    page_title="Helmet Compliance Detector",
    page_icon="⛑️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #6C757D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #D4EDDA;
        border-left-color: #28A745;
    }
    .error-card {
        background: #F8D7DA;
        border-left-color: #DC3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model() -> Optional[tf.keras.Model]:
    """Load the TensorFlow model with proper error handling."""
    try:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image_resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def predict_helmet_compliance(model: tf.keras.Model, image: Image.Image) -> Tuple[str, float]:
    """Make prediction on helmet compliance."""
    try:
        img_array = preprocess_image(image)
        prediction = model.predict(img_array, verbose=0)
        
        label_idx = np.argmax(prediction)
        label = CLASS_NAMES[label_idx]
        confidence = float(np.max(prediction))
        
        return label, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "total_predictions" not in st.session_state:
        st.session_state.total_predictions = 0
    if "compliance_rate" not in st.session_state:
        st.session_state.compliance_rate = 0.0

def add_to_history(result: str, confidence: float, filename: str, threshold: float):
    """Add prediction to history if confidence meets threshold."""
    if confidence >= threshold:
        timestamp = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "result": result,
            "confidence": f"{confidence * 100:.2f}%",
            "filename": filename,
            "raw_confidence": confidence
        }
        
        # Add to beginning of history and limit size
        st.session_state.history.insert(0, entry)
        if len(st.session_state.history) > MAX_HISTORY_ENTRIES:
            st.session_state.history = st.session_state.history[:MAX_HISTORY_ENTRIES]
        
        # Update statistics
        st.session_state.total_predictions += 1
        compliant_count = sum(1 for h in st.session_state.history if h["result"] == "ON Helmet")
        st.session_state.compliance_rate = (compliant_count / len(st.session_state.history)) * 100

def display_prediction_results(label: str, confidence: float):
    """Display prediction results with improved styling."""
    confidence_percent = confidence * 100
    
    if label == "ON Helmet":
        st.markdown(f"""
        <div class="metric-card success-card">
            <h3>✅ Compliance Status: COMPLIANT</h3>
            <p><strong>Confidence:</strong> {confidence_percent:.2f}%</p>
            <p>Worker is properly wearing safety helmet</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card error-card">
            <h3>❌ Compliance Status: NON-COMPLIANT</h3>
            <p><strong>Confidence:</strong> {confidence_percent:.2f}%</p>
            <p>Worker is not wearing required safety helmet</p>
        </div>
        """, unsafe_allow_html=True)

def display_statistics():
    """Display compliance statistics."""
    if st.session_state.history:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", st.session_state.total_predictions)
        
        with col2:
            st.metric("Compliance Rate", f"{st.session_state.compliance_rate:.1f}%")
        
        with col3:
            recent_predictions = len(st.session_state.history)
            st.metric("Recent Logs", recent_predictions)

def export_history():
    """Export history as JSON."""
    if st.session_state.history:
        history_data = {
            "export_timestamp": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
            "total_predictions": st.session_state.total_predictions,
            "compliance_rate": st.session_state.compliance_rate,
            "history": st.session_state.history
        }
        return json.dumps(history_data, indent=2)
    return None

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load the helmet detection model. Please check the model file.")
        st.info("Make sure 'model.savedmodel' exists in the application directory.")
        st.stop()
    
    # Header
    st.markdown("<h1 class='main-header'>⛑️ Helmet Compliance Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-powered safety compliance monitoring system</p>", unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.7, 
            step=0.01,
            help="Minimum confidence required to log predictions"
        )
        
        st.header("📊 Statistics")
        display_statistics()
        
        if st.session_state.history:
            export_data = export_history()
            if export_data:
                st.download_button(
                    label="📥 Export History",
                    data=export_data,
                    file_name=f"helmet_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.total_predictions = 0
            st.session_state.compliance_rate = 0.0
            st.rerun()
    
    # Input method selection
    st.subheader("📸 Input Method")
    col1, col2 = st.columns(2)
    
    with col1:
        input_type = st.selectbox(
            "Choose Input Method", 
            ["Upload Image", "Camera Input"],
            help="Select how you want to provide the image for analysis"
        )
    
    with col2:
        show_confidence = st.checkbox("Show Confidence Details", value=True)
    
    # Image input handling
    image = None
    filename = ""
    
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader(
            "📤 Upload an image",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS).upper()}"
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            filename = uploaded_file.name
    
    elif input_type == "Camera Input":
        camera_file = st.camera_input("📷 Take a photo")
        if camera_file:
            image = Image.open(camera_file)
            filename = f"camera_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    # Process image if available
    if image:
        # Display image
        st.subheader("🖼️ Image Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption=filename, use_column_width=True)
        
        with col2:
            st.write("**Image Details:**")
            st.write(f"• **Filename:** {filename}")
            st.write(f"• **Size:** {image.size[0]} × {image.size[1]} px")
            st.write(f"• **Format:** {image.format}")
            st.write(f"• **Mode:** {image.mode}")
        
        # Make prediction
        try:
            with st.spinner("🧠 Analyzing image..."):
                label, confidence = predict_helmet_compliance(model, image)
            
            # Display results
            st.subheader("📋 Analysis Results")
            display_prediction_results(label, confidence)
            
            # Show detailed confidence if enabled
            if show_confidence:
                st.subheader("📊 Confidence Analysis")
                conf_col1, conf_col2 = st.columns(2)
                
                with conf_col1:
                    st.metric("Helmet Detected", f"{confidence * 100:.2f}%" if label == "ON Helmet" else f"{(1-confidence) * 100:.2f}%")
                
                with conf_col2:
                    st.metric("No Helmet", f"{(1-confidence) * 100:.2f}%" if label == "ON Helmet" else f"{confidence * 100:.2f}%")
                
                # Confidence bar
                st.progress(confidence if label == "ON Helmet" else 1-confidence)
            
            # Add to history
            add_to_history(label, confidence, filename, threshold)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}")
    
    # Display history
    if st.session_state.history:
        st.subheader("📝 Detection History")
        
        # Create a more detailed history view
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"#{i+1} - {entry['timestamp']} - {entry['result']} ({entry['confidence']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Result:** {entry['result']}")
                with col2:
                    st.write(f"**Confidence:** {entry['confidence']}")
                with col3:
                    st.write(f"**File:** {entry['filename']}")
    else:
        st.info("📋 No predictions logged yet. Upload an image or use camera to start analyzing.")

if __name__ == "__main__":
    main()
