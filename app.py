import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title=" Plant Disease Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS STYLING
# ===============================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #27ae60;
        font-weight: 600;
        border-bottom: 2px solid #27ae60;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #34495e;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #ecf0f1;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #27ae60;
        color: white;
    }
    .uploadedFile {
        border: 2px dashed #27ae60;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR NAVIGATION
# ===============================
with st.sidebar:
    #st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title(" Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 Model Comparison", "🔍 Disease Prediction", "📈 Training Insights"]
    )
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.info("""
    **Dataset:** PlantVillage  
    **Classes:** 38  
    **Image Size:** 224×224  
    **Total Images:** ~54,000+  
    **Split:** 80-20 (Train-Val)
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.write("• TensorFlow / Keras")
    st.write("• Streamlit")
    st.write("• NumPy")
    st.write("• Matplotlib / Plotly")

# ===============================
# LOAD DATA AND MODELS
# ===============================
import os

# Check if files exist
def check_files():
    required_files = {
        'baseline_cnn_history.npy': 'required',
        'mobilenetv2_transfer_history.npy': 'optional',
        'baseline_cnn_best.h5': 'required'
    }
    
    current_dir = os.getcwd()
    st.sidebar.markdown(f"**📂 Current Directory:** `{current_dir}`")
    
    missing_required = []
    for file, requirement in required_files.items():
        if os.path.exists(file):
            st.sidebar.success(f"✅ {file}")
        else:
            if requirement == 'required':
                st.sidebar.error(f"❌ {file} (Required)")
                missing_required.append(file)
            else:
                st.sidebar.warning(f"⚠️ {file} (Optional)")
    
    return len(missing_required) == 0

files_exist = check_files()

@st.cache_resource
def load_histories():
    baseline_history = None
    mobilenet_history = None
    
    try:
        baseline_history = np.load('baseline_cnn_history.npy', allow_pickle=True).item()
        st.sidebar.info("✅ Baseline history loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Baseline history error: {str(e)}")
    
    try:
        mobilenet_history = np.load('mobilenetv2_transfer_history.npy', allow_pickle=True).item()
        st.sidebar.info("✅ MobileNet history loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ MobileNet history not available: {str(e)}")
    
    return baseline_history, mobilenet_history

@st.cache_resource
def load_trained_model():
    try:
        baseline_model = tf.keras.models.load_model("baseline_cnn_best.h5", compile=False)
        st.sidebar.info("✅ Baseline model loaded")
        return baseline_model
    except Exception as e:
        st.sidebar.error(f"❌ Baseline model error: {str(e)}")
        st.sidebar.warning("💡 Trying alternative loading method...")
        
        # Alternative: Try loading with different settings
        try:
            baseline_model = tf.keras.models.load_model("baseline_cnn_best.h5", compile=False, safe_mode=False)
            st.sidebar.success("✅ Baseline model loaded via alternative method")
            return baseline_model
        except:
            try:
                # Try loading weights only
                baseline_model = tf.keras.models.load_model("baseline_cnn_best.h5")
                st.sidebar.success("✅ Baseline model loaded with compile=True")
                return baseline_model
            except Exception as e2:
                st.sidebar.error(f"❌ All loading methods failed: {str(e2)}")
                return None

# Load model and histories
baseline_history = None
mobilenet_history = None
baseline_model = None
model_loaded = False

if files_exist:
    try:
        baseline_history, mobilenet_history = load_histories()
        baseline_model = load_trained_model()
        
        if baseline_model is not None:
            model_loaded = True
            st.sidebar.success("🎉 Baseline model loaded successfully!")
            if mobilenet_history is not None:
                st.sidebar.success("📊 Both training histories available for comparison")
            else:
                st.sidebar.info("📊 Only baseline history available")
        else:
            model_loaded = False
            st.sidebar.error("❌ Model could not be loaded")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading files: {str(e)}")
        st.sidebar.code(f"Error details: {type(e).__name__}")
        model_loaded = False
        baseline_model = None
else:
    st.sidebar.warning("⚠️ Required files are missing. Please check above.")

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ===============================
# PAGE: HOME
# ===============================
if page == "🏠 Home":
    st.title("🌿 Plant Disease Classification System")
    st.markdown("### *AI-Powered Agricultural Disease Detection*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Total Classes", "38", delta="diseases")
    with col2:
        st.metric("🔬 Best Accuracy", "98.03%", delta="+2.99%")
    with col3:
        st.metric("📦 Model Size", "27M", delta="params")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("###  Project Overview")
        st.markdown("""
        This application leverages **deep learning** to automatically identify plant diseases from leaf images. 
        The system has been trained on the comprehensive **PlantVillage dataset** containing thousands of images 
        across 38 different disease categories affecting various crops.
        
        ####  Key Features:
        - **Custom CNN Architecture**: High-performance 4-block CNN achieving 98.03% accuracy
        - **Model Comparison**: Compare Baseline CNN vs. MobileNetV2 transfer learning performance
        - **Real-time Prediction**: Upload images for instant disease classification using Baseline CNN
        - **Interactive Visualization**: Explore training metrics and model performance
        - **User-Friendly Interface**: Designed for researchers, farmers, and agricultural professionals
        
        #### 🌱 Supported Plants:
        Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, 
        Soybean, Squash, Strawberry, and Tomato
        """)
    
    with col2:
        st.markdown("###  Quick Start")
        st.info("""
        **Step 1:** Navigate to 'Disease Prediction'  
        **Step 2:** Upload a clear leaf image  
        **Step 3:** Get instant diagnosis  
        **Step 4:** View confidence scores
        """)
        
        st.markdown("### 📊 Model Details")
        st.success("""
        **Architecture:** Custom 4-Block CNN  
        **Best Accuracy:** 98.03%  
        **Epochs:** 20  
        **Batch Size:** 32  
        **Optimizer:** Adam  
        **Image Size:** 224×224×3
        """)
    
    st.markdown("---")
    st.markdown("### 🎓 Research & Educational Use")
    st.write("""
    This platform is designed for both practical agricultural applications and educational purposes. 
    It demonstrates state-of-the-art computer vision techniques in precision agriculture, 
    helping to promote sustainable farming practices through early disease detection.
    """)

# ===============================
# PAGE: MODEL COMPARISON
# ===============================
elif page == "📊 Model Comparison":
    st.title("📊 Model Architecture Comparison")
    
    if baseline_history is None:
        st.warning("⚠️ Baseline model history not loaded. Cannot display comparison.")
        st.stop()
    
    if mobilenet_history is None:
        st.warning("⚠️ MobileNetV2 history not available. Showing baseline metrics only.")
    
    # Performance Metrics Comparison
    st.markdown("###  Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧩 Baseline CNN (Active Model)")
        st.markdown("""
        <div class="metric-card">
        <h4>Custom 4-Block CNN Architecture</h4>
        <ul>
            <li><b>Total Parameters:</b> 27,010,886</li>
            <li><b>Best Val Accuracy:</b> 98.03%</li>
            <li><b>Best Val Loss:</b> 0.0579</li>
            <li><b>Precision:</b> 98.25%</li>
            <li><b>Recall:</b> 97.97%</li>
            <li><b>Training Time:</b> ~45 min/epoch</li>
            <li><b>Status:</b> ✅ Used for predictions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if mobilenet_history is not None:
            st.markdown("#### ⚡ MobileNetV2 Transfer Learning (Reference)")
            st.markdown("""
            <div class="metric-card">
            <h4>Pre-trained on ImageNet</h4>
            <ul>
                <li><b>Total Parameters:</b> 3,058,022</li>
                <li><b>Trainable Parameters:</b> 798,502</li>
                <li><b>Best Val Accuracy:</b> 95.04%</li>
                <li><b>Best Val Loss:</b> 0.1490</li>
                <li><b>Precision:</b> 95.91%</li>
                <li><b>Recall:</b> 93.92%</li>
                <li><b>Training Time:</b> ~15 min/epoch</li>
                <li><b>Status:</b> 📊 Comparison only</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚠️ MobileNetV2 history not available for comparison")
    
    if mobilenet_history is not None:
        # Visual Comparison
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ **Winner: Baseline CNN**")
            st.write("• Higher accuracy (+2.99%)")
            st.write("• Better precision & recall")
            st.write("• More stable training")
            st.write("• Used for all predictions")
        
        with col2:
            st.info("⚡ **Advantage: MobileNetV2**")
            st.write("• 89% fewer parameters")
            st.write("• 3× faster training")
            st.write("• Better for mobile deployment")
            st.write("• Reference model only")
        
        st.markdown("---")
        
        # Interactive Metrics Chart
        st.markdown("### 📈 Metrics Comparison Chart")
        
        metrics = ['Accuracy', 'Precision', 'Recall']
        baseline_vals = [98.03, 98.25, 97.97]
        mobilenet_vals = [95.04, 95.91, 93.92]
        
        fig = go.Figure(data=[
            go.Bar(name='Baseline CNN (Active)', x=metrics, y=baseline_vals, marker_color='#27ae60'),
            go.Bar(name='MobileNetV2 (Reference)', x=metrics, y=mobilenet_vals, marker_color='#3498db')
        ])
        fig.update_layout(
            title='Model Performance Comparison (%)',
            yaxis_title='Percentage',
            barmode='group',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# PAGE: DISEASE PREDICTION
# ===============================
elif page == "🔍 Disease Prediction":
    st.title("🔍 Plant Disease Prediction")
    st.markdown("### Upload a leaf image to detect diseases")
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please ensure 'baseline_cnn_best.h5' is available.")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file is not None:
            image_data = uploaded_file.read()
            st.image(image_data, caption="📷 Uploaded Image", use_column_width=True)
            
            st.info(" **Active Model:** Baseline CNN (98.03% accuracy)")
    
    with col2:
        if uploaded_file is not None:
            st.markdown("####  Prediction Results")
            
            with st.spinner('🔄 Analyzing image with Baseline CNN...'):
                # Preprocess image
                img = image.load_img(io.BytesIO(image_data), target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = baseline_model.predict(img_array, verbose=0)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)
                
                # Format class name
                plant, disease = predicted_class.split('___')
                disease = disease.replace('_', ' ').title()
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                🌿 {plant}<br>
                🔬 {disease}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown("#### 📊 Confidence Score")
            st.progress(float(confidence))
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            # Top 3 predictions
            st.markdown("#### 🔝 Top 3 Predictions")
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            for idx in top_3_idx:
                plant_name, disease_name = class_names[idx].split('___')
                prob = prediction[0][idx] * 100
                st.write(f"**{plant_name}** - {disease_name.replace('_', ' ')}: {prob:.2f}%")
            
            st.success(f"✅ **Model:** Baseline CNN (98.03% accuracy)")
        else:
            st.info("👆 Please upload an image to get started")
            st.markdown("""
            #### 💡 Tips for Best Results:
            - Use clear, well-lit images
            - Focus on the leaf with visible symptoms
            - Avoid blurry or dark images
            - Ensure the leaf fills most of the frame
            """)

# ===============================
# PAGE: TRAINING INSIGHTS
# ===============================
elif page == "📈 Training Insights":
    st.title("📈 Training Performance Analysis")
    
    if baseline_history is None:
        st.warning("⚠️ History files not loaded.")
        st.stop()
    
    # Tabs for different models
    if mobilenet_history is not None:
        tab1, tab2 = st.tabs(["🧩 Baseline CNN (Active Model)", "⚡ MobileNetV2 (Reference)"])
    else:
        tab1, = st.tabs(["🧩 Baseline CNN (Active Model)"])
        tab2 = None
    
    with tab1:
        st.markdown("### Baseline CNN Training History")
        st.info("✅ This is the active model used for all predictions")
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy Over Epochs', 'Loss Over Epochs')
        )
        
        epochs = range(1, len(baseline_history['accuracy']) + 1)
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=baseline_history['accuracy'], 
                      name='Train Accuracy', line=dict(color='#27ae60', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=baseline_history['val_accuracy'], 
                      name='Val Accuracy', line=dict(color='#e74c3c', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=baseline_history['loss'], 
                      name='Train Loss', line=dict(color='#27ae60', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(epochs), y=baseline_history['val_loss'], 
                      name='Val Loss', line=dict(color='#e74c3c', width=2)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Val Acc", f"{max(baseline_history['val_accuracy'])*100:.2f}%")
        with col2:
            st.metric("Best Val Loss", f"{min(baseline_history['val_loss']):.4f}")
        with col3:
            st.metric("Final Train Acc", f"{baseline_history['accuracy'][-1]*100:.2f}%")
        with col4:
            st.metric("Final Val Acc", f"{baseline_history['val_accuracy'][-1]*100:.2f}%")
    
    if tab2 is not None and mobilenet_history is not None:
        with tab2:
            st.markdown("### MobileNetV2 Training History")
            st.warning("📊 This model is for comparison only - not used for predictions")
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy Over Epochs', 'Loss Over Epochs')
            )
            
            epochs = range(1, len(mobilenet_history['accuracy']) + 1)
            
            fig.add_trace(
                go.Scatter(x=list(epochs), y=mobilenet_history['accuracy'], 
                          name='Train Accuracy', line=dict(color='#3498db', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=mobilenet_history['val_accuracy'], 
                          name='Val Accuracy', line=dict(color='#e67e22', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(epochs), y=mobilenet_history['loss'], 
                          name='Train Loss', line=dict(color='#3498db', width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=mobilenet_history['val_loss'], 
                          name='Val Loss', line=dict(color='#e67e22', width=2)),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=2)
            
            fig.update_layout(height=500, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Val Acc", f"{max(mobilenet_history['val_accuracy'])*100:.2f}%")
            with col2:
                st.metric("Best Val Loss", f"{min(mobilenet_history['val_loss']):.4f}")
            with col3:
                st.metric("Final Train Acc", f"{mobilenet_history['accuracy'][-1]*100:.2f}%")
            with col4:
                st.metric("Final Val Acc", f"{mobilenet_history['val_accuracy'][-1]*100:.2f}%")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>🌿 <b>Plant Disease Classification System</b> | Powered by TensorFlow & Streamlit</p>
    <p>© 2025 | Developed for Agricultural Innovation and Research</p>
</div>
""", unsafe_allow_html=True)