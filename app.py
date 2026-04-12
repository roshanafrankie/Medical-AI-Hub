import os
import logging

# 1. SILENCE TENSORFLOW WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px

# 2. PAGE CONFIGURATION
st.set_page_config(page_title="NEURAL MED HUB", page_icon="logo.png", layout="wide")

# 3. ADVANCED UI STYLING (Mobile Responsive Fixes)
st.markdown("""
    <style>
    /* Fixed Sidebar Width for Desktop, Flexible for Mobile */
    @media (min-width: 768px) {
        [data-testid="stSidebar"] {
            width: 310px !important;
        }
    }

    /* Force all elements in sidebar to align center */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        align-items: center;
        text-align: center;
    }

    .main-title-block { text-align: center; margin-bottom: 2rem; }
    .main-title-block h1 { font-size: clamp(1.5rem, 5vw, 2.2rem); margin-bottom: 0; }
    .main-title-block p { color: #64748b; font-size: 1rem; }
    
    .instruction-card {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.2);
        margin-bottom: 20px;
        font-size: 0.9rem;
        text-align: left; /* Keep instructions readable */
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #2563eb;
        color: white;
        font-weight: 700;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# 4. SIDEBAR CONFIGURATION
with st.sidebar:
    # Direct centering for the logo
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    
    # Adding text-align:center directly back here
    st.markdown("<h1 style='text-align:center; font-size:1.4rem; margin-top:4px;'>NEURAL MED v2.0</h1>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Analysis Settings")
    st.markdown("""
    <div class="instruction-card">
    <strong>Diagnostic Sensitivity:</strong><br>
    Sets the confidence gate for AI findings.
    <ul>
    <li><b>High (0.8+):</b> Clinical confirmation.</li>
    <li><b>Low (0.4+):</b> Initial screening.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.6, help="Min probability for confirmed diagnosis.")
    
    st.markdown("---")
    st.warning("**⚠️ Medical Disclaimer:** This AI assistant provides diagnostic suggestions. Always consult a certified medical professional for official diagnosis.")

# 5. HEADER
st.markdown("""
<div class="main-title-block">
  <h1>Advanced Medical AI Hub</h1>
  <p>Hierarchical Diagnostic System | Clinical Support Dashboard</p>
</div>
""", unsafe_allow_html=True)

# 6. MODEL LOADING
@st.cache_resource
def load_all_models():
    base_path = "models"
    return {
        "type": load_model(os.path.join(base_path, "type_check_TL.h5")),
        "chest": load_model(os.path.join(base_path, "Chest_model.h5")),
        "mri": load_model(os.path.join(base_path, "MRI_model.h5")),
        "cells": load_model(os.path.join(base_path, "cells_model.h5")),
        "cancer": load_model(os.path.join(base_path, "Blood_Cancer_model.h5"))
    }

with st.spinner("Initializing Clinical Intelligence..."):
    models = load_all_models()

classes = {
    "type": {0: "Blood Cells", 1: "Cancer Cells", 2: "Chest X-Ray", 3: "MRI"},
    "chest": {0: "NORMAL", 1: "PNEUMONIA"},
    "mri": {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"},
    "cells": {0: "EOSINOPHIL", 1: "LYMPHOCYTE", 2: "MONOCYTE", 3: "NEUTROPHIL"},
    "cancer": {0: "Benign", 1: "[Malignant] Pre-B", 2: "[Malignant] Pro-B", 3: "[Malignant] early Pre-B"}
}

# 7. DASHBOARD LAYOUT
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("📤 Upload Specimen")
    uploaded_file = st.file_uploader("Select medical image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    analysis_triggered = False
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
        if st.button("RUN DIAGNOSIS"):
            analysis_triggered = True

with col2:
    if analysis_triggered:
        st.subheader("🔬 Clinical Findings")
        with st.status("Analyzing Specimen...", expanded=False) as status:
            img_resized = img.resize((224, 224)).convert('RGB') 
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Step 1: Router
            type_probs = models["type"].predict(img_array)[0]
            type_label = classes["type"][np.argmax(type_probs)]
            
            # Step 2: Expert Analysis
            if type_label == "Chest X-Ray":
                expert_probs = models["chest"].predict(img_array)[0]
                expert_classes = classes["chest"]
            elif type_label == "MRI":
                expert_probs = models["mri"].predict(img_array)[0]
                expert_classes = classes["mri"]
            else:
                cell_probs = models["cells"].predict(img_array)[0]
                cancer_probs = models["cancer"].predict(img_array)[0]
                expert_probs = None
            status.update(label="Analysis Complete", state="complete")

        m1, m2 = st.columns(2)
        m1.metric("Domain", type_label)
        m2.metric("Triage Confidence", f"{np.max(type_probs):.1%}")
        st.divider()

        if type_label in ["Chest X-Ray", "MRI"]:
            res_idx = np.argmax(expert_probs)
            res_conf = expert_probs[res_idx]
            res_label = expert_classes[res_idx]

            if res_conf >= threshold:
                st.success(f"Final Finding: **{res_label}**")
            else:
                st.warning(f"Low Confidence: **{res_label}** ({res_conf:.1%})")

            fig = px.bar(x=expert_probs, y=list(expert_classes.values()), orientation="h", 
                         color=expert_probs, color_continuous_scale="Blues",
                         labels={'x': 'Confidence Score', 'y': 'Diagnosis'})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        elif type_label in ["Blood Cells", "Cancer Cells"]:
            cell_label = classes["cells"][np.argmax(cell_probs)]
            malig_idx = np.argmax(cancer_probs)
            malig_label = classes["cancer"][malig_idx]
            malig_conf = cancer_probs[malig_idx]

            if malig_conf >= threshold:
                st.success(f"Pathology: {cell_label} | {malig_label}")
            else:
                st.warning(f"Pathology: {cell_label} | Indeterminate ({malig_conf:.1%})")

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                fig1 = px.pie(values=cell_probs, names=list(classes["cells"].values()), title="<b>Morphology</b>", hole=0.4)
                fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=20, r=20, t=50, b=80),
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
                fig1.update_traces(textposition='inside', textinfo='percent')
                st.plotly_chart(fig1, use_container_width=True)
                
            with c2:
                fig2 = px.bar(x=cancer_probs, y=list(classes["cancer"].values()), orientation="h", 
                              title="<b>Malignancy Status</b>", color=cancer_probs, color_continuous_scale="Blues",
                              labels={'x': 'Probability (%)', 'y': 'Classification'})
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)