import os
# Silence TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="MED AI Hub", page_icon="logo.png", layout="wide")

<<<<<<< HEAD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
=======
# 2. Styling: Lock Sidebar Width & Center Main Titles
st.markdown("""
    <style>
    /* Sidebar width + disable resize */
    [data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }
    [data-testid="stSidebarResizer"] {
        display: none !important;
    }
>>>>>>> 838b3c1 (Updated UI, fixed RGB color depth error, and added Plotly charts)

    .main-title-block {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-title-block h1 {
        margin: 0;
        font-size: 2rem;
    }
    .main-title-block p {
        margin: 0.2rem 0 0 0;
        color: grey;
        font-size: 0.95rem;
    }

    .instruction-card {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.2);
        margin-bottom: 20px;
        font-size: 0.9rem;
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

# 3. Sidebar Configuration
with st.sidebar:
    left, center, right = st.columns([1, 2, 1])

    with center:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=120)
        st.markdown(
            "<h1 style='text-align:center; font-size:1.4rem; margin-top:4px;'>MED AI v2.0</h1>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("⚙️ Analysis Settings")
    
    st.markdown("""
    <div class="instruction-card">
    <strong>Diagnostic Sensitivity:</strong><br>
    Sets the confidence gate for AI findings.
    <ul>
    <li><b>High (0.8+):</b> Use for clinical confirmation.</li>
    <li><b>Low (0.4+):</b> Use for initial screening.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

<<<<<<< HEAD
cancer_classes = {
    0: "Benign",
    1: "[Malignant] Pre-B",
    2: "[Malignant] Pro-B",
    3: "[Malignant] early Pre-B"
}

IMG_SIZE_TYPE = (224, 224)
IMG_SIZE_OTHER = (224, 224)

TYPE_CONF_THRESHOLD = 0.6
DETAIL_CONF_THRESHOLD = 0.6


def load_and_preprocess(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected")

    upload_path = os.path.join("static", "uploads")
    os.makedirs(upload_path, exist_ok=True)
    img_path = os.path.join(upload_path, file.filename)
    file.save(img_path)

    x_type = load_and_preprocess(img_path, IMG_SIZE_TYPE)
    type_probs = type_model.predict(x_type)[0]

    sorted_indices = np.argsort(type_probs)[::-1]
    best_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])

    best_conf = float(type_probs[best_idx])
    second_conf = float(type_probs[second_idx])
    gap = best_conf - second_conf

    type_label = type_classes.get(best_idx, "Unknown")
    type_conf = best_conf

    detail_label = "Not predicted"
    detail_probs = None        
    cancer_probs_dict = None   
    
    if best_conf < 0.7 or gap < 0.25:
        type_label = "Unknown / Not a recognized medical image"
    else:
        if type_label == "Chest X ray":
            x = load_and_preprocess(img_path, IMG_SIZE_OTHER)
            probs = chest_model.predict(x)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if conf < DETAIL_CONF_THRESHOLD:
                detail_label = "Uncertain chest X-ray finding"
            else:
                detail_label = chest_classes[idx]
            detail_probs = {chest_classes[i]: float(probs[i]) for i in range(len(probs))}

        elif type_label == "MRI":
            x = load_and_preprocess(img_path, IMG_SIZE_OTHER)
            probs = mri_model.predict(x)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if conf < DETAIL_CONF_THRESHOLD:
                detail_label = "Uncertain MRI finding"
            else:
                detail_label = mri_classes[idx]
            detail_probs = {mri_classes[i]: float(probs[i]) for i in range(len(probs))}

        elif type_label == "Blood Cells":
            x = load_and_preprocess(img_path, IMG_SIZE_OTHER)
            cell_probs = cells_model.predict(x)[0]
            cell_idx = int(np.argmax(cell_probs))
            cell_conf = float(cell_probs[cell_idx])

            cell_probs_dict = {cells_classes[i]: float(cell_probs[i]) for i in range(len(cell_probs))}
            cancer_probs_dict = None

            if cell_conf < DETAIL_CONF_THRESHOLD:
                detail_label = "Uncertain blood cell type"
            else:
                cell_label = cells_classes[cell_idx]
                # run cancer model
                cancer_probs = cancer_model.predict(x)[0]
                cancer_idx = int(np.argmax(cancer_probs))
                cancer_conf = float(cancer_probs[cancer_idx])
                cancer_probs_dict = {cancer_classes[i]: float(cancer_probs[i]) for i in range(len(cancer_probs))}

                if cancer_conf < DETAIL_CONF_THRESHOLD:
                    detail_label = f"{cell_label} (Uncertain cancer status)"
                else:
                    detail_label = f"{cell_label} - {cancer_classes[cancer_idx]}"

            detail_probs = cell_probs_dict

        elif type_label == "Cancer Cells":
            x = load_and_preprocess(img_path, IMG_SIZE_OTHER)
            cancer_probs = cancer_model.predict(x)[0]
            cancer_idx = int(np.argmax(cancer_probs))
            cancer_conf = float(cancer_probs[cancer_idx])

            cancer_probs_dict = {cancer_classes[i]: float(cancer_probs[i]) for i in range(len(cancer_probs))}
            if cancer_conf < DETAIL_CONF_THRESHOLD:
                detail_label = "Uncertain cancer cell type"
            else:
                detail_label = cancer_classes[cancer_idx]
            detail_probs = None

    return render_template(
        "index.html",
        image_path=img_path,
        type_label=type_label,
        type_confidence=type_conf,
        detail_label=detail_label,
        detail_probs=detail_probs,
        cancer_probs=cancer_probs_dict,
        type_probs={type_classes[i]: float(type_probs[i]) for i in range(len(type_probs)) if i in type_classes}
=======
    threshold = st.slider(
        "Sensitivity Threshold", 
        0.0, 1.0, 0.6,
        help="The minimum probability required for a confirmed diagnosis."
    )
    
    st.markdown("---")
    st.warning(
        "**⚠️ Medical Disclaimer:** This tool provides diagnostic suggestions. "
        "Always consult a certified medical professional for official diagnosis."
>>>>>>> 838b3c1 (Updated UI, fixed RGB color depth error, and added Plotly charts)
    )

# 4. Main Header
st.markdown("""
<div class="main-title-block">
  <h1>Advanced Medical AI Hub</h1>
  <p>Hierarchical Diagnostic System | Clinical Support Dashboard</p>
</div>
""", unsafe_allow_html=True)

<<<<<<< HEAD
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
=======
# 5. Optimized Model Loading
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

# 6. Dashboard Layout
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("📤 Upload Specimen")
    uploaded_file = st.file_uploader(
        "Select medical image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    
    analysis_triggered = False
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width='stretch')
        if st.button("RUN DIAGNOSIS"):
            analysis_triggered = True

with col2:
    if analysis_triggered:
        st.subheader("🔬 Clinical Findings")
        with st.status("Analyzing Specimen...", expanded=False) as status:
            # FIX: Convert to RGB to ensure 3 channels for MobileNetV2
            img_resized = img.resize((224, 224)).convert('RGB') 
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Step 1: Router
            type_probs = models["type"].predict(img_array)[0]
            type_idx = np.argmax(type_probs)
            type_label = classes["type"][type_idx]
            
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

        # Visualizing Results
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
                st.warning(f"Low Confidence Finding: **{res_label}** ({res_conf:.1%})")

            fig = px.bar(
                x=expert_probs,
                y=list(expert_classes.values()),
                orientation="h",
                color=expert_probs,
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="grey",
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, width='stretch')

        elif type_label in ["Blood Cells", "Cancer Cells"]:
            cell_label = classes["cells"][np.argmax(cell_probs)]
            malig_idx = np.argmax(cancer_probs)
            malig_label = classes["cancer"][malig_idx]
            malig_conf = cancer_probs[malig_idx]

            if malig_conf >= threshold:
                st.success(f"Pathology: {cell_label} | {malig_label}")
            else:
                st.warning(
                    f"Pathology: {cell_label} | Indeterminate Malignancy ({malig_conf:.1%})"
                )

            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.pie(
                    values=cell_probs,
                    names=list(classes["cells"].values()),
                    title="Morphology",
                    hole=0.4 # Modern donut style
                )
                fig1.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    # Legend moved to bottom to prevent squashing the pie
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=30, b=80),
                    height=400, # Increased height to prevent cuts
                )
                # Keep percentages inside the slices
                fig1.update_traces(textposition='inside', textinfo='percent')
                st.plotly_chart(fig1, width='stretch')
                
            with c2:
                fig2 = px.bar(
                    x=cancer_probs,
                    y=list(classes["cancer"].values()),
                    orientation="h",
                    title="<b>Malignancy Status</b>",
                    color=cancer_probs,
                    color_continuous_scale="Blues",
                    # ADD THIS LINE BELOW TO CHANGE AXIS NAMES
                    labels={'x': 'Detection Probability', 'y': 'Cell Classification'} 
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10),
                    font_color="grey",
                    coloraxis_showscale=False
                )
                # This forces the X-axis to show the specific name you chose
                fig2.update_xaxes(title_text="Probability (%)") 
                fig2.update_yaxes(title_text="Diagnosis")
                st.plotly_chart(fig2, width='stretch')
>>>>>>> 838b3c1 (Updated UI, fixed RGB color depth error, and added Plotly charts)
