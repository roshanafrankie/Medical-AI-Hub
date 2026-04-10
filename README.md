# Med-AI: Hierarchical Medical Image Diagnosis Hub

Med-AI is a full-stack medical diagnostic application that utilizes a **Cascading Neural Network Architecture** to analyze various medical specimens. Unlike traditional single-purpose models, this system first triages the image modality and then routes it to a specialized "expert" model for granular diagnosis.

## 🧠 System Architecture
The system follows a hierarchical decision-making process to ensure high diagnostic accuracy and prevent cross-domain errors.

1.  **Level 1 (The Router):** A `Type_Model` identifies the specimen category (Chest X-ray, MRI, Blood Cells, or Cancer Cells).
2.  **Level 2 (The Expert):** The image is routed to one of four specialized models:
    * **Chest X-Ray Model:** Distinguishes between Normal and Pneumonia.
    * **MRI Model:** Classifies brain tumors (Glioma, Meningioma, Pituitary, or No Tumor).
    * **Blood Cells Model:** Identifies cell types (Eosinophil, Lymphocyte, Monocyte, Neutrophil).
    * **Cancer Cells Model:** Detects malignancy and identifies subtypes (Benign, Pre-B, Pro-B, early Pre-B).



## 📊 Technical Specifications
* **Base Architecture:** MobileNetV2 (Pre-trained on ImageNet).
* **Input Resolution:** 224x224 pixels across all models for consistent feature extraction.
* **Optimization:** Adam Optimizer with dynamic learning rate reduction (`ReduceLROnPlateau`).
* **Deployment:** Flask-based web interface with real-time probability visualization.

## 📂 Datasets Used
This project integrates multiple high-quality datasets to cover a broad spectrum of medical imaging:

* **Chest X-ray:** [Kaggle - Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **MRI:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* **Cancer Cells:** [Kaggle - Blood Cell Cancer (ALL)](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
* **Blood Cells:** [Kaggle - Blood Cells Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
* **Combined New Dataset:** [Google Drive - Combined Research Data](https://drive.google.com/drive/folders/1tohNKrzLkQZX4ybgt590U9NkTIKwUWZB?usp=sharing)

## 🛠️ Project Structure
```text
Medical_Image_Recognition/
├── app.py              # Flask Backend & Routing Logic
├── models/             # Contains all 5 .h5 Model Files
├── notebooks/          # Training pipelines for each model
├── static/             # CSS and Uploaded Images
└── templates/          # index.html (Frontend)
=======
# MED-AI Hub v2.0: Advanced Medical Imaging Diagnostic System 🏥

An intelligent medical imaging dashboard built with **Python**, **Streamlit**, and **Tensorflow**. This system uses a hierarchical deep learning approach to classify medical specimens including Chest X-rays, MRI scans, and Hematological samples.

## 🚀 Live Demo
[Insert your Streamlit Cloud Link Here]

## 🛠️ System Architecture
The application employs a **Router-Expert** architecture to maximize diagnostic accuracy:
1. **Triage Model (MobileNetV2):** Analyzes the input to determine the modality (Chest, MRI, or Blood).
2. **Expert Models:** Once the modality is identified, the image is routed to a specialized model for fine-grained diagnosis:
   - **Chest Expert:** Normal vs. Pneumonia.
   - **MRI Expert:** Glioma, Meningioma, Pituitary, or No Tumor.
   - **Hematology Experts:** Identifies cell types (Eosinophil, Lymphocyte, etc.) and performs a **Malignancy Check** for Cancer Cells.

## ✨ Key Features
- **Dynamic Sensitivity Threshold:** Users can adjust the "Confidence Gate" (0.0 - 1.0) to control diagnostic sensitivity.
- **Hierarchical Inference:** Intelligent routing ensures images are processed by the correct specialized model.
- **Clinical Dashboard:** Interactive visualizations using Plotly including Morphology Donut Charts and Malignancy Probability Bars.
- **Theme-Adaptive UI:** Seamlessly switches between Light and Dark modes with transparent, high-contrast charts.

## 📊 Datasets Used
* [Original Data Sources] - https://drive.google.com/drive/folders/1RObgiaza_J5VydhyuXll9wjOVJAF0FYy?usp=sharing
* [Combined Research Dataset] - https://drive.google.com/drive/folders/1k2X6yKKwo0Asy5n3wNk9fDpuF322s8qv?usp=drive_link

## 💻 Tech Stack
- **Frontend:** Streamlit (Custom CSS injected for clinical-grade UI)
- **Deep Learning:** TensorFlow / Keras (MobileNetV2 Transfer Learning)
- **Data Viz:** Plotly Express
- **Image Processing:** Pillow (PIL)

## 🔧 Installation & Local Setup
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure model files are in the `models/` directory.
4. Run the app: `streamlit run app.py`

---
**⚠️ Medical Disclaimer:** This tool is for clinical support and educational purposes only. It is not intended for official medical diagnosis.
