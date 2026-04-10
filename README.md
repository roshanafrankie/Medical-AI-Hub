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
