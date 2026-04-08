import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# -------------------------
# 1) Load models once
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

type_model_path = os.path.join(BASE_DIR, "models", "type_check_TL.h5")
chest_model_path = os.path.join(BASE_DIR, "models", "Chest_model.h5")
mri_model_path = os.path.join(BASE_DIR, "models", "MRI_model.h5")
cells_model_path = os.path.join(BASE_DIR, "models", "cells_model.h5")
cancer_model_path = os.path.join(BASE_DIR, "models", "Blood_Cancer_model.h5")

type_model = load_model(type_model_path)
chest_model = load_model(chest_model_path)
mri_model = load_model(mri_model_path)
cells_model = load_model(cells_model_path)
cancer_model = load_model(cancer_model_path)

type_classes = {
    0: "Blood Cells",
    1: "Cancer Cells",
    2: "Chest X ray",
    3: "MRI"
}

chest_classes = {
    0: "NORMAL",
    1: "PNEUMONIA",
}

mri_classes = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

cells_classes = {
    0: "EOSINOPHIL",
    1: "LYMPHOCYTE",
    2: "MONOCYTE",
    3: "NEUTROPHIL"
}

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

    # 1) Type prediction
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
    detail_probs = None        # main detail probs (cells / xray / mri / cancer)
    cancer_probs_dict = None   # cancer subtype probs (for blood or cancer cells)

    # Out-of-distribution rule
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

            # Only use the cancer_probs section (no duplicate)
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
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)