from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import time
import uuid
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static"
)

# Load model
MODEL_PATH = os.path.join(BASE_DIR, "trained.h5")
model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# Create static folder if missing
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Auto-delete generated files
def delete_file(path):
    time.sleep(60)

    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"Could not delete {path}: {e}")


# Image preprocessing
def preprocess_image(image_path, target_size=(300, 300)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)

    img_array = np.array(img) / 255.0

    return img, img_array


# Required for LIME binary classification
def lime_predict(images):
    images = np.array(images)

    # normalize if needed
    images = images.astype(np.float32)

    if images.max() > 1:
        images = images / 255.0

    preds = model.predict(images, verbose=0)

    return np.hstack([
        1 - preds,
        preds
    ])


# Generate LIME explanation
def generate_lime_explanation(image_array):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_array,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=200
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    temp = (temp * 255).astype(np.uint8)

    lime_image_result = mark_boundaries(
        temp,
        mask
    )

    return lime_image_result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only JPG, JPEG and PNG files allowed"
        }), 400

    unique_filename = f"{uuid.uuid4()}.jpg"

    file_path = os.path.join(
        STATIC_DIR,
        unique_filename
    )

    try:

        file.save(file_path)

        original_image, image_array = preprocess_image(
            file_path
        )

        prediction = model.predict(
            np.expand_dims(
                image_array,
                axis=0
            ),
            verbose=0
        )

        prob = float(prediction[0][0])

        predicted_class = (
            "Pneumonia"
            if prob > 0.5
            else "Normal"
        )

        confidence = max(
            prob,
            1 - prob
        )

        lime_explanation = generate_lime_explanation(
            image_array
        )

        lime_filename = (
            str(uuid.uuid4()) +
            "_lime.jpg"
        )

        lime_path = os.path.join(
            STATIC_DIR,
            lime_filename
        )

        plt.imsave(
            lime_path,
            np.clip(
                lime_explanation,
                0,
                1
            )
        )

        # cleanup in background
        threading.Thread(
            target=delete_file,
            args=(file_path,),
            daemon=True
        ).start()

        threading.Thread(
            target=delete_file,
            args=(lime_path,),
            daemon=True
        ).start()

        timestamp = int(
            time.time()
        )

        return jsonify({

            "prediction":
            predicted_class,

            "confidence":
            confidence,

            "image_url":
            f"/static/{unique_filename}?{timestamp}",

            "lime_url":
            f"/static/{lime_filename}?{timestamp}"

        })

    except Exception as e:

        return jsonify({
            "error":
            f"Prediction failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)