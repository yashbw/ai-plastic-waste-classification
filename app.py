import io
import os
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_PATH = os.path.join(os.getcwd(), "saved_models", "plastic_classifier.keras")
CLASS_NAMES = ["PET", "HDPE", "LDPE", "PP", "PS", "Other Plastic", "Not Plastic Waste"]
NON_PLASTIC_LABEL = "Not Plastic Waste"
CONFIDENCE_THRESHOLD = 0.55

PLASTIC_INFO = {
    "PET": {
        "title": "Polyethylene Terephthalate",
        "description": "Commonly found in beverage bottles and food containers.",
        "recycle": "Recycle in clear plastic bottle streams when clean and empty.",
        "tips": "Rinse and remove caps when possible before recycling."
    },
    "HDPE": {
        "title": "High-Density Polyethylene",
        "description": "Used in milk jugs, detergent bottles, and sturdy containers.",
        "recycle": "Accepted in most curbside plastic recycling programs.",
        "tips": "Empty and rinse containers to reduce contamination."
    },
    "LDPE": {
        "title": "Low-Density Polyethylene",
        "description": "Found in plastic bags, wraps, and squeeze bottles.",
        "recycle": "Often accepted at retail drop-off locations rather than municipal bins.",
        "tips": "Flatten bags and keep them clean and dry."
    },
    "PP": {
        "title": "Polypropylene",
        "description": "Common in food containers, yogurt tubs, bottle caps, and straws.",
        "recycle": "Accepted by some recycling programs; check local rules.",
        "tips": "Scrape out residues and sort by material type before disposal."
    },
    "PS": {
        "title": "Polystyrene",
        "description": "Often seen in foam cups, trays, and takeaway packaging.",
        "recycle": "Not widely accepted in standard recycling; may require special facilities.",
        "tips": "Avoid mixing with general recyclable plastics."
    },
    "Other Plastic": {
        "title": "Other or Mixed Plastic",
        "description": "Includes mixed-material or composite plastic waste that may be hard to recycle.",
        "recycle": "Reduce use when possible and dispose of according to local waste guidance.",
        "tips": "If in doubt, separate it from recyclable plastics to avoid contamination."
    },
    "Not Plastic Waste": {
        "title": "Not Plastic Waste",
        "description": "The detected object does not appear to be plastic waste or the image is too unclear for classification.",
        "recycle": "No plastic recycling required. Reassess the object or try another image.",
        "tips": "Use a clear image of the actual object and avoid background clutter."
    },
    "Uncertain": {
        "title": "Uncertain / Not Clear",
        "description": "The image could not be confidently matched to a specific plastic type.",
        "recycle": "Use caution and check the item label or local waste system.",
        "tips": "Capture a clearer image or try a different angle for better classification."
    }
}


def load_model():
    if tf is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None


def preprocess_image_bytes(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def classify_image(image_bytes: bytes):
    model = load_model()
    if model is None:
        return {
            "success": False,
            "message": "The trained model is not available yet. Please train the model using the training pipeline first.",
            "label": "Model Missing",
            "confidence": 0.0,
            "metadata": PLASTIC_INFO["Uncertain"],
        }

    processed = preprocess_image_bytes(image_bytes)
    probabilities = model.predict(processed, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = "Uncertain"
        confidence = min(float(confidence), 100.0)

    metadata = PLASTIC_INFO.get(predicted_label, PLASTIC_INFO["Uncertain"])
    return {
        "success": True,
        "message": "Prediction completed successfully.",
        "label": predicted_label,
        "confidence": round(float(confidence * 100), 2),
        "probabilities": [round(float(p), 4) for p in probabilities],
        "metadata": metadata,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image was received."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"success": False, "message": "No image was selected."}), 400

    image_bytes = image_file.read()
    result = classify_image(image_bytes)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting Flask app on http://0.0.0.0:{port}")
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
