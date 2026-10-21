"""
NeuroScan AI: Standalone MRI Prediction Script
Loads the trained MobileNetV2 model and predicts a given brain MRI image.

Usage:
    python predict_sample.py --image static/samples/glioma_sample.jpg
    python predict_sample.py
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

DEFAULT_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "models", "brain_tumor_model.keras"),
    os.path.join(os.path.dirname(__file__), "models", "brain_tumor_mobilenetv2.keras")
]
DEFAULT_CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "class_names.json")
FALLBACK_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def load_class_names(class_file=DEFAULT_CLASS_NAMES_PATH):
    """Load class names from JSON or fallback to defaults."""
    if os.path.exists(class_file):
        try:
            with open(class_file, "r") as f:
                classes = json.load(f)
                if isinstance(classes, list) and len(classes) == 4:
                    return [c.replace("_", " ").title() if c.lower() not in ["notumor", "no tumor"] else "No Tumor" for c in classes]
        except Exception as e:
            print(f"[!] Warning: Could not read {class_file}: {e}")
    return FALLBACK_CLASSES

def preprocess_image(image_path, target_size=(224, 224), apply_scaling=True):
    """
    Loads and normalizes an MRI image according to MobileNetV2 standards.
    Scales pixel values to [-1.0, 1.0].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: '{image_path}'")
        
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    
    if apply_scaling:
        # MobileNetV2 normalization: scale to [-1, 1]
        arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

def predict(image_path, model_path=None):
    """Runs prediction on a single MRI image using the trained Keras model."""
    print("=" * 65)
    print(" NeuroScan AI: Brain MRI Inference CLI")
    print("=" * 65)
    print(f" Target Image : {image_path}")

    # Determine model file to use
    selected_model_path = None
    if model_path and os.path.exists(model_path):
        selected_model_path = model_path
    else:
        for candidate in DEFAULT_MODEL_PATHS:
            if os.path.exists(candidate):
                selected_model_path = candidate
                break

    class_names = load_class_names()

    # Load Model & Predict
    if selected_model_path:
        print(f" Model Loaded : {selected_model_path}")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(selected_model_path)
            has_preprocess = any(
                "preprocess" in getattr(layer, "name", "").lower() or "rescaling" in getattr(layer, "name", "").lower()
                for layer in getattr(model, "layers", [])
            )
            input_tensor = preprocess_image(image_path, apply_scaling=not has_preprocess)
            raw_preds = model.predict(input_tensor, verbose=0)[0]
        except Exception as e:
            print(f"[!] Error running Keras model: {e}")
            sys.exit(1)
    else:
        print(" [!] Note: No trained Keras weight file found at models/brain_tumor_model.keras.")
        print("     To train the model: python train.py --data_dir dataset/")
        print("     Running with demonstration heuristic classifier...")
        import model as demo_model
        result = demo_model.predict_tumor(image_path)
        print("\n" + "-" * 65)
        print(f" PREDICTION RESULT: {result['predicted_class'].upper()}")
        print(f" CONFIDENCE SCORE : {result['confidence']}%")
        print("-" * 65)
        print(" Softmax Class Probabilities:")
        for cls_name, prob in result['probabilities'].items():
            bar_len = int(prob / 2.5)
            print(f"   - {cls_name:12}: {prob:6.2f}% | {'#' * bar_len}")
        print("=" * 65)
        return

    # Process Keras model output
    pred_idx = int(np.argmax(raw_preds))
    confidence = float(raw_preds[pred_idx] * 100.0)
    predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"

    print("\n" + "-" * 65)
    print(f" PREDICTION RESULT: {predicted_class.upper()}")
    print(f" CONFIDENCE SCORE : {confidence:.2f}%")
    print("-" * 65)
    print(" Softmax Class Probabilities:")
    for i, prob in enumerate(raw_preds):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        percentage = float(prob * 100.0)
        bar_len = int(percentage / 2.5)
        marker = " <== (Highest)" if i == pred_idx else ""
        print(f"   - {name:12}: {percentage:6.2f}% | {'#' * bar_len}{marker}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Tumor Class for a Brain MRI Image")
    default_sample = os.path.join(os.path.dirname(__file__), "static", "samples", "glioma_sample.jpg")
    parser.add_argument("--image", type=str, default=default_sample, help="Path to input brain MRI scan")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .keras model")
    args = parser.parse_args()

    predict(image_path=args.image, model_path=args.model)
