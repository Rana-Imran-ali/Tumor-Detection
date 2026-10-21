"""
NeuroScan AI: Brain MRI Model Inference Module
Loads the trained MobileNetV2 Keras model, validates inputs, and executes genuine predictions.
Includes Grad-CAM tumor localization for spatial attention visualization.
Never hardcodes predictions.
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Class definitions & labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_KEYS = ["glioma", "meningioma", "notumor", "pituitary"]

# Medical Disclaimer definition
MEDICAL_DISCLAIMER = (
    "MEDICAL DISCLAIMER: NeuroScan AI is an investigational decision-support tool "
    "designed for educational and research purposes only. It is not approved as a primary "
    "clinical diagnostic device. All AI classification findings must be verified and "
    "correlated with clinical examination, histological data, and official radiologist review."
)

# Custom Exceptions for clean backend error handling
class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained Keras model file does not exist."""
    pass

class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be parsed or is corrupted."""
    pass

# Candidate paths where trained weights and class mappings are stored
MODEL_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "models", "brain_tumor_model.keras"),
    os.path.join(os.path.dirname(__file__), "models", "brain_tumor_mobilenetv2.keras")
]
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "class_names.json")

# Load saved custom class names if present
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            custom_classes = json.load(f)
            if isinstance(custom_classes, list) and len(custom_classes) == 4:
                formatted_classes = []
                for c in custom_classes:
                    clean_c = c.replace("_", " ").title()
                    if clean_c.lower() in ["notumor", "no tumor"]:
                        formatted_classes.append("No Tumor")
                    else:
                        formatted_classes.append(clean_c)
                CLASS_NAMES = formatted_classes
    except Exception as e:
        print(f"Notice: Could not load class names from {CLASS_NAMES_PATH}: {e}")

# Clinical diagnostic context
CLINICAL_DETAILS = {
    "Glioma": {
        "title": "Glioma Tumor Detected",
        "severity": "High / Urgent Evaluation",
        "badge_class": "badge-danger",
        "description": "Gliomas originate in glial cells (astrocytes, oligodendrocytes). They are common primary neoplasms ranging from low-grade to Grade IV Glioblastoma.",
        "mri_features": "Infiltrative borders, central necrotic core, surrounding vasogenic edema on T2/FLAIR, and heterogeneous contrast enhancement.",
        "precautions": "Urgent neurosurgical consultation. Advanced neuroimaging (contrast T1, T2-FLAIR, MR spectroscopy). Monitor mass effect and midline shift."
    },
    "Meningioma": {
        "title": "Meningioma Tumor Detected",
        "severity": "Moderate / Specialist Review",
        "badge_class": "badge-warning",
        "description": "Primary extra-axial tumor arising from the arachnoid cap cells of the meninges. Predominantly benign (>80%), but can exert compressive mass effect.",
        "mri_features": "Extra-axial, well-circumscribed, homogeneous intense contrast enhancement with classic 'dural tail' sign.",
        "precautions": "Neurosurgical and neuro-oncological evaluation. Serial MRI monitoring for slow-growing lesions, or surgical resection if symptomatic."
    },
    "Pituitary": {
        "title": "Pituitary Tumor Detected",
        "severity": "Moderate / Endocrine & Surgical Review",
        "badge_class": "badge-info",
        "description": "Neoplasm situated in the sella turcica originating from pituitary cells. Can cause hormonal abnormalities or bitemporal hemianopsia via optic chiasm compression.",
        "mri_features": "Sellar or suprasellar expansion, possible optic chiasm elevation or sellar floor erosion.",
        "precautions": "Complete serum pituitary hormone panel (prolactin, ACTH, GH, cortisol, TSH). Formal visual field perimetry. Endocrinology consult."
    },
    "No Tumor": {
        "title": "No Tumor Detected (Healthy Scan)",
        "severity": "Normal / Routine Follow-up",
        "badge_class": "badge-success",
        "description": "No definitive abnormal intracranial mass lesion, significant midline shift, or pathologic focal enhancement observed in this MRI slice.",
        "mri_features": "Symmetric cerebral hemispheres, intact gray-white matter differentiation, normal ventricular morphology.",
        "precautions": "No immediate oncology intervention indicated. Correlate with clinical symptom history and complete orthogonal imaging series if headaches persist."
    }
}

_loaded_model = None

def get_model_path():
    """Finds the first existing model candidate path, or None."""
    for path in MODEL_CANDIDATE_PATHS:
        if os.path.exists(path):
            return path
    return None

def load_prediction_model():
    """
    Loads and caches the trained MobileNetV2 Keras model.
    Raises ModelNotFoundError if no model file is present.
    """
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model

    model_path = get_model_path()
    if not model_path:
        target_path = MODEL_CANDIDATE_PATHS[0]
        raise ModelNotFoundError(
            f"Trained model not found at '{target_path}'. "
            "Please train the model first by running: python train.py --data_dir dataset/"
        )

    try:
        import tensorflow as tf
        print(f"[+] Loading trained Keras model from: {model_path}...")
        _loaded_model = tf.keras.models.load_model(model_path)
        return _loaded_model
    except ImportError:
        raise ImportError("TensorFlow is not installed in the environment. Please run: pip install tensorflow")
    except Exception as e:
        raise RuntimeError(f"Failed to load Keras model at {model_path}: {str(e)}")

def preprocess_image(image_path_or_file, apply_scaling=True):
    """
    Validates input image integrity, resizes to 224x224 RGB, and applies MobileNetV2 normalization.
    Raises InvalidImageError if the file is corrupt or unreadable.
    
    Returns:
        numpy.ndarray: Tensor of shape (1, 224, 224, 3) normalized to [-1.0, 1.0].
    """
    if not image_path_or_file:
        raise InvalidImageError("No image path or file provided.")

    try:
        # Integrity verification
        with Image.open(image_path_or_file) as img:
            img.verify()

        # Reopen for transformation
        with Image.open(image_path_or_file) as img:
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize((224, 224))
            arr = np.array(img_resized, dtype=np.float32)

            if apply_scaling:
                # MobileNetV2 preprocessing: scale pixels to [-1.0, 1.0]
                arr = (arr / 127.5) - 1.0
            return np.expand_dims(arr, axis=0)

    except Exception as e:
        raise InvalidImageError(f"File could not be parsed as a valid image: {str(e)}")


def _get_anatomical_location(center_x, center_y, img_size=224):
    """
    Maps pixel coordinates to approximate anatomical brain region names.
    Uses a grid-based approach dividing the brain into 9 quadrant regions.
    
    Args:
        center_x (int): X coordinate of the tumor center (0=left, 224=right)
        center_y (int): Y coordinate of the tumor center (0=top, 224=bottom)
        img_size (int): Image dimension (default 224)
    
    Returns:
        dict: Anatomical location description with region, hemisphere, and quadrant.
    """
    # Normalize to [0, 1]
    nx = center_x / img_size
    ny = center_y / img_size

    # Determine hemisphere (left/right in the image maps to contralateral brain)
    if nx < 0.4:
        hemisphere = "Right Hemisphere"
        h_side = "right"
    elif nx > 0.6:
        hemisphere = "Left Hemisphere"
        h_side = "left"
    else:
        hemisphere = "Midline / Central"
        h_side = "central"

    # Determine vertical region
    if ny < 0.33:
        vertical = "superior"
        v_region = "Superior"
    elif ny < 0.66:
        vertical = "middle"
        v_region = "Mid-level"
    else:
        vertical = "inferior"
        v_region = "Inferior"

    # Map to anatomical regions
    region_map = {
        ("right", "superior"): "Right Frontal / Parietal Lobe",
        ("right", "middle"): "Right Temporal / Insular Region",
        ("right", "inferior"): "Right Temporal / Occipital Lobe",
        ("left", "superior"): "Left Frontal / Parietal Lobe",
        ("left", "middle"): "Left Temporal / Insular Region",
        ("left", "inferior"): "Left Temporal / Occipital Lobe",
        ("central", "superior"): "Frontal Midline / Falx Cerebri",
        ("central", "middle"): "Central Deep Structures (Thalamus / Basal Ganglia)",
        ("central", "inferior"): "Posterior Fossa / Sella Turcica Region",
    }

    region_name = region_map.get((h_side, vertical), "Unspecified Region")

    return {
        "region": region_name,
        "hemisphere": hemisphere,
        "vertical_level": v_region,
        "description": f"The focal activation zone is concentrated in the {region_name.lower()}, "
                        f"within the {hemisphere.lower()} at {v_region.lower()} level of the axial slice.",
        "center_x": center_x,
        "center_y": center_y,
    }


def generate_gradcam(image_path, target_class_idx=None):
    """
    Generates a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap 
    for tumor localization on the MRI image.
    
    Uses the gradients of the target class flowing into the last convolutional 
    layer of MobileNetV2 to produce a coarse spatial attention map highlighting 
    the regions most important for the prediction.
    
    Args:
        image_path (str): Path to the MRI image file.
        target_class_idx (int, optional): Class index to generate heatmap for. 
                                          If None, uses the predicted class.
    
    Returns:
        dict: Contains heatmap_overlay (base64 PNG), bounding_box, 
              anatomical_location, and tumor_detected flag.
    """
    import tensorflow as tf

    loaded_model = load_prediction_model()

    # Determine preprocessing mode
    has_preprocess_layer = any(
        "preprocess" in getattr(layer, "name", "").lower() or "rescaling" in getattr(layer, "name", "").lower()
        for layer in getattr(loaded_model, "layers", [])
    )

    # Preprocess image
    input_tensor = preprocess_image(image_path, apply_scaling=not has_preprocess_layer)
    input_tf = tf.constant(input_tensor, dtype=tf.float32)

    # Find the MobileNetV2 base model layer and its last conv layer
    base_model_layer = None
    for layer in loaded_model.layers:
        if hasattr(layer, 'layers') and len(getattr(layer, 'layers', [])) > 10:
            base_model_layer = layer
            break

    if base_model_layer is None:
        return {"error": "Could not find base convolutional model layer", "tumor_detected": False}

    # Find the last convolutional/activation layer in the base model
    last_conv_layer = None
    for layer in reversed(base_model_layer.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Activation)):
            last_conv_layer = layer
            break
        if 'relu' in layer.name.lower() or 'conv' in layer.name.lower():
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return {"error": "Could not find last convolutional layer", "tumor_detected": False}

    # Build the feature extractor sub-model
    conv_output_model = tf.keras.Model(
        inputs=base_model_layer.input,
        outputs=last_conv_layer.output
    )

    # Build the classifier head sub-model
    # Find layers after the base model in the full model
    head_layers = []
    found_base = False
    for layer in loaded_model.layers:
        if layer == base_model_layer:
            found_base = True
            continue
        if found_base:
            head_layers.append(layer)

    # Construct head model
    head_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = head_input
    for layer in head_layers:
        x = layer(x)
    head_model = tf.keras.Model(inputs=head_input, outputs=x)

    # Extract features from the input up to the base model input
    # We need to pass through any layers before the base model (e.g., data augmentation, preprocessing)
    pre_base_layers = []
    for layer in loaded_model.layers:
        if layer == base_model_layer:
            break
        pre_base_layers.append(layer)

    # Process input through pre-base layers
    processed = input_tf
    for layer in pre_base_layers:
        if layer == loaded_model.layers[0]:  # Skip input layer
            continue
        processed = layer(processed, training=False)

    # Compute Grad-CAM with GradientTape
    with tf.GradientTape() as tape:
        conv_outputs = conv_output_model(processed, training=False)
        tape.watch(conv_outputs)
        predictions = head_model(conv_outputs, training=False)

        if target_class_idx is None:
            target_class_idx = int(tf.argmax(predictions[0]))
        class_score = predictions[:, target_class_idx]

    # Compute gradients
    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        return {"error": "Gradient computation failed", "tumor_detected": False}

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)  # ReLU
    heatmap = heatmap / (np.max(heatmap) + 1e-8)  # Normalize to [0, 1]

    # Resize heatmap to image dimensions (224x224)
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_img.resize((224, 224), resample=Image.Resampling.BILINEAR)
    heatmap_arr = np.array(heatmap_resized, dtype=np.float32) / 255.0

    # Compute tumor bounding box from thresholded heatmap
    threshold = 0.40
    binary_mask = (heatmap_arr > threshold).astype(np.uint8)
    y_indices, x_indices = np.where(binary_mask)

    tumor_detected = len(x_indices) > 0 and len(y_indices) > 0
    bounding_box = None
    anatomical_location = None

    if tumor_detected:
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        width = xmax - xmin
        height = ymax - ymin

        bounding_box = {
            "x_min": xmin, "x_max": xmax,
            "y_min": ymin, "y_max": ymax,
            "center_x": center_x, "center_y": center_y,
            "width": width, "height": height,
            "area_percent": round((width * height) / (224 * 224) * 100, 1)
        }

        anatomical_location = _get_anatomical_location(center_x, center_y)

    # Generate colorized heatmap overlay on original image
    original_img = Image.open(image_path).convert("RGB").resize((224, 224))
    original_arr = np.array(original_img, dtype=np.float32)

    # Create jet colormap heatmap
    heatmap_colored = _apply_jet_colormap(heatmap_arr)

    # Blend: overlay = alpha * heatmap + (1-alpha) * original
    alpha = 0.45
    overlay_arr = (alpha * heatmap_colored + (1.0 - alpha) * original_arr).astype(np.uint8)
    overlay_img = Image.fromarray(overlay_arr)

    # Draw bounding box and label on overlay
    if tumor_detected and bounding_box:
        draw = ImageDraw.Draw(overlay_img)
        # Draw bounding box rectangle
        box_color = (0, 255, 128)  # Bright green
        for offset in range(2):  # 2px thick border
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=box_color
            )
        # Draw crosshair at center
        cross_size = 6
        draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y], fill=(255, 255, 0), width=1)
        draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size], fill=(255, 255, 0), width=1)

        # Add label
        predicted_class = CLASS_NAMES[target_class_idx] if target_class_idx < len(CLASS_NAMES) else f"Class {target_class_idx}"
        label_text = f"{predicted_class}"
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except (IOError, OSError):
            font = ImageFont.load_default()
        
        # Draw label background
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        label_x = max(0, xmin)
        label_y = max(0, ymin - text_h - 6)
        draw.rectangle([label_x, label_y, label_x + text_w + 8, label_y + text_h + 4], fill=(0, 0, 0, 200))
        draw.text((label_x + 4, label_y + 2), label_text, fill=box_color, font=font)

    # Convert overlay to base64 PNG
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG", quality=95)
    buffer.seek(0)
    overlay_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Also generate a clean heatmap-only image (for reference)
    heatmap_only = Image.fromarray(_apply_jet_colormap(heatmap_arr).astype(np.uint8))
    heatmap_buffer = io.BytesIO()
    heatmap_only.save(heatmap_buffer, format="PNG")
    heatmap_buffer.seek(0)
    heatmap_b64 = base64.b64encode(heatmap_buffer.read()).decode("utf-8")

    return {
        "tumor_detected": tumor_detected,
        "heatmap_overlay_b64": overlay_b64,
        "heatmap_raw_b64": heatmap_b64,
        "bounding_box": bounding_box,
        "anatomical_location": anatomical_location,
        "target_class_idx": target_class_idx,
        "target_class_name": CLASS_NAMES[target_class_idx] if target_class_idx < len(CLASS_NAMES) else f"Class {target_class_idx}",
    }


def _apply_jet_colormap(heatmap_arr):
    """
    Applies a JET-style colormap to a normalized [0,1] 2D heatmap array.
    Returns an RGB float array (0-255).
    """
    h, w = heatmap_arr.shape
    colored = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            v = heatmap_arr[i, j]
            # JET colormap approximation
            if v < 0.125:
                r, g, b = 0, 0, 0.5 + v * 4
            elif v < 0.375:
                r, g, b = 0, (v - 0.125) * 4, 1.0
            elif v < 0.625:
                r, g, b = (v - 0.375) * 4, 1.0, 1.0 - (v - 0.375) * 4
            elif v < 0.875:
                r, g, b = 1.0, 1.0 - (v - 0.625) * 4, 0
            else:
                r, g, b = 1.0 - (v - 0.875) * 4, 0, 0

            colored[i, j] = [r * 255, g * 255, b * 255]

    return colored


def predict_tumor(image_path):
    """
    Executes real neural network inference on a given brain MRI image.
    Never hardcodes predictions.
    
    Args:
        image_path (str): File system path to the MRI scan image.
        
    Returns:
        dict: Prediction summary containing predicted class, confidence, probabilities, and disclaimer.
    """
    # 1. Load model (raises ModelNotFoundError if missing)
    loaded_model = load_prediction_model()

    # Automatically check if the model graph already contains internal normalization
    has_preprocess_layer = any(
        "preprocess" in getattr(layer, "name", "").lower() or "rescaling" in getattr(layer, "name", "").lower()
        for layer in getattr(loaded_model, "layers", [])
    )

    # 2. Preprocess and validate image (guaranteeing exact match with training)
    input_tensor = preprocess_image(image_path, apply_scaling=not has_preprocess_layer)

    # 3. Real inference through trained layers
    raw_preds = loaded_model.predict(input_tensor, verbose=0)[0]
    raw_preds = np.array(raw_preds, dtype=float)

    # Ensure valid probabilities via softmax if output layer was linear logits
    if np.any(raw_preds < 0) or not np.isclose(np.sum(raw_preds), 1.0, atol=0.05):
        exp_preds = np.exp(raw_preds - np.max(raw_preds))
        raw_preds = exp_preds / np.sum(exp_preds)

    best_idx = int(np.argmax(raw_preds))
    predicted_class = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else f"Class {best_idx}"
    confidence = float(raw_preds[best_idx] * 100.0)

    # 4-class probabilities mapping
    probabilities = {}
    for i, p in enumerate(raw_preds):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        probabilities[name] = round(float(p * 100.0), 2)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "details": CLINICAL_DETAILS.get(predicted_class, {}),
        "disclaimer": MEDICAL_DISCLAIMER
    }
