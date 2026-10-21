"""
NeuroScan AI: Brain Tumor MRI Classification Web Application Backend
Flask server handling real MRI predictions, secure file uploads, SQLite history, and CSV exports.
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from PIL import Image

import database
import model

# Flask App Configuration
app = Flask(__name__)
app.secret_key = "neuroscan_secret_key_production_grade"
app.config["UPLOAD_FOLDER"] = os.path.abspath(os.path.join(app.root_path, "static", "uploads"))
app.config["SAMPLES_FOLDER"] = os.path.abspath(os.path.join(app.root_path, "static", "samples"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Prominent Medical Disclaimer
MEDICAL_DISCLAIMER = (
    "MEDICAL DISCLAIMER: NeuroScan AI is an investigational decision-support tool "
    "intended for educational and scientific research purposes only. It is not approved "
    "as a primary diagnostic device. All deep learning outputs must be interpreted in "
    "conjunction with patient history, physical examination, and official board-certified radiologist review."
)

# Ensure required directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["SAMPLES_FOLDER"], exist_ok=True)
os.makedirs(os.path.abspath(os.path.join(app.root_path, "models")), exist_ok=True)
os.makedirs(os.path.abspath(os.path.join(app.root_path, "outputs")), exist_ok=True)

# Initialize SQLite database
database.init_db()

def allowed_file(filename):
    """Validate file extension against allowed formats."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_safe_path(base_dir, path):
    """Prevent directory traversal attacks by verifying path resolves within base_dir."""
    return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, os.path.abspath(path)])

def validate_image_file(file_storage_or_path):
    """
    Validates that the uploaded file contains genuine, non-corrupt image bytes using PIL.
    Returns (True, None) if valid, or (False, error_message).
    """
    try:
        with Image.open(file_storage_or_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid or corrupted image format: {str(e)}"

@app.context_processor
def inject_global_context():
    """Inject medical disclaimer into all templates automatically."""
    return {"medical_disclaimer": MEDICAL_DISCLAIMER}

@app.route("/")
def index():
    """Diagnostic Workstation Dashboard."""
    recent_scans = database.get_all_scans()[:5]
    stats = database.get_statistics()
    
    # Load available sample scans
    samples = []
    if os.path.exists(app.config["SAMPLES_FOLDER"]):
        for f in sorted(os.listdir(app.config["SAMPLES_FOLDER"])):
            if allowed_file(f):
                samples.append({
                    "filename": f,
                    "name": f.rsplit(".", 1)[0].replace("_", " ").title(),
                    "path": f"static/samples/{f}"
                })
                
    return render_template(
        "index.html",
        recent_scans=recent_scans,
        stats=stats,
        samples=samples,
        disclaimer=MEDICAL_DISCLAIMER
    )

@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint:
    - Validates file upload or sample selection
    - Performs secure file handling
    - Validates image integrity
    - Executes real neural model prediction
    - Logs results to SQLite
    - Returns 4-class probabilities, predicted class, and medical disclaimer
    """
    patient_name = request.form.get("patient_name", "").strip() or "Anonymous Case"
    sample_file = request.form.get("sample_file", "").strip()

    file_to_analyze = None
    saved_rel_path = None
    original_name = None

    # Case 1: Pre-packaged sample image selected
    if sample_file:
        safe_sample_name = secure_filename(sample_file)
        sample_path = os.path.join(app.config["SAMPLES_FOLDER"], safe_sample_name)
        
        if not is_safe_path(app.config["SAMPLES_FOLDER"], sample_path) or not os.path.exists(sample_path):
            return jsonify({
                "success": False,
                "error": "The specified sample image does not exist or is unauthorized.",
                "error_code": "INVALID_SAMPLE",
                "disclaimer": MEDICAL_DISCLAIMER
            }), 400
            
        file_to_analyze = sample_path
        saved_rel_path = f"static/samples/{safe_sample_name}"
        original_name = safe_sample_name

    # Case 2: New uploaded file
    elif "mri_image" in request.files:
        file = request.files["mri_image"]
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file was selected for upload.",
                "error_code": "NO_FILE_SELECTED",
                "disclaimer": MEDICAL_DISCLAIMER
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file format. Allowed extensions are: {', '.join(ALLOWED_EXTENSIONS).upper()}",
                "error_code": "INVALID_EXTENSION",
                "disclaimer": MEDICAL_DISCLAIMER
            }), 400

        # Validate genuine image bytes with PIL
        is_valid, validation_error = validate_image_file(file.stream)
        file.stream.seek(0)  # Rewind stream after inspection
        if not is_valid:
            return jsonify({
                "success": False,
                "error": f"Corrupted or invalid image: {validation_error}",
                "error_code": "CORRUPT_IMAGE",
                "disclaimer": MEDICAL_DISCLAIMER
            }), 400

        # Secure file saving with collision avoidance
        original_name = secure_filename(file.filename)
        unique_prefix = uuid.uuid4().hex[:10]
        unique_filename = f"{unique_prefix}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        if not is_safe_path(app.config["UPLOAD_FOLDER"], save_path):
            return jsonify({
                "success": False,
                "error": "Path traversal detected in upload filename.",
                "error_code": "UNSAFE_PATH",
                "disclaimer": MEDICAL_DISCLAIMER
            }), 400

        file.save(save_path)
        file_to_analyze = save_path
        saved_rel_path = f"static/uploads/{unique_filename}"

    else:
        return jsonify({
            "success": False,
            "error": "Please provide an MRI image upload or select a sample.",
            "error_code": "MISSING_INPUT",
            "disclaimer": MEDICAL_DISCLAIMER
        }), 400

    # Execute Prediction and Handle Errors
    try:
        prediction_result = model.predict_tumor(file_to_analyze)

        # Generate Grad-CAM heatmap for tumor localization
        gradcam_data = None
        try:
            gradcam_result = model.generate_gradcam(file_to_analyze)
            if gradcam_result and "error" not in gradcam_result:
                gradcam_data = gradcam_result
        except Exception as gc_err:
            print(f"[!] Grad-CAM generation warning (non-fatal): {gc_err}")

        # Log to SQLite
        scan_id = database.save_scan(
            patient_name=patient_name,
            filename=original_name,
            image_path=saved_rel_path,
            predicted_class=prediction_result["predicted_class"],
            confidence=prediction_result["confidence"],
            probabilities=prediction_result["probabilities"],
            clinical_notes=prediction_result["details"].get("precautions", "")
        )

        response_data = {
            "success": True,
            "scan_id": scan_id,
            "patient_name": patient_name,
            "image_url": url_for("static", filename=saved_rel_path.replace("static/", "")),
            "predicted_class": prediction_result["predicted_class"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result["probabilities"],
            "details": prediction_result["details"],
            "disclaimer": MEDICAL_DISCLAIMER
        }

        # Attach Grad-CAM localization data if available
        if gradcam_data:
            response_data["gradcam"] = {
                "tumor_detected": gradcam_data.get("tumor_detected", False),
                "heatmap_overlay": f"data:image/png;base64,{gradcam_data['heatmap_overlay_b64']}" if gradcam_data.get("heatmap_overlay_b64") else None,
                "heatmap_raw": f"data:image/png;base64,{gradcam_data['heatmap_raw_b64']}" if gradcam_data.get("heatmap_raw_b64") else None,
                "bounding_box": gradcam_data.get("bounding_box"),
                "anatomical_location": gradcam_data.get("anatomical_location"),
            }

        return jsonify(response_data)

    except model.ModelNotFoundError as mne:
        return jsonify({
            "success": False,
            "error": str(mne),
            "error_code": "MODEL_NOT_FOUND",
            "resolution": "Run 'python train.py --data_dir dataset/' to train the model, or place 'brain_tumor_model.keras' in models/.",
            "disclaimer": MEDICAL_DISCLAIMER
        }), 503

    except model.InvalidImageError as iie:
        return jsonify({
            "success": False,
            "error": str(iie),
            "error_code": "INVALID_IMAGE",
            "disclaimer": MEDICAL_DISCLAIMER
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}",
            "error_code": "INFERENCE_ERROR",
            "disclaimer": MEDICAL_DISCLAIMER
        }), 500

@app.route("/history")
def history():
    """View patient history logs from SQLite."""
    scans = database.get_all_scans()
    stats = database.get_statistics()
    return render_template("history.html", scans=scans, stats=stats, disclaimer=MEDICAL_DISCLAIMER)

@app.route("/history/export")
def export_history_csv():
    """Export the complete SQLite prediction history as a downloadable CSV."""
    csv_content = database.export_scans_to_csv()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_filename = f"neuroscan_history_{timestamp}.csv"

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={download_filename}",
            "Content-Type": "text/csv; charset=utf-8"
        }
    )

@app.route("/history/delete/<int:scan_id>", methods=["POST"])
def delete_scan(scan_id):
    """Delete a scan record from SQLite database."""
    success = database.delete_scan(scan_id)
    if success:
        flash("Record deleted successfully.", "success")
    else:
        flash("Record could not be found.", "danger")
    return redirect(url_for("history"))

@app.route("/performance")
def performance():
    """Model evaluation metrics, confusion matrix, and training curves."""
    metrics_path = os.path.join(app.root_path, "outputs", "metrics.json")
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Notice: Could not read {metrics_path}: {e}")
    return render_template(
        "performance.html",
        metrics=metrics_data,
        disclaimer=MEDICAL_DISCLAIMER
    )

@app.route("/about")
def about():
    """Pathology guide, architecture overview, and medical disclaimer."""
    return render_template(
        "about.html",
        clinical_info=model.CLINICAL_DETAILS,
        disclaimer=MEDICAL_DISCLAIMER
    )

@app.route("/api/stats")
def api_stats():
    """API endpoint providing current scan count statistics."""
    return jsonify(database.get_statistics())

@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    """
    Standalone Grad-CAM API endpoint.
    Accepts an image path (from a previously uploaded file) and returns
    heatmap overlay with tumor localization data.
    """
    image_path = request.json.get("image_path", "") if request.is_json else ""
    
    if not image_path:
        return jsonify({"success": False, "error": "No image path provided."}), 400
    
    # Resolve relative paths
    if not os.path.isabs(image_path):
        image_path = os.path.join(app.root_path, image_path)
    
    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": "Image file not found."}), 404
    
    if not is_safe_path(app.root_path, image_path):
        return jsonify({"success": False, "error": "Unauthorized file path."}), 403
    
    try:
        gradcam_result = model.generate_gradcam(image_path)
        if "error" in gradcam_result:
            return jsonify({"success": False, "error": gradcam_result["error"]}), 500
        
        return jsonify({
            "success": True,
            "tumor_detected": gradcam_result.get("tumor_detected", False),
            "heatmap_overlay": f"data:image/png;base64,{gradcam_result['heatmap_overlay_b64']}" if gradcam_result.get("heatmap_overlay_b64") else None,
            "heatmap_raw": f"data:image/png;base64,{gradcam_result['heatmap_raw_b64']}" if gradcam_result.get("heatmap_raw_b64") else None,
            "bounding_box": gradcam_result.get("bounding_box"),
            "anatomical_location": gradcam_result.get("anatomical_location"),
            "disclaimer": MEDICAL_DISCLAIMER
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Grad-CAM generation failed: {str(e)}"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle uploads exceeding MAX_CONTENT_LENGTH."""
    return jsonify({
        "success": False,
        "error": "File size exceeds the 16MB maximum limit.",
        "error_code": "FILE_TOO_LARGE",
        "disclaimer": MEDICAL_DISCLAIMER
    }), 413

if __name__ == "__main__":
    print("=======================================================")
    print(" NeuroScan AI: Brain Tumor MRI Classification Server")
    print(" Backend running on http://127.0.0.1:5000")
    print("=======================================================")
    app.run(debug=True, host="127.0.0.1", port=5000)
