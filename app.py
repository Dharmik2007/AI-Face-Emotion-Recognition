"""
Flask Web Application for Human Facial Expression Recognition using CNN.
BCA Semester 5 - Neural Networks Mini Project.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from utils.image_processing import (
    base64_to_image,
    check_face_quality,
    crop_and_preprocess_face,
    detect_faces,
    draw_face_annotations,
    image_to_base64,
    is_allowed_file,
)
from utils.prediction import (
    get_model_summary_info,
    is_model_available,
    predict_expression,
)
from utils.feature_extractor import extract_features_from_face
from utils.sequence_buffer import get_session_manager
from utils.lstm_prediction import (
    get_comparison_specs,
    get_lstm_summary_info,
    is_lstm_available,
    predict_temporal_sequence,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH


@app.context_processor
def inject_global_template_vars():
    """Inject configuration data accessible in all Jinja templates."""
    return {
        "project_name": config.PROJECT_NAME,
        "project_subtitle": config.PROJECT_SUBTITLE,
        "class_names": config.CLASS_NAMES,
        "class_colors": config.CLASS_COLORS,
        "class_descriptions": config.CLASS_DESCRIPTIONS,
        "disclaimer": config.ETHICAL_DISCLAIMER,
        "model_available": is_model_available(),
        "lstm_available": is_lstm_available(),
        "lstm_info": get_lstm_summary_info(),
        "sequence_length": config.SEQUENCE_LENGTH,
        "feature_dim": config.FEATURE_DIM,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "allowed_extensions": sorted(list(config.ALLOWED_EXTENSIONS)),
    }


@app.route("/", methods=["GET"])
def index():
    """
    Render main FaceSense AI application dashboard.
    Supports deep linking via query param ?tab=dashboard|webcam|upload|compare|performance.
    """
    active_tab = request.args.get("tab", "dashboard")
    model_info = get_model_summary_info()
    return render_template("index.html", model_info=model_info, active_tab=active_tab)


@app.route("/live", methods=["GET"])
def live_route():
    """Direct route for live webcam facial analysis."""
    model_info = get_model_summary_info()
    return render_template("index.html", model_info=model_info, active_tab="webcam")


@app.route("/upload", methods=["GET"])
def upload_route():
    """Direct route for image upload expression analysis."""
    model_info = get_model_summary_info()
    return render_template("index.html", model_info=model_info, active_tab="upload")


@app.route("/compare", methods=["GET"])
def compare_route():
    """Direct route for CNN vs CNN+LSTM spatiotemporal model comparison."""
    model_info = get_model_summary_info()
    return render_template("index.html", model_info=model_info, active_tab="compare")


@app.route("/performance", methods=["GET"])
def performance_route():
    """Direct route for empirical model performance and evaluation metrics."""
    model_info = get_model_summary_info()
    return render_template("index.html", model_info=model_info, active_tab="performance")


@app.route("/demo", methods=["GET"])
def demo_route():
    """Direct route for the Faculty Demonstration & Viva Presentation Guide."""
    return redirect(url_for("about"))


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle user image upload, validate file, detect face, preprocess,
    execute CNN inference, and display results.
    """
    # 1. Verify Model Availability
    if not is_model_available():
        error_msg = (
            "Model file not found! A trained neural network model is required. "
            "Please run 'python training/train.py' to train the model first."
        )
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": error_msg}), 503
        flash(error_msg, "error")
        return redirect(url_for("index"))

    # 2. Check File in Request
    if "image" not in request.files:
        error_msg = "No image file provided in upload request."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        error_msg = "No file selected. Please choose a valid image file."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    # 3. Validate File Extension
    safe_name = secure_filename(file.filename)
    if not is_allowed_file(safe_name):
        error_msg = "Invalid file format. Only JPG, JPEG, PNG, and WEBP images are supported."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    # 4. Decode Image from Memory
    try:
        file_bytes = file.read()
        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image. File may be corrupted.")
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    # 5. Face Detection via OpenCV
    faces = detect_faces(img_bgr)
    face_detected = len(faces) > 0
    primary_bbox = faces[0] if face_detected else None

    # Face Quality & Illumination check
    quality = check_face_quality(img_bgr, primary_bbox)

    # 6. Preprocessing (Crop face or fallback to full image)
    tensor, visual_crop = crop_and_preprocess_face(
        image_bgr=img_bgr,
        bbox=primary_bbox,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    )

    # 7. Model Inference
    try:
        pred_result = predict_expression(tensor)
    except Exception as e:
        error_msg = f"Neural network prediction error: {str(e)}"
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for("index"))

    # 8. Annotate Image with Bounding Box
    if face_detected:
        annotated_bgr = draw_face_annotations(
            img_bgr,
            faces=faces,
            label=pred_result["predicted_label"],
            confidence=pred_result["confidence"]
        )
    else:
        annotated_bgr = img_bgr.copy()

    # 9. Convert to Base64 for Inline Browser Display
    annotated_b64 = image_to_base64(annotated_bgr)
    crop_b64 = image_to_base64(visual_crop)

    # Return JSON for AJAX calls
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "status": "success",
            "face_detected": face_detected,
            "faces_count": len(faces),
            "bbox": primary_bbox,
            "predicted_label": pred_result["predicted_label"],
            "confidence": pred_result["confidence"],
            "is_uncertain": pred_result.get("is_uncertain", False),
            "uncertainty_note": pred_result.get("uncertainty_note", ""),
            "probabilities": pred_result["probabilities"],
            "color": pred_result["color"],
            "description": pred_result["description"],
            "quality": quality,
            "annotated_image": annotated_b64,
            "crop_image": crop_b64
        })

    # Render results page for standard form submit
    return render_template(
        "result.html",
        filename=safe_name,
        face_detected=face_detected,
        bbox=primary_bbox,
        faces_count=len(faces),
        predicted_label=pred_result["predicted_label"],
        confidence=pred_result["confidence"],
        is_uncertain=pred_result.get("is_uncertain", False),
        uncertainty_note=pred_result.get("uncertainty_note", ""),
        probabilities=pred_result["probabilities"],
        color=pred_result["color"],
        description=pred_result["description"],
        quality=quality,
        annotated_image=annotated_b64,
        crop_image=crop_b64
    )


@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    """
    Handle live webcam frames transmitted via AJAX.
    Supports Mode A (CNN), Mode B (CNN + LSTM), and Mode C (Compare Both).
    Includes temporal sliding buffer, multi-session management, quality diagnostics,
    and processing latency reporting.
    """
    t0 = time.time()

    if not is_model_available():
        return jsonify({
            "status": "error",
            "message": "CNN model not loaded. Please train the model first."
        }), 503

    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"status": "error", "message": "Missing webcam image frame."}), 400

    frame_b64 = data["frame"]
    img_bgr = base64_to_image(frame_b64)
    if img_bgr is None:
        return jsonify({"status": "error", "message": "Failed to decode webcam frame."}), 400

    # Session & mode configurations
    session_id = str(data.get("session_id", "default_session"))
    mode = str(data.get("mode", "cnn")).lower()
    if mode not in ("cnn", "lstm", "compare"):
        mode = "cnn"

    prev_probs = data.get("previous_probabilities")
    try:
        conf_thresh = float(data.get("confidence_threshold", config.CONFIDENCE_THRESHOLD))
    except (ValueError, TypeError):
        conf_thresh = config.CONFIDENCE_THRESHOLD

    smoothing_enabled = bool(data.get("enable_smoothing", True))
    if not smoothing_enabled:
        prev_probs = None

    session_manager = get_session_manager()
    seq_buffer = session_manager.get_buffer(session_id)

    # Face detection with robust neighbor clustering
    faces = detect_faces(img_bgr, min_neighbors=5)
    if not faces:
        seq_buffer.record_missing_frame()
        latency_ms = round((time.time() - t0) * 1000, 1)
        return jsonify({
            "status": "no_face",
            "face_detected": False,
            "message": "No frontal face detected in current frame. Center face in light.",
            "latency_ms": latency_ms,
            "mode": mode,
            "temporal_buffer": seq_buffer.get_progress_status()
        })

    primary_bbox = faces[0]
    quality = check_face_quality(img_bgr, primary_bbox)

    tensor, visual_crop = crop_and_preprocess_face(
        image_bgr=img_bgr,
        bbox=primary_bbox,
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    )

    all_bboxes = [{"x": f[0], "y": f[1], "w": f[2], "h": f[3]} for f in faces]

    # Feature extraction for temporal sequence buffering
    feature_vector = extract_features_from_face(tensor)
    seq_buffer.add_frame(feature_vector)
    buffer_status = seq_buffer.get_progress_status()

    # CNN Inference
    cnn_t0 = time.time()
    pred_result = predict_expression(
        tensor,
        confidence_threshold=conf_thresh,
        previous_probabilities=prev_probs
    )
    cnn_latency_ms = round((time.time() - cnn_t0) * 1000, 1)

    # LSTM Inference
    lstm_res = None
    if mode in ("lstm", "compare"):
        if seq_buffer.is_ready():
            sequence_tensor = seq_buffer.get_sequence()
            lstm_res = predict_temporal_sequence(sequence_tensor)
        else:
            lstm_res = {
                "status": "buffering",
                "trained": is_lstm_available(),
                "message": f"Collecting sequence: {buffer_status['current_frames']}/{buffer_status['sequence_length']} frames",
                "predicted_label": "Collecting Frames",
                "confidence": 0.0,
                "probabilities": {},
                "color": "#6366f1",
                "description": f"Accumulating temporal frames ({buffer_status['current_frames']}/{buffer_status['sequence_length']})...",
                "latency_ms": 0.0,
                "is_uncertain": True,
                "uncertainty_note": "Awaiting full temporal sequence window."
            }

    latency_ms = round((time.time() - t0) * 1000, 1)

    # MODE B: CNN + LSTM
    if mode == "lstm":
        primary_label = lstm_res.get("predicted_label") or pred_result["predicted_label"]
        primary_conf = lstm_res.get("confidence", 0.0) if lstm_res.get("status") == "success" else pred_result["confidence"]
        primary_color = lstm_res.get("color") or pred_result["color"]

        return jsonify({
            "status": "success",
            "mode": "lstm",
            "face_detected": True,
            "faces_count": len(faces),
            "all_bboxes": all_bboxes,
            "bbox": {
                "x": primary_bbox[0],
                "y": primary_bbox[1],
                "w": primary_bbox[2],
                "h": primary_bbox[3]
            },
            "predicted_label": primary_label,
            "confidence": primary_conf,
            "is_uncertain": lstm_res.get("is_uncertain", False),
            "uncertainty_note": lstm_res.get("uncertainty_note", ""),
            "probabilities": lstm_res.get("probabilities", {}) if lstm_res.get("status") == "success" else pred_result["probabilities"],
            "color": primary_color,
            "description": lstm_res.get("description", pred_result["description"]),
            "quality": quality,
            "latency_ms": latency_ms,
            "temporal_buffer": buffer_status,
            "lstm_status": lstm_res.get("status", "success"),
            "lstm_trained": lstm_res.get("trained", False),
            "lstm_message": lstm_res.get("message", "")
        })

    # MODE C: Compare Both
    if mode == "compare":
        return jsonify({
            "status": "success",
            "mode": "compare",
            "face_detected": True,
            "faces_count": len(faces),
            "all_bboxes": all_bboxes,
            "bbox": {
                "x": primary_bbox[0],
                "y": primary_bbox[1],
                "w": primary_bbox[2],
                "h": primary_bbox[3]
            },
            "predicted_label": pred_result["predicted_label"],
            "confidence": pred_result["confidence"],
            "is_uncertain": pred_result["is_uncertain"],
            "uncertainty_note": pred_result.get("uncertainty_note", ""),
            "probabilities": pred_result["probabilities"],
            "color": pred_result["color"],
            "description": pred_result["description"],
            "quality": quality,
            "latency_ms": latency_ms,
            "temporal_buffer": buffer_status,
            "comparison": {
                "cnn": {
                    "predicted_label": pred_result["predicted_label"],
                    "confidence": pred_result["confidence"],
                    "probabilities": pred_result["probabilities"],
                    "color": pred_result["color"],
                    "latency_ms": cnn_latency_ms,
                    "analysis_type": "Spatial Single-Frame"
                },
                "lstm": {
                    "status": lstm_res.get("status", "success"),
                    "trained": lstm_res.get("trained", False),
                    "predicted_label": lstm_res.get("predicted_label"),
                    "confidence": lstm_res.get("confidence", 0.0),
                    "probabilities": lstm_res.get("probabilities", {}),
                    "color": lstm_res.get("color", "#6366f1"),
                    "latency_ms": lstm_res.get("latency_ms", 0.0),
                    "analysis_type": f"Spatiotemporal ({config.SEQUENCE_LENGTH} Frames)",
                    "message": lstm_res.get("message", "")
                },
                "agreement": (
                    pred_result["predicted_label"] == lstm_res.get("predicted_label")
                    if lstm_res.get("status") == "success"
                    else None
                )
            }
        })

    # MODE A: Standard CNN
    return jsonify({
        "status": "success",
        "mode": "cnn",
        "face_detected": True,
        "faces_count": len(faces),
        "all_bboxes": all_bboxes,
        "bbox": {
            "x": primary_bbox[0],
            "y": primary_bbox[1],
            "w": primary_bbox[2],
            "h": primary_bbox[3]
        },
        "predicted_label": pred_result["predicted_label"],
        "confidence": pred_result["confidence"],
        "is_uncertain": pred_result["is_uncertain"],
        "uncertainty_note": pred_result.get("uncertainty_note", ""),
        "probabilities": pred_result["probabilities"],
        "color": pred_result["color"],
        "description": pred_result["description"],
        "quality": quality,
        "latency_ms": latency_ms,
        "temporal_buffer": buffer_status
    })


@app.route("/api/lstm_info", methods=["GET"])
def api_lstm_info():
    """Return CNN + LSTM model metadata and training status in JSON format."""
    return jsonify(get_lstm_summary_info())


@app.route("/api/compare_models", methods=["GET"])
def api_compare_models():
    """Return theoretical and empirical comparison specs between CNN and CNN+LSTM."""
    return jsonify(get_comparison_specs())


@app.route("/api/sequence_buffer_reset", methods=["POST"])
def api_sequence_buffer_reset():
    """Reset temporal sequence buffer for a specific client session."""
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "default_session"))
    session_manager = get_session_manager()
    buf = session_manager.get_buffer(session_id)
    buf.reset()
    return jsonify({
        "status": "success",
        "session_id": session_id,
        "message": "Temporal sequence buffer reset successfully."
    })


@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    """Return model summary and evaluation metrics in JSON format."""
    return jsonify(get_model_summary_info())


@app.route("/health", methods=["GET"])
def health():
    """System health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": is_model_available(),
        "classes": config.CLASS_NAMES,
        "cv2_version": cv2.__version__
    })


@app.route("/about", methods=["GET"])
def about():
    """Information page detailing project architecture, FER-2013 dataset, and BCA submission."""
    model_info = get_model_summary_info()
    return render_template("about.html", model_info=model_info)


# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    msg = "Uploaded file exceeds the maximum allowed size limit of 5 MB."
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"status": "error", "message": msg}), 413
    flash(msg, "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    print(f"Starting FaceSense AI Web App on http://127.0.0.1:5000")
    print(f"Model available: {is_model_available()}")
    app.run(host="127.0.0.1", port=5000, debug=False)
